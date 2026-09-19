"""Request an exhaustive, frozen Jev risk feature bank using existing API allowance.

Dry run: python scripts/build_jev_feature_table.py --dry-run
Build: python scripts/build_jev_feature_table.py --out runs/jev-features.json --env-file .env
Resume explicitly after a failure with --resume. There are no automatic retries.
This script never purchases credits or changes account billing settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from mario_play.envs.jev_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    MODEL,
    SCHEMA_VERSION,
    FeatureCase,
    canonical_json,
    case_bank,
    specification_sha256,
    validate_response,
)

API_URL = "https://api.typesafe.ai/v1/systemone"
HARD_MAX_CALLS = 3000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_api_key(env_file: Path | None) -> str:
    """Read one credential without executing shell syntax or printing its value."""
    value = os.environ.get("TYPESAFE_API_KEY", "")
    if not value and env_file is not None:
        for line in env_file.read_text().splitlines():
            name, separator, candidate = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() == "TYPESAFE_API_KEY":
                value = candidate.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
    if not value or any(char in value for char in "\n\r"):
        raise ValueError("Set TYPESAFE_API_KEY or provide an env file containing that key")
    return value


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _valid_usage(response: Any) -> dict[str, int] | None:
    if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
        return None
    usage = response["usage"]
    keys = ("input_tokens", "output_tokens")
    if any(
        isinstance(usage.get(key), bool) or not isinstance(usage.get(key), int) or usage[key] < 0
        for key in keys
    ):
        return None
    return {key: usage[key] for key in keys}


def request_case(case: FeatureCase, api_key: str, timeout: float) -> dict[str, Any]:
    """Make one HTTP attempt and return sanitized accounting even when validation fails."""
    started = time.perf_counter()
    result: dict[str, Any] = {
        "case_key": case.key,
        "started_at": _now(),
        "status": "error",
        "usage": None,
        "record": None,
    }
    request = Request(
        API_URL,
        data=canonical_json(case.request()),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with build_opener(_NoRedirects()).open(request, timeout=timeout) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("oversized_response")
        if api_key.encode() in raw:
            raise ValueError("credential_in_response")
        response = json.loads(raw)
        result["usage"] = _valid_usage(response)
        probability, usage = validate_response(response)
        answer = response["answers"]["risk"]
        # Preserve exact documented output values, excluding unneeded server metadata.
        raw_response = {
            "model": response["model"],
            "answers": {"risk": {"type": answer["type"], "noul": answer["noul"]}},
            "usage": usage,
        }
        result["record"] = {
            "feature": case.feature,
            "context": case.context(),
            "request": case.request(),
            "raw_response": raw_response,
            "probability": probability,
            "usage": usage,
        }
        result["status"] = "ok"
    except HTTPError as error:
        result["http_status"] = int(error.code)
        result["error_type"] = "http_error"
        error.close()
    except (URLError, TimeoutError, OSError):
        result["error_type"] = "connection_error"
    except (ValueError, TypeError, KeyError):
        result["error_type"] = "invalid_response"
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    return result


def _refresh_totals(data: dict[str, Any], bank: tuple[FeatureCase, ...]) -> None:
    records, attempts = data["records"], data["attempts"]
    data["coverage"] = {
        "expected_cases": len(bank),
        "completed_cases": len(records),
        "complete": len(records) == len(bank),
        "by_feature": {
            name: {
                "expected": sum(case.feature == name for case in bank),
                "completed": sum(record["feature"] == name for record in records.values()),
            }
            for name in FEATURE_NAMES
        },
    }
    data["usage"] = {
        "attempted_calls": len(attempts),
        "input_tokens": sum(
            attempt["usage"]["input_tokens"]
            for attempt in attempts
            if attempt.get("usage") is not None
        ),
        "output_tokens": sum(
            attempt["usage"]["output_tokens"]
            for attempt in attempts
            if attempt.get("usage") is not None
        ),
        "unaccounted_calls": sum(attempt.get("usage") is None for attempt in attempts),
    }
    data.setdefault("timing", {})["request_latency_ms_total"] = sum(
        attempt.get("latency_ms", 0.0) for attempt in attempts
    )
    data["updated_at"] = _now()


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _initial_table() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "model": MODEL,
        "feature_names": list(FEATURE_NAMES),
        "spec_sha256": specification_sha256(),
        "created_at": _now(),
        "records": {},
        "attempts": [],
        "provenance": {
            "source": "Actual Jev Noul API outputs; exhaustive direct cache; no learned surrogate",
            "endpoint": API_URL,
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "data_inputs": "Bucketed current grid geometry, motion, vertical phase and size only",
            "labels_excluded": "No actions, rewards, history, expert demonstrations "
            "or policy labels",
            "bucket_specification": "src/mario_play/envs/jev_features.py module documentation",
            "shared_precomputation": "Count these physical API calls once across all runs using "
            "this table; cached feature refreshes send no API requests",
        },
    }


def _validate_partial(data: dict[str, Any], bank: tuple[FeatureCase, ...]) -> None:
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("model") != MODEL
        or data.get("feature_schema") != FEATURE_SCHEMA
        or data.get("spec_sha256") != specification_sha256()
        or data.get("feature_names") != list(FEATURE_NAMES)
    ):
        raise ValueError("Existing table has a different schema, model or assessment specification")
    cases = {case.key: case for case in bank}
    if not isinstance(data.get("records"), dict) or not isinstance(data.get("attempts"), list):
        raise ValueError("Existing table has invalid records or accounting")
    if set(data["records"]) - set(cases):
        raise ValueError("Existing table has unknown assessment contexts")
    for key, record in data["records"].items():
        probability, usage = validate_response(record["raw_response"])
        case = cases[key]
        if (
            record.get("request") != case.request()
            or record.get("context") != case.context()
            or record.get("feature") != case.feature
            or record.get("probability") != probability
            or record.get("usage") != usage
        ):
            raise ValueError("Existing table record does not match its actual request and response")


def build_table(args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    """Build/resume with at most eight in-flight calls; an error stops new submissions."""
    bank = case_bank()
    if not 1 <= args.concurrency <= 8 or not 1 <= args.max_calls <= HARD_MAX_CALLS:
        raise ValueError("Concurrency must be 1-8 and max calls must be 1-3000")
    if not 0 < args.timeout <= 120:
        raise ValueError("Timeout must be greater than zero and at most 120 seconds")
    if args.out.exists():
        if not args.resume:
            raise ValueError(
                "Output already exists; use --resume to continue its recorded attempts"
            )
        data = json.loads(args.out.read_text())
        _validate_partial(data, bank)
        for attempt in data["attempts"]:
            if attempt.get("status") == "in_flight":
                attempt["status"] = "interrupted_unknown_result"
    else:
        data = _initial_table()
    _refresh_totals(data, bank)
    if len(data["attempts"]) > args.max_calls:
        raise ValueError("Existing attempted calls already exceed the requested cap")
    pending = iter(case for case in bank if case.key not in data["records"])
    prior_wall = data.get("timing", {}).get("builder_wall_seconds", 0.0)
    started = time.perf_counter()
    submitted = len(data["attempts"])
    stopped = False
    _atomic_write(args.out, data)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        running = {}

        def submit_next() -> None:
            nonlocal submitted
            if stopped or submitted >= args.max_calls:
                return
            case = next(pending, None)
            if case is not None:
                submitted += 1
                index = len(data["attempts"])
                data["attempts"].append(
                    {
                        "case_key": case.key,
                        "started_at": _now(),
                        "status": "in_flight",
                        "usage": None,
                        "latency_ms": 0.0,
                    }
                )
                _refresh_totals(data, bank)
                # Reserve durably before transmission so an interrupted run cannot reset its cap.
                _atomic_write(args.out, data)
                running[executor.submit(request_case, case, api_key, args.timeout)] = (case, index)

        for _ in range(args.concurrency):
            submit_next()
        while running:
            ready, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in ready:
                case, index = running.pop(future)
                try:
                    result = future.result()
                except Exception:
                    result = {
                        "case_key": case.key,
                        "status": "error",
                        "usage": None,
                        "record": None,
                        "error_type": "unexpected_client_error",
                        "latency_ms": 0.0,
                    }
                record = result.pop("record")
                data["attempts"][index] = result
                if record is not None:
                    data["records"][case.key] = record
                if result["status"] != "ok":
                    stopped = True
                _refresh_totals(data, bank)
                data["timing"]["builder_wall_seconds"] = prior_wall + time.perf_counter() - started
                _atomic_write(args.out, data)
                if len(data["attempts"]) % 25 == 0 or stopped or data["coverage"]["complete"]:
                    print(
                        json.dumps(
                            {
                                "completed": len(data["records"]),
                                "expected": len(bank),
                                "attempted_calls": len(data["attempts"]),
                                "input_tokens": data["usage"]["input_tokens"],
                                "stopped_after_error": stopped,
                            }
                        ),
                        flush=True,
                    )
            for _ in ready:
                submit_next()
    _refresh_totals(data, bank)
    data["timing"]["builder_wall_seconds"] = prior_wall + time.perf_counter() - started
    _atomic_write(args.out, data)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    bank = case_bank()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": MODEL,
                    "cases": len(bank),
                    "features": list(FEATURE_NAMES),
                    "cases_by_feature": {
                        name: sum(case.feature == name for case in bank) for name in FEATURE_NAMES
                    },
                    "spec_sha256": specification_sha256(),
                    "hard_max_calls": HARD_MAX_CALLS,
                    "api_calls_made": 0,
                },
                indent=2,
            )
        )
        return 0
    if args.out is None:
        parser.error("--out is required unless --dry-run is selected")
    try:
        data = build_table(args, _read_api_key(args.env_file))
    except (ValueError, OSError, KeyError, TypeError):
        print(
            "Feature-table build could not start or resume; check paths, credentials, "
            "schema and configured limits. No credential values are displayed."
        )
        return 2
    print(
        json.dumps(
            {
                "path": str(args.out),
                "coverage": data["coverage"],
                "usage": data["usage"],
                "timing": data["timing"],
            },
            indent=2,
        )
    )
    return 0 if data["coverage"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
