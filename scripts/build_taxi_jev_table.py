"""Freeze actual Jev assessments for all 500 Taxi-v4 states; no training or purchases.

Use --dry-run to inspect coverage without reading credentials or calling the API.
Actual construction requires --out and TYPESAFE_API_KEY or --env-file. Existing
files are never overwritten without explicit --resume. Failed calls never retry
automatically, and every submitted call is reserved durably before transmission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from mario_play.envs import taxi_jev

# Share already-tested HTTP credential parsing and atomic persistence without
# mutating the Mario builder. Loading definitions does not execute its CLI.
_SHARED_PATH = Path(__file__).with_name("build_jev_feature_table.py")
_SPEC = importlib.util.spec_from_file_location("_jev_builder_io", _SHARED_PATH)
_IO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IO)
API_URL = "https://api.typesafe.ai/v1/systemone"
HARD_MAX_CALLS = 1500


def request_state(state: int, api_key: str, timeout: float) -> dict[str, Any]:
    """One HTTP request containing four assessments, with sanitized accounting on failures."""
    started = time.perf_counter()
    result: dict[str, Any] = {
        "state_id": state,
        "started_at": _IO._now(),
        "status": "error",
        "usage": None,
        "record": None,
    }
    payload = taxi_jev.request_for_state(state)
    request = Request(
        API_URL,
        data=taxi_jev.canonical_json(payload),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with build_opener(_IO._NoRedirects()).open(request, timeout=timeout) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576 or api_key.encode() in raw:
            raise ValueError("Unsafe or oversized response")
        response = json.loads(raw)
        result["usage"] = _IO._valid_usage(response)
        probabilities, usage = taxi_jev.validate_response(response)
        result["record"] = {
            "state_id": state,
            "request": payload,
            "raw_response": response,
            "probabilities": probabilities,
            "usage": usage,
        }
        result["status"] = "ok"
    except HTTPError as error:
        result["error_type"] = "http_error"
        result["http_status"] = int(error.code)
        error.close()
    except (URLError, OSError, TimeoutError):
        result["error_type"] = "connection_error"
    except (ValueError, TypeError, KeyError):
        result["error_type"] = "invalid_response"
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    return result


def initial_table() -> dict[str, Any]:
    """Describe immutable prompts, source files and independently accounted Taxi requests."""
    return {
        "schema_version": taxi_jev.SCHEMA_VERSION,
        "feature_schema": taxi_jev.FEATURE_SCHEMA,
        "model": taxi_jev.MODEL,
        "feature_names": list(taxi_jev.FEATURE_NAMES),
        "spec_sha256": taxi_jev.specification_sha256(),
        "created_at": _IO._now(),
        "records": {},
        "attempts": [],
        "provenance": {
            "source": "Actual Jev API Noul outputs for all 500 Taxi states, "
            "four questions per call",
            "endpoint": API_URL,
            "base_environment": "Taxi-v4 default deterministic map",
            "gymnasium_version": taxi_jev.gym.__version__,
            "source_sha256": {
                str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (
                    Path(__file__).resolve(),
                    _SHARED_PATH.resolve(),
                    Path(taxi_jev.__file__).resolve(),
                )
            },
            "inputs": "Current decoded taxi row, column, passenger location, destination "
            "and the same static map facts exposed to every PPO arm",
            "controls": "Same four quantities have exact inexpensive rule definitions; "
            "compare Jev features with rules and zero auxiliary values",
            "excluded": "No actions, action masks, rewards, history, optimal policies, "
            "shortest paths, future simulation or expert demonstrations",
            "accounting": "Taxi physical API requests are separate from the Mario bank. "
            "All Taxi training/evaluation feature lookups use this frozen cache.",
        },
    }


def refresh_totals(data: dict[str, Any]) -> None:
    """Recompute accounting from durable attempts, never from guessed token counts."""
    attempts = data["attempts"]
    data["coverage"] = {
        "expected_states": 500,
        "completed_states": len(data["records"]),
        "questions_per_state": 4,
        "expected_assessments": 2000,
        "complete": set(data["records"]) == {str(state) for state in range(500)},
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
    data["updated_at"] = _IO._now()


def validate_partial(data: dict[str, Any]) -> None:
    """Resume only the same frozen assessment definition and recorded actual outputs."""
    if (
        data.get("schema_version") != taxi_jev.SCHEMA_VERSION
        or data.get("feature_schema") != taxi_jev.FEATURE_SCHEMA
        or data.get("model") != taxi_jev.MODEL
        or data.get("feature_names") != list(taxi_jev.FEATURE_NAMES)
        or data.get("spec_sha256") != taxi_jev.specification_sha256()
        or not isinstance(data.get("records"), dict)
        or not isinstance(data.get("attempts"), list)
    ):
        raise ValueError("Taxi table has different schema, model, prompts or accounting")
    for key, record in data["records"].items():
        state = int(key)
        if key != str(state):
            raise ValueError("Taxi table has invalid state IDs")
        probabilities, usage = taxi_jev.validate_response(record["raw_response"])
        if (
            record.get("state_id") != state
            or record.get("request") != taxi_jev.request_for_state(state)
            or record.get("probabilities") != probabilities
            or record.get("usage") != usage
        ):
            raise ValueError("Taxi table record differs from its actual request/response")


def build_table(args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    """Populate all states with bounded concurrent calls and explicit resumable failures."""
    if not 1 <= args.concurrency <= 8 or not 1 <= args.max_calls <= HARD_MAX_CALLS:
        raise ValueError("Taxi concurrency must be 1-8 and max calls must be 1-1500")
    if not 0 < args.timeout <= 120:
        raise ValueError("Timeout must be positive and at most 120 seconds")
    if args.out.exists():
        if not args.resume:
            raise ValueError("Taxi output exists; explicit --resume is required")
        data = json.loads(args.out.read_text())
        validate_partial(data)
        for attempt in data["attempts"]:
            if attempt.get("status") == "in_flight":
                attempt["status"] = "interrupted_unknown_result"
    else:
        data = initial_table()
    if len(data["attempts"]) > args.max_calls:
        raise ValueError("Taxi existing attempted calls exceed the requested cap")
    pending = iter(state for state in range(500) if str(state) not in data["records"])
    started = time.perf_counter()
    prior_wall = data.get("timing", {}).get("builder_wall_seconds", 0.0)
    stopped = False
    refresh_totals(data)
    _IO._atomic_write(args.out, data)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        running = {}

        def submit_next() -> None:
            if stopped or len(data["attempts"]) >= args.max_calls:
                return
            state = next(pending, None)
            if state is None:
                return
            index = len(data["attempts"])
            data["attempts"].append(
                {
                    "state_id": state,
                    "started_at": _IO._now(),
                    "status": "in_flight",
                    "usage": None,
                    "latency_ms": 0.0,
                }
            )
            refresh_totals(data)
            _IO._atomic_write(args.out, data)
            running[executor.submit(request_state, state, api_key, args.timeout)] = (state, index)

        for _ in range(args.concurrency):
            submit_next()
        while running:
            ready, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in ready:
                state, index = running.pop(future)
                try:
                    result = future.result()
                except Exception:
                    result = {
                        "state_id": state,
                        "status": "error",
                        "record": None,
                        "usage": None,
                        "error_type": "unexpected_client_error",
                        "latency_ms": 0.0,
                    }
                record = result.pop("record")
                data["attempts"][index] = result
                if record is not None:
                    data["records"][str(state)] = record
                if result["status"] != "ok":
                    stopped = True
                refresh_totals(data)
                data["timing"]["builder_wall_seconds"] = prior_wall + time.perf_counter() - started
                _IO._atomic_write(args.out, data)
                if len(data["records"]) % 25 == 0 or stopped or data["coverage"]["complete"]:
                    print(
                        json.dumps(
                            {
                                "completed_states": len(data["records"]),
                                "expected": 500,
                                "attempted_calls": len(data["attempts"]),
                                "input_tokens": data["usage"]["input_tokens"],
                                "stopped_after_error": stopped,
                            }
                        ),
                        flush=True,
                    )
            for _ in ready:
                submit_next()
    refresh_totals(data)
    data["timing"]["builder_wall_seconds"] = prior_wall + time.perf_counter() - started
    _IO._atomic_write(args.out, data)
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
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": taxi_jev.MODEL,
                    "states": 500,
                    "assessments": 2000,
                    "questions_per_request": 4,
                    "expected_physical_api_calls": 500,
                    "features": taxi_jev.FEATURE_NAMES,
                    "observation_size": taxi_jev.OBSERVATION_SIZE,
                    "valid_start_states": len(taxi_jev.valid_start_states()),
                    "spec_sha256": taxi_jev.specification_sha256(),
                    "api_calls_made": 0,
                },
                indent=2,
            )
        )
        return 0
    if args.out is None:
        parser.error("--out is required except for --dry-run")
    try:
        data = build_table(args, _IO._read_api_key(args.env_file))
    except (ValueError, OSError, KeyError, TypeError):
        print(
            "Taxi table build could not start/resume; check paths, limits and matching schema. "
            "Credentials and server error bodies are not displayed."
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
