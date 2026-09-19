"""Offline feature-table coverage, provenance and bounded-builder regression tests.

Synthetic responses in these tests are explicitly fixtures; no API calls are made.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest

from mario_play.envs.jev_features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    MODEL,
    SCHEMA_VERSION,
    case_bank,
    cases_for_observation,
    load_table,
    specification_sha256,
)


@pytest.fixture
def observation():
    grid = np.zeros((14, 15, 16), dtype=np.float32)
    grid[0, 13:, :] = 1
    grid[7, 12, 4] = 1
    grid[8] = 0.6
    grid[10] = 1
    grid[13] = 1
    return grid


def synthetic_record(case, probability=0.37):
    usage = {"input_tokens": 10, "output_tokens": 2}
    return {
        "feature": case.feature,
        "context": case.context(),
        "request": case.request(),
        "probability": probability,
        "usage": usage,
        "raw_response": {
            "model": MODEL,
            "answers": {"risk": {"type": "noul", "noul": probability}},
            "usage": usage,
        },
    }


@pytest.fixture
def table_data():
    bank = case_bank()
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "model": MODEL,
        "feature_names": list(FEATURE_NAMES),
        "spec_sha256": specification_sha256(),
        "coverage": {"complete": True},
        "provenance": {"source": "SYNTHETIC OFFLINE TEST FIXTURE; NOT REAL JEV RESULTS"},
        "records": {
            case.key: synthetic_record(case, 0.01 + (index % 97) / 100)
            for index, case in enumerate(bank)
        },
    }


def test_exhaustive_bank_has_unique_action_free_risk_questions():
    bank = case_bank()
    assert len(bank) == 476
    assert len({case.key for case in bank}) == 476
    assert [sum(case.feature == name for case in bank) for name in FEATURE_NAMES] == [
        261,
        81,
        101,
        33,
    ]
    for case in bank:
        request = case.request()
        assert request["model"] == MODEL
        assert set(request["questions"]) == {"risk"}
        assert request["questions"]["risk"]["type"] == "noul"
        assert "criteria" not in request["questions"]["risk"]
        assert not set(case.context()) & {"action", "reward", "history", "label"}


def test_walk_speed_float32_boundary_remains_slow(observation):
    cases = cases_for_observation(observation)
    assert all(case.motion == "right_slow" for case in cases[:3])
    observation[8] = -0.6
    assert all(case.motion == "left_slow" for case in cases_for_observation(observation)[:3])
    observation[8] = np.nextafter(np.float32(0.6), np.float32(1))
    assert all(case.motion == "right_fast" for case in cases_for_observation(observation)[:3])


def test_spatial_cases_use_only_current_grid(observation):
    observation[3, 12, 6] = 1
    observation[0, 12, 8] = 1
    observation[0, 13:, 7] = 0
    observation[0, 9, 4] = 1
    cases = cases_for_observation(observation)
    assert [case.geometry for case in cases] == ["near_body", "far", "near", "near"]
    observation[3] = 0
    observation[3, 10, 5] = 1
    assert cases_for_observation(observation)[0].geometry == "adjacent_above"


def test_random_valid_grids_and_empty_player_have_covered_cases(observation):
    rng = np.random.default_rng(93)
    bank_keys = {case.key for case in case_bank()}
    for _ in range(80):
        grid = observation.copy()
        grid[:7] = rng.integers(0, 2, size=grid[:7].shape)
        grid[7] = 0
        row, col = int(rng.integers(0, 15)), int(rng.integers(0, 16))
        grid[7, row, col] = 1
        grid[8] = rng.uniform(-1, 1)
        grid[9] = rng.uniform(-1, 1)
        grid[10] = rng.integers(0, 2)
        grid[11] = rng.integers(0, 2)
        assert all(case.key in bank_keys for case in cases_for_observation(grid))
    observation[7] = 0
    terminal = cases_for_observation(observation)
    assert len(terminal) == FEATURE_COUNT
    assert all(case.geometry == "player_not_visible" and case.key in bank_keys for case in terminal)


def test_loaded_features_are_exact_cached_values_and_hash_verified(
    tmp_path, table_data, observation
):
    path = tmp_path / "synthetic-test-table.json"
    path.write_text(json.dumps(table_data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    table = load_table(path, digest)
    assert table is load_table(path, digest)
    values = table.features(observation)
    assert values.shape == (4,) and values.dtype == np.float32
    expected = [
        table_data["records"][case.key]["raw_response"]["answers"]["risk"]["noul"]
        for case in cases_for_observation(observation)
    ]
    np.testing.assert_array_equal(values, np.asarray(expected, dtype=np.float32))
    with pytest.raises(TypeError):
        table.probabilities[case_bank()[0].key] = 0
    observation[7] = 0
    assert np.all(table.features(observation) > 0), "terminal features must not be invented zeros"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        load_table(path, digest)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_case",
        "false_raw_output",
        "wrong_request",
        "wrong_schema",
        "incomplete",
        "invalid_probability",
    ],
)
def test_corrupt_or_incomplete_table_never_gets_fallback_values(tmp_path, table_data, fault):
    key = next(iter(table_data["records"]))
    record = table_data["records"][key]
    if fault == "missing_case":
        del table_data["records"][key]
    elif fault == "false_raw_output":
        record["probability"] = 0.99
    elif fault == "wrong_request":
        record["request"]["state"] = "unrelated prompt"
    elif fault == "wrong_schema":
        table_data["feature_schema"] = "different"
    elif fault == "incomplete":
        table_data["coverage"]["complete"] = False
    else:
        record["raw_response"]["answers"]["risk"]["noul"] = float("nan")
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(table_data))
    with pytest.raises(ValueError):
        load_table(path)


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_jev_feature_table.py"
    spec = importlib.util.spec_from_file_location("jev_feature_builder_offline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_args(path, *, max_calls=2, resume=False):
    return argparse.Namespace(
        out=path, concurrency=1, max_calls=max_calls, timeout=1, resume=resume
    )


def test_builder_durably_caps_requests_and_resumes_without_repeating_success(
    tmp_path, builder, monkeypatch
):
    cases = case_bank()[:3]
    monkeypatch.setattr(builder, "case_bank", lambda: cases)
    path = tmp_path / "partial.json"
    requested = []

    def offline_request(case, api_key, timeout):
        disk = json.loads(path.read_text())
        assert any(
            attempt["case_key"] == case.key and attempt["status"] == "in_flight"
            for attempt in disk["attempts"]
        ), "reserve must reach disk before API dispatch"
        requested.append(case.key)
        record = synthetic_record(case)
        return {
            "case_key": case.key,
            "status": "ok",
            "record": record,
            "usage": record["usage"],
            "latency_ms": 5.0,
        }

    monkeypatch.setattr(builder, "request_case", offline_request)
    partial = builder.build_table(build_args(path), "OFFLINE_TEST_ONLY")
    assert partial["usage"]["attempted_calls"] == 2
    assert len(requested) == 2 and partial["coverage"]["complete"] is False
    complete = builder.build_table(build_args(path, max_calls=3, resume=True), "OFFLINE_TEST_ONLY")
    assert complete["coverage"]["complete"] is True
    assert requested == [case.key for case in cases]
    assert complete["usage"] == {
        "attempted_calls": 3,
        "input_tokens": 30,
        "output_tokens": 6,
        "unaccounted_calls": 0,
    }
    assert complete["timing"]["request_latency_ms_total"] == 15


def test_builder_stops_on_failure_and_counts_unknown_response(tmp_path, builder, monkeypatch):
    monkeypatch.setattr(builder, "case_bank", lambda: case_bank()[:3])
    calls = []

    def offline_failure(case, api_key, timeout):
        calls.append(case.key)
        return {
            "case_key": case.key,
            "status": "error",
            "record": None,
            "usage": None,
            "latency_ms": 4.0,
            "error_type": "connection_error",
        }

    monkeypatch.setattr(builder, "request_case", offline_failure)
    data = builder.build_table(build_args(tmp_path / "failed.json"), "OFFLINE_TEST_ONLY")
    assert len(calls) == 1 and data["usage"]["attempted_calls"] == 1
    assert data["usage"]["unaccounted_calls"] == 1
    assert data["coverage"]["complete"] is False


def test_dry_run_never_reads_credentials_or_calls_api(builder, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not access credentials or network")

    monkeypatch.setattr(builder, "_read_api_key", forbidden)
    monkeypatch.setattr(builder, "request_case", forbidden)
    assert builder.main(["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["api_calls_made"] == 0


def test_http_failure_has_no_credential_or_server_body(builder, monkeypatch):
    class FakeOpener:
        def open(self, request, **kwargs):
            raise HTTPError(builder.API_URL, 401, "secret_server_message", {}, None)

    monkeypatch.setattr(builder, "build_opener", lambda *_: FakeOpener())
    result = builder.request_case(case_bank()[0], "SECRET_TEST_KEY", 1)
    serialized = json.dumps(result)
    assert result["http_status"] == 401 and result["record"] is None
    assert "secret_server_message" not in serialized and "SECRET_TEST_KEY" not in serialized
