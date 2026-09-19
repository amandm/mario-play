"""All-state Taxi observation proofs and offline-only API accounting tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from mario_play.envs.taxi_jev import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    MODEL,
    OBSERVATION_SIZE,
    RAW_FEATURE_COUNT,
    SCHEMA_VERSION,
    TaxiJevEnv,
    decode_state,
    encode_state,
    load_table,
    make_observation_table,
    request_for_state,
    rule_features,
    specification_sha256,
    valid_start_states,
)


def synthetic_record(state):
    """Explicitly synthetic fixture, never presented as real model output."""
    probabilities = [0.13 + ((state + index) % 7) / 10 for index in range(4)]
    usage = {"input_tokens": 10, "output_tokens": 2}
    response = {
        "model": MODEL,
        "answers": {
            name: {"type": "noul", "noul": value}
            for name, value in zip(FEATURE_NAMES, probabilities, strict=True)
        },
        "usage": usage,
    }
    return {
        "state_id": state,
        "request": request_for_state(state),
        "raw_response": response,
        "probabilities": probabilities,
        "usage": usage,
    }


@pytest.fixture
def table_data():
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "model": MODEL,
        "feature_names": list(FEATURE_NAMES),
        "spec_sha256": specification_sha256(),
        "coverage": {"complete": True},
        "provenance": {"source": "SYNTHETIC OFFLINE TEST FIXTURE; NOT REAL JEV RESULTS"},
        "records": {str(state): synthetic_record(state) for state in range(500)},
    }


@pytest.fixture
def table_file(tmp_path, table_data):
    path = tmp_path / "synthetic-taxi-table.json"
    path.write_text(json.dumps(table_data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_500_states_and_300_starts_match_gymnasium_public_encoding():
    env = gym.make("Taxi-v4")
    for state in range(500):
        assert decode_state(state) == tuple(env.unwrapped.decode(state))
        assert encode_state(*decode_state(state)) == state
        request = request_for_state(state)
        assert set(request["state"]) == {"taxi", "passenger_location", "destination", "map"}
        assert set(request["questions"]) == set(FEATURE_NAMES)
        assert all(question["type"] == "noul" for question in request["questions"].values())
    valid = valid_start_states()
    assert len(valid) == len(set(valid)) == 300
    np.testing.assert_array_equal(valid, np.flatnonzero(env.unwrapped.initial_state_distrib))
    env.close()


def test_pickup_dropoff_rule_truth_matches_actual_gym_transitions_all_states():
    env = gym.make("Taxi-v4")
    for state in range(500):
        _, _, passenger, _ = decode_state(state)
        pickup_transition = env.unwrapped.P[state][4][0]
        dropoff_transition = env.unwrapped.P[state][5][0]
        features = rule_features(state)
        legal_pickup = passenger != 4 and decode_state(pickup_transition[1])[2] == 4
        assert bool(features[0]) == legal_pickup
        assert bool(features[1]) == (dropoff_transition[2] == 20)
    env.close()


@pytest.mark.parametrize(
    "state,nearby,blocked",
    [
        (encode_state(0, 2, 0, 1), True, True),
        (encode_state(0, 3, 0, 1), False, True),
        (encode_state(2, 0, 0, 1), True, False),
        (encode_state(0, 0, 0, 1), True, False),
        (encode_state(4, 4, 4, 3), True, False),
        (encode_state(4, 2, 4, 3), True, True),
    ],
)
def test_geometry_features_have_exact_threshold_and_wall_semantics(state, nearby, blocked):
    values = rule_features(state)
    assert bool(values[2]) == nearby
    assert bool(values[3]) == blocked


def test_all_arms_receive_identical_raw_facts_and_actual_cached_auxiliary_values(table_file):
    path, digest = table_file
    table = load_table(path, digest)
    assert table is load_table(path, digest)
    observations = {mode: make_observation_table(mode, table) for mode in ("zeros", "rules", "jev")}
    for values in observations.values():
        assert values.shape == (500, OBSERVATION_SIZE) and values.dtype == np.float32
        assert not values.flags.writeable
        np.testing.assert_array_equal(
            values[:, :RAW_FEATURE_COUNT], observations["zeros"][:, :RAW_FEATURE_COUNT]
        )
    for state in range(500):
        np.testing.assert_array_equal(observations["jev"][state, -4:], table.features(state))
        np.testing.assert_array_equal(observations["rules"][state, -4:], rule_features(state))
    assert np.all(observations["zeros"][:, -4:] == 0)
    with pytest.raises(ValueError):
        table.probabilities.setflags(write=True)
    with pytest.raises(ValueError):
        observations["jev"].setflags(write=True)
    with pytest.raises(ValueError, match="SHA256"):
        load_table(path, "wrong")


@pytest.mark.parametrize("mode", ["zeros", "rules", "jev"])
def test_reset_encodes_every_state_and_never_exposes_masks(mode, table_file):
    path, digest = table_file
    env = TaxiJevEnv(feature_mode=mode, table_path=path, table_sha256=digest)
    for state in range(500):
        obs, info = env.reset(seed=123, options={"state": state})
        assert env.unwrapped.s == state
        assert "action_mask" not in info
        np.testing.assert_array_equal(obs, env.observation_table[state])
        assert env.observation_space.contains(obs)
    assert env.feature_refreshes == (0 if mode == "zeros" else 500)
    assert env.physical_api_calls == 0
    env.close()


def test_terminal_observation_and_time_limit_preserve_gym_semantics(table_file):
    path, digest = table_file
    env = TaxiJevEnv(feature_mode="jev", table_path=path, table_sha256=digest)
    env.reset(seed=0, options={"state": encode_state(0, 0, 4, 0)})
    obs, reward, terminated, truncated, info = env.step(5)
    assert reward == 20 and terminated is True and truncated is False
    np.testing.assert_array_equal(obs, env.observation_table[encode_state(0, 0, 0, 0)])
    assert "action_mask" not in info
    env.reset(seed=0, options={"state": encode_state(2, 4, 0, 1)})
    for _ in range(199):
        _, reward, terminated, truncated, _ = env.step(2)
        assert reward == -1 and not terminated and not truncated
    _, reward, terminated, truncated, _ = env.step(2)
    assert reward == -1 and not terminated and truncated
    env.close()


def test_custom_registered_environment_factory_contract():
    env = gym.make("mario_play.envs.taxi_jev:TaxiJev-v0", feature_mode="rules", render_mode=None)
    observation, info = env.reset(seed=32)
    assert observation.shape == (71,) and "action_mask" not in info
    assert env.action_space.n == 6
    env.close()


@pytest.mark.parametrize(
    "fault", ["missing", "changed_probability", "wrong_request", "wrong_model"]
)
def test_invalid_table_cannot_silently_supply_features(tmp_path, table_data, fault):
    record = table_data["records"]["0"]
    if fault == "missing":
        del table_data["records"]["499"]
    elif fault == "changed_probability":
        record["probabilities"][0] = 0.99
    elif fault == "wrong_request":
        record["request"]["state"] = {"answer": True}
    else:
        record["raw_response"]["model"] = "other-model"
    path = tmp_path / "broken-table.json"
    path.write_text(json.dumps(table_data))
    with pytest.raises(ValueError):
        load_table(path)


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_taxi_jev_table.py"
    spec = importlib.util.spec_from_file_location("taxi_builder_offline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_never_reads_credentials_or_calls_api(builder, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not access credentials or network")

    monkeypatch.setattr(builder._IO, "_read_api_key", forbidden)
    monkeypatch.setattr(builder, "request_state", forbidden)
    assert builder.main(["--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["api_calls_made"] == 0 and result["expected_physical_api_calls"] == 500


def test_builder_reserves_requests_before_transmission_and_resumes_without_duplicate_success(
    tmp_path, builder, monkeypatch
):
    path = tmp_path / "partial.json"
    calls = []

    def offline_request(state, api_key, timeout):
        disk = json.loads(path.read_text())
        assert any(
            attempt["state_id"] == state and attempt["status"] == "in_flight"
            for attempt in disk["attempts"]
        )
        calls.append(state)
        record = synthetic_record(state)
        return {
            "state_id": state,
            "status": "ok",
            "record": record,
            "usage": record["usage"],
            "latency_ms": 3.0,
        }

    monkeypatch.setattr(builder, "request_state", offline_request)
    args = argparse.Namespace(out=path, concurrency=1, max_calls=2, timeout=1, resume=False)
    partial = builder.build_table(args, "OFFLINE_TEST_ONLY")
    assert calls == [0, 1] and partial["usage"]["attempted_calls"] == 2
    args.max_calls = 3
    args.resume = True
    resumed = builder.build_table(args, "OFFLINE_TEST_ONLY")
    assert calls == [0, 1, 2]
    assert resumed["usage"] == {
        "attempted_calls": 3,
        "input_tokens": 30,
        "output_tokens": 6,
        "unaccounted_calls": 0,
    }
    assert not resumed["coverage"]["complete"]


def test_builder_stops_new_requests_after_error(tmp_path, builder, monkeypatch):
    def offline_failure(state, api_key, timeout):
        return {
            "state_id": state,
            "status": "error",
            "record": None,
            "usage": None,
            "latency_ms": 1.0,
            "error_type": "connection_error",
        }

    monkeypatch.setattr(builder, "request_state", offline_failure)
    args = argparse.Namespace(
        out=tmp_path / "failed.json", concurrency=1, max_calls=500, timeout=1, resume=False
    )
    result = builder.build_table(args, "OFFLINE_TEST_ONLY")
    assert result["usage"]["attempted_calls"] == 1
    assert result["usage"]["unaccounted_calls"] == 1
