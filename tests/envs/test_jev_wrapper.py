"""Offline integration proofs for frozen advice, cadence, and final observations."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from mario_play.envs.factory import make_env
from mario_play.envs.jev_features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    MODEL,
    SCHEMA_VERSION,
    case_bank,
    load_table,
    specification_sha256,
)
from mario_play.envs.jev_wrapper import JevFeatureWrapper
from mario_play.envs.mario_env import MarioEnv
from mario_play.rl.config import EnvConfig, config_from_dict, config_to_dict
from mario_play.rl.vec_env import SyncVecEnv


@pytest.fixture
def fake_table(tmp_path):
    """A complete, explicitly artificial table for offline tests; never a training artifact."""
    bank = case_bank()
    records = {}
    for index, case in enumerate(bank):
        probability = (index + 1) / (len(bank) + 1)
        usage = {"input_tokens": 10, "output_tokens": 1}
        records[case.key] = {
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
    data = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "spec_sha256": specification_sha256(),
        "coverage": {"complete": True},
        "records": records,
        "provenance": "ARTIFICIAL OFFLINE TEST FIXTURE; NOT JEV OUTPUT",
    }
    path = tmp_path / "artificial-table.json"
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    return str(path), hashlib.sha256(raw).hexdigest()


def table_config(fake_table, **kwargs):
    path, digest = fake_table
    return EnvConfig(
        level="flat",
        jev_features_mode="table",
        jev_features_path=path,
        jev_features_sha256=digest,
        **kwargs,
    )


def assert_features(observation, values):
    expected = np.broadcast_to(np.asarray(values)[:, None, None], (FEATURE_COUNT, 15, 16))
    np.testing.assert_array_equal(observation[14:], expected)


def test_real_factory_preserves_grid_and_matches_zero_control_shape(fake_table):
    assisted = make_env(table_config(fake_table), seed=0)
    control = make_env(EnvConfig(level="flat", jev_features_mode="zeros"), seed=0)
    plain = make_env(EnvConfig(level="flat"), seed=0)
    try:
        actual, info = assisted.reset(seed=0)
        zero, zero_info = control.reset(seed=0)
        raw, _ = plain.reset(seed=0)
        assert actual.shape == zero.shape == (14 + FEATURE_COUNT, 15, 16)
        assert assisted.observation_space == control.observation_space
        assert actual.dtype == zero.dtype == np.float32
        assert assisted.observation_space.contains(actual)
        np.testing.assert_array_equal(actual[:14], raw)
        np.testing.assert_array_equal(zero[:14], raw)
        assert_features(actual, load_table(*fake_table).features(raw))
        assert_features(zero, np.zeros(FEATURE_COUNT, dtype=np.float32))
        assert info["jev_features_age"] == 0 and info["jev_features_refreshes"] == 1
        assert zero_info["jev_features_refreshes"] == 0
        actual[:] = -1  # Returned arrays must not corrupt cached advice or raw game state.
        next_obs, *_ = assisted.step(1)
        assert assisted.observation_space.contains(next_obs)
        np.testing.assert_array_equal(next_obs[:14], assisted.unwrapped.observe())
    finally:
        assisted.close()
        control.close()
        plain.close()


def test_sparse_advice_is_held_then_refreshed_and_reset_clears_age(fake_table):
    env = make_env(table_config(fake_table, jev_features_interval=3))
    try:
        initial, _ = env.reset(seed=0)
        first, _, _, _, info = env.step(3)
        assert info["jev_features_age"] == 1 and info["jev_features_refreshes"] == 1
        np.testing.assert_array_equal(first[14:], initial[14:])
        assert not np.array_equal(first[:14], initial[:14])
        second, _, _, _, info = env.step(3)
        np.testing.assert_array_equal(second[14:], initial[14:])
        assert info["jev_features_age"] == 2
        third, _, _, _, info = env.step(3)
        assert info["jev_features_age"] == 0 and info["jev_features_refreshes"] == 2
        assert_features(third, load_table(*fake_table).features(third[:14]))
        assert not np.array_equal(third[14:], initial[14:])
        reset, info = env.reset(seed=0)
        np.testing.assert_array_equal(reset, initial)
        assert info["jev_features_age"] == 0 and info["jev_features_refreshes"] == 3
        assert info["jev_features_episode_refreshes"] == 1
        assert env.get_wrapper_attr("jev_features_refreshes") == 3
    finally:
        env.close()


def test_truncated_final_observation_is_enriched_before_vector_auto_reset(fake_table):
    cfg = table_config(fake_table, max_episode_steps=2)
    with SyncVecEnv([partial(make_env, cfg)]) as vec:
        initial = vec.reset(seed=77)
        vec.step(np.array([3]))
        transition = vec.step(np.array([3]))
        assert transition.truncated.tolist() == [True]
        assert transition.terminated.tolist() == [False]
        assert transition.final_obs.shape == initial.shape == (1, 18, 15, 16)
        assert_features(
            transition.final_obs[0], load_table(*fake_table).features(transition.final_obs[0, :14])
        )
        np.testing.assert_array_equal(transition.obs, initial)
        assert not np.array_equal(transition.final_obs, transition.obs)
        assert transition.infos[0]["jev_features_refreshes"] == 3
        assert vec.envs[0].get_wrapper_attr("jev_features_refreshes") == 4


def test_pit_terminal_observation_can_have_no_visible_player(fake_table, tmp_path):
    level = tmp_path / "pit.txt"
    level.write_text(
        "; time=100\n"
        "....................F...\n"
        "....................F...\n"
        "..S.................F...\n"
        "####....################\n"
        "####....################\n",
        encoding="utf-8",
    )
    cfg = table_config(fake_table)
    cfg.level = str(level)
    env = make_env(cfg)
    try:
        env.reset(seed=0)
        for _ in range(300):
            obs, _, terminated, truncated, info = env.step(1)
            if terminated or truncated:
                break
        assert terminated and not truncated and info["death_cause"] == "pit"
        assert env.observation_space.contains(obs)
        assert_features(obs, load_table(*fake_table).features(obs[:14]))
    finally:
        env.close()


def test_expected_hash_is_checked_even_after_table_was_cached(fake_table):
    path, digest = fake_table
    env = make_env(table_config(fake_table))
    env.close()
    with Path(path).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        make_env(table_config((path, digest)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"jev_features_mode": "unknown"},
        {"jev_features_mode": "table"},
        {"jev_features_mode": "table", "jev_features_path": "ignored"},
        {
            "jev_features_mode": "table",
            "jev_features_path": "ignored",
            "jev_features_sha256": "bad",
        },
        {"jev_features_mode": "zeros", "jev_features_path": "ignored"},
        {"jev_features_mode": "zeros", "obs_mode": "pixels"},
        {"jev_features_mode": "zeros", "action_set": "complex"},
        {"jev_features_mode": "zeros", "frame_stack": 2},
        {"jev_features_mode": "zeros", "id": "CartPole-v1"},
        {"jev_features_interval": 0},
        {"jev_features_interval": True},
    ],
)
def test_incompatible_settings_fail_before_any_environment_or_api_use(overrides):
    with pytest.raises(ValueError):
        make_env(EnvConfig(**overrides))


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "range"])
def test_invalid_observation_rejected(fault):
    env = JevFeatureWrapper(MarioEnv("flat", obs_mode="grid"), mode="zeros")
    try:
        raw, _ = env.unwrapped.reset(seed=0)
        if fault == "shape":
            raw = raw[:13]
        elif fault == "dtype":
            raw = raw.astype(np.float64)
        elif fault == "nan":
            raw[0, 0, 0] = np.nan
        else:
            raw[0, 0, 0] = 2
        with pytest.raises(ValueError):
            env.observation(raw)
    finally:
        env.close()


def test_legacy_config_defaults_and_new_settings_round_trip(fake_table):
    legacy = config_from_dict({"env": {"level": "flat"}})
    assert legacy.env.jev_features_mode == "off"
    assert legacy.env.jev_features_interval == 1
    legacy.env = table_config(fake_table, jev_features_interval=16)
    restored = config_from_dict(config_to_dict(legacy))
    assert restored == legacy
