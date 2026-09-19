"""Config dataclasses: YAML loading, strict keys, type coercion, dotted overrides."""

import pytest

from mario_play.rl.config import (
    EnvConfig,
    TrainConfig,
    apply_overrides,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)


def test_defaults_build_without_a_file():
    cfg = load_config(None)
    assert isinstance(cfg, TrainConfig)
    assert cfg.algo == "ppo"
    assert isinstance(cfg.env, EnvConfig)
    assert cfg.env.id == "MarioPlay-v0"


def test_yaml_round_trip(tmp_path):
    cfg = load_config(None, ["algo=dqn", "env.level=[1-1,1-2]", "ppo.target_kl=0.02"])
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_dict_round_trip_is_plain_python():
    cfg = load_config(None)
    data = config_to_dict(cfg)
    assert isinstance(data["env"], dict)
    assert config_from_dict(data) == cfg


def test_scientific_notation_strings_are_coerced(tmp_path):
    # PyYAML (YAML 1.1) reads `1e-4` and `1e7` as strings; the field type fixes that.
    path = tmp_path / "cfg.yaml"
    path.write_text("total_timesteps: 1e7\nppo:\n  lr: 1e-4\n")
    cfg = load_config(path)
    assert cfg.total_timesteps == 10_000_000 and isinstance(cfg.total_timesteps, int)
    assert cfg.ppo.lr == pytest.approx(1e-4) and isinstance(cfg.ppo.lr, float)


def test_overrides_are_typed():
    cfg = load_config(
        None,
        [
            "ppo.lr=3e-4",
            "n_envs=2",
            "env.level=flat",
            "env.stall_steps=null",
            "env.resize=[64,64]",
            "ppo.anneal_lr=false",
            "env.reward.flag_bonus=100",
        ],
    )
    assert cfg.ppo.lr == pytest.approx(3e-4)
    assert cfg.n_envs == 2
    assert cfg.env.level == "flat"
    assert cfg.env.stall_steps is None
    assert cfg.env.resize == [64, 64]
    assert cfg.ppo.anneal_lr is False
    assert cfg.env.reward == {"flag_bonus": 100}


def test_level_accepts_a_list():
    cfg = config_from_dict({"env": {"level": ["1-1", "1-2"]}})
    assert cfg.env.level == ["1-1", "1-2"]


def test_optional_int_accepts_numeric_string_and_none():
    assert config_from_dict({"env": {"max_episode_steps": "500"}}).env.max_episode_steps == 500
    assert config_from_dict({"env": {"max_episode_steps": None}}).env.max_episode_steps is None


@pytest.mark.parametrize(
    "data",
    [
        {"nope": 1},
        {"ppo": {"learning_rate": 1e-3}},
        {"env": {"levle": "1-1"}},
    ],
)
def test_unknown_keys_are_rejected(data):
    with pytest.raises(ValueError, match="unknown config key"):
        config_from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        {"n_envs": 2.5},
        {"n_envs": True},
        {"ppo": {"lr": "fast"}},
        {"ppo": {"anneal_lr": "maybe"}},
        {"env": "MarioPlay-v0"},
        {"network": {"mlp_hidden": 64}},
        {"algo": 3},
    ],
)
def test_bad_types_are_rejected(data):
    with pytest.raises(ValueError):
        config_from_dict(data)


def test_override_without_equals_is_rejected():
    with pytest.raises(ValueError, match="key.subkey=value"):
        apply_overrides({}, ["ppo.lr"])


def test_override_through_a_non_section_is_rejected():
    with pytest.raises(ValueError, match="not a section"):
        apply_overrides({"algo": "ppo"}, ["algo.name=x"])


def test_top_level_must_be_a_mapping(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
