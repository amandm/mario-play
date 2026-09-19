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


@pytest.mark.parametrize(
    ("override", "field", "expected"),
    [
        ("run_name=007", "run_name", "007"),
        ("run_name=2026-09-19", "run_name", "2026-09-19"),
        ("run_name=1.0", "run_name", "1.0"),
        ("run_name=on", "run_name", "on"),
        ("run_name= 42 ", "run_name", "42"),
        ("run_dir=2026", "run_dir", "2026"),
        ("run_name=smoke", "run_name", "smoke"),
        ("run_name='007'", "run_name", "007"),
        ("run_name=null", "run_name", None),
        ("run_name=~", "run_name", None),
    ],
)
def test_overrides_of_string_fields_keep_their_text(override, field, expected):
    """`run_name=2026-09-19` names a run; YAML's date / int / bool reading must not reject it."""
    assert getattr(load_config(None, [override]), field) == expected


def test_string_overrides_in_nested_sections_keep_their_text():
    cfg = load_config(None, ["env.id=123", "env.level=1", "network.encoder=off"])
    assert (cfg.env.id, cfg.env.level, cfg.network.encoder) == ("123", "1", "off")
    assert load_config(None, ["env.level=[1-1,1-2]"]).env.level == ["1-1", "1-2"]


def test_only_string_fields_keep_the_override_text():
    """Everything else stays YAML-typed: numbers, bools and the untyped reward / kwargs dicts."""
    data = apply_overrides(
        {},
        [
            "env.reward.flag_bonus=100",
            "env.kwargs.continuous=on",
            "n_envs=4",
            "ppo.anneal_lr=false",
            "ppo.target_kl=0.02",
            "env.stall_steps=null",
            "nope=007",
        ],
    )
    assert data == {
        "env": {"reward": {"flag_bonus": 100}, "kwargs": {"continuous": True}, "stall_steps": None},
        "n_envs": 4,
        "ppo": {"anneal_lr": False, "target_kl": 0.02},
        "nope": 7,
    }
    with pytest.raises(ValueError, match="unknown config key"):
        config_from_dict(data)


def test_a_yaml_file_still_needs_quotes_for_a_numeric_string():
    """Only the command line is lenient: in a file, `run_name: 7` is an int, as in any YAML."""
    with pytest.raises(ValueError, match="run_name"):
        config_from_dict({"run_name": 7})


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
