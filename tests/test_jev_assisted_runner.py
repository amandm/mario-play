"""Offline proofs for fair staged learning, RNG isolation and durable resume artifacts."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mario_play.rl.checkpoint import load_checkpoint
from mario_play.rl.config import config_to_dict, load_config
from mario_play.rl.utils import get_rng_state


def runner():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/train_jev_assisted_ppo.py")
    )


def config(tmp_path, name="baseline"):
    cfg = load_config("configs/ppo_grid.yaml")
    cfg.run_dir, cfg.run_name = str(tmp_path), name
    cfg.device, cfg.tensorboard = "cpu", False
    cfg.total_timesteps, cfg.n_envs = 120, 2
    cfg.ppo.n_steps, cfg.ppo.n_epochs, cfg.ppo.n_minibatches = 8, 1, 1
    cfg.network.hidden_size = 16
    cfg.log_interval = cfg.checkpoint_interval = cfg.eval.interval = 12
    cfg.env.max_episode_steps = 4
    cfg.env.jev_features_mode = "zeros"
    return cfg


def same_state(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            same_state(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            same_state(left, right)
    else:
        assert actual == expected


def test_stages_preserve_rollouts_and_rng_despite_interleaved_other_training(tmp_path):
    script = runner()
    cls = script["StagedTrainer"]
    trainers = [
        cls(config(tmp_path, name), stage_steps=12, eval_episodes=1, eval_max_steps=3)
        for name in ("split", "other", "continuous")
    ]
    split, other, continuous = trainers
    try:
        assert len({trainer.initial_model_sha256 for trainer in trainers}) == 1
        assert split.venv.single_observation_space.shape == (18, 15, 16)
        split.run_stage(12)
        assert len(split.algo.buffer) == 6  # Below one 8-vector-step PPO rollout.
        other.run_stage(24)  # Substantial unrelated RNG consumption.
        split.run_stage(24)
        continuous.run_stage(24)
        assert split.algo.n_updates == continuous.algo.n_updates == 1
        assert len(split.algo.buffer) == len(continuous.algo.buffer) == 4
        same_state(split.algo.state_dict(), continuous.algo.state_dict())
        same_state(split._rng, continuous._rng)
        np.testing.assert_array_equal(split._obs, continuous._obs)
        assert split.training_stats["train/lr"] == pytest.approx(split.cfg.ppo.lr)
        latest = load_checkpoint(split.ckpt_dir / "latest.pt")
        assert latest["config"]["total_timesteps"] == 120
        assert latest["global_step"] == 24
        assert load_checkpoint(split.ckpt_dir / "initial.pt")["global_step"] == 0
        assert [row["global_step"] for row in split.milestones] == [12, 24]
    finally:
        for trainer in trainers:
            trainer.close()


def test_three_conditions_match_initial_weights_and_resume_cap_without_horizon_change(
    tmp_path,
    monkeypatch,
):
    import mario_play.envs.jev_wrapper as wrapper

    table = tmp_path / "table.json"
    table.write_text(json.dumps({"usage": {"attempted_calls": 476}}))
    # Explicit artificial offline table: the actual exhaustive-table loader has its own tests.
    monkeypatch.setattr(
        wrapper,
        "load_table",
        lambda *args: SimpleNamespace(
            features=lambda grid: np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        ),
    )
    script = runner()
    configs = script["condition_configs"](config(tmp_path), tmp_path, table)
    reference = config_to_dict(configs[0])
    for cfg in configs:
        value = config_to_dict(cfg)
        value["run_name"] = reference["run_name"]
        for field in ("mode", "interval", "path", "sha256"):
            key = f"jev_features_{field}"
            value["env"][key] = reference["env"][key]
        assert value == reference
    result = script["run_experiment"](
        configs,
        max_steps=12,
        stage_steps=12,
        eval_episodes=1,
        eval_max_steps=3,
        make_gifs=True,
    )
    assert result["common_precomputation"]["physical_api_requests"] == 476
    assert result["training_physical_api_requests"] == 0
    initial_hashes = {row["initial_model_sha256"] for row in result["conditions"].values()}
    assert len(initial_hashes) == 1
    for name in result["conditions"]:
        assert (tmp_path / name / "selected.gif").is_file()
        playback = json.loads((tmp_path / name / "selected_playback.json").read_text())
        assert playback["seed"] == 4000000 and playback["deterministic"]
    updates = {name: row["logical_advice_updates"] for name, row in result["conditions"].items()}
    assert updates["baseline"] == 0 < updates["jev_interval16"] < updates["jev_interval1"]
    result = script["run_experiment"](
        configs,
        max_steps=24,
        resume=True,
        stage_steps=12,
        eval_episodes=1,
        eval_max_steps=3,
        make_gifs=False,
    )
    for name, row in result["conditions"].items():
        assert row["global_step"] == 24 and row["fixed_lr_horizon"] == 120
        latest = tmp_path / name / "checkpoints/latest.pt"
        assert load_checkpoint(latest)["config"]["total_timesteps"] == 120
        changed = copy.deepcopy(next(cfg for cfg in configs if cfg.run_name == name))
        changed.total_timesteps = 240
        with pytest.raises(ValueError, match="resume config differs"):
            script["check_resume"](changed, latest)


def test_selection_uses_earliest_crossing_and_greedy_requirement():
    script = runner()

    def row(step, rate, greedy):
        return {
            "global_step": step,
            "sampled": {"flag_rate": rate, "mean_progress": 0.9, "mean_return": 50},
            "greedy": {"flag_rate": greedy},
        }

    high_but_greedy_fails = row(100000, 0.95, 0.0)
    crossing = row(200000, 0.6, 1.0)
    later_better = row(300000, 1.0, 1.0)
    assert not script["ready"](high_but_greedy_fails)
    assert script["ready"](crossing)
    assert script["select_milestone"]([later_better, high_but_greedy_fails, crossing]) == crossing


def test_exception_saves_partial_step_and_evaluation_restores_rng(tmp_path, monkeypatch):
    script = runner()
    trainer = script["StagedTrainer"](
        config(tmp_path), stage_steps=12, eval_episodes=1, eval_max_steps=3
    )
    try:
        state = get_rng_state()
        result = script["evaluate_policy"](
            trainer.algo,
            trainer.cfg.env,
            episodes=2,
            seed=2000000,
            deterministic=False,
            max_steps=3,
        )
        assert [row["seed"] for row in result["episode_results"]] == [2000000, 2000001]
        same_state(get_rng_state(), state)
        original = trainer.venv.step
        calls = 0

        def fail(actions):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("offline injected failure")
            return original(actions)

        monkeypatch.setattr(trainer.venv, "step", fail)
        with pytest.raises(RuntimeError, match="injected"):
            trainer.run_stage(12)
        assert load_checkpoint(trainer.ckpt_dir / "latest.pt")["global_step"] == 4
        progress = json.loads((trainer.run_dir / "progress.json").read_text())
        assert progress["status"] == "failed" and progress["global_step"] == 4
    finally:
        trainer.close()


def test_fixed_budget_resumes_latest_past_old_crossing_and_counts_actual_updates(
    tmp_path, monkeypatch
):
    script = runner()
    cfg = config(tmp_path)
    # Make the first milestone qualify, independently of this tiny plumbing run's ability.
    monkeypatch.setitem(script["run_experiment"].__globals__, "ready", lambda record: True)
    first = script["run_experiment"](
        [cfg],
        max_steps=36,
        stage_steps=12,
        eval_episodes=1,
        eval_max_steps=3,
        make_gifs=False,
    )
    assert first["conditions"]["baseline"]["global_step"] == 12
    checkpoint = tmp_path / "baseline/checkpoints/latest.pt"
    initial = load_checkpoint(checkpoint)
    assert initial["extra"]["pending_rollout_transitions"] == 12
    assert initial["algo_state"]["n_updates"] == 0
    original_initial = (checkpoint.parent / "initial.pt").read_bytes()
    # An older selected policy must never initialize continuation.
    (checkpoint.parent / "selected.pt").write_bytes(original_initial)
    resumed = script["StagedTrainer"](
        cfg, resume=checkpoint, stage_steps=12, eval_episodes=1, eval_max_steps=3
    )
    try:
        same_state(resumed._rng, initial["rng_state"])
        assert resumed.global_step == 12
        assert resumed.discarded_rollout_transitions == 12
    finally:
        resumed.close()
    result = script["run_experiment"](
        [cfg],
        max_steps=36,
        resume=True,
        fixed_budget=True,
        stage_steps=12,
        eval_episodes=1,
        eval_max_steps=3,
        make_gifs=False,
    )
    snapshot = result["conditions"]["baseline"]
    assert result["fixed_budget"] and result["stop_reason"] == "fixed_budget"
    assert snapshot["global_step"] == 36 and snapshot["fixed_lr_horizon"] == 120
    assert snapshot["ppo_updates"] == 1  # Not floor(36 / 16): 12 transitions were discarded.
    assert snapshot["pending_rollout_transitions"] == 8
    assert snapshot["discarded_rollout_transitions"] == 12
    assert snapshot["process_resume_count"] == 1
    final = load_checkpoint(checkpoint)
    assert final["algo_state"]["n_updates"] == final["extra"]["ppo_updates"] == 1
    assert final["config"]["total_timesteps"] == 120
    assert (checkpoint.parent / "initial.pt").read_bytes() == original_initial
