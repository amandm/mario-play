"""Protect the single-policy learning run's stopping, selection and resume guarantees."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

import mario_play.rl.trainer as trainer_module
from mario_play.rl.checkpoint import load_checkpoint
from mario_play.rl.config import load_config
from mario_play.rl.utils import get_rng_state


def _runner():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train_ppo_agent.py"))


def _config(tmp_path):
    cfg = load_config("configs/ppo_grid.yaml")
    cfg.run_dir, cfg.run_name = str(tmp_path), "agent"
    cfg.device, cfg.tensorboard = "cpu", False
    cfg.total_timesteps, cfg.n_envs = 64, 2
    cfg.ppo.n_steps, cfg.ppo.n_epochs, cfg.ppo.n_minibatches = 4, 1, 1
    cfg.network.hidden_size = 16
    cfg.log_interval = cfg.checkpoint_interval = cfg.eval.interval = 8
    cfg.eval.episodes, cfg.eval.seed, cfg.eval.deterministic = 2, 2_000_000, False
    cfg.env.max_episode_steps = 4
    return cfg


def _same_state(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _same_state(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _same_state(left, right)
    else:
        assert actual == expected


def test_readiness_selects_playable_policy_and_preserves_horizon_and_rng(tmp_path, monkeypatch):
    runner = _runner()
    train_agent = runner["train_agent"]
    cfg = _config(tmp_path)
    sampled_calls = 0
    snapshots = []
    calls = []

    def evaluate(algo, env_cfg, *, episodes, seed, deterministic):
        nonlocal sampled_calls
        calls.append((seed, episodes, deterministic))
        if not deterministic and seed == cfg.eval.seed:
            sampled_calls += 1
            snapshots.append(get_rng_state())
        random.random()
        np.random.random()
        torch.rand(3)
        # A higher sampled success with a broken greedy policy must not win selection.
        flag = (0.95 if sampled_calls == 1 else 0.9) if not deterministic else 0.0
        if deterministic and sampled_calls > 1:
            flag = 1.0
        return {
            "mean_return": 100.0 if sampled_calls == 1 else 20.0,
            "std_return": 0.0,
            "mean_length": 4.0,
            "episodes": episodes,
            "flag_rate": flag,
            "mean_progress": 1.0,
        }

    monkeypatch.setattr(trainer_module, "evaluate", evaluate)
    monkeypatch.setitem(train_agent.__globals__, "evaluate", evaluate)
    progress = train_agent(cfg, held_out_episodes=3)
    run_dir = tmp_path / "agent"
    assert progress["status"] == "complete"
    assert progress["stop_reason"] == "validation_criterion"
    assert progress["global_step"] == progress["selected_checkpoint_step"] == 16
    assert progress["target_steps"] == 64
    selected = load_checkpoint(run_dir / "checkpoints/selected.pt")
    assert selected["config"]["total_timesteps"] == 64
    assert selected["global_step"] == 16
    assert load_checkpoint(run_dir / "checkpoints/initial.pt")["extra"]["elapsed"] == 0
    _same_state(get_rng_state(), snapshots[-1])
    _same_state(selected["rng_state"], snapshots[-1])
    milestones = json.loads((run_dir / "learning_milestones.json").read_text())
    assert [row["global_step"] for row in milestones] == [8, 16]
    assert milestones[-1]["training"]["train/lr"] == pytest.approx(cfg.ppo.lr * (1 - 8 / 64))
    assert calls[-2:] == [(3_000_000, 3, False), (3_000_000, 1, True)]
    assert progress["final_assessment"]["checkpoint_step"] == 16


def test_interrupted_run_resumes_without_duplicate_milestones(tmp_path, monkeypatch):
    runner = _runner()
    train_agent, trainer_type = runner["train_agent"], runner["LearningTrainer"]
    cfg = _config(tmp_path)
    cfg.total_timesteps = 32
    cfg.eval.interval = 16
    original_update = trainer_type._record_update

    def stop_after_update(self, metrics):
        original_update(self, metrics)
        self.request_stop()

    monkeypatch.setattr(trainer_type, "_record_update", stop_after_update)
    first = train_agent(cfg, held_out_episodes=2)
    assert first["status"] == "interrupted"
    assert first["global_step"] == 8
    run_dir = tmp_path / "agent"
    latest = run_dir / "checkpoints/latest.pt"
    assert load_checkpoint(latest)["global_step"] == 8
    assert not (run_dir / "final_assessment.json").exists()

    monkeypatch.setattr(trainer_type, "_record_update", original_update)
    completed = train_agent(cfg, resume=latest, held_out_episodes=2)
    assert completed["status"] == "complete"
    assert completed["global_step"] == 32
    milestones = json.loads((run_dir / "learning_milestones.json").read_text())
    assert [row["global_step"] for row in milestones] == [16, 32]
    selected = run_dir / "checkpoints/selected.pt"
    original_bytes = selected.read_bytes()
    assert train_agent(cfg, resume=latest, held_out_episodes=2) == completed
    assert selected.read_bytes() == original_bytes

    cfg.total_timesteps = 64
    with pytest.raises(ValueError, match="learning-rate horizon"):
        train_agent(cfg, resume=latest, held_out_episodes=2)
    assert selected.read_bytes() == original_bytes


def test_legacy_resume_uses_new_field_defaults_but_rejects_changed_settings(tmp_path):
    runner = _runner()
    cfg = _config(tmp_path)
    trainer = runner["LearningTrainer"](cfg)
    try:
        payload = load_checkpoint(trainer.ckpt_dir / "initial.pt")
    finally:
        trainer.close()
    for key in tuple(payload["config"]["env"]):
        if key.startswith("jev_features_"):
            del payload["config"]["env"][key]
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)

    runner["_check_resume"](cfg, legacy)
    assert not any(
        key.startswith("jev_features_") for key in load_checkpoint(legacy)["config"]["env"]
    )

    cfg.env.jev_features_mode = "zeros"
    with pytest.raises(ValueError, match="resume training settings differ"):
        runner["_check_resume"](cfg, legacy)
    cfg.env.jev_features_mode = "off"
    cfg.ppo.lr *= 2
    with pytest.raises(ValueError, match="resume training settings differ"):
        runner["_check_resume"](cfg, legacy)
