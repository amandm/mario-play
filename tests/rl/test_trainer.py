"""Trainer: run-dir artefacts, step intervals, evaluation/best tracking, resume, SIGINT, cleanup."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import mario_play.rl.trainer as trainer_module
from mario_play.rl.algos.dqn import DQN
from mario_play.rl.algos.ppo import PPO
from mario_play.rl.checkpoint import list_checkpoints, load_checkpoint
from mario_play.rl.config import config_from_dict, config_to_dict
from mario_play.rl.evaluate import evaluate, load_algorithm
from mario_play.rl.trainer import Trainer
from mario_play.rl.types import VecStep

pytestmark = pytest.mark.timeout(120)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def read_rows(run_dir: Path) -> list[dict[str, str]]:
    with open(Path(run_dir) / "metrics.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def steps_with(rows: list[dict[str, str]], column: str) -> list[int]:
    """Steps of the rows that have a value in `column`."""
    return [int(row["step"]) for row in rows if row.get(column)]


def first_multiples_reached(interval: int, n_envs: int, total: int) -> list[int]:
    """The global steps (multiples of `n_envs`) at which each multiple of `interval` is crossed."""
    final = math.ceil(total / n_envs) * n_envs
    reached = [math.ceil(k / n_envs) * n_envs for k in range(interval, final + 1, interval)]
    return sorted(set(reached))


def scripted_evaluate(scores: list[float], calls: list[dict] | None = None):
    """A stand-in for `evaluate` that returns the given mean returns one after another."""
    remaining = list(scores)

    def fake(algo, env_cfg, episodes, seed, deterministic=True, **kwargs):
        if calls is not None:
            calls.append({"episodes": episodes, "seed": seed, "deterministic": deterministic})
        score = remaining.pop(0)
        return {"mean_return": score, "std_return": 0.0, "mean_length": 1.0, "episodes": episodes}

    return fake


def interrupt_on_update(trainer: Trainer, nth: int, times: int = 1) -> list[int]:
    """Send SIGINT to this process from inside the `nth` call of `algo.update`."""
    calls: list[int] = []
    real_update = trainer.algo.update

    def update(global_step: int, progress: float) -> dict[str, float]:
        calls.append(global_step)
        if len(calls) == nth:
            for _ in range(times):
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.01)  # the Python-level handler runs before the next signal is sent
        return real_update(global_step, progress)

    trainer.algo.update = update
    return calls


def model_tensors(algo) -> dict[str, torch.Tensor]:
    state = algo.state_dict()
    return state["model"] if "model" in state else state["q_net"]


def assert_same_weights(a, b) -> None:
    tensors_a, tensors_b = model_tensors(a), model_tensors(b)
    assert tensors_a.keys() == tensors_b.keys()
    for key in tensors_a:
        torch.testing.assert_close(tensors_a[key], tensors_b[key], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# a plain run
# --------------------------------------------------------------------------- #


def test_run_directory_artefacts_and_result(tiny_cfg):
    cfg = tiny_cfg()
    trainer = Trainer(cfg)
    result = trainer.train()

    run_dir = Path(cfg.run_dir) / "run"
    assert trainer.run_dir == run_dir
    assert set(result) >= {"global_step", "best_eval", "run_dir", "final_eval"}
    assert result["global_step"] == 1_000
    assert result["run_dir"] == str(run_dir)
    assert result["interrupted"] is False
    assert result["final_eval"]["episodes"] == cfg.eval.episodes
    assert result["best_eval"] >= result["final_eval"]["mean_return"]

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "log.txt").read_text(encoding="utf-8").strip()
    assert not (run_dir / "tb").exists()  # tensorboard is off in the tiny config
    ckpt_dir = run_dir / "checkpoints"
    assert [p.name for p in list_checkpoints(ckpt_dir)] == [
        "ckpt_400.pt",
        "ckpt_800.pt",
        "ckpt_1000.pt",
    ]
    assert (ckpt_dir / "latest.pt").is_file()
    assert (ckpt_dir / "best.pt").is_file()
    assert not [p for p in ckpt_dir.iterdir() if p.name.endswith(".tmp")]


def test_config_dump_round_trips_and_names_the_run(tiny_cfg):
    cfg = tiny_cfg()
    trainer = Trainer(cfg)
    trainer.train()
    with open(trainer.run_dir / "config.yaml", encoding="utf-8") as fh:
        dumped = config_from_dict(yaml.safe_load(fh))
    assert dumped == trainer.cfg
    assert dumped.run_name == "run"
    assert dataclasses.replace(dumped, run_name=cfg.run_name) == cfg


def test_the_callers_config_is_not_modified(tiny_cfg):
    cfg = tiny_cfg(run_name=None)
    trainer = Trainer(cfg)
    trainer.close()
    assert cfg.run_name is None
    assert trainer.cfg.run_name == trainer.run_dir.name


def test_metrics_csv_has_rollout_loss_time_and_eval_columns(tiny_cfg):
    trainer = Trainer(tiny_cfg())
    trainer.train()
    rows = read_rows(trainer.run_dir)
    columns = set(rows[0])
    assert {
        "step",
        "rollout/ep_return_mean",
        "rollout/ep_length_mean",
        "rollout/episodes",
        "time/sps",
        "time/elapsed",
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "train/approx_kl",
        "train/lr",
        "eval/mean_return",
        "eval/std_return",
        "eval/mean_length",
    } <= columns
    assert "rollout/flag_rate" not in columns  # CartPole has no Mario extras
    last = rows[-1]
    assert float(last["rollout/ep_return_mean"]) == pytest.approx(
        float(last["rollout/ep_length_mean"])
    )
    assert float(last["time/sps"]) > 0
    for row in rows:
        for key, value in row.items():
            if value:
                assert math.isfinite(float(value)), (key, value)


def test_console_lines_are_kept_in_log_txt(tiny_cfg, capsys):
    trainer = Trainer(tiny_cfg())
    trainer.train()
    printed = capsys.readouterr().out
    log_text = (trainer.run_dir / "log.txt").read_text(encoding="utf-8")
    assert "sps" in printed and "return" in printed and "eval" in printed
    assert "policy_loss" in printed
    assert str(trainer.run_dir) in printed
    for line in printed.strip().splitlines():
        assert line in log_text


def test_default_run_name_is_algo_env_timestamp_and_unique(tiny_cfg):
    first = Trainer(tiny_cfg(run_name=None))
    second = Trainer(tiny_cfg(run_name=None))  # created within the same second
    try:
        assert first.run_dir.name.startswith("ppo_CartPole-v1_")
        assert first.run_dir.parent == second.run_dir.parent
        assert first.run_dir != second.run_dir
        assert first.run_dir.is_dir() and second.run_dir.is_dir()
    finally:
        first.close()
        second.close()


def test_an_existing_run_is_never_overwritten(tiny_cfg):
    first = Trainer(tiny_cfg(total_timesteps=200))
    first.train()
    rows_before = read_rows(first.run_dir)

    second = Trainer(tiny_cfg(total_timesteps=200))  # same run_name, no resume
    second.close()
    assert second.run_dir != first.run_dir
    assert second.run_dir.name.startswith("run")
    assert read_rows(first.run_dir) == rows_before


def test_tensorboard_events_are_written_when_enabled(tiny_cfg):
    trainer = Trainer(tiny_cfg(total_timesteps=200, tensorboard=True))
    trainer.train()
    events = list((trainer.run_dir / "tb").glob("events.out.tfevents.*"))
    assert events and events[0].stat().st_size > 0


def test_dqn_runs_end_to_end(tiny_cfg):
    trainer = Trainer(tiny_cfg("dqn"))
    assert isinstance(trainer.algo, DQN)
    result = trainer.train()
    assert result["global_step"] == 1_000
    assert trainer.algo.n_updates > 0
    columns = set(read_rows(trainer.run_dir)[-1])
    assert {"train/loss", "train/q_mean", "train/epsilon", "rollout/ep_return_mean"} <= columns
    last = read_rows(trainer.run_dir)[-1]
    assert float(last["train/epsilon"]) == pytest.approx(trainer.algo.epsilon(1_000))

    algo, cfg = load_algorithm(trainer.run_dir / "checkpoints" / "latest.pt", device="cpu")
    assert isinstance(algo, DQN)
    assert set(evaluate(algo, cfg.env, episodes=2, seed=0)) >= {"mean_return", "mean_length"}


def test_checkpoint_alone_is_enough_to_evaluate(tiny_cfg):
    trainer = Trainer(tiny_cfg())
    trainer.train()
    algo, cfg = load_algorithm(trainer.run_dir / "checkpoints" / "best.pt", device="cpu")
    assert isinstance(algo, PPO)
    assert cfg == trainer.cfg
    scores = evaluate(algo, cfg.env, episodes=3, seed=cfg.eval.seed)
    assert set(scores) >= {"mean_return", "std_return", "mean_length"}


def test_same_seed_reproduces_the_same_policy(tiny_cfg):
    first = Trainer(tiny_cfg(run_name="a"))
    first.train()
    second = Trainer(tiny_cfg(run_name="b"))
    second.train()
    assert_same_weights(first.algo, second.algo)
    different = Trainer(tiny_cfg(run_name="c", seed=2))
    different.train()
    with pytest.raises(AssertionError):
        assert_same_weights(first.algo, different.algo)


def test_mario_extras_are_tracked_when_the_env_reports_them(tiny_cfg, flag_run_env_id):
    trainer = Trainer(tiny_cfg(env={"id": flag_run_env_id}, total_timesteps=600))
    result = trainer.train()
    rows = read_rows(trainer.run_dir)
    for column in ("rollout/flag_rate", "rollout/mean_progress"):
        values = [float(row[column]) for row in rows if row[column]]
        assert values and all(0.0 <= value <= 1.0 for value in values)
    assert steps_with(rows, "eval/flag_rate") == steps_with(rows, "eval/mean_return")
    assert 0.0 <= result["final_eval"]["mean_progress"] <= 1.0
    log_text = (trainer.run_dir / "log.txt").read_text(encoding="utf-8")
    assert "flag" in log_text and "progress" in log_text


def test_episode_statistics_cover_the_last_100_episodes_and_accept_numpy_values(tiny_cfg):
    trainer = Trainer(tiny_cfg())
    try:
        for episode in range(150):
            infos = [
                {
                    "episode": {"r": np.array([episode], dtype=np.float32), "l": np.int64(episode)},
                    "flag_get": np.bool_(episode >= 100),
                    "progress": np.float32(0.5),
                },
                {"episode": {"r": 1e9, "l": 1e9}},  # env 1 is not done: must be ignored
            ]
            obs = np.zeros((2, 4), dtype=np.float32)
            trainer._record_episodes(
                VecStep(
                    obs=obs,
                    rewards=np.zeros(2, dtype=np.float32),
                    terminated=np.array([episode % 2 == 0, False]),
                    truncated=np.array([episode % 2 == 1, False]),
                    final_obs=obs,
                    infos=infos,
                )
            )
        row = trainer._training_row()
    finally:
        trainer.close()
    assert row["rollout/episodes"] == 150
    assert row["rollout/ep_return_mean"] == pytest.approx(np.mean(range(50, 150)))
    assert row["rollout/ep_length_mean"] == pytest.approx(np.mean(range(50, 150)))
    assert row["rollout/flag_rate"] == pytest.approx(0.5)
    assert row["rollout/mean_progress"] == pytest.approx(0.5)
    assert all(type(value) in (int, float) for value in row.values())


# --------------------------------------------------------------------------- #
# step intervals
# --------------------------------------------------------------------------- #


def test_intervals_trigger_when_a_multiple_is_crossed_even_if_n_envs_does_not_divide_it(tiny_cfg):
    cfg = tiny_cfg(
        n_envs=3,
        log_interval=100,
        checkpoint_interval=250,
        keep_checkpoints=10,
        eval={"interval": 400},
    )
    trainer = Trainer(cfg)
    result = trainer.train()
    assert result["global_step"] == 1_002

    rows = read_rows(trainer.run_dir)
    assert steps_with(rows, "time/sps") == first_multiples_reached(100, 3, 1_000)
    assert steps_with(rows, "eval/mean_return") == first_multiples_reached(400, 3, 1_000) + [1_002]
    numbered = [p.name for p in list_checkpoints(trainer.run_dir / "checkpoints")]
    expected = first_multiples_reached(250, 3, 1_000)
    assert expected == [252, 501, 750, 1_002]
    assert numbered == [f"ckpt_{step}.pt" for step in expected]
    all_steps = [int(row["step"]) for row in rows]
    assert all_steps == sorted(set(all_steps)), "one row per step, in order"


def test_an_interval_smaller_than_one_vector_step_fires_once_per_step(tiny_cfg):
    trainer = Trainer(tiny_cfg(n_envs=4, log_interval=3, total_timesteps=40, eval={"interval": 0}))
    trainer.train()
    assert steps_with(read_rows(trainer.run_dir), "time/sps") == list(range(4, 41, 4))


def test_the_end_of_training_is_always_logged_and_checkpointed(tiny_cfg):
    cfg = tiny_cfg(total_timesteps=500, log_interval=300, checkpoint_interval=300)
    trainer = Trainer(cfg)
    trainer.train()
    assert steps_with(read_rows(trainer.run_dir), "time/sps") == [300, 500]
    ckpt_dir = trainer.run_dir / "checkpoints"
    assert [p.name for p in list_checkpoints(ckpt_dir)] == ["ckpt_300.pt", "ckpt_500.pt"]
    assert load_checkpoint(ckpt_dir / "latest.pt")["global_step"] == 500


def test_non_positive_intervals_disable_periodic_work(tiny_cfg):
    cfg = tiny_cfg(log_interval=0, checkpoint_interval=0, eval={"interval": 0})
    trainer = Trainer(cfg)
    result = trainer.train()
    rows = read_rows(trainer.run_dir)
    assert [int(row["step"]) for row in rows] == [1_000]  # only the closing row
    assert [p.name for p in list_checkpoints(trainer.run_dir / "checkpoints")] == ["ckpt_1000.pt"]
    assert result["final_eval"] is not None  # the final evaluation still happens


def test_rotation_keeps_only_the_newest_numbered_checkpoints(tiny_cfg):
    cfg = tiny_cfg(checkpoint_interval=200, keep_checkpoints=2)
    trainer = Trainer(cfg)
    trainer.train()
    ckpt_dir = trainer.run_dir / "checkpoints"
    assert [p.name for p in list_checkpoints(ckpt_dir)] == ["ckpt_800.pt", "ckpt_1000.pt"]
    assert (ckpt_dir / "latest.pt").is_file() and (ckpt_dir / "best.pt").is_file()


# --------------------------------------------------------------------------- #
# evaluation and best.pt
# --------------------------------------------------------------------------- #


def test_best_checkpoint_follows_the_best_periodic_eval(tiny_cfg, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate([5.0, 9.0, 7.0], calls))
    cfg = tiny_cfg(eval={"interval": 400, "episodes": 4, "seed": 77, "deterministic": False})
    trainer = Trainer(cfg)
    result = trainer.train()

    assert calls == [{"episodes": 4, "seed": 77, "deterministic": False}] * 3
    assert result["best_eval"] == 9.0
    assert result["final_eval"]["mean_return"] == 7.0
    ckpt_dir = trainer.run_dir / "checkpoints"
    best = load_checkpoint(ckpt_dir / "best.pt")
    assert best["global_step"] == 800
    assert best["best_eval"] == 9.0
    assert load_checkpoint(ckpt_dir / "latest.pt")["best_eval"] == 9.0
    rows = read_rows(trainer.run_dir)
    assert [float(row["eval/mean_return"]) for row in rows if row["eval/mean_return"]] == [5, 9, 7]


def test_final_eval_can_become_the_best(tiny_cfg, monkeypatch):
    monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate([5.0, 3.0, 11.0]))
    trainer = Trainer(tiny_cfg())
    result = trainer.train()
    assert result["best_eval"] == 11.0
    best = load_checkpoint(trainer.run_dir / "checkpoints" / "best.pt")
    assert best["global_step"] == 1_000


def test_an_eval_due_at_the_last_step_is_not_repeated_as_final_eval(tiny_cfg, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate([1.0, 2.0], calls))
    trainer = Trainer(tiny_cfg(eval={"interval": 500}))
    result = trainer.train()
    assert len(calls) == 2  # at 500 and at 1000; the latter doubles as the final evaluation
    assert result["final_eval"]["mean_return"] == 2.0


def test_no_evaluation_at_all_with_zero_episodes(tiny_cfg, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluate must not be called")

    monkeypatch.setattr(trainer_module, "evaluate", forbidden)
    trainer = Trainer(tiny_cfg(eval={"episodes": 0}))
    result = trainer.train()
    assert result["final_eval"] is None
    assert result["best_eval"] is None
    assert not (trainer.run_dir / "checkpoints" / "best.pt").exists()


def test_evaluation_never_changes_the_training_trajectory(tiny_cfg):
    quiet = Trainer(tiny_cfg(run_name="quiet", eval={"interval": 0}))
    quiet.train()
    noisy = Trainer(tiny_cfg(run_name="noisy", eval={"interval": 200, "deterministic": False}))
    noisy.train()
    assert_same_weights(quiet.algo, noisy.algo)


# --------------------------------------------------------------------------- #
# schedules
# --------------------------------------------------------------------------- #


def test_lr_anneal_starts_at_the_configured_rate_and_the_last_update_still_learns(tiny_cfg):
    """Update i of N runs at `lr * (1 - (i - 1) / N)` (the CleanRL schedule).

    `progress` is the fraction completed when the update's rollout *began*: measured after
    the rollout, the first update never saw the configured rate and the last one ran at 0.
    """
    cfg = tiny_cfg(total_timesteps=4 * 16 * 2, log_interval=16 * 2, eval={"interval": 0})
    trainer = Trainer(cfg)
    seen: list[float] = []
    weights_before_last: dict[str, torch.Tensor] = {}
    real_update = trainer.algo.update

    def update(global_step: int, progress: float) -> dict[str, float]:
        seen.append(progress)
        if len(seen) == 4:
            weights_before_last.update(
                {key: value.clone() for key, value in model_tensors(trainer.algo).items()}
            )
        return real_update(global_step, progress)

    trainer.algo.update = update
    trainer.train()

    assert seen == [0.0, 0.25, 0.5, 0.75]
    rows = read_rows(trainer.run_dir)
    rates = [float(row["train/lr"]) for row in rows if row["train/lr"]]
    assert rates == pytest.approx([cfg.ppo.lr * left for left in (1.0, 0.75, 0.5, 0.25)])
    after = model_tensors(trainer.algo)
    assert any(not torch.equal(after[key], weights_before_last[key]) for key in after), (
        "the final update must change the weights"
    )


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #


def test_resume_continues_the_step_count_and_appends_to_the_logs(tiny_cfg):
    first = Trainer(tiny_cfg(total_timesteps=600))
    first_result = first.train()
    assert first_result["global_step"] == 600
    updates_before = first.algo.n_updates
    assert updates_before == 600 // (16 * 2)
    first_rows = read_rows(first.run_dir)
    first_log = (first.run_dir / "log.txt").read_text(encoding="utf-8")

    latest = first.run_dir / "checkpoints" / "latest.pt"
    second = Trainer(tiny_cfg(total_timesteps=1_200), resume=str(latest))
    assert second.run_dir == first.run_dir
    assert second.global_step == 600
    assert second.best_eval == first_result["best_eval"]
    result = second.train()

    assert result["global_step"] == 1_200
    assert result["run_dir"] == first_result["run_dir"]
    assert second.algo.n_updates == updates_before + 600 // (16 * 2)
    rows = read_rows(second.run_dir)
    assert rows[: len(first_rows)] == [
        {**row, **{k: "" for k in rows[0] if k not in row}} for row in first_rows
    ]
    assert steps_with(rows, "time/sps") == [200, 400, 600, 800, 1_000, 1_200]
    assert (second.run_dir / "log.txt").read_text(encoding="utf-8").startswith(first_log)
    assert [p.name for p in list_checkpoints(second.run_dir / "checkpoints")] == [
        "ckpt_600.pt",
        "ckpt_800.pt",
        "ckpt_1200.pt",
    ]
    # The LR anneal continues from 600/1200 instead of restarting at the full rate.
    lr_after_resume = float(next(row for row in rows if row["step"] == "800")["train/lr"])
    assert 0.0 < lr_after_resume < 0.5 * second.cfg.ppo.lr


def test_resume_restores_weights_counters_and_rng(tiny_cfg):
    first = Trainer(tiny_cfg(total_timesteps=400))
    first.train()
    latest = first.run_dir / "checkpoints" / "latest.pt"
    saved = load_checkpoint(latest)

    torch.manual_seed(12345)
    resumed = Trainer(tiny_cfg(total_timesteps=800), resume=latest)
    try:
        assert_same_weights(first.algo, resumed.algo)
        assert resumed.algo.n_updates == first.algo.n_updates
        assert torch.equal(torch.get_rng_state(), saved["rng_state"]["torch"])
    finally:
        resumed.close()


def test_resume_keeps_the_best_score_and_best_checkpoint(tiny_cfg, monkeypatch):
    monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate([50.0, 40.0]))
    first = Trainer(tiny_cfg(total_timesteps=600))
    assert first.train()["best_eval"] == 50.0

    monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate([10.0, 20.0, 30.0]))
    second = Trainer(tiny_cfg(total_timesteps=1_200), resume=first.run_dir / "checkpoints")
    result = second.train()
    assert result["best_eval"] == 50.0
    best = load_checkpoint(second.run_dir / "checkpoints" / "best.pt")
    assert (best["global_step"], best["best_eval"]) == (400, 50.0)


def test_resume_accepts_a_run_directory(tiny_cfg):
    first = Trainer(tiny_cfg(total_timesteps=200))
    first.train()
    resumed = Trainer(tiny_cfg(total_timesteps=400), resume=str(first.run_dir))
    try:
        assert resumed.run_dir == first.run_dir
        assert resumed.global_step == 200
    finally:
        resumed.close()


def test_resume_accepts_a_relative_path(tiny_cfg, monkeypatch):
    first = Trainer(tiny_cfg(total_timesteps=200))
    first.train()
    monkeypatch.chdir(first.run_dir)
    resumed = Trainer(tiny_cfg(total_timesteps=400), resume="checkpoints/latest.pt")
    try:
        assert resumed.run_dir.is_absolute()
        assert resumed.run_dir.samefile(first.run_dir)
        assert resumed.cfg.run_name == "run"
    finally:
        resumed.close()


def test_resume_without_a_config_uses_the_one_in_the_checkpoint(tiny_cfg):
    first = Trainer(tiny_cfg(total_timesteps=200))
    first.train()
    resumed = Trainer(resume=first.run_dir / "checkpoints" / "latest.pt")
    try:
        assert resumed.cfg == first.cfg
        assert resumed.global_step == 200
    finally:
        resumed.close()
    with pytest.raises(ValueError, match="cfg"):
        Trainer()


def test_resume_next_thresholds_are_the_next_multiples(tiny_cfg):
    cfg = tiny_cfg(n_envs=3, total_timesteps=450, log_interval=100, eval={"interval": 0})
    first = Trainer(cfg)
    assert first.train()["global_step"] == 450

    second = Trainer(dataclasses.replace(cfg, total_timesteps=800), resume=first.run_dir)
    second.train()
    log_steps = steps_with(read_rows(second.run_dir), "time/sps")
    assert log_steps == [102, 201, 300, 402, 450, 501, 600, 702, 801]


def test_resume_of_a_finished_run_trains_no_further(tiny_cfg):
    first = Trainer(tiny_cfg(total_timesteps=400))
    first.train()
    again = Trainer(tiny_cfg(total_timesteps=400), resume=first.run_dir)
    result = again.train()
    assert result["global_step"] == 400
    assert again.algo.n_updates == first.algo.n_updates
    assert_same_weights(first.algo, again.algo)


def test_rewinding_to_an_older_checkpoint_sets_later_ones_aside(tiny_cfg):
    first = Trainer(tiny_cfg(checkpoint_interval=200, keep_checkpoints=10))
    first.train()
    ckpt_dir = first.run_dir / "checkpoints"
    second = Trainer(
        tiny_cfg(checkpoint_interval=200, keep_checkpoints=2, total_timesteps=800),
        resume=ckpt_dir / "ckpt_400.pt",
    )
    second.train()
    # Without setting ckpt_600..1000 aside, rotation would delete the new checkpoints instead.
    assert [p.name for p in list_checkpoints(ckpt_dir)] == ["ckpt_600.pt", "ckpt_800.pt"]
    assert load_checkpoint(ckpt_dir / "latest.pt")["global_step"] == 800
    assert sorted(p.name for p in ckpt_dir.glob("ckpt_*.superseded.pt")) == [
        "ckpt_1000.superseded.pt",
        "ckpt_600.superseded.pt",
        "ckpt_800.superseded.pt",
    ]
    assert load_checkpoint(ckpt_dir / "latest.superseded.pt")["global_step"] == 1_000
    assert steps_with(read_rows(second.run_dir), "time/sps") == [200, 400, 600, 800]


def test_rewinding_makes_latest_follow_so_a_crash_cannot_jump_back_into_the_abandoned_future(
    tiny_cfg,
):
    """`resume=<run_dir>` means `latest.pt`: after a rewind it must be the rewound state at once,
    not only after the next periodic checkpoint (which a crash may never let happen)."""
    cfg = tiny_cfg(checkpoint_interval=200, keep_checkpoints=10)
    first = Trainer(cfg)
    first.train()
    ckpt_dir = first.run_dir / "checkpoints"
    old_latest = (ckpt_dir / "latest.pt").read_bytes()

    Trainer(cfg, resume=ckpt_dir / "ckpt_400.pt").close()  # dies before its first checkpoint
    assert load_checkpoint(ckpt_dir / "latest.pt")["global_step"] == 400
    assert (ckpt_dir / "latest.pt").read_bytes() == (ckpt_dir / "ckpt_400.pt").read_bytes()
    assert (ckpt_dir / "latest.superseded.pt").read_bytes() == old_latest  # nothing is deleted
    assert "latest.superseded.pt" in (first.run_dir / "log.txt").read_text(encoding="utf-8")

    again = Trainer(cfg, resume=first.run_dir)
    again.close()
    assert again.global_step == 400
    assert sorted(p.name for p in ckpt_dir.glob("latest*")) == ["latest.pt", "latest.superseded.pt"]

    Trainer(cfg, resume=ckpt_dir / "ckpt_200.pt").close()  # a second rewind finds a free name
    assert load_checkpoint(ckpt_dir / "latest.pt")["global_step"] == 200
    assert load_checkpoint(ckpt_dir / "latest.superseded-2.pt")["global_step"] == 400
    assert not [p for p in ckpt_dir.iterdir() if p.name.endswith(".tmp")]


def test_resuming_from_the_newest_checkpoint_sets_nothing_aside(tiny_cfg):
    cfg = tiny_cfg(checkpoint_interval=200, keep_checkpoints=10)
    first = Trainer(cfg)
    first.train()
    ckpt_dir = first.run_dir / "checkpoints"
    before = sorted(p.name for p in ckpt_dir.iterdir())
    for source in (ckpt_dir / "ckpt_1000.pt", ckpt_dir / "latest.pt", first.run_dir):
        Trainer(cfg, resume=source).close()
        assert sorted(p.name for p in ckpt_dir.iterdir()) == before


def crash_after_periodic_work_at(trainer: Trainer, step: int) -> None:
    """Make `train()` die like a killed process: right after the periodic work of `step`."""
    real_periodic_work = trainer._periodic_work

    def periodic_work(finished: bool) -> None:
        real_periodic_work(finished)
        if trainer.global_step >= step:
            raise RuntimeError("killed")

    trainer._periodic_work = periodic_work


def test_crash_resume_does_not_replace_a_better_best_checkpoint_it_never_knew(
    tiny_cfg, monkeypatch
):
    """best.pt from an evaluation after the last checkpoint is newer than `latest.pt`'s bar."""
    cfg = tiny_cfg(eval={"interval": 200}, checkpoint_interval=400)
    monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate([5.0, 6.0, 9.0]))
    first = Trainer(cfg)
    crash_after_periodic_work_at(first, 600)  # evaluated (9.0 -> best.pt) but not checkpointed
    with pytest.raises(RuntimeError, match="killed"):
        first.train()
    ckpt_dir = first.run_dir / "checkpoints"
    best_bytes = (ckpt_dir / "best.pt").read_bytes()
    assert load_checkpoint(ckpt_dir / "best.pt")["best_eval"] == 9.0
    assert load_checkpoint(ckpt_dir / "latest.pt")["best_eval"] == 6.0

    monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate([7.0, 3.0, 4.0]))
    second = Trainer(cfg, resume=first.run_dir)
    assert (second.global_step, second.best_eval) == (400, 6.0)
    result = second.train()

    assert result["best_eval"] == 7.0  # a truthful "new best" of the resumed timeline ...
    best = load_checkpoint(ckpt_dir / "best.pt")
    assert (best["global_step"], best["best_eval"]) == (600, 7.0)
    # ... that did not destroy the better policy of the timeline that was lost.
    survivors = [p.name for p in ckpt_dir.glob("best*.pt") if p.read_bytes() == best_bytes]
    assert survivors == ["best.superseded.pt"]
    assert "best.superseded.pt" in (first.run_dir / "log.txt").read_text(encoding="utf-8")


def test_rewinding_sets_the_best_checkpoint_of_the_abandoned_future_aside(tiny_cfg, monkeypatch):
    monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate([5.0, 9.0, 7.0]))
    first = Trainer(tiny_cfg(keep_checkpoints=10))
    first.train()
    ckpt_dir = first.run_dir / "checkpoints"
    best_bytes = (ckpt_dir / "best.pt").read_bytes()
    assert load_checkpoint(ckpt_dir / "best.pt")["global_step"] == 800

    monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate([6.0]))
    second = Trainer(tiny_cfg(total_timesteps=800), resume=ckpt_dir / "ckpt_400.pt")
    assert second.best_eval == 5.0
    assert second.train()["best_eval"] == 6.0
    best = load_checkpoint(ckpt_dir / "best.pt")
    assert (best["global_step"], best["best_eval"]) == (800, 6.0)
    survivors = [p.name for p in ckpt_dir.glob("best*.pt") if p.read_bytes() == best_bytes]
    assert survivors == ["best.superseded.pt"]


def test_unreadable_best_or_latest_files_do_not_block_a_resume(tiny_cfg):
    first = Trainer(tiny_cfg(total_timesteps=400, checkpoint_interval=200))
    first.train()
    ckpt_dir = first.run_dir / "checkpoints"
    (ckpt_dir / "best.pt").write_bytes(b"truncated")
    (ckpt_dir / "latest.pt").write_bytes(b"truncated")
    resumed = Trainer(tiny_cfg(total_timesteps=600), resume=ckpt_dir / "ckpt_200.pt")
    assert resumed.global_step == 200
    assert resumed.train()["global_step"] == 600
    assert load_checkpoint(ckpt_dir / "latest.pt")["global_step"] == 600


def test_resume_rejects_a_checkpoint_of_another_algorithm(tiny_cfg):
    first = Trainer(tiny_cfg("ppo", total_timesteps=200))
    first.train()
    with pytest.raises(ValueError, match="ppo.*dqn|dqn.*ppo"):
        Trainer(tiny_cfg("dqn"), resume=first.run_dir)


def test_resume_reports_a_missing_checkpoint(tiny_cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        Trainer(tiny_cfg(), resume=tmp_path / "missing.pt")


def test_a_checkpoint_outside_a_run_directory_starts_a_new_run_dir(tiny_cfg, tmp_path):
    first = Trainer(tiny_cfg(total_timesteps=200))
    first.train()
    loose = tmp_path / "somewhere" / "policy.pt"
    loose.parent.mkdir()
    loose.write_bytes((first.run_dir / "checkpoints" / "latest.pt").read_bytes())

    resumed = Trainer(tiny_cfg(total_timesteps=400, run_name="continued"), resume=loose)
    result = resumed.train()
    assert resumed.run_dir == Path(resumed.cfg.run_dir) / "continued"
    assert result["global_step"] == 400
    assert steps_with(read_rows(resumed.run_dir), "time/sps") == [400]


def test_dqn_resume_restores_schedule_and_counters(tiny_cfg):
    first = Trainer(tiny_cfg("dqn", total_timesteps=600))
    first.train()
    second = Trainer(tiny_cfg("dqn", total_timesteps=1_000), resume=first.run_dir)
    assert second.algo.n_updates == first.algo.n_updates
    result = second.train()
    assert result["global_step"] == 1_000
    assert second.algo.n_updates > first.algo.n_updates
    last = read_rows(second.run_dir)[-1]
    assert float(last["train/epsilon"]) == pytest.approx(second.cfg.dqn.eps_end)


# --------------------------------------------------------------------------- #
# SIGINT and cleanup
# --------------------------------------------------------------------------- #


def test_sigint_saves_latest_and_returns_normally(tiny_cfg):
    cfg = tiny_cfg(total_timesteps=10_000_000, checkpoint_interval=0, eval={"interval": 0})
    trainer = Trainer(cfg)
    handler_before = signal.getsignal(signal.SIGINT)
    calls = interrupt_on_update(trainer, nth=3)

    result = trainer.train()

    assert result["interrupted"] is True
    assert result["final_eval"] is None
    assert len(calls) == 3
    assert result["global_step"] == calls[-1]
    latest = load_checkpoint(trainer.run_dir / "checkpoints" / "latest.pt")
    assert latest["global_step"] == result["global_step"]
    assert signal.getsignal(signal.SIGINT) is handler_before
    assert trainer.closed
    assert "interrupt" in (trainer.run_dir / "log.txt").read_text(encoding="utf-8").lower()


def test_an_interrupted_run_can_be_resumed(tiny_cfg):
    cfg = tiny_cfg(total_timesteps=10_000_000, eval={"interval": 0})
    trainer = Trainer(cfg)
    interrupt_on_update(trainer, nth=2)
    stopped_at = trainer.train()["global_step"]

    resumed = Trainer(
        dataclasses.replace(cfg, total_timesteps=stopped_at + 200), resume=trainer.run_dir
    )
    assert resumed.global_step == stopped_at
    assert resumed.train()["global_step"] == stopped_at + 200


def test_a_second_sigint_aborts_immediately_but_still_cleans_up(tiny_cfg):
    trainer = Trainer(tiny_cfg(total_timesteps=10_000_000))
    handler_before = signal.getsignal(signal.SIGINT)
    interrupt_on_update(trainer, nth=2, times=2)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    assert signal.getsignal(signal.SIGINT) is handler_before
    assert trainer.closed


def test_request_stop_ends_training_like_sigint(tiny_cfg):
    trainer = Trainer(tiny_cfg(total_timesteps=10_000_000))
    real_update = trainer.algo.update

    def update(global_step, progress):
        trainer.request_stop()
        return real_update(global_step, progress)

    trainer.algo.update = update
    result = trainer.train()
    assert result["interrupted"] is True
    assert result["global_step"] == 16 * 2
    assert (trainer.run_dir / "checkpoints" / "latest.pt").is_file()


def test_training_in_a_worker_thread_leaves_signal_handlers_alone(tiny_cfg):
    trainer = Trainer(tiny_cfg(total_timesteps=200))
    handler_before = signal.getsignal(signal.SIGINT)
    seen: list[object] = []
    real_update = trainer.algo.update

    def update(global_step, progress):
        seen.append(signal.getsignal(signal.SIGINT))
        return real_update(global_step, progress)

    trainer.algo.update = update
    results: list[dict] = []
    thread = threading.Thread(target=lambda: results.append(trainer.train()))
    thread.start()
    thread.join(60)
    assert results and results[0]["global_step"] == 200
    assert seen and all(handler is handler_before for handler in seen)


def test_the_sigint_handler_is_only_installed_while_training(tiny_cfg):
    handler_before = signal.getsignal(signal.SIGINT)
    trainer = Trainer(tiny_cfg(total_timesteps=200))
    assert signal.getsignal(signal.SIGINT) is handler_before
    seen: list[object] = []
    real_update = trainer.algo.update

    def update(global_step, progress):
        seen.append(signal.getsignal(signal.SIGINT))
        return real_update(global_step, progress)

    trainer.algo.update = update
    trainer.train()
    assert seen and all(handler is not handler_before for handler in seen)
    assert signal.getsignal(signal.SIGINT) is handler_before


def test_envs_and_logger_are_closed_when_training_fails(tiny_cfg):
    trainer = Trainer(tiny_cfg())
    handler_before = signal.getsignal(signal.SIGINT)

    def broken_update(global_step, progress):
        raise RuntimeError("update exploded")

    trainer.algo.update = broken_update
    with pytest.raises(RuntimeError, match="update exploded"):
        trainer.train()
    assert trainer.closed
    with pytest.raises(RuntimeError, match="closed"):
        trainer.venv.reset()
    with pytest.raises(RuntimeError, match="closed"):
        trainer.logger.log({"x": 1.0}, 1)
    assert signal.getsignal(signal.SIGINT) is handler_before


def test_a_failing_constructor_does_not_leak_envs(tiny_cfg, monkeypatch):
    created = []
    real_make_vec_env = trainer_module.make_vec_env

    def spy(*args, **kwargs):
        venv = real_make_vec_env(*args, **kwargs)
        created.append(venv)
        return venv

    monkeypatch.setattr(trainer_module, "make_vec_env", spy)
    with pytest.raises(ValueError, match="n_minibatches"):
        Trainer(tiny_cfg(ppo={"n_minibatches": 10_000}))
    assert len(created) == 1
    with pytest.raises(RuntimeError, match="closed"):
        created[0].reset()


# Workers started with "spawn" re-import __main__: everything heavy stays under the guard.
CTRL_C_SCRIPT = """
import json
import sys

if __name__ == "__main__":
    from mario_play.rl.config import config_from_dict
    from mario_play.rl.trainer import Trainer

    with open(sys.argv[1], encoding="utf-8") as fh:
        cfg = config_from_dict(json.load(fh))
    result = Trainer(cfg).train()
    print("RESULT " + json.dumps(result), flush=True)
"""


@pytest.mark.timeout(90)
def test_ctrl_c_on_a_real_training_process_with_subprocess_envs(tiny_cfg, tmp_path):
    cfg = tiny_cfg(total_timesteps=50_000_000, vec_env="subproc", eval={"interval": 0})
    (tmp_path / "cfg.json").write_text(json.dumps(config_to_dict(cfg)), encoding="utf-8")
    (tmp_path / "train.py").write_text(textwrap.dedent(CTRL_C_SCRIPT), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(tmp_path / "train.py"), str(tmp_path / "cfg.json")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    seen: list[str] = []
    try:
        for line in process.stdout:  # wait until training is demonstrably under way
            seen.append(line)
            if line.startswith("step"):
                break
        assert process.poll() is None, "".join(seen)
        process.send_signal(signal.SIGINT)
        rest, _ = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    output = "".join(seen) + rest

    assert process.returncode == 0, output
    assert "Traceback" not in output, output
    result = json.loads(next(ln for ln in output.splitlines() if ln.startswith("RESULT "))[7:])
    assert result["interrupted"] is True
    assert 0 < result["global_step"] < cfg.total_timesteps
    latest = load_checkpoint(Path(result["run_dir"]) / "checkpoints" / "latest.pt")
    assert latest["global_step"] == result["global_step"]


def test_train_cannot_run_twice(tiny_cfg):
    trainer = Trainer(tiny_cfg(total_timesteps=200))
    trainer.train()
    with pytest.raises(RuntimeError, match="closed"):
        trainer.train()


def test_unknown_algorithm_is_rejected_before_anything_is_created(tiny_cfg):
    cfg = tiny_cfg("sarsa")
    with pytest.raises(ValueError, match="sarsa"):
        Trainer(cfg)
    assert not (Path(cfg.run_dir) / "run").exists()
