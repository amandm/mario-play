"""Algorithm correctness: PPO and Double DQN solve CartPole-v1 with the shipped configs (CPU).

The assertion is on the best periodic evaluation (`Trainer.train()["best_eval"]`,
the policy kept as `best.pt`), each a mean over 20 deterministic episodes. The
configs were tuned so that many seeds pass with a wide margin; see the comments
next to the thresholds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mario_play.rl.checkpoint import load_checkpoint
from mario_play.rl.config import TrainConfig, load_config
from mario_play.rl.evaluate import evaluate, load_algorithm
from mario_play.rl.trainer import Trainer

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

pytestmark = [pytest.mark.slow, pytest.mark.timeout(600)]


def shipped_config(name: str, tmp_path: Path) -> TrainConfig:
    overrides = [f"run_dir={tmp_path}", "run_name=cartpole", "tensorboard=false"]
    cfg = load_config(CONFIG_DIR / name, overrides)
    assert cfg.env.id == "CartPole-v1"
    assert cfg.device == "cpu", "tiny MLPs are slower on MPS/CUDA; the shipped default must be cpu"
    assert cfg.eval.episodes == 20 and cfg.eval.deterministic
    return cfg


def train_and_check(cfg: TrainConfig, threshold: float) -> None:
    result = Trainer(cfg).train()
    assert not result["interrupted"]
    assert result["global_step"] >= cfg.total_timesteps
    assert result["best_eval"] >= threshold, (
        f"{cfg.algo} reached a best mean evaluation return of {result['best_eval']:.1f} "
        f"in {result['global_step']:,} steps, expected >= {threshold}"
    )

    # best.pt really is that policy: reloaded from disk alone, it reproduces the score.
    best_path = Path(result["run_dir"]) / "checkpoints" / "best.pt"
    assert load_checkpoint(best_path)["best_eval"] == result["best_eval"]
    algo, loaded_cfg = load_algorithm(best_path, device="cpu")
    score = evaluate(algo, loaded_cfg.env, cfg.eval.episodes, cfg.eval.seed, cfg.eval.deterministic)
    assert score["mean_return"] == pytest.approx(result["best_eval"])


def test_ppo_solves_cartpole(tmp_path: Path) -> None:
    cfg = shipped_config("ppo_cartpole.yaml", tmp_path)
    assert cfg.algo == "ppo" and cfg.total_timesteps <= 150_000
    # Seeds 0-9 all reach a best evaluation of 500 (the maximum) and pass 195 by 20k steps.
    train_and_check(cfg, threshold=195.0)


def test_dqn_solves_cartpole(tmp_path: Path) -> None:
    cfg = shipped_config("dqn_cartpole.yaml", tmp_path)
    assert cfg.algo == "dqn" and cfg.total_timesteps <= 100_000
    # Seeds 0-15 all reach a best evaluation of 500 and pass 150 within the first 25k steps.
    train_and_check(cfg, threshold=150.0)
