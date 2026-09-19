"""Where all subsystems meet: game -> env -> vec env -> algorithm -> trainer -> checkpoint -> CLI.

Every training run here is a smoke test (2 envs, a few thousand steps at most,
tiny networks, CPU): it proves that the pieces fit together, not that anything
is learned. Runs go through `mario_play.cli.main`, exactly like a user's would.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from mario_play.cli import main
from mario_play.rl.config import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
MARIO_CONFIGS = ("ppo_grid", "ppo_pixels", "dqn_grid", "dqn_pixels")

# Scale any shipped config down to a smoke test. Episodes on `flat` have no natural end
# for a policy that dawdles (no enemies, no pits, 400 s on the clock), hence the step limit.
SMOKE = [
    "n_envs=2",
    "vec_env=sync",
    "device=cpu",
    "tensorboard=false",
    "network.hidden_size=32",
    "env.level=flat",
    "env.max_episode_steps=60",
    "eval.episodes=1",
    "eval.interval=0",
    "log_interval=500",
    "checkpoint_interval=1000",
    "ppo.n_steps=32",
    "ppo.n_epochs=2",
    "ppo.n_minibatches=2",
    "dqn.buffer_size=2000",
    "dqn.learning_starts=200",
    "dqn.batch_size=16",
    "dqn.train_freq=4",
    "dqn.gradient_steps=1",
    "dqn.target_update_interval=200",
    "dqn.eps_decay_steps=1000",
]


def read_metrics(run_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def train(config: str, run_dir: Path, run_name: str, *overrides: str) -> Path:
    argv = ["train", "--config", str(CONFIGS / f"{config}.yaml"), *SMOKE]
    assert main([*argv, f"run_dir={run_dir}", f"run_name={run_name}", *overrides]) == 0
    return run_dir / run_name


@pytest.fixture(scope="module")
def ppo_run(tmp_path_factory) -> Path:
    """Run directory of a 2k-step PPO / grid / `flat` run; shared, so tests must not modify it."""
    return train("ppo_grid", tmp_path_factory.mktemp("ppo"), "smoke", "total_timesteps=2000")


@pytest.fixture(scope="module")
def dqn_run(tmp_path_factory) -> Path:
    """Run directory of a short DQN / grid / `flat` run; shared, so tests must not modify it."""
    return train("dqn_grid", tmp_path_factory.mktemp("dqn"), "smoke", "total_timesteps=1200")


# --- shipped configs ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MARIO_CONFIGS)
def test_shipped_config_loads_with_the_documented_settings(name):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    algo, obs_mode = name.split("_")
    assert cfg.algo == algo
    assert cfg.env.id == "MarioPlay-v0"
    assert cfg.env.obs_mode == obs_mode
    assert cfg.env.level == "1-1"
    assert cfg.env.stall_steps is not None
    assert cfg.device == "auto"
    if obs_mode == "pixels":
        assert (cfg.env.grayscale, cfg.env.resize, cfg.env.frame_stack) == (True, [84, 84], 4)
        assert cfg.vec_env == "subproc"
    else:
        assert cfg.vec_env == "sync"
    if name == "ppo_grid":
        assert cfg.n_envs == 8
    if name == "ppo_pixels":
        assert (cfg.n_envs, cfg.total_timesteps) == (16, 10_000_000)
        assert (cfg.ppo.n_steps * cfg.n_envs) % cfg.ppo.n_minibatches == 0
    if name == "dqn_pixels":
        assert cfg.dqn.buffer_size == 100_000


def test_shipped_configs_spell_out_every_section():
    """Catches a config that silently falls back to defaults for its own algorithm."""
    for name in MARIO_CONFIGS:
        data = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
        assert {"env", "network", "eval", data["algo"]} <= set(data), name
        other = "dqn" if data["algo"] == "ppo" else "ppo"
        assert other not in data, f"{name} configures the algorithm it does not use"


@pytest.mark.parametrize("name", ["ppo_pixels", "dqn_pixels"])
def test_pixel_configs_train_and_record(name, tmp_path, capsys):
    """Builds env + algorithm from the shipped pixel config and runs real updates on (4, 84, 84)."""
    overrides = [
        "total_timesteps=96",
        "ppo.n_steps=16",
        "dqn.learning_starts=32",
        "eval.episodes=0",
    ]
    run_dir = train(name, tmp_path, "pixels", *overrides)
    columns, rows = read_metrics(run_dir)
    assert any(column.startswith("train/") for column in columns), columns
    assert int(float(rows[-1]["step"])) == 96

    # The policy sees 84x84 gray stacks; the recording must still be the full game frame.
    out = tmp_path / "pixels.gif"
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    argv = ["record", "--checkpoint", str(checkpoint), "--out", str(out), "--max-steps", "5"]
    assert main(argv) == 0
    with Image.open(out) as image:
        assert image.size == (256, 240)
        assert image.n_frames == 6


# --- PPO: train -> artefacts -> eval -> resume ----------------------------------------------------


def test_ppo_run_directory_is_complete(ppo_run):
    names = {p.name for p in (ppo_run / "checkpoints").iterdir()}
    assert {"latest.pt", "best.pt", "ckpt_1000.pt", "ckpt_2000.pt"} <= names
    assert (ppo_run / "config.yaml").is_file()
    assert (ppo_run / "log.txt").is_file()
    stored = load_config(ppo_run / "config.yaml")
    assert (stored.env.level, stored.n_envs, stored.network.hidden_size) == ("flat", 2, 32)


def test_trainer_logs_the_mario_extras(ppo_run):
    columns, rows = read_metrics(ppo_run)
    for column in (
        "rollout/ep_return_mean",
        "rollout/flag_rate",
        "rollout/mean_progress",
        "eval/mean_return",
        "eval/flag_rate",
        "eval/mean_progress",
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
    ):
        assert column in columns, (column, columns)
    last = rows[-1]
    assert int(float(last["step"])) == 2000
    assert 0.0 <= float(last["rollout/flag_rate"]) <= 1.0
    assert 0.0 < float(last["rollout/mean_progress"]) <= 1.0
    assert 0.0 <= float(last["eval/mean_progress"]) <= 1.0
    assert np.isfinite(float(last["train/policy_loss"]))


def test_eval_checkpoint_matches_the_library_evaluation(ppo_run, capsys):
    from mario_play.rl.evaluate import evaluate, load_algorithm

    checkpoint = ppo_run / "checkpoints" / "best.pt"
    assert main(["eval", "--checkpoint", str(checkpoint), "--episodes", "2", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["level"] == "flat"
    assert result["episodes"] == 2

    algo, cfg = load_algorithm(checkpoint, device="cpu")
    expected = evaluate(algo, cfg.env, episodes=2, seed=cfg.eval.seed, deterministic=True)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value), key


def test_eval_checkpoint_table_level_override_and_stochastic(ppo_run, capsys):
    argv = ["eval", "--checkpoint", str(ppo_run), "--level", "1-1", "--stochastic"]
    assert main([*argv, "--episodes", "1", "--max-steps", "30"]) == 0  # a run dir: latest.pt
    out = capsys.readouterr().out
    assert "latest.pt" in out
    assert "1-1" in out
    assert "mean_return" in out


def test_watch_a_trained_policy(ppo_run):
    checkpoint = ppo_run / "checkpoints" / "latest.pt"
    argv = ["watch", "--checkpoint", str(checkpoint), "--episodes", "1", "--max-steps", "3"]
    assert main(argv) == 0  # a real window on SDL's dummy driver


def test_resume_continues_the_run(ppo_run, tmp_path, capsys):
    run_dir = tmp_path / "resumed"
    shutil.copytree(ppo_run, run_dir)
    # No --config: the checkpoint's own config, with a longer budget on top.
    assert main(["train", "--resume", str(run_dir), "total_timesteps=3000"]) == 0
    out = capsys.readouterr().out
    assert "resumed from" in out
    assert str(run_dir) in out

    names = {p.name for p in (run_dir / "checkpoints").iterdir()}
    assert "ckpt_3000.pt" in names
    steps = [int(float(row["step"])) for row in read_metrics(run_dir)[1]]
    assert steps == sorted(steps)
    assert steps[-1] == 3000
    assert 2000 in steps  # the first run's rows are still there
    assert load_config(run_dir / "config.yaml").total_timesteps == 3000
    # The shared fixture is untouched.
    assert not (ppo_run / "checkpoints" / "ckpt_3000.pt").exists()


def test_resume_with_an_explicit_config(ppo_run, tmp_path):
    run_dir = tmp_path / "resumed"
    shutil.copytree(ppo_run, run_dir)
    checkpoint = run_dir / "checkpoints" / "ckpt_1000.pt"
    argv = ["train", "--config", str(run_dir / "config.yaml"), "--resume", str(checkpoint)]
    assert main([*argv, "total_timesteps=1500"]) == 0
    assert (run_dir / "checkpoints" / "ckpt_1500.pt").is_file()


def test_resume_with_the_wrong_algorithm_is_a_one_line_error(ppo_run, tmp_path, capsys):
    run_dir = tmp_path / "resumed"
    shutil.copytree(ppo_run, run_dir)
    assert main(["train", "--resume", str(run_dir), "algo=dqn"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_interrupted_training_exits_130_and_resumes(tmp_path, monkeypatch, capsys):
    """Ctrl-C: `latest.pt` is saved, the exit code says so, and `--resume` finishes the run."""
    from mario_play.rl.trainer import Trainer

    real_train = Trainer.train

    def interrupted_train(self):
        self.request_stop()  # what the first SIGINT does
        return real_train(self)

    argv = ["train", "--config", str(CONFIGS / "ppo_grid.yaml"), *SMOKE]
    argv += [f"run_dir={tmp_path}", "run_name=ctrl_c", "total_timesteps=128"]
    with monkeypatch.context() as patched:
        patched.setattr(Trainer, "train", interrupted_train)
        assert main(argv) == 130
    run_dir = tmp_path / "ctrl_c"
    assert (run_dir / "checkpoints" / "latest.pt").is_file()
    assert f"mario-play train --resume {run_dir}" in capsys.readouterr().out

    assert main(["train", "--resume", str(run_dir)]) == 0
    assert (run_dir / "checkpoints" / "ckpt_128.pt").is_file()


# --- DQN ------------------------------------------------------------------------------------------


def test_dqn_trains_and_its_checkpoint_plays(dqn_run, tmp_path, capsys):
    columns, rows = read_metrics(dqn_run)
    assert {"train/loss", "train/epsilon", "rollout/mean_progress"} <= set(columns), columns
    assert int(float(rows[-1]["step"])) == 1200
    assert 0.0 < float(rows[-1]["train/epsilon"]) < 1.0  # the schedule ran on global_step
    assert (dqn_run / "checkpoints" / "best.pt").is_file()

    checkpoint = dqn_run / "checkpoints" / "latest.pt"
    assert main(["eval", "--checkpoint", str(checkpoint), "--episodes", "1", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert {"mean_return", "std_return", "mean_length", "flag_rate", "mean_progress"} <= set(result)

    out = tmp_path / "dqn.gif"
    argv = ["record", "--checkpoint", str(checkpoint), "--out", str(out), "--max-steps", "8"]
    assert main(argv) == 0
    with Image.open(out) as image:
        assert image.size == (256, 240)


# --- subprocess vec env with the real game --------------------------------------------------------


@pytest.mark.timeout(180)
def test_training_with_subprocess_envs(tmp_path):
    """Spawned workers must be able to import mario_play and build the Mario env by themselves."""
    run_dir = train("ppo_grid", tmp_path, "subproc", "vec_env=subproc", "total_timesteps=256")
    columns, rows = read_metrics(run_dir)
    assert "train/policy_loss" in columns
    assert int(float(rows[-1]["step"])) == 256
    assert load_config(run_dir / "config.yaml").vec_env == "subproc"


# --- baselines as reference points ----------------------------------------------------------------


def test_baselines_rank_as_expected(capsys):
    """Sanity of game + env + reward: heading for the flag pays, flailing does not."""

    def score(agent: str, level: str) -> dict:
        argv = ["eval", "--agent", agent, "--level", level, "--episodes", "3", "--json"]
        assert main([*argv, "--max-steps", "300"]) == 0
        return json.loads(capsys.readouterr().out)

    # `flat` is the sanity level: even random play stumbles into the flag, only later.
    heuristic, random = score("heuristic", "flat"), score("random", "flat")
    assert heuristic["flag_rate"] == 1.0
    assert heuristic["mean_length"] < random["mean_length"]
    assert heuristic["mean_return"] > random["mean_return"]

    # On a real level the reward must separate them clearly.
    heuristic, random = score("heuristic", "1-1"), score("random", "1-1")
    assert heuristic["mean_progress"] > 2 * random["mean_progress"]
    assert heuristic["mean_return"] > random["mean_return"] + 20
