"""Train replicated Taxi PPO controls with zeros, exact rules, or frozen Jev features.

The API is never contacted here. A shared immutable table is prepared separately.
Training uses ordinary Gym transitions. Evaluation exhaustively simulates all
300 valid initial states using the same deterministic Gym transition dynamics;
only the observation encoder's public state representation reaches the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from mario_play.envs.factory import make_env
from mario_play.rl.algos import get_algorithm
from mario_play.rl.checkpoint import atomic_copy, load_checkpoint, save_checkpoint
from mario_play.rl.config import (
    EnvConfig,
    NetworkConfig,
    PPOConfig,
    TrainConfig,
    config_from_dict,
    config_to_dict,
)
from mario_play.rl.trainer import Trainer
from mario_play.rl.utils import get_rng_state, set_rng_state

TAXI_ID = "mario_play.envs.taxi_jev:TaxiJev-v0"
CONDITIONS = ("zeros", "rules", "jev")
SEEDS = (0, 1, 2, 3, 4)
HORIZON = 200_000
MILESTONE_STEPS = 20_000
EPISODE_CAP = 200
GIF_STARTS = (461, 91, 244)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_digest(model: torch.nn.Module) -> str:
    """Compare weights by tensor content, without checkpoint metadata."""
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        result.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def make_config(
    root: Path,
    table: Path,
    condition: str,
    seed: int,
    *,
    horizon: int = HORIZON,
    milestone_steps: int = MILESTONE_STEPS,
) -> TrainConfig:
    """Identical PPO setup for every arm; only the auxiliary inputs differ."""
    if condition not in CONDITIONS:
        raise ValueError("unknown Taxi experimental condition")
    cfg = TrainConfig(
        algo="ppo",
        total_timesteps=horizon,
        n_envs=8,
        vec_env="sync",
        seed=seed,
        device="cpu",
        torch_deterministic=True,
        run_dir=str(root / condition),
        run_name=f"seed_{seed}",
        log_interval=milestone_steps,
        checkpoint_interval=milestone_steps,
        keep_checkpoints=max(3, horizon // milestone_steps + 1),
        tensorboard=False,
        env=EnvConfig(
            id=TAXI_ID,
            kwargs={
                "feature_mode": condition,
                "table_path": str(table) if condition == "jev" else None,
                "table_sha256": digest(table) if condition == "jev" else None,
            },
        ),
        network=NetworkConfig(encoder="mlp", mlp_hidden=[64, 64]),
        ppo=PPOConfig(lr=0.0003, n_steps=128, n_epochs=4, n_minibatches=4),
    )
    cfg.eval.interval = milestone_steps
    cfg.eval.episodes = 300
    cfg.eval.deterministic = True
    return cfg


class TaxiEvaluator:
    """Exact deterministic evaluation; the transition model is never policy input."""

    def __init__(self, env_cfg: EnvConfig) -> None:
        env = make_env(env_cfg)
        try:
            self.observations = np.asarray(
                env.get_wrapper_attr("observation_table"), dtype=np.float32
            ).copy()
            raw = env.unwrapped
            self.starts = np.flatnonzero(raw.initial_state_distrib > 0).astype(np.int64)
            self.next_state = np.empty((500, 6), dtype=np.int64)
            self.reward = np.empty((500, 6), dtype=np.float32)
            self.done = np.empty((500, 6), dtype=np.bool_)
            for state in range(500):
                for action in range(6):
                    transitions = raw.P[state][action]
                    if len(transitions) != 1 or transitions[0][0] != 1:
                        raise ValueError("Taxi evaluator requires default deterministic dynamics")
                    _, following, reward, done = transitions[0]
                    self.next_state[state, action] = following
                    self.reward[state, action] = reward
                    self.done[state, action] = done
            if len(self.starts) != 300 or self.observations.shape[0] != 500:
                raise ValueError("Taxi state space or initial distribution changed")
            self.mode = env_cfg.kwargs["feature_mode"]
        finally:
            env.close()

    def evaluate(
        self, algo: Any, *, starts: np.ndarray | None = None, cap: int = EPISODE_CAP
    ) -> dict[str, Any]:
        """All outcomes include failures at the same fixed episode cap."""
        initial = self.starts if starts is None else np.asarray(starts, dtype=np.int64)
        if cap < 1 or not len(initial) or not np.isin(initial, self.starts).all():
            raise ValueError("evaluation requires valid Taxi initial states and a positive cap")
        rng = get_rng_state()
        try:
            states = initial.copy()
            returns = np.zeros(len(states), dtype=np.float64)
            lengths = np.zeros(len(states), dtype=np.int64)
            illegal = np.zeros(len(states), dtype=np.int64)
            successes = np.zeros(len(states), dtype=np.bool_)
            active = np.ones(len(states), dtype=np.bool_)
            for _ in range(cap):
                indices = np.flatnonzero(active)
                if not len(indices):
                    break
                current = states[indices]
                actions = np.asarray(
                    algo.predict(self.observations[current], deterministic=True), dtype=np.int64
                )
                if actions.shape != indices.shape or np.any(actions < 0) or np.any(actions > 5):
                    raise ValueError("policy returned invalid Taxi actions")
                rewards = self.reward[current, actions]
                terminated = self.done[current, actions]
                states[indices] = self.next_state[current, actions]
                returns[indices] += rewards
                lengths[indices] += 1
                illegal[indices] += (actions >= 4) & (rewards == -10)
                successes[indices] = terminated
                active[indices] = ~terminated
            rows = [
                {
                    "initial_state": int(start),
                    "final_state": int(final),
                    "return": float(ret),
                    "steps": int(length),
                    "illegal_pickup_dropoff": int(bad),
                    "success": bool(success),
                    "terminated": bool(success),
                    "truncated": int(length) == cap,
                }
                for start, final, ret, length, bad, success in zip(
                    initial, states, returns, lengths, illegal, successes, strict=True
                )
            ]
            return {
                "episodes": len(rows),
                "successes": int(successes.sum()),
                "success_rate": float(successes.mean()),
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std()),
                "mean_length": float(lengths.mean()),
                "mean_illegal_pickup_dropoff": float(illegal.mean()),
                "logical_feature_refreshes": (int(lengths.sum()) if self.mode != "zeros" else 0),
                "feature_count_semantics": "actual batched policy observation lookups",
                "physical_api_requests": 0,
                "episode_cap": cap,
                "episode_results": rows,
            }
        finally:
            set_rng_state(rng)


def logical_refreshes(trainer: Trainer) -> int:
    """Diagnostic counter supplied by each wrapper, including episode resets."""
    return sum(int(env.get_wrapper_attr("feature_refreshes")) for env in trainer.venv.envs)


class TaxiTrainer(Trainer):
    """Ordinary PPO training with deterministic exhaustive milestone assessment."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        try:
            self.initial_model_sha256 = model_digest(self.algo.model)
            self.evaluator = TaxiEvaluator(self.cfg.env)
            self.milestones: list[dict[str, Any]] = []
            self.evaluation_seconds = 0.0
            self.training_stats: dict[str, Any] = {}
            self._save(self.ckpt_dir / "initial.pt")
            self.initial_checkpoint_sha256 = digest(self.ckpt_dir / "initial.pt")
        except BaseException:
            self.close()
            raise

    def _save(self, path: Path) -> None:
        elapsed = (
            self._elapsed_before + time.perf_counter() - self._train_started
            if self._train_started
            else 0.0
        )
        save_checkpoint(
            path,
            algo_name=self.cfg.algo,
            algo_state=self.algo.state_dict(),
            config_dict=config_to_dict(self.cfg),
            global_step=self.global_step,
            best_eval=self.best_eval,
            rng_state=get_rng_state(),
            extra={
                "episodes": self._episodes,
                "elapsed": elapsed,
                "initial_model_sha256": self.initial_model_sha256,
                "ppo_updates": self.algo.n_updates,
                "logical_feature_refreshes": logical_refreshes(self),
            },
        )

    def _training_row(self) -> dict[str, Any]:
        row = super()._training_row()
        self.training_stats = {"global_step": self.global_step, **row}
        return row

    def _evaluate(self) -> dict[str, float | int]:
        started = time.perf_counter()
        assessment = self.evaluator.evaluate(self.algo)
        self.evaluation_seconds += time.perf_counter() - started
        checkpoint = self.run_dir / "milestones" / f"checkpoint_{self.global_step}.pt"
        self._save(checkpoint)
        row = {
            "global_step": self.global_step,
            "ppo_updates": self.algo.n_updates,
            "optimizer_steps": self.algo.n_updates
            * self.cfg.ppo.n_epochs
            * self.cfg.ppo.n_minibatches,
            "checkpoint": str(checkpoint.relative_to(self.run_dir)),
            "checkpoint_sha256": digest(checkpoint),
            "assessment": assessment,
            "logical_feature_refreshes": logical_refreshes(self),
            "training": self.training_stats,
            "elapsed_seconds": time.perf_counter() - self._train_started,
            "evaluation_seconds": self.evaluation_seconds,
        }
        self.milestones.append(row)
        write_json(self.run_dir / "milestones.json", self.milestones)
        score = assessment["mean_return"]
        if self.best_eval is None or score > self.best_eval:
            self.best_eval = score
        summary = {
            key: assessment[key]
            for key in (
                "episodes",
                "success_rate",
                "mean_return",
                "std_return",
                "mean_length",
                "mean_illegal_pickup_dropoff",
            )
        }
        self._last_eval = summary
        self.logger.print(
            f"Taxi {self.cfg.env.kwargs['feature_mode']} seed {self.cfg.seed} "
            f"@ {self.global_step:,}: "
            f"{assessment['successes']}/300 successes; return {score:.2f}; "
            f"illegal {assessment['mean_illegal_pickup_dropoff']:.2f}/episode"
        )
        write_json(self.run_dir / "progress.json", self.snapshot())
        return summary

    def snapshot(self) -> dict[str, Any]:
        return {
            "condition": self.cfg.env.kwargs["feature_mode"],
            "training_seed": self.cfg.seed,
            "global_step": self.global_step,
            "target_steps": self.cfg.total_timesteps,
            "initial_model_sha256": self.initial_model_sha256,
            "ppo_updates": self.algo.n_updates,
            "logical_feature_refreshes": logical_refreshes(self),
            "physical_api_requests": 0,
            "evaluation_seconds": self.evaluation_seconds,
            "last_assessment": self.milestones[-1]["assessment"] if self.milestones else None,
        }


def load_portable_algorithm(checkpoint: Path, table: Path) -> tuple[Any, TrainConfig]:
    """Rebuild a saved policy using an explicitly relocated, digest-verified table."""
    payload = load_checkpoint(checkpoint)
    cfg = config_from_dict(payload["config"])
    if cfg.env.kwargs["feature_mode"] == "jev":
        if digest(table) != cfg.env.kwargs["table_sha256"]:
            raise ValueError("Taxi feature table changed since training")
        cfg.env.kwargs["table_path"] = str(table)
    env = make_env(cfg.env)
    try:
        algo = get_algorithm("ppo")(
            env.observation_space, env.action_space, cfg, torch.device("cpu"), cfg.n_envs
        )
    finally:
        env.close()
    algo.load_state_dict(payload["algo_state"])
    return algo, cfg


def run_experiment(
    run_dir: Path,
    table: Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
    horizon: int = HORIZON,
    milestone_steps: int = MILESTONE_STEPS,
) -> dict[str, Any]:
    """Run all predeclared conditions; completed runs are verified and reused, never overwritten."""
    results = {}
    initial_hashes: dict[int, str] = {}
    started = time.perf_counter()
    for seed in seeds:
        order = CONDITIONS[seed % 3 :] + CONDITIONS[: seed % 3]
        for condition in order:
            cfg = make_config(
                run_dir, table, condition, seed, horizon=horizon, milestone_steps=milestone_steps
            )
            directory = Path(cfg.run_dir) / cfg.run_name
            result_path = directory / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                final = directory / "checkpoints/final.pt"
                if (
                    result["status"] != "complete"
                    or result["global_step"] != horizon
                    or result["checkpoint_sha256"] != digest(final)
                    or load_checkpoint(final)["config"] != config_to_dict(cfg)
                ):
                    raise ValueError("existing Taxi result does not match this fixed experiment")
            else:
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError(f"incomplete existing run requires inspection: {directory}")
                trainer = TaxiTrainer(cfg)
                try:
                    expected = initial_hashes.setdefault(seed, trainer.initial_model_sha256)
                    if trainer.initial_model_sha256 != expected:
                        raise ValueError(
                            "paired conditions must start with identical model weights"
                        )
                    outcome = trainer.train()
                    result = trainer.snapshot()
                    result["status"] = "interrupted" if outcome["interrupted"] else "complete"
                    if not outcome["interrupted"]:
                        final = trainer.ckpt_dir / "final.pt"
                        atomic_copy(trainer.ckpt_dir / "latest.pt", final)
                        result["checkpoint_sha256"] = digest(final)
                    write_json(result_path, result)
                    if outcome["interrupted"]:
                        return {"status": "interrupted", "latest": result}
                finally:
                    trainer.close()
            expected = initial_hashes.setdefault(seed, result["initial_model_sha256"])
            if result["initial_model_sha256"] != expected:
                raise ValueError("paired initial-weight hashes differ")
            results[f"{condition}/seed_{seed}"] = result
            write_json(
                run_dir / "progress.json",
                {
                    "status": "running",
                    "completed_runs": len(results),
                    "planned_runs": len(seeds) * len(CONDITIONS),
                    "results": results,
                },
            )
    result = {
        "status": "complete",
        "completed_runs": len(results),
        "results": results,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "paired_initial_hashes": {str(seed): value for seed, value in initial_hashes.items()},
        "physical_api_requests_during_training": 0,
    }
    write_json(run_dir / "progress.json", result)
    return result


def benchmark(run_dir: Path, table: Path, steps: int = 4096) -> dict[str, Any]:
    """Measure local throughput on a disjoint seed, without selecting learning settings."""
    cfg = make_config(run_dir, table, "zeros", 100_000, horizon=steps, milestone_steps=steps)
    cfg.eval.interval = 0
    cfg.eval.episodes = 0
    started = time.perf_counter()
    trainer = Trainer(cfg)
    try:
        trainer.train()
        training_seconds = time.perf_counter() - started
        evaluator = TaxiEvaluator(cfg.env)
        started = time.perf_counter()
        evaluator.evaluate(trainer.algo)
        eval_seconds = time.perf_counter() - started
    finally:
        trainer.close()
    result = {
        "benchmark_steps": steps,
        "training_seconds": training_seconds,
        "training_steps_per_second": steps / training_seconds,
        "all_300_start_evaluation_seconds": eval_seconds,
        "estimated_15_run_seconds": training_seconds / steps * HORIZON * 15 + eval_seconds * 150,
        "purpose": "throughput only; seed excluded from experiment",
    }
    write_json(run_dir / "benchmark.json", result)
    return result


def source_snapshot(root: Path) -> dict[str, str]:
    """Archive only public source/config/protocol files; never credentials or run outputs."""
    repository = Path(__file__).resolve().parents[1]
    paths = sorted((repository / "src").rglob("*.py"))
    paths.extend(
        repository / name
        for name in (
            "scripts/train_taxi_jev.py",
            "scripts/report_taxi_jev.py",
            "scripts/build_taxi_jev_table.py",
            "scripts/build_jev_feature_table.py",
            "docs/taxi-jev-protocol.md",
            "pyproject.toml",
            "uv.lock",
        )
    )
    hashes = {}
    snapshot = root / "source_snapshot.zip"
    if snapshot.exists():
        raise ValueError("source snapshot already exists without an experiment manifest")
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source snapshot requires a regular file: {path.name}")
            name = str(path.relative_to(repository))
            archive.write(path, name)
            hashes[name] = digest(path)
    return hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    root, source = args.run_dir.resolve(), args.table.resolve()
    root.mkdir(parents=True, exist_ok=True)
    table = root / "table.json"
    if table.exists() and digest(table) != digest(source):
        raise ValueError("existing run uses another frozen table")
    if table != source and not table.exists():
        shutil.copy2(source, table)
    if args.benchmark_only:
        print(json.dumps(benchmark(root / "throughput-only", table)), flush=True)
        return 0
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": "Taxi-v4",
        "gymnasium_version": gym.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "horizon": HORIZON,
        "milestone_steps": MILESTONE_STEPS,
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "table_sha256": digest(table),
        "action_mask": "unused in every condition",
        "evaluation": "greedy policy from every one of the 300 valid initial states",
        "primary": "normalized trapezoidal success AUC over 20k..200k and final success",
        "secondary": "90% success at three consecutive evaluated milestones",
        "gif_initial_states": list(GIF_STARTS),
        "gif_training_seed": 0,
        "precomputation": json.loads(table.read_text()).get("usage", {}),
        "physical_api_requests_during_training": 0,
        "configs": [
            config_to_dict(make_config(root, table, condition, seed))
            for seed in SEEDS
            for condition in CONDITIONS
        ],
    }
    if (root / "experiment.json").exists():
        previous = json.loads((root / "experiment.json").read_text())
        for key, value in manifest.items():
            if key != "created_utc" and previous.get(key) != value:
                raise ValueError(f"fixed Taxi experiment manifest differs: {key}")
        repository = Path(__file__).resolve().parents[1]
        for name, expected in previous["source_hashes"].items():
            if digest(repository / name) != expected:
                raise ValueError(f"fixed Taxi experiment source changed: {name}")
    else:
        manifest["source_hashes"] = source_snapshot(root)
        write_json(root / "experiment.json", manifest)
    result = run_experiment(root, table)
    print(json.dumps({key: value for key, value in result.items() if key != "results"}), flush=True)
    return 130 if result["status"] == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
