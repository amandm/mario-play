"""Train one PPO policy on level 1-1 and record how its behavior improves.

Run on Colab with ``python scripts/train_ppo_agent.py --run-dir /content/mario-ppo-agent``.
Resume the same run with ``--resume /content/mario-ppo-agent``. The 5M-step learning-rate
horizon stays fixed when validation reaches the stopping criterion earlier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mario_play.rl.checkpoint import atomic_copy, load_checkpoint, save_checkpoint
from mario_play.rl.config import TrainConfig, config_from_dict, config_to_dict, load_config
from mario_play.rl.evaluate import evaluate, load_algorithm
from mario_play.rl.trainer import Trainer, _resolve_checkpoint
from mario_play.rl.utils import get_rng_state, resolve_device, set_rng_state

VALIDATION_SEED = 2_000_000
HELD_OUT_SEED = 3_000_000
TARGET_STEPS = 5_000_000


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ready(record: dict[str, Any]) -> bool:
    return record["sampled"]["flag_rate"] >= 0.9 and record["greedy"]["flag_rate"] == 1.0


def _rank(record: dict[str, Any]) -> tuple[float, ...]:
    """Select using validation success first; held-out episodes never select a policy."""
    return (
        float(_ready(record)),
        float(record["sampled"]["flag_rate"]),
        float(record["greedy"]["flag_rate"]),
        float(record["sampled"]["mean_progress"]),
        float(record["sampled"]["mean_return"]),
    )


class LearningTrainer(Trainer):
    """Add learning milestones to the existing PPO loop without changing its updates."""

    def __init__(self, cfg: TrainConfig, resume: Path | None = None) -> None:
        self.training_stats: dict[str, Any] = {}
        self.milestones: list[dict[str, Any]] = []
        self.reached_criterion = False
        self.status = "running"
        super().__init__(cfg, resume=resume)
        try:
            (self.run_dir / "milestones").mkdir(exist_ok=True)
            records = _read_json(self.run_dir / "learning_milestones.json", [])
            self.milestones = [row for row in records if row["global_step"] <= self.global_step]
            if len(self.milestones) != len(records):
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                _write_json(self.run_dir / f"learning_milestones.superseded-{stamp}.json", records)
                self._write_milestones()
            if self.milestones:
                selected = max(self.milestones, key=_rank)
                atomic_copy(self.run_dir / selected["checkpoint"], self.ckpt_dir / "selected.pt")
                if self.milestones[-1]["global_step"] == self.global_step:
                    self.reached_criterion = _ready(self.milestones[-1])
                    if self.reached_criterion:
                        self.request_stop()
            if self.global_step == 0:
                # Trainer's elapsed clock starts in train(), so save step zero explicitly.
                save_checkpoint(
                    self.ckpt_dir / "initial.pt",
                    algo_name=self.cfg.algo,
                    algo_state=self.algo.state_dict(),
                    config_dict=config_to_dict(self.cfg),
                    global_step=0,
                    best_eval=None,
                    rng_state=get_rng_state(),
                    extra={"episodes": 0, "elapsed": 0.0},
                )
                atomic_copy(self.ckpt_dir / "initial.pt", self.ckpt_dir / "latest.pt")
                self._last_ckpt_step = 0
            self.write_progress()
        except BaseException:
            self.close()
            raise

    def _training_row(self) -> dict[str, Any]:
        row = super()._training_row()
        self.training_stats = {"global_step": self.global_step, **row}
        return row

    def _periodic_work(self, finished: bool) -> None:
        super()._periodic_work(finished)
        if (
            self.training_stats.get("global_step") == self.global_step
            or self._last_ckpt_step == self.global_step
        ):
            self.write_progress()

    def _evaluate(self) -> dict[str, float | int]:
        state = get_rng_state()
        started = time.perf_counter()
        try:
            sampled = super()._evaluate()
            greedy = evaluate(
                self.algo,
                self.cfg.env,
                episodes=1,
                seed=self.cfg.eval.seed,
                deterministic=True,
            )
        finally:
            set_rng_state(state)
        path = self.run_dir / "milestones" / f"checkpoint_{self.global_step}.pt"
        self._save(path)
        self._save_numbered()
        record = {
            "global_step": self.global_step,
            "checkpoint": str(path.relative_to(self.run_dir)),
            "checkpoint_sha256": _digest(path),
            "validation_seed": self.cfg.eval.seed,
            "sampled": sampled,
            "greedy": greedy,
            "training": self.training_stats,
            "evaluation_seconds": time.perf_counter() - started,
        }
        self.milestones = [row for row in self.milestones if row["global_step"] != self.global_step]
        self.milestones.append(record)
        self.milestones.sort(key=lambda row: row["global_step"])
        selected = max(self.milestones, key=_rank)
        atomic_copy(self.run_dir / selected["checkpoint"], self.ckpt_dir / "selected.pt")
        self._write_milestones()
        self.reached_criterion = _ready(record)
        self.logger.print(
            f"learning milestone {self.global_step:,}: sampled completions "
            f"{sampled['flag_rate']:.0%}, progress {sampled['mean_progress']:.1%}; "
            f"greedy completion {bool(greedy['flag_rate'])}"
        )
        if self.reached_criterion:
            self.logger.print("Validation criterion reached; saving this single PPO run.")
            self.request_stop()
        return sampled

    def _write_milestones(self) -> None:
        _write_json(self.run_dir / "learning_milestones.json", self.milestones)
        rows = []
        for milestone in self.milestones:
            row = {"global_step": milestone["global_step"], "checkpoint": milestone["checkpoint"]}
            for mode in ("sampled", "greedy"):
                row.update({f"{mode}/{key}": value for key, value in milestone[mode].items()})
            row.update({f"training/{key}": value for key, value in milestone["training"].items()})
            rows.append(row)
        path = self.run_dir / "learning_milestones.csv"
        temporary = path.with_suffix(".csv.tmp")
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or ["global_step", "checkpoint"])
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)

    def write_progress(self, **extra: Any) -> dict[str, Any]:
        """Publish a complete snapshot for local artifact sync and user updates."""
        selected = max(self.milestones, key=_rank) if self.milestones else None
        progress = {
            "status": self.status,
            "method": "PPO",
            "level": self.cfg.env.level,
            "training_seed": self.cfg.seed,
            "global_step": self.global_step,
            "target_steps": self.cfg.total_timesteps,
            "training": self.training_stats,
            "last_validation": self.milestones[-1] if self.milestones else None,
            "selected_checkpoint": "checkpoints/selected.pt" if selected else None,
            "selected_checkpoint_step": selected["global_step"] if selected else None,
            "stop_criterion": "sampled validation flag rate >= 0.90 and greedy completion",
            "validation_seed": self.cfg.eval.seed,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        _write_json(self.run_dir / "progress.json", progress)
        return progress


def _check_resume(cfg: TrainConfig, checkpoint: Path) -> None:
    # Older checkpoints omit fields introduced later; compare their effective defaults.
    previous = config_to_dict(config_from_dict(load_checkpoint(checkpoint)["config"]))
    current = config_to_dict(cfg)
    for key in ("run_dir", "run_name", "device"):
        previous.pop(key, None)
        current.pop(key, None)
    if previous != current:
        raise ValueError("resume training settings differ, including the learning-rate horizon")


def train_agent(
    cfg: TrainConfig, resume: Path | None = None, *, held_out_episodes: int = 100
) -> dict[str, Any]:
    """Train one policy, then assess the validation-selected checkpoint independently."""
    run_dir = Path(cfg.run_dir) / cfg.run_name
    if resume is not None:
        _check_resume(cfg, resume)
        completed = _read_json(run_dir / "progress.json", {})
        if (
            completed.get("status") == "complete"
            and completed.get("global_step") == load_checkpoint(resume)["global_step"]
        ):
            selected = run_dir / "checkpoints" / "selected.pt"
            assessment = _read_json(run_dir / "final_assessment.json", {})
            if assessment.get("checkpoint_sha256") == _digest(selected):
                print(json.dumps(completed), flush=True)
                return completed
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"{run_dir} already contains data; pass --resume to continue it")

    trainer = LearningTrainer(cfg, resume=resume)
    if trainer.run_dir.resolve() != run_dir.resolve():
        trainer.close()
        raise RuntimeError(f"trainer claimed an unexpected run directory: {trainer.run_dir}")
    try:
        result = trainer.train()
        if result["interrupted"] and not trainer.reached_criterion:
            trainer.status = "interrupted"
            return trainer.write_progress(stop_reason="interrupted")
        stop_reason = "validation_criterion" if trainer.reached_criterion else "step_budget"
        trainer.status = "assessing"
        trainer.write_progress(stop_reason=stop_reason)
        checkpoint = trainer.ckpt_dir / "selected.pt"
        # Rebuilding a network consumes RNG; isolate it as well as sampled evaluation.
        state = get_rng_state()
        started = time.perf_counter()
        try:
            algo, selected_cfg = load_algorithm(checkpoint, device=cfg.device)
            held_out = {
                mode: evaluate(
                    algo,
                    selected_cfg.env,
                    episodes=held_out_episodes if mode == "sampled" else 1,
                    seed=HELD_OUT_SEED,
                    deterministic=mode == "greedy",
                )
                for mode in ("sampled", "greedy")
            }
        finally:
            set_rng_state(state)
        assessment = {
            "checkpoint": "checkpoints/selected.pt",
            "checkpoint_step": load_checkpoint(checkpoint)["global_step"],
            "checkpoint_sha256": _digest(checkpoint),
            "seed": HELD_OUT_SEED,
            "held_out": held_out,
            "evaluation_seconds": time.perf_counter() - started,
            "selection": (
                "meets both readiness criteria, sampled success, greedy completion, "
                "sampled progress, sampled return"
            ),
            "scope": "One original Mario-style level, 1-1; not unseen-level generalization.",
        }
        _write_json(run_dir / "final_assessment.json", assessment)
        trainer.status = "complete"
        progress = trainer.write_progress(stop_reason=stop_reason, final_assessment=assessment)
        print(json.dumps(progress), flush=True)
        return progress
    except BaseException as exc:
        trainer.status = "failed"
        trainer.write_progress(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        trainer.close()


def main(argv: list[str] | None = None) -> int:
    """Start or resume the fixed single-policy learning run; use only existing compute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="exact output directory for this one policy")
    parser.add_argument("--resume", type=Path, help="existing run, checkpoints directory, or .pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.run_dir is None and args.resume is None:
        parser.error("--run-dir is required for a fresh run")
    checkpoint = _resolve_checkpoint(args.resume) if args.resume else None
    if checkpoint is not None and checkpoint.parent.name != "checkpoints":
        parser.error("resume a run's checkpoints/latest.pt so its learning history stays attached")
    run_dir = (args.run_dir or checkpoint.parent.parent).resolve()
    if checkpoint is not None and checkpoint.parent.parent != run_dir:
        parser.error("--run-dir must identify the same run as --resume")
    torch.set_num_threads(1)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "ppo_grid.yaml")
    cfg.device = str(resolve_device(args.device))
    cfg.run_dir, cfg.run_name = str(run_dir.parent), run_dir.name
    cfg.total_timesteps = TARGET_STEPS
    cfg.seed = 0
    cfg.env.level, cfg.env.obs_mode = "1-1", "grid"
    cfg.checkpoint_interval = 100_000
    cfg.eval.interval, cfg.eval.episodes = 500_000, 20
    cfg.eval.seed, cfg.eval.deterministic = VALIDATION_SEED, False
    result = train_agent(cfg, resume=checkpoint)
    return 130 if result["status"] == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
