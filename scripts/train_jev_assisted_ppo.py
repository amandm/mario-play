"""Compare from-scratch PPO with zero, sparse, or frequent frozen Jev advice.

    python scripts/train_jev_assisted_ppo.py --run-dir /content/jev-assisted-ppo \
        --table /content/jev-table.json --max-steps 1000000 --device cuda

Repeat with --resume and a larger --max-steps to extend the bounded experiment.
Add --fixed-budget to complete that budget even after all conditions cross the
criterion. Without it, the original criterion-based stopping behavior remains.
The learning-rate horizon remains five million transitions. Each live condition
keeps its environments, partial rollout, optimizer and independent RNG streams
between matched stages. Process restarts follow the framework's normal resume
semantics: environments and any incomplete PPO rollout restart.

The script makes no API requests. Jev's frozen risk features augment observations;
PPO learns and chooses every action, with the original actions and reward.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from mario_play.cli import gif_durations
from mario_play.envs.factory import make_env
from mario_play.rl.algos import get_algorithm
from mario_play.rl.checkpoint import atomic_copy, load_checkpoint, save_checkpoint
from mario_play.rl.config import TrainConfig, config_from_dict, config_to_dict, load_config
from mario_play.rl.trainer import Trainer
from mario_play.rl.utils import get_rng_state, resolve_device, set_rng_state

HORIZON = 5_000_000
STAGE_STEPS = 100_000
SAMPLED_SEED = 2_000_000
GREEDY_SEED = 4_000_000
CONDITIONS = (
    ("baseline", "zeros", 1),
    ("jev_interval16", "table", 16),
    ("jev_interval1", "table", 1),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def model_digest(model: torch.nn.Module) -> str:
    """Hash named tensor values, independent of checkpoint metadata/serialization."""
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        result.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def ready(record: dict[str, Any]) -> bool:
    return record["sampled"]["flag_rate"] >= 0.6 and record["greedy"]["flag_rate"] == 1.0


def select_milestone(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the first successful milestone, otherwise the best validation result."""
    crossings = [record for record in records if ready(record)]
    if crossings:
        return min(crossings, key=lambda record: record["global_step"])
    return max(
        records,
        key=lambda record: (
            record["sampled"]["flag_rate"],
            record["greedy"]["flag_rate"],
            record["sampled"]["mean_progress"],
            record["sampled"]["mean_return"],
            -record["global_step"],
        ),
        default=None,
    )


def refreshes(env: Any) -> int:
    try:
        return int(env.get_wrapper_attr("jev_features_refreshes"))
    except AttributeError:
        return 0


def evaluate_policy(
    algo: Any,
    cfg: Any,
    *,
    episodes: int,
    seed: int,
    deterministic: bool,
    max_steps: int = 6000,
    frames: list[Image.Image] | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Seeded evaluation including individual outcomes and logical advice counts."""
    state = get_rng_state()
    env = None
    rows = []
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        env = make_env(cfg, seed=seed, render_mode="rgb_array" if frames is not None else None)
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            reward_sum = 0.0
            for step in range(max_steps):
                if frames is not None:
                    frames.append(_frame(env.render(), label, step, info))
                action = int(algo.predict(np.asarray(obs)[None], deterministic=deterministic)[0])
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                if terminated or truncated:
                    break
            if frames is not None:
                frames.append(_frame(env.render(), label, step + 1, info))
            rows.append(
                {
                    "seed": seed + episode,
                    "return": reward_sum,
                    "length": step + 1,
                    "flag_get": bool(info.get("flag_get", False)),
                    "progress": float(info.get("progress", 0.0)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "decision_cap_reached": not (terminated or truncated),
                }
            )
        return {
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "std_return": float(np.std([row["return"] for row in rows])),
            "mean_length": float(np.mean([row["length"] for row in rows])),
            "episodes": episodes,
            "flags": sum(row["flag_get"] for row in rows),
            "flag_rate": float(np.mean([row["flag_get"] for row in rows])),
            "mean_progress": float(np.mean([row["progress"] for row in rows])),
            "logical_advice_updates": refreshes(env),
            "episode_results": rows,
        }
    finally:
        if env is not None:
            env.close()
        set_rng_state(state)


def _frame(rgb: np.ndarray, label: str, step: int, info: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + 42), (17, 24, 39))
    image.paste(Image.fromarray(rgb), (0, 42))
    ImageDraw.Draw(image).multiline_text(
        (4, 3),
        f"{label}\nDecision {step} | progress {float(info.get('progress', 0)):.1%}",
        fill=(241, 245, 249),
        spacing=3,
    )
    return image.quantize(colors=256, dither=Image.Dither.NONE)


def condition_configs(base: TrainConfig, run_dir: Path, table: Path) -> list[TrainConfig]:
    """Keep all learning/environment settings shared, changing only advice exposure."""
    configs = []
    table_sha = digest(table)
    for name, mode, interval in CONDITIONS:
        cfg = copy.deepcopy(base)
        cfg.run_dir, cfg.run_name = str(run_dir), name
        cfg.env.jev_features_mode = mode
        cfg.env.jev_features_interval = interval
        cfg.env.jev_features_path = str(table) if mode == "table" else None
        cfg.env.jev_features_sha256 = table_sha if mode == "table" else None
        configs.append(cfg)
    return configs


class StagedTrainer(Trainer):
    """The normal PPO update loop, pausable without closing envs or discarding rollouts."""

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        resume: Path | None = None,
        stage_steps: int = STAGE_STEPS,
        eval_episodes: int = 20,
        eval_max_steps: int = 6000,
    ) -> None:
        self.stage_steps, self.eval_episodes = stage_steps, eval_episodes
        self.eval_max_steps = eval_max_steps
        self.training_stats: dict[str, Any] = {}
        self.status = "initialized"
        self.active_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.advice_before = 0
        self._stage_active = False
        super().__init__(cfg, resume=resume)
        try:
            payload = load_checkpoint(resume) if resume else None
            extra = payload["extra"] if payload else {}
            self.active_seconds = float(extra.get("active_seconds", 0.0))
            self.evaluation_seconds = float(extra.get("evaluation_seconds", 0.0))
            self.advice_before = int(extra.get("logical_advice_updates", 0))
            self.training_stats = extra.get("training", {})
            self.process_resume_count = int(extra.get("process_resume_count", 0))
            self.discarded_rollout_transitions = int(extra.get("discarded_rollout_transitions", 0))
            if resume is not None:
                batch = cfg.n_envs * cfg.ppo.n_steps
                pending = int(
                    extra.get(
                        "pending_rollout_transitions",
                        self.global_step
                        - self.algo.n_updates * batch
                        - self.discarded_rollout_transitions,
                    )
                )
                if not 0 <= pending < batch:
                    raise ValueError("checkpoint PPO update/rollout accounting is inconsistent")
                self.discarded_rollout_transitions += pending
                self.process_resume_count += 1
            self.initial_model_sha256 = extra.get("initial_model_sha256") or model_digest(
                self.algo.model
            )
            self.milestones = [
                record
                for record in read_json(self.run_dir / "milestones.json", [])
                if record["global_step"] <= self.global_step
            ]
            self._update_start_step = self.global_step
            self._obs = self.venv.reset(seed=self._env_seed)
            self._rng = get_rng_state()
            if resume is None:
                self._save(self.ckpt_dir / "initial.pt")
                atomic_copy(self.ckpt_dir / "initial.pt", self.ckpt_dir / "latest.pt")
            write_json(self.run_dir / "milestones.json", self.milestones)
            self.write_progress()
        except BaseException:
            self.close()
            raise

    def advice_updates(self) -> int:
        return self.advice_before + sum(refreshes(env) for env in self.venv.envs)

    def elapsed(self) -> float:
        return self.active_seconds + (
            time.perf_counter() - self._train_started if self._stage_active else 0.0
        )

    def _save(self, path: Path) -> None:
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
                "elapsed": self.elapsed(),
                "active_seconds": self.elapsed(),
                "evaluation_seconds": self.evaluation_seconds,
                "logical_advice_updates": self.advice_updates(),
                "initial_model_sha256": self.initial_model_sha256,
                "training": self.training_stats,
                "ppo_updates": self.algo.n_updates,
                "pending_rollout_transitions": len(self.algo.buffer) * self.n_envs,
                "discarded_rollout_transitions": self.discarded_rollout_transitions,
                "process_resume_count": self.process_resume_count,
            },
        )

    def _training_row(self) -> dict[str, Any]:
        row = super()._training_row()
        row["advice/logical_updates"] = self.advice_updates()
        row["train/ppo_updates"] = self.algo.n_updates
        self.training_stats = {"global_step": self.global_step, **row}
        return row

    def run_stage(self, target: int) -> bool:
        """Return False after interruption; preserve live rollout state between calls."""
        if target > self.cfg.total_timesteps or target < self.global_step:
            raise ValueError("stage target must follow the current step within the fixed horizon")
        if target % self.n_envs:
            raise ValueError("stage target must be divisible by n_envs")
        if target == self.global_step:
            return True
        set_rng_state(self._rng)
        self.status = "training"
        self._train_started = self._window_started = time.perf_counter()
        self._elapsed_before = self.active_seconds
        self._window_steps = 0
        self._window_overhead = 0.0
        self._stage_active = True
        previous = self._install_sigint_handler()
        self.write_progress()
        try:
            while self.global_step < target and not self._stop_requested:
                actions, extras = self.algo.select_actions(self._obs, self.global_step)
                step = self.venv.step(actions)
                self.algo.observe(self._obs, actions, extras, step)
                self.global_step += self.n_envs
                self._window_steps += self.n_envs
                self._record_episodes(step)
                if self.algo.ready_to_update(self.global_step):
                    progress = min(1.0, self._update_start_step / self.cfg.total_timesteps)
                    self._record_update(self.algo.update(self.global_step, progress))
                    self._update_start_step = self.global_step
                self._obs = step.obs
                if self._crossed("log", self.cfg.log_interval):
                    row = self._training_row()
                    self.logger.log(row, self.global_step)
                    self.logger.print(self._format_training_line(row))
                    self.write_progress()
                if self.global_step % self.stage_steps == 0:
                    self.milestone()
                elif self._crossed("ckpt", self.cfg.checkpoint_interval):
                    self._save_numbered()
            self.status = "stage_complete" if self.global_step == target else "interrupted"
            self._save(self.ckpt_dir / "latest.pt")
            self.logger.flush()
            return self.global_step == target
        except BaseException as exc:
            self.status = "failed"
            self._save(self.ckpt_dir / "latest.pt")
            self.write_progress(error_type=type(exc).__name__)
            raise
        finally:
            self.active_seconds = self.elapsed()
            self._stage_active = False
            self._rng = get_rng_state()
            self._restore_sigint_handler(previous)
            self.write_progress()

    def milestone(self) -> dict[str, Any]:
        started = time.perf_counter()
        sampled = evaluate_policy(
            self.algo,
            self.cfg.env,
            episodes=self.eval_episodes,
            seed=SAMPLED_SEED,
            deterministic=False,
            max_steps=self.eval_max_steps,
        )
        greedy = evaluate_policy(
            self.algo,
            self.cfg.env,
            episodes=1,
            seed=GREEDY_SEED,
            deterministic=True,
            max_steps=self.eval_max_steps,
        )
        self.evaluation_seconds += time.perf_counter() - started
        self.best_eval = max(sampled["mean_return"], self.best_eval or -float("inf"))
        path = self.run_dir / "milestones" / f"checkpoint_{self.global_step}.pt"
        self._save(path)
        self._save_numbered()
        record = {
            "global_step": self.global_step,
            "checkpoint": str(path.relative_to(self.run_dir)),
            "checkpoint_sha256": digest(path),
            "sampled_seed": SAMPLED_SEED,
            "greedy_seed": GREEDY_SEED,
            "sampled": sampled,
            "greedy": greedy,
            "logical_advice_updates": self.advice_updates(),
            "active_seconds": self.elapsed(),
            "evaluation_seconds": self.evaluation_seconds,
            "training": self.training_stats,
            "ppo_updates": self.algo.n_updates,
            "discarded_rollout_transitions": self.discarded_rollout_transitions,
        }
        self.milestones = [row for row in self.milestones if row["global_step"] != self.global_step]
        self.milestones.append(record)
        self.milestones.sort(key=lambda row: row["global_step"])
        selected = select_milestone(self.milestones)
        atomic_copy(self.run_dir / selected["checkpoint"], self.ckpt_dir / "selected.pt")
        write_json(self.run_dir / "milestones.json", self.milestones)
        self.logger.log(
            {
                "eval/flag_rate": sampled["flag_rate"],
                "eval/mean_progress": sampled["mean_progress"],
                "eval/mean_return": sampled["mean_return"],
                "eval/greedy_flag_rate": greedy["flag_rate"],
            },
            self.global_step,
        )
        self.logger.print(
            f"{self.cfg.run_name} @ {self.global_step:,}: sampled {sampled['flags']}/"
            f"{self.eval_episodes} flags; greedy {bool(greedy['flags'])}; "
            f"logical advice updates {self.advice_updates():,}"
        )
        self._window_overhead += time.perf_counter() - started
        self.write_progress()
        return record

    def snapshot(self) -> dict[str, Any]:
        selected = select_milestone(self.milestones)
        crossings = [row["global_step"] for row in self.milestones if ready(row)]
        return {
            "condition": self.cfg.run_name,
            "status": self.status,
            "global_step": self.global_step,
            "initial_model_sha256": self.initial_model_sha256,
            "observation_shape": list(self.venv.single_observation_space.shape),
            "feature_mode": self.cfg.env.jev_features_mode,
            "advice_interval": self.cfg.env.jev_features_interval,
            "logical_advice_updates": self.advice_updates(),
            "training_physical_api_requests": 0,
            "active_seconds": self.elapsed(),
            "evaluation_seconds": self.evaluation_seconds,
            "training": self.training_stats,
            "first_crossing_step": min(crossings) if crossings else None,
            "last_validation": self.milestones[-1] if self.milestones else None,
            "selected_checkpoint": "checkpoints/selected.pt" if selected else None,
            "selected_checkpoint_step": selected["global_step"] if selected else None,
            "fixed_lr_horizon": self.cfg.total_timesteps,
            "ppo_updates": self.algo.n_updates,
            "pending_rollout_transitions": len(self.algo.buffer) * self.n_envs,
            "discarded_rollout_transitions": self.discarded_rollout_transitions,
            "process_resume_count": self.process_resume_count,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }

    def write_progress(self, **extra: Any) -> dict[str, Any]:
        result = {**self.snapshot(), **extra}
        write_json(self.run_dir / "progress.json", result)
        return result


def check_resume(cfg: TrainConfig, checkpoint: Path) -> None:
    previous, current = load_checkpoint(checkpoint)["config"], config_to_dict(cfg)
    for data in (previous, current):
        for field in ("run_dir", "run_name", "device"):
            data.pop(field, None)
        # The hash pins table contents; relocation does not change the experiment.
        data["env"].pop("jev_features_path", None)
    if previous != current:
        raise ValueError("resume config differs from the original fixed experiment")


def load_portable_algorithm(checkpoint: Path, table: Path, device: str) -> tuple[Any, TrainConfig]:
    """Load a downloaded condition checkpoint using the adjacent verified frozen table."""
    payload = load_checkpoint(checkpoint)
    cfg = config_from_dict(payload["config"])
    if cfg.env.jev_features_mode == "table":
        if digest(table) != cfg.env.jev_features_sha256:
            raise ValueError("portable table SHA-256 does not match the checkpoint")
        cfg.env.jev_features_path = str(table.resolve())
    env = make_env(cfg.env)
    try:
        algo = get_algorithm(cfg.algo)(
            env.observation_space, env.action_space, cfg, resolve_device(device), cfg.n_envs
        )
    finally:
        env.close()
    algo.load_state_dict(payload["algo_state"])
    return algo, cfg


def record_selected(trainer: StagedTrainer) -> None:
    selected = select_milestone(trainer.milestones)
    if selected is None:
        return
    path = trainer.ckpt_dir / "selected.pt"
    previous = read_json(trainer.run_dir / "selected_playback.json", {})
    if previous.get("checkpoint_sha256") == digest(path):
        return
    state = get_rng_state()
    try:
        algo, cfg = load_portable_algorithm(
            path, trainer.run_dir.parent / "table.json", trainer.cfg.device
        )
        frames: list[Image.Image] = []
        result = evaluate_policy(
            algo,
            cfg.env,
            episodes=1,
            seed=GREEDY_SEED,
            deterministic=True,
            max_steps=trainer.eval_max_steps,
            frames=frames,
            label=f"{trainer.cfg.run_name} | PPO {selected['global_step']:,} steps",
        )
        frames[0].save(
            trainer.run_dir / "selected.gif",
            save_all=True,
            append_images=frames[1:],
            duration=gif_durations(len(frames), 60 / cfg.env.frame_skip),
            loop=0,
            optimize=False,
        )
        write_json(
            trainer.run_dir / "selected_playback.json",
            {
                "checkpoint_sha256": digest(path),
                "checkpoint_step": selected["global_step"],
                "seed": GREEDY_SEED,
                "deterministic": True,
                "result": result,
                "selection": "first criterion crossing, else best validation; GIFs do not select",
            },
        )
    finally:
        set_rng_state(state)


def run_experiment(
    configs: list[TrainConfig],
    *,
    max_steps: int,
    resume: bool = False,
    stage_steps: int = STAGE_STEPS,
    eval_episodes: int = 20,
    eval_max_steps: int = 6000,
    make_gifs: bool = True,
    fixed_budget: bool = False,
) -> dict[str, Any]:
    run_dir = Path(configs[0].run_dir)
    trainers: list[StagedTrainer] = []
    prior = read_json(run_dir / "progress.json", {})
    prior_seconds = float(prior.get("wall_seconds", 0.0))
    started = time.perf_counter()
    table_metadata = read_json(run_dir / "table.json", {})
    precompute = {
        "physical_api_requests": table_metadata.get("usage", {}).get("attempted_calls"),
        "usage": table_metadata.get("usage", {}),
        "timing": table_metadata.get("timing", {}),
        "model": table_metadata.get("model"),
        "shared_once_across_conditions": True,
    }

    def publish(status: str) -> dict[str, Any]:
        records = {trainer.cfg.run_name: trainer.snapshot() for trainer in trainers}
        result = {
            "status": status,
            "max_steps_per_condition": max_steps,
            "stage_steps": stage_steps,
            "fixed_lr_horizon": configs[0].total_timesteps,
            "fixed_budget": fixed_budget,
            "wall_seconds": prior_seconds + time.perf_counter() - started,
            "conditions": records,
            "training_physical_api_requests": 0,
            "common_precomputation": precompute,
            "criterion": "sampled flags >= 12/20 and greedy flag; first crossing per condition",
            "scope": "One training seed, one level; exploratory comparison only.",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "progress.json", result)
        return result

    try:
        for cfg in configs:
            checkpoint = Path(cfg.run_dir) / cfg.run_name / "checkpoints" / "latest.pt"
            if checkpoint.exists():
                if not resume:
                    raise ValueError(f"{checkpoint.parent.parent} already exists; use --resume")
                check_resume(cfg, checkpoint)
            elif checkpoint.parent.parent.exists() and any(checkpoint.parent.parent.iterdir()):
                raise ValueError(f"run has files but no checkpoint: {checkpoint.parent.parent}")
            trainer = StagedTrainer(
                cfg,
                resume=checkpoint if checkpoint.exists() else None,
                stage_steps=stage_steps,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
            )
            trainers.append(trainer)
        if len({trainer.initial_model_sha256 for trainer in trainers}) != 1:
            raise ValueError("conditions did not start with identical model weights")
        if len({trainer.venv.single_observation_space.shape for trainer in trainers}) != 1:
            raise ValueError("condition observation shapes differ")
        if any(trainer.global_step > max_steps for trainer in trainers):
            raise ValueError("--max-steps is below an existing condition's checkpoint")
        publish("training")
        for target in range(stage_steps, max_steps + 1, stage_steps):
            for trainer in trainers:
                if trainer.global_step < target and not trainer.run_stage(target):
                    return publish("interrupted")
                publish("training")
            if not fixed_budget and all(
                trainer.snapshot()["first_crossing_step"] is not None for trainer in trainers
            ):
                break
        if make_gifs:
            publish("recording")
            for trainer in trainers:
                record_selected(trainer)
        for trainer in trainers:
            trainer.status = "complete"
            trainer.write_progress()
        result = publish("complete")
        result["stop_reason"] = (
            "fixed_budget"
            if fixed_budget
            else "all_conditions_crossed"
            if all(trainer.snapshot()["first_crossing_step"] is not None for trainer in trainers)
            else "step_cap"
        )
        write_json(run_dir / "progress.json", result)
        return result
    except BaseException:
        publish("failed")
        raise
    finally:
        for trainer in trainers:
            trainer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fixed-budget",
        action="store_true",
        help="complete --max-steps even after all conditions cross the criterion",
    )
    parser.add_argument("--no-gifs", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.max_steps <= HORIZON or args.max_steps % STAGE_STEPS:
        parser.error("--max-steps must be a positive multiple of 100000, at most 5000000")
    run_dir, source_table = args.run_dir.resolve(), args.table.resolve()
    if not source_table.is_file():
        parser.error("--table must identify the precomputed frozen Jev table")
    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    table = run_dir / "table.json"
    if table.exists() and digest(table) != digest(source_table):
        parser.error("run table differs from --table; choose a new run directory")
    if table != source_table and not table.exists():
        shutil.copyfile(source_table, table)
    base = load_config(Path(__file__).resolve().parents[1] / "configs" / "ppo_grid.yaml")
    base.device = str(resolve_device(args.device))
    base.total_timesteps, base.seed = HORIZON, 0
    base.checkpoint_interval = STAGE_STEPS
    base.eval.interval, base.eval.episodes = STAGE_STEPS, 20
    base.eval.seed, base.eval.deterministic = SAMPLED_SEED, False
    configs = condition_configs(base, run_dir, table)
    manifest = {
        "table": "table.json",
        "table_sha256": digest(table),
        "conditions": [config_to_dict(cfg) for cfg in configs],
        "initialization": (
            "continue each condition from its latest checkpoint"
            if args.resume
            else "from scratch, seed 0, identical 18-channel architecture and weights"
        ),
        "common_precomputation": {
            "physical_api_requests": read_json(table, {}).get("usage", {}).get("attempted_calls"),
            "usage": read_json(table, {}).get("usage", {}),
            "timing": read_json(table, {}).get("timing", {}),
            "shared_once_across_conditions": True,
        },
        "training_physical_api_requests": 0,
        "stage_semantics": "live environments and partial rollouts persist between matched stages",
        "process_resume_limit": "envs and incomplete PPO rollouts restart after process exit",
        "selection": "earliest 12/20 sampled flags plus greedy flag, otherwise best validation",
        "fixed_budget": args.fixed_budget,
        "max_steps_per_condition": args.max_steps,
    }
    write_json(run_dir / "experiment.json", manifest)
    result = run_experiment(
        configs,
        max_steps=args.max_steps,
        resume=args.resume,
        make_gifs=not args.no_gifs,
        fixed_budget=args.fixed_budget,
    )
    print(json.dumps(result), flush=True)
    return 130 if result["status"] == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
