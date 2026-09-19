"""The training loop: one `Trainer` drives any `Algorithm` over a vectorized env.

Run directory layout (`<run_dir>/<run_name>/`)::

    config.yaml               the resolved config of the (latest) run
    metrics.csv  log.txt      scalar metrics and the console lines
    tb/                       TensorBoard events (when enabled)
    checkpoints/
        ckpt_<step>.pt        periodic + final checkpoints (the newest `keep_checkpoints`)
        latest.pt             copy of the newest checkpoint; also written on Ctrl-C
        best.pt               the policy with the best evaluation score so far

Periodic work (logging, evaluation, checkpoints) is driven by `global_step`, the
number of env transitions summed over all envs. `global_step` grows by `n_envs`
per vector step, so an interval is rarely hit exactly: each kind of work keeps
the next multiple of its interval as a threshold and fires on the first step at
or beyond it. After a resume the thresholds are the next multiples above the
restored step.

Resuming restores the algorithm (weights, optimizer, counters), `global_step`,
the best evaluation score and the global RNG streams, and appends to the logs of
the run the checkpoint belongs to. Environments cannot be restored mid-episode:
they start new episodes (seeded differently from the run's first ones).
"""

from __future__ import annotations

import copy
import itertools
import os
import re
import signal
import threading
import time
from collections import deque
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch

from mario_play.rl.algos import get_algorithm
from mario_play.rl.algos.base import Algorithm
from mario_play.rl.checkpoint import (
    atomic_copy,
    list_checkpoints,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from mario_play.rl.config import TrainConfig, config_from_dict, config_to_dict, save_config
from mario_play.rl.evaluate import as_float, evaluate
from mario_play.rl.logger import Logger
from mario_play.rl.types import VecStep
from mario_play.rl.utils import get_rng_state, resolve_device, set_rng_state, set_seed
from mario_play.rl.vec_env import VecEnv, make_vec_env

CHECKPOINT_DIR = "checkpoints"
LATEST = "latest.pt"
BEST = "best.pt"
CONFIG_NAME = "config.yaml"
EPISODE_WINDOW = 100  # episodes the rolling rollout statistics are averaged over

# Update metrics that describe a current value; all others are averaged between two log rows.
_LAST_VALUE_METRICS = frozenset({"lr", "epsilon", "buffer_size", "n_updates"})
# Update metrics shown on the console line, in this order (the CSV holds all of them).
_CONSOLE_METRICS = (
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "explained_variance",
    "q_mean",
    "epsilon",
    "lr",
)
_HANDLER_NOT_INSTALLED = object()


def _next_multiple(step: int, interval: int) -> int:
    """Smallest multiple of `interval` strictly greater than `step`."""
    return (step // interval + 1) * interval


def _resolve_checkpoint(resume: str | Path) -> Path:
    """Accept a checkpoint file, a `checkpoints/` directory or a run directory."""
    # Absolute, because the run directory is derived from the checkpoint's location.
    path = Path(os.path.abspath(resume))
    if path.is_dir():
        for candidate in (path / LATEST, path / CHECKPOINT_DIR / LATEST):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no {LATEST} found in {path} or {path / CHECKPOINT_DIR}")
    return path


def _default_run_name(cfg: TrainConfig) -> str:
    env_name = re.sub(r"[^A-Za-z0-9._-]+", "-", cfg.env.id.split(":")[-1]).strip("-") or "env"
    return f"{cfg.algo}_{env_name}_{time.strftime('%Y%m%d-%H%M%S')}"


def _claim_run_dir(base: Path, name: str) -> Path:
    """Create and return `base/name`, or `base/name_2`, `_3`, ... if that run already exists.

    A run directory that holds anything is never reused: its logs would be
    truncated and its checkpoints mixed with the new run's.
    """
    base.mkdir(parents=True, exist_ok=True)
    for attempt in itertools.count(1):
        candidate = base / (name if attempt == 1 else f"{name}_{attempt}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            if candidate.is_dir() and not any(candidate.iterdir()):
                return candidate
    raise AssertionError("unreachable")  # pragma: no cover


class Trainer:
    """Trains `cfg.algo` on `cfg.env` and owns the run directory.

    `resume` is a checkpoint file, a run directory or its `checkpoints/` directory
    (the latter two mean `latest.pt`). A checkpoint that lives in a run directory
    continues *that* run - same directory, logs appended - whatever `cfg.run_dir`
    and `cfg.run_name` say; a loose checkpoint file starts a new run directory.
    `cfg` stays authoritative on resume (raise `total_timesteps` to train longer);
    with `cfg=None` the config stored in the checkpoint is used.

    Everything is built in the constructor; `train()` runs the loop once and
    closes the envs and the logger when it returns or raises. Public attributes:
    `cfg` (resolved copy), `run_dir`, `ckpt_dir`, `device`, `venv`, `algo`,
    `logger`, `global_step`, `best_eval`.
    """

    def __init__(self, cfg: TrainConfig | None = None, resume: str | Path | None = None) -> None:
        if cfg is None and resume is None:
            raise ValueError("cfg is required unless resume is given")
        payload: dict[str, Any] | None = None
        resume_path: Path | None = None
        if resume is not None:
            resume_path = _resolve_checkpoint(resume)
            payload = load_checkpoint(resume_path, map_location="cpu")
        self.cfg: TrainConfig = (
            copy.deepcopy(cfg) if cfg is not None else config_from_dict(payload["config"])
        )
        cfg = self.cfg
        algo_cls = get_algorithm(cfg.algo)
        if payload is not None and str(payload["algo_name"]).lower() != cfg.algo.lower():
            raise ValueError(
                f"{resume_path} is a {payload['algo_name']!r} checkpoint but the config asks "
                f"for algo={cfg.algo!r}"
            )
        if cfg.total_timesteps < 0:
            raise ValueError(f"total_timesteps must be >= 0, got {cfg.total_timesteps}")

        self.global_step = int(payload["global_step"]) if payload else 0
        self.best_eval: float | None = None
        if payload and payload["best_eval"] is not None:
            self.best_eval = float(payload["best_eval"])
        extra = payload["extra"] if payload else {}
        self._episodes = int(extra.get("episodes", 0))
        self._elapsed_before = float(extra.get("elapsed", 0.0))

        self.device = resolve_device(cfg.device)
        self.n_envs = cfg.n_envs
        self._closed = False
        self._stop_requested = False
        self.venv: VecEnv | None = None
        self.logger: Logger | None = None

        set_seed(cfg.seed, cfg.torch_deterministic)
        # Envs restart from scratch on resume; shifting the seed avoids replaying the
        # very episodes the run began with.
        self._env_seed = cfg.seed + self.global_step
        try:
            self.venv = make_vec_env(cfg.env, cfg.n_envs, seed=self._env_seed, kind=cfg.vec_env)
            self.algo: Algorithm = algo_cls(
                self.venv.single_observation_space,
                self.venv.single_action_space,
                cfg,
                self.device,
                cfg.n_envs,
            )
            if payload is not None:
                self.algo.load_state_dict(payload["algo_state"])

            continues_run = resume_path is not None and resume_path.parent.name == CHECKPOINT_DIR
            if continues_run:
                self.run_dir = resume_path.parent.parent
            else:
                self.run_dir = _claim_run_dir(
                    Path(cfg.run_dir), cfg.run_name or _default_run_name(cfg)
                )
            cfg.run_dir, cfg.run_name = str(self.run_dir.parent), self.run_dir.name
            self.ckpt_dir = self.run_dir / CHECKPOINT_DIR
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_config(cfg, self.run_dir / CONFIG_NAME)
            self.logger = Logger(
                self.run_dir,
                use_tensorboard=cfg.tensorboard,
                resume=continues_run,
                resume_step=self.global_step if continues_run else None,
            )
            if continues_run:
                self._set_aside_later_checkpoints()
        except BaseException:
            self.close()
            raise

        self._resumed_from = resume_path
        self._next: dict[str, int | None] = {
            "log": self._first_threshold(cfg.log_interval),
            "eval": self._first_threshold(cfg.eval.interval),
            "ckpt": self._first_threshold(cfg.checkpoint_interval),
        }
        self._returns: deque[float] = deque(maxlen=EPISODE_WINDOW)
        self._lengths: deque[float] = deque(maxlen=EPISODE_WINDOW)
        self._flags: deque[float] = deque(maxlen=EPISODE_WINDOW)
        self._progress: deque[float] = deque(maxlen=EPISODE_WINDOW)
        self._update_metrics: dict[str, list[float]] = {}
        self._last_eval: dict[str, float | int] | None = None
        self._last_ckpt_step = -1
        self._train_started = 0.0
        self._window_started = 0.0
        self._window_steps = 0
        self._window_overhead = 0.0

        if payload is not None:
            # Last, so that nothing above (network init, env creation) shifts the streams.
            set_rng_state(payload["rng_state"])

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    @property
    def closed(self) -> bool:
        """Whether the envs and the logger have been released."""
        return self._closed

    def request_stop(self) -> None:
        """Ask `train()` to stop after the current step, exactly like a first Ctrl-C."""
        self._stop_requested = True

    def close(self) -> None:
        """Close the envs and the logger. Idempotent; `train()` always calls it."""
        if self._closed:
            return
        self._closed = True
        for resource in (self.venv, self.logger):
            if resource is not None:
                try:
                    resource.close()
                except Exception:  # cleanup must not mask the error that led here
                    pass

    def train(self) -> dict[str, Any]:
        """Run the training loop until `total_timesteps`, or until interrupted.

        Returns `global_step`, `best_eval` (best mean evaluation return so far, or
        None), `run_dir`, `final_eval` (result dict of the evaluation at the end, or
        None when interrupted / evaluation is off) and `interrupted`.

        A first SIGINT (Ctrl-C) finishes the current step, saves `latest.pt` and
        returns normally; a second one raises `KeyboardInterrupt` right away. The
        handler is installed only when called from the main thread and the previous
        one is restored on the way out. Envs and logger are closed in any case.
        """
        if self._closed:
            raise RuntimeError("this Trainer is closed; build a new one (resume=...) to go on")
        total = self.cfg.total_timesteps
        previous_handler = self._install_sigint_handler()
        try:
            self._announce()
            self._train_started = self._window_started = time.perf_counter()
            start_step = self.global_step
            obs = self.venv.reset(seed=self._env_seed)
            while self.global_step < total and not self._stop_requested:
                actions, extras = self.algo.select_actions(obs, self.global_step)
                step = self.venv.step(actions)
                self.algo.observe(obs, actions, extras, step)
                self.global_step += self.n_envs
                self._window_steps += self.n_envs
                self._record_episodes(step)
                if self.algo.ready_to_update(self.global_step):
                    progress = min(1.0, self.global_step / total)
                    self._record_update(self.algo.update(self.global_step, progress))
                obs = step.obs

                finished = self.global_step >= total
                if self._stop_requested and not finished:
                    break
                self._periodic_work(finished)

            interrupted = self.global_step < total
            if interrupted:
                self._save(self.ckpt_dir / LATEST)
                self.logger.print(
                    f"interrupted at step {self.global_step:,}: saved {self.ckpt_dir / LATEST}; "
                    f"continue with resume={self.run_dir}"
                )
            elif self.global_step == start_step:
                self._finish_without_training()
            final_eval = None if interrupted else self._last_eval
            if not interrupted:
                self.logger.print(
                    f"done: {self.global_step:,} steps, best eval return "
                    f"{_fmt(self.best_eval)}, run dir {self.run_dir}"
                )
            return {
                "global_step": self.global_step,
                "best_eval": self.best_eval,
                "run_dir": str(self.run_dir),
                "final_eval": final_eval,
                "interrupted": interrupted,
            }
        finally:
            try:
                self.close()
            finally:
                self._restore_sigint_handler(previous_handler)

    # ------------------------------------------------------------------ #
    # periodic work
    # ------------------------------------------------------------------ #

    def _first_threshold(self, interval: int) -> int | None:
        return _next_multiple(self.global_step, interval) if interval > 0 else None

    def _crossed(self, kind: str, interval: int) -> bool:
        """Whether `global_step` reached the threshold of `kind`; if so, move it on."""
        threshold = self._next[kind]
        if threshold is None or self.global_step < threshold:
            return False
        self._next[kind] = _next_multiple(self.global_step, interval)
        return True

    def _periodic_work(self, finished: bool) -> None:
        """Log / evaluate / checkpoint if due; all three happen when training `finished`."""
        cfg = self.cfg
        log_due = self._crossed("log", cfg.log_interval) or finished
        eval_due = (self._crossed("eval", cfg.eval.interval) or finished) and cfg.eval.episodes > 0
        ckpt_due = self._crossed("ckpt", cfg.checkpoint_interval) or finished
        if not (log_due or eval_due or ckpt_due):
            return

        row: dict[str, Any] = {}
        if log_due:
            row.update(self._training_row())
            self.logger.print(self._format_training_line(row))
        started = time.perf_counter()
        if eval_due:
            row.update({f"eval/{key}": value for key, value in self._evaluate().items()})
        if row:
            self.logger.log(row, self.global_step)
        if ckpt_due:
            self._save_numbered()
        # Evaluation and checkpoint time must not count against the env-steps-per-second figure.
        self._window_overhead += time.perf_counter() - started

    def _finish_without_training(self) -> None:
        """Resumed at or beyond `total_timesteps`: evaluate, and make the run dir self-contained.

        Nothing new happened, so no CSV row is written (the step already has one).
        """
        if self.cfg.eval.episodes > 0:
            self._evaluate()
        if not (self.ckpt_dir / LATEST).is_file():
            self._save_numbered()

    def _evaluate(self) -> dict[str, float | int]:
        """Evaluate on fresh envs, track the best score and write `best.pt` on improvement."""
        cfg = self.cfg
        # Stochastic evaluation samples from torch's global generator; putting the streams
        # back keeps training identical no matter how often (or whether) it is evaluated.
        rng_state = get_rng_state()
        try:
            result = evaluate(
                self.algo,
                cfg.env,
                episodes=cfg.eval.episodes,
                seed=cfg.eval.seed,
                deterministic=cfg.eval.deterministic,
            )
        finally:
            set_rng_state(rng_state)
        self._last_eval = result

        score = float(result["mean_return"])
        improved = self.best_eval is None or score > self.best_eval
        if improved:
            self.best_eval = score
            self._save(self.ckpt_dir / BEST)
        line = (
            f"eval @ {self.global_step:,}: return {score:.2f} +/- {result['std_return']:.2f}"
            f" | length {result['mean_length']:.1f}"
        )
        if "flag_rate" in result:
            line += f" | flag_rate {result['flag_rate']:.2f}"
        if "mean_progress" in result:
            line += f" | progress {result['mean_progress']:.3f}"
        line += " | new best" if improved else f" | best {_fmt(self.best_eval)}"
        self.logger.print(line)
        return result

    def _save(self, path: Path) -> None:
        elapsed = self._elapsed_before + (time.perf_counter() - self._train_started)
        save_checkpoint(
            path,
            algo_name=self.cfg.algo,
            algo_state=self.algo.state_dict(),
            config_dict=config_to_dict(self.cfg),
            global_step=self.global_step,
            best_eval=self.best_eval,
            rng_state=get_rng_state(),
            extra={"episodes": self._episodes, "elapsed": elapsed},
        )

    def _save_numbered(self) -> None:
        """Write `ckpt_<step>.pt`, refresh `latest.pt` and rotate old numbered checkpoints."""
        if self._last_ckpt_step == self.global_step:
            return
        path = self.ckpt_dir / f"ckpt_{self.global_step}.pt"
        self._save(path)
        atomic_copy(path, self.ckpt_dir / LATEST)
        rotate_checkpoints(self.ckpt_dir, self.cfg.keep_checkpoints)
        self._last_ckpt_step = self.global_step

    def _set_aside_later_checkpoints(self) -> None:
        """Rename numbered checkpoints from beyond the resumed step (`*.superseded.pt`).

        Resuming from an older checkpoint rewinds the run. Rotation keeps the
        highest step numbers, so the abandoned future would otherwise outlive every
        checkpoint written from here on. Nothing is deleted.
        """
        for path in list_checkpoints(self.ckpt_dir):
            step = int(path.stem.split("_")[1])
            if step <= self.global_step:
                continue
            for attempt in itertools.count(1):
                suffix = ".superseded" if attempt == 1 else f".superseded-{attempt}"
                target = path.with_name(f"{path.stem}{suffix}.pt")
                if not target.exists():
                    break
            path.rename(target)
            self.logger.print(f"resume rewinds the run: {path.name} set aside as {target.name}")

    # ------------------------------------------------------------------ #
    # statistics
    # ------------------------------------------------------------------ #

    def _record_episodes(self, step: VecStep) -> None:
        """Pick up the statistics of episodes that ended on this step."""
        for index in np.flatnonzero(step.dones):
            info = step.infos[index]
            episode = info.get("episode")
            if episode is None:
                continue
            self._episodes += 1
            self._returns.append(as_float(episode["r"]))
            self._lengths.append(as_float(episode["l"]))
            if "flag_get" in info:
                self._flags.append(float(bool(as_float(info["flag_get"]))))
            if "progress" in info:
                self._progress.append(as_float(info["progress"]))

    def _record_update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._update_metrics.setdefault(key, []).append(float(value))

    def _training_row(self) -> dict[str, Any]:
        """Rollout, timing and update metrics since the previous row; starts a new window."""
        now = time.perf_counter()
        busy = now - self._window_started - self._window_overhead
        row: dict[str, Any] = {
            "rollout/ep_return_mean": float(np.mean(self._returns)) if self._returns else None,
            "rollout/ep_length_mean": float(np.mean(self._lengths)) if self._lengths else None,
            "rollout/episodes": self._episodes,
        }
        if self._flags:
            row["rollout/flag_rate"] = float(np.mean(self._flags))
        if self._progress:
            row["rollout/mean_progress"] = float(np.mean(self._progress))
        row["time/sps"] = self._window_steps / max(busy, 1e-9)
        row["time/elapsed"] = self._elapsed_before + (now - self._train_started)
        for key, values in self._update_metrics.items():
            last_value = key in _LAST_VALUE_METRICS
            row[f"train/{key}"] = values[-1] if last_value else float(np.mean(values))

        self._update_metrics = {}
        self._window_started, self._window_steps, self._window_overhead = now, 0, 0.0
        return row

    def _format_training_line(self, row: dict[str, Any]) -> str:
        total = f"{self.cfg.total_timesteps:,}"
        parts = [
            f"step {self.global_step:>{len(total)},}/{total}",
            f"{row['time/sps']:,.0f} sps",
            f"episodes {row['rollout/episodes']:,}",
            f"return {_fmt(row['rollout/ep_return_mean'])}",
            f"length {_fmt(row['rollout/ep_length_mean'], 1)}",
        ]
        if "rollout/flag_rate" in row:
            parts.append(f"flag_rate {row['rollout/flag_rate']:.2f}")
        if "rollout/mean_progress" in row:
            parts.append(f"progress {row['rollout/mean_progress']:.3f}")
        parts += [
            f"{key} {row[f'train/{key}']:.4g}" for key in _CONSOLE_METRICS if f"train/{key}" in row
        ]
        return " | ".join(parts)

    def _announce(self) -> None:
        cfg = self.cfg
        n_params = _count_trainable_parameters(self.algo)
        self.logger.print(
            f"run dir {self.run_dir} | {cfg.algo} on {cfg.env.id} | device {self.device} | "
            f"{cfg.n_envs} {cfg.vec_env} envs | {n_params:,} parameters | "
            f"{cfg.total_timesteps:,} steps"
        )
        if self._resumed_from is not None:
            self.logger.print(
                f"resumed from {self._resumed_from} at step {self.global_step:,} "
                f"(best eval return {_fmt(self.best_eval)})"
            )

    # ------------------------------------------------------------------ #
    # SIGINT
    # ------------------------------------------------------------------ #

    def _install_sigint_handler(self) -> Any:
        # Python only lets the main thread set signal handlers.
        if threading.current_thread() is not threading.main_thread():
            return _HANDLER_NOT_INSTALLED
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._on_sigint)
        return previous

    def _restore_sigint_handler(self, previous: Any) -> None:
        if previous is _HANDLER_NOT_INSTALLED:
            return
        # getsignal() yields None for a handler that was not installed from Python.
        signal.signal(signal.SIGINT, signal.SIG_DFL if previous is None else previous)

    def _on_sigint(self, signum: int, frame: FrameType | None) -> None:
        if self._stop_requested:
            raise KeyboardInterrupt
        self._stop_requested = True


def _count_trainable_parameters(algo: Algorithm) -> int:
    modules = [value for value in vars(algo).values() if isinstance(value, torch.nn.Module)]
    return sum(p.numel() for module in modules for p in module.parameters() if p.requires_grad)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"
