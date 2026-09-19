"""Run a separate PPO benchmark and seeded experiments, resuming durable checkpoints.

Example (from the repository root, with the package installed)::

    python scripts/run_experiments.py --run-root /content/drive/MyDrive/mario-experiments

Use a new output root when changing training settings. Increasing --steps or
changing --device is allowed on resume; interrupted environments restart, and
PPO recollects its current rollout, as documented by Trainer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from mario_play.rl.checkpoint import load_checkpoint
from mario_play.rl.config import TrainConfig, config_to_dict, load_config
from mario_play.rl.evaluate import evaluate, load_algorithm
from mario_play.rl.trainer import Trainer
from mario_play.rl.utils import resolve_device


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _source_metadata(root: Path) -> dict[str, Any]:
    files = [root / name for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE")]
    for folder, suffixes in (
        ("src", {".py", ".txt"}),
        ("configs", {".yaml"}),
        ("scripts", {".py"}),
    ):
        files.extend(path for path in (root / folder).rglob("*") if path.suffix in suffixes)
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
        if path.is_file() and not path.is_symlink()
    }
    manifest_path = root / "source_manifest.json"
    manifest = _read_json(manifest_path)
    return {
        "code_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "files_sha256": hashes,
        "manifest_sha256": (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.exists()
            else None
        ),
        "manifest_git_commit": manifest.get("git_commit"),
        "matches_manifest": hashes == manifest["files_sha256"] if manifest else None,
    }


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "gpus": [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
        "packages": {name: version(name) for name in ("torch", "numpy", "gymnasium", "mario-play")},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--porcelain"),
        "source": _source_metadata(Path(__file__).resolve().parents[1]),
    }


def _training_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"run_dir", "run_name", "device", "total_timesteps"}
    }


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(cfg: TrainConfig, evaluation: dict[str, int], runtime: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(cfg.run_dir) / cfg.run_name
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    payload = load_checkpoint(checkpoint, map_location="cpu") if checkpoint.exists() else None
    start_step = int(payload["global_step"]) if payload else 0
    if payload is not None:
        if _training_settings(payload["config"]) != _training_settings(config_to_dict(cfg)):
            raise ValueError(f"{run_dir}: training settings differ; use a new --run-root")
        if cfg.total_timesteps < payload["config"]["total_timesteps"]:
            raise ValueError(f"{run_dir}: target steps decreased; use a new --run-root")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"{run_dir}: existing data has no latest checkpoint; use a new --run-root")
    if summary and summary["evaluation_settings"] != evaluation:
        raise ValueError(f"{run_dir}: evaluation settings differ; use a new --run-root")

    if payload and start_step >= cfg.total_timesteps:
        if summary.get("status") == "complete" and summary.get(
            "checkpoint_sha256"
        ) == _checkpoint_digest(checkpoint):
            print(f"Already complete: {run_dir} ({start_step:,} steps)", flush=True)
            return summary
    else:
        trainer = Trainer(cfg, resume=checkpoint if payload else None)
        if trainer.run_dir != run_dir:
            trainer.close()
            raise RuntimeError(f"run directory was claimed concurrently: {run_dir}")
        session: dict[str, Any] = {"runtime": runtime, "start_step": start_step}
        summary = {
            **summary,
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "status": "running",
            "target_steps": cfg.total_timesteps,
            "evaluation_settings": evaluation,
            "sessions": [*summary.get("sessions", []), session],
        }
        _write_json(summary_path, summary)
        print(f"{'Resuming' if payload else 'Starting'}: {run_dir}", flush=True)
        started = time.perf_counter()
        try:
            result = trainer.train()
        except BaseException as exc:
            session.update(wall_seconds=time.perf_counter() - started, error=str(exc))
            summary["status"] = "failed"
            _write_json(summary_path, summary)
            raise
        wall_seconds = time.perf_counter() - started
        session.update(
            wall_seconds=wall_seconds,
            end_step=result["global_step"],
            steps_per_second=(result["global_step"] - start_step) / wall_seconds,
        )
        summary.update(
            status="interrupted" if result["interrupted"] else "evaluating",
            global_step=result["global_step"],
            training_eval=result["final_eval"],
        )
        _write_json(summary_path, summary)
        if result["interrupted"]:
            return summary
        payload = load_checkpoint(checkpoint, map_location="cpu")

    # Report both sampled and greedy policies on the same held-out episode seeds.
    algo, checkpoint_cfg = load_algorithm(checkpoint, device=cfg.device)
    started = time.perf_counter()
    held_out = {
        mode: evaluate(
            algo,
            checkpoint_cfg.env,
            episodes=evaluation["episodes"],
            seed=evaluation["seed"],
            deterministic=mode == "greedy",
        )
        for mode in ("sampled", "greedy")
    }
    elapsed = float(payload["extra"].get("elapsed", 0.0))
    summary.update(
        run_name=cfg.run_name,
        seed=cfg.seed,
        status="complete",
        target_steps=cfg.total_timesteps,
        global_step=int(payload["global_step"]),
        training_wall_seconds=elapsed,
        training_steps_per_second=payload["global_step"] / elapsed if elapsed else None,
        throughput_includes="training, periodic evaluation, and checkpoint overhead",
        checkpoint=str(checkpoint),
        checkpoint_sha256=_checkpoint_digest(checkpoint),
        evaluation_settings=evaluation,
        held_out=held_out,
        evaluation_wall_seconds=time.perf_counter() - started,
    )
    _write_json(summary_path, summary)
    print(json.dumps({"run": cfg.run_name, "held_out": held_out}), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the requested suite; return 130 after saving an interrupted run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="durable output directory")
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_grid.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--benchmark-steps", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--skip-benchmark", action="store_true")
    phases.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1_000_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    for name in (
        "steps",
        "benchmark_steps",
        "checkpoint_interval",
        "eval_episodes",
        "torch_threads",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if min(*args.seeds, args.eval_seed) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be nonnegative and training seeds must be unique")

    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    base = load_config(args.config, args.set)
    base.device = str(device)
    base.run_dir = str(args.run_root.resolve())
    base.checkpoint_interval = args.checkpoint_interval
    args.run_root.mkdir(parents=True, exist_ok=True)
    runtime = _runtime_metadata(device)
    evaluation = {"episodes": args.eval_episodes, "seed": args.eval_seed}
    jobs = [] if args.skip_benchmark else [("benchmark", args.seeds[0], args.benchmark_steps)]
    if not args.benchmark_only:
        jobs.extend((f"seed_{seed}", seed, args.steps) for seed in args.seeds)
    summaries = {
        path.parent.name: _read_json(path) for path in sorted(args.run_root.glob("*/summary.json"))
    }
    _write_json(args.run_root / "runtime.json", runtime)
    for name, seed, steps in jobs:
        cfg = copy.deepcopy(base)
        cfg.run_name, cfg.seed, cfg.total_timesteps = name, seed, steps
        result = _run(cfg, evaluation, runtime)
        summaries[name] = result
        _write_json(
            args.run_root / "summary.json", {"runtime": runtime, "runs": list(summaries.values())}
        )
        if result["status"] == "interrupted":
            return 130
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
