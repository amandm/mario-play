"""Record one bounded episode each: hosted Jev inference and the saved PPO policy.

    uv run python scripts/compare_jev_ppo.py --agent ppo --out runs/jev-comparison
    uv run python scripts/compare_jev_ppo.py --agent jev --out runs/jev-comparison

The Jev command reads TYPESAFE_API_KEY from the ignored project-root .env file.

Both episodes use the checkpoint's environment and the same seed/decision cap.
The simulator pauses during inference; GIF playback uses simulation time. No
training, distillation, automatic retries, or additional episodes occur.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageSequence

from mario_play.cli import gif_durations
from mario_play.envs.actions import action_names
from mario_play.envs.factory import make_env
from mario_play.rl.checkpoint import load_checkpoint
from mario_play.rl.config import MARIO_ENV_ID, EnvConfig, config_from_dict
from mario_play.rl.evaluate import load_algorithm

DEFAULT_CHECKPOINT = Path("runs/colab/ppo-agent-v1/checkpoints/selected.pt")
LABELS = {"ppo": "PPO | trained local network", "jev": "Jev | pretrained hosted inference"}


def load_api_key(env_file: Path) -> str:
    """Read only the API-key assignment without sourcing or evaluating dotenv code."""
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    if not env_file.is_file():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and key.strip() == "TYPESAFE_API_KEY":
            try:
                parts = shlex.split(value, comments=True)
            except ValueError:
                raise ValueError("invalid TYPESAFE_API_KEY assignment in env file") from None
            if len(parts) != 1 or not parts[0]:
                raise ValueError("empty or invalid TYPESAFE_API_KEY assignment in env file")
            return parts[0]
    return ""


class Controller(Protocol):
    """Inference-only controller; metadata contains no credentials."""

    last_decision: dict[str, Any]

    def act(self, obs: np.ndarray) -> int: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class PPOController:
    """Expose greedy PPO probabilities without collecting training experience."""

    def __init__(self, checkpoint: Path, device: str, names: list[str]) -> None:
        self.algo, _ = load_algorithm(checkpoint, device=device)
        self.names = names
        self.last_decision: dict[str, Any] = {}

    def reset(self) -> None:
        self.last_decision = {}

    def close(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        started = time.perf_counter()
        logits, value = self.algo.model(self.algo.obs_to_tensor(np.asarray(obs)[None]))
        probabilities = logits.softmax(dim=-1)[0].cpu().tolist()
        action = int(logits.argmax(dim=-1)[0].item())
        self.last_decision = {
            "choice": self.names[action],
            "probabilities": dict(zip(self.names, probabilities, strict=True)),
            "confidence": probabilities[action],
            "value": float(value[0].item()),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "mode": "greedy",
        }
        return action


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _frame(
    rgb: np.ndarray, name: str, step: int, action: str, info: dict[str, Any], note: str = ""
) -> Image.Image:
    frame = Image.fromarray(rgb).convert("RGB")
    canvas = Image.new("RGB", (frame.width, frame.height + 80), (17, 24, 39))
    canvas.paste(frame, (0, 80))
    title = "PPO | trained local network" if name == "ppo" else "Jev | pretrained model"
    subtitle = "Greedy local inference" if name == "ppo" else "Hosted decision inference"
    caption = (
        f"{title}\n{subtitle}\nDecision {step} | {action}\n"
        f"Progress {float(info.get('progress', 0)):.1%}"
    )
    if note:
        caption += f"\n{note}"
    ImageDraw.Draw(canvas).multiline_text((4, 3), caption, fill=(241, 245, 249), spacing=2)
    return canvas.quantize(colors=256, dither=Image.Dither.NONE)


def _gif(path: Path, frames: list[Image.Image], fps: float) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=gif_durations(len(frames), fps),
        loop=0,
        optimize=False,
    )


def record_episode(
    name: str,
    controller: Controller,
    cfg: EnvConfig,
    out: Path,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Record the first episode, preserving partial results after inference failure."""
    directory = out / name
    # An exclusive directory prevents accidental duplicate billable episodes too.
    directory.mkdir(parents=True, exist_ok=False)
    env = make_env(cfg, seed=seed, render_mode="rgb_array")
    frames: list[Image.Image] = []
    names = action_names(cfg.action_set)
    total_reward = decision_seconds = 0.0
    steps = 0
    terminated = truncated = False
    status = "complete"
    error_type = None
    started = time.perf_counter()
    info: dict[str, Any] = {}
    initial_hash = None
    fps = 60.0 / cfg.frame_skip
    try:
        obs, info = env.reset(seed=seed)
        initial_hash = hashlib.sha256(np.asarray(obs).tobytes()).hexdigest()
        controller.reset()
        frames.append(_frame(env.render(), name, 0, "initial state", info))
        with (directory / "decisions.jsonl").open("x", encoding="utf-8") as log:
            for decision in range(1, max_steps + 1):
                before = hashlib.sha256(np.asarray(obs).tobytes()).hexdigest()
                decision_start = time.perf_counter()
                try:
                    action = controller.act(obs)
                finally:
                    wall_seconds = time.perf_counter() - decision_start
                    decision_seconds += wall_seconds
                if not env.action_space.contains(action):
                    raise ValueError("controller returned an illegal action")
                obs, reward, terminated, truncated, info = env.step(action)
                steps = decision
                total_reward += float(reward)
                entry = {
                    "decision": decision,
                    "observation_sha256": before,
                    "action": int(action),
                    "action_name": names[action],
                    "reward": float(reward),
                    "progress": float(info.get("progress", 0)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "inference_wall_seconds": wall_seconds,
                    "controller": controller.last_decision,
                }
                log.write(json.dumps(entry) + "\n")
                log.flush()
                note = ""
                if info.get("flag_get"):
                    note = "ENDED: REACHED FLAG"
                elif terminated:
                    note = f"ENDED: {info.get('death_cause') or 'terminal state'}"
                elif truncated:
                    note = "ENDED: stall or environment limit"
                elif decision == max_steps:
                    note = "ENDED: decision limit"
                frames.append(_frame(env.render(), name, decision, names[action], info, note))
                if decision % 20 == 0 or terminated or truncated:
                    print(
                        f"{name}: {decision}/{max_steps} decisions, "
                        f"progress={info.get('progress', 0):.1%}",
                        flush=True,
                    )
                if terminated or truncated:
                    break
    except (Exception, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        # Provider exceptions may include request details. Persist the type only.
        error_type = type(exc).__name__
        if frames:
            frames[-1] = _frame(env.render(), name, steps, "stopped", info, f"ENDED: {status}")
    finally:
        elapsed = time.perf_counter() - started
        env.close()
        controller.close()
    cap_reached = steps >= max_steps and not terminated and not truncated
    result = {
        "agent": name,
        "label": LABELS[name],
        "status": status,
        "error_type": error_type,
        "seed": seed,
        "max_steps": max_steps,
        "decisions": steps,
        "return": total_reward,
        "progress": float(info.get("progress", 0)),
        "flag_get": bool(info.get("flag_get", False)),
        "death_cause": info.get("death_cause"),
        "terminated": bool(terminated),
        "truncated": bool(truncated or cap_reached),
        "decision_cap_reached": cap_reached,
        "initial_observation_sha256": initial_hash,
        "inference_wall_seconds": decision_seconds,
        "episode_wall_seconds": elapsed,
        "simulated_seconds_upper_bound": steps * cfg.frame_skip / 60,
        "gif_fps": fps,
        "api_calls": int(getattr(controller, "calls", 0)),
        "api_usage": getattr(controller, "usage", {}),
        "selection": "first fixed-seed episode; no retries or successful-take search",
    }
    _json(directory / "result.json", result)
    if frames:
        _gif(directory / "gameplay.gif", frames, fps)
        frames[-1].save(directory / "final-frame.png")
    print(json.dumps(result), flush=True)
    return result


def write_report(out: Path, settings: dict[str, Any]) -> None:
    """Join available episode artifacts; shorter clips hold their last frame."""
    results = {
        name: json.loads((out / name / "result.json").read_text())
        for name in LABELS
        if (out / name / "result.json").is_file()
    }
    _json(out / "comparison.json", {"settings": settings, "episodes": results})
    lines = [
        "# One episode: Jev and the saved PPO agent",
        "",
        "Jev uses a pretrained hosted decision model. PPO uses our saved locally runnable "
        "network trained with environment rewards. Both are doing inference here; "
        "no training or distillation occurs.",
        "",
        f"Both start on level **{settings['env']['level']}**, seed **{settings['seed']}**, "
        f"with the same seven actions, observation grid, rewards, "
        f"**{settings['env']['frame_skip']} frames per decision**, and "
        f"**{settings['max_steps']} decision limit**. The PPO policy takes its most likely "
        "action; Jev chooses from its Choice response. Jev's prompt and structured encoding "
        "provide domain guidance. Jev additionally receives its previous four actions and "
        "previous jump-button state; PPO receives only the current unstacked grid. This "
        "history is extra information, relevant to releasing jump before another takeoff.",
        "",
        "Each recording is the first episode, without retries or selection for success. "
        "One episode shows behavior, not a reliable success-rate estimate. The simulator "
        "pauses during inference. GIFs play at normal simulation speed; API waiting time "
        "is reported separately below.",
        "",
        "| Controller | Reached flag | Progress | Decisions | Inference wall time | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {LABELS[name]} | {'Yes' if result['flag_get'] else 'No'} | "
            f"{result['progress']:.1%} | {result['decisions']} | "
            f"{result['inference_wall_seconds']:.2f}s | {result['status']} |"
        )
    for name in results:
        lines.extend(["", f"![{LABELS[name]}]({name}/gameplay.gif)"])
    if len(results) == 2 and all((out / name / "gameplay.gif").is_file() for name in LABELS):
        clips = []
        for name in LABELS:
            with Image.open(out / name / "gameplay.gif") as gif:
                clips.append([frame.convert("RGB") for frame in ImageSequence.Iterator(gif)])
        combined = []
        for index in range(max(map(len, clips))):
            left, right = (clip[min(index, len(clip) - 1)] for clip in clips)
            image = Image.new("RGB", (left.width + right.width + 8, left.height), "#111827")
            image.paste(left, (0, 0))
            image.paste(right, (left.width + 8, 0))
            combined.append(image.quantize(colors=256, dither=Image.Dither.NONE))
        _gif(out / "side-by-side.gif", combined, 60 / settings["env"]["frame_skip"])
        lines.extend(["", "![PPO left, Jev right](side-by-side.gif)"])
    lines.extend(
        [
            "",
            f"PPO checkpoint: `{settings['checkpoint']}` at "
            f"{settings['checkpoint_step']:,} training interactions. "
            f"SHA-256: `{settings['checkpoint_sha256']}`.",
            "",
            "Exact environment settings are in `settings.json`; each agent folder contains "
            "`result.json` and per-decision `decisions.jsonl`. API tokens/call counts are "
            "recorded, but this script does not infer billing from token counts.",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Record selected controllers; PPO-only execution never needs an API key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agent", choices=("ppo", "jev", "both"), default="both")
    parser.add_argument("--seed", type=int, default=4_000_000)
    parser.add_argument("--max-calls", "--max-steps", dest="max_steps", type=int, default=400)
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    if args.seed < 0 or args.max_steps < 1 or args.timeout <= 0:
        parser.error("seed must be nonnegative; max-calls and timeout must be positive")
    selected = list(LABELS) if args.agent == "both" else [args.agent]
    for name in selected:
        if (args.out / name).exists():
            parser.error(f"refusing to overwrite or repeat existing {name} episode")
    api_key = load_api_key(args.env_file) if "jev" in selected else ""
    if "jev" in selected and not api_key:
        parser.error("TYPESAFE_API_KEY must be set for Jev; use --agent ppo for offline recording")
    checkpoint = args.checkpoint.resolve()
    payload = load_checkpoint(checkpoint)
    cfg = config_from_dict(payload["config"])
    if (
        payload["algo_name"] != "ppo"
        or cfg.env.id != MARIO_ENV_ID
        or not isinstance(cfg.env.level, str)
        or cfg.env.obs_mode != "grid"
        or cfg.env.frame_stack != 1
        or cfg.env.action_set != "simple"
    ):
        parser.error(
            "requires a PPO checkpoint on one MarioPlay level, unstacked grid, simple actions"
        )
    settings = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": int(payload["global_step"]),
        "env": dataclasses.asdict(cfg.env),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "jev_model": args.model,
        "inference_timeout_seconds": args.timeout,
        "action_names": action_names(cfg.env.action_set),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    settings_path = args.out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        parser.error("existing output uses different settings; choose a new output directory")
    _json(settings_path, settings)
    torch.set_num_threads(1)
    failed = False
    for name in selected:
        controller: Controller
        if name == "ppo":
            controller = PPOController(checkpoint, args.device, settings["action_names"])
        else:
            from mario_play.agents.jev_agent import JevAgent

            controller = JevAgent(
                api_key,
                max_calls=args.max_steps,
                timeout=args.timeout,
                frame_skip=cfg.env.frame_skip,
                model=args.model,
            )
        result = record_episode(
            name, controller, cfg.env, args.out, seed=args.seed, max_steps=args.max_steps
        )
        write_report(args.out, settings)
        failed |= result["status"] != "complete"
        if result["status"] == "interrupted":
            return 130
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
