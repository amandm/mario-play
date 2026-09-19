"""Record the first fixed-seed greedy episode at one identical PPO training budget.

    uv run python scripts/record_jev_comparison.py \
        --run-dir runs/colab/jev-assisted-v1 --steps 100000 --device cpu

Uses existing checkpoints and the adjacent frozen table, with no API requests or
training. Every episode uses seed 4,000,000 and a 6,000-decision cap. The combined
GIF aligns simulator decisions at their original rate; finished episodes hold
their actual final frame with an explicit end label.
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

LABELS = {
    "baseline": "PPO baseline | zero advice",
    "jev_interval16": "PPO + Jev | every 16 decisions",
    "jev_interval1": "PPO + Jev | every decision",
}
SEED = 4_000_000


def combined_frame(
    index: int,
    clips: dict[str, list[Image.Image]],
    results: dict[str, Any],
    steps: int,
    fps: float,
) -> Image.Image:
    width, height = next(iter(clips.values()))[0].size
    canvas = Image.new("RGB", (3 * width + 16, height + 64), (17, 24, 39))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 5),
        f"Same training budget: {steps:,} transitions | first greedy episode | seed {SEED:,}",
        fill=(241, 245, 249),
    )
    draw.text(
        (8, 22),
        f"Simulation time {index / fps:.2f}s | original {fps:g} decisions/s | no speed changes",
        fill=(203, 213, 225),
    )
    for column, (name, frames) in enumerate(clips.items()):
        x = column * (width + 8)
        canvas.paste(frames[min(index, len(frames) - 1)].convert("RGB"), (x, 42))
        ended = index >= len(frames) - 1
        outcome = results[name]["result"]["episode_results"][0]
        if ended:
            detail = "FLAG REACHED" if outcome["flag_get"] else "NO FLAG"
            if outcome["decision_cap_reached"]:
                detail += " | decision cap"
            status = f"EPISODE ENDED | {detail}"
        else:
            status = "PLAYING"
        draw.text((x + 4, height + 46), status, fill=(253, 224, 71) if ended else (148, 163, 184))
    return canvas.quantize(colors=256, dither=Image.Dither.NONE)


def record(run_dir: Path, steps: int, device: str) -> dict[str, Any]:
    runner = runpy.run_path(str(Path(__file__).with_name("train_jev_assisted_ppo.py")))
    checkpoints = {
        name: run_dir / name / "milestones" / f"checkpoint_{steps}.pt" for name in LABELS
    }
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"matched checkpoint not downloaded: {checkpoint}")
    table = run_dir / "table.json"
    if not table.is_file():
        raise FileNotFoundError(f"frozen table not downloaded: {table}")
    torch.set_num_threads(1)
    clips: dict[str, list[Image.Image]] = {}
    results: dict[str, Any] = {}
    fps = None
    for name, checkpoint in checkpoints.items():
        payload = runner["load_checkpoint"](checkpoint)
        if payload["global_step"] != steps:
            raise ValueError(f"checkpoint content does not match requested budget: {checkpoint}")
        algo, cfg = runner["load_portable_algorithm"](checkpoint, table, device)
        current_fps = 60 / cfg.env.frame_skip
        if fps is not None and fps != current_fps:
            raise ValueError("condition decision rates differ; cannot align this comparison")
        fps = current_fps
        frames: list[Image.Image] = []
        outcome = runner["evaluate_policy"](
            algo,
            cfg.env,
            episodes=1,
            seed=SEED,
            deterministic=True,
            max_steps=6000,
            frames=frames,
            label=LABELS[name],
        )
        gif = run_dir / name / f"matched_{steps}.gif"
        frames[0].save(
            gif,
            save_all=True,
            append_images=frames[1:],
            duration=runner["gif_durations"](len(frames), fps),
            loop=0,
            optimize=False,
        )
        results[name] = {
            "checkpoint": str(checkpoint.relative_to(run_dir)),
            "checkpoint_sha256": runner["digest"](checkpoint),
            "global_step": steps,
            "table_sha256": runner["digest"](table),
            "seed": SEED,
            "deterministic": True,
            "playback_device": device,
            "frames": len(frames),
            "fps": fps,
            "physical_api_requests": 0,
            "result": outcome,
            "gif": str(gif.relative_to(run_dir)),
        }
        runner["write_json"](gif.with_suffix(".json"), results[name])
        clips[name] = frames
        del algo, payload
    length = max(len(frames) for frames in clips.values())
    first = combined_frame(0, clips, results, steps, fps)
    combined = run_dir / f"matched_{steps}.gif"
    first.save(
        combined,
        save_all=True,
        append_images=(combined_frame(i, clips, results, steps, fps) for i in range(1, length)),
        duration=runner["gif_durations"](length, fps),
        loop=0,
        optimize=False,
    )
    summary = {
        "global_step": steps,
        "seed": SEED,
        "deterministic": True,
        "playback_device": device,
        "physical_api_requests": 0,
        "fps": fps,
        "frames": length,
        "alignment": "aligned decision index; ended runs hold final frame marked EPISODE ENDED",
        "conditions": results,
        "combined_gif": str(combined.relative_to(run_dir)),
    }
    runner["write_json"](combined.with_suffix(".json"), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    result = record(args.run_dir.resolve(), args.steps, args.device)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
