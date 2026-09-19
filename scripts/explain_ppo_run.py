"""Explain one PPO run and record its first fixed-seed evaluation episode.

    uv run python scripts/explain_ppo_run.py --run-dir runs/my_ppo_run

The report uses actual checkpoint contents and logged evaluations. The GIF is
the first episode, without searching for a successful or unusually good take.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from mario_play.cli import gif_durations
from mario_play.envs.factory import make_env
from mario_play.envs.rewards import make_reward_config
from mario_play.rl.checkpoint import load_checkpoint
from mario_play.rl.config import MARIO_ENV_ID, TrainConfig
from mario_play.rl.evaluate import evaluate, load_algorithm


def _logged_evaluations(path: Path, until_step: int) -> list[dict[str, float]]:
    milestones = path.parent / "learning_milestones.json"
    if milestones.is_file():
        return [
            {"step": row["global_step"], **row["sampled"]}
            for row in json.loads(milestones.read_text(encoding="utf-8"))
            if row["global_step"] <= until_step
        ]
    if not path.is_file():
        return []
    points = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("eval/flag_rate"):
                continue
            step = int(row["step"])
            if step <= until_step:
                points[step] = {
                    "step": step,
                    **{
                        name.removeprefix("eval/"): float(value)
                        for name, value in row.items()
                        if name.startswith("eval/") and value
                    },
                }
    return [points[step] for step in sorted(points)]


def _milestones(points: list[dict[str, float]]) -> list[dict[str, float]]:
    if len(points) <= 12:
        return points
    selected = {round(i * (len(points) - 1) / 9) for i in range(10)}
    first_flag = next((i for i, point in enumerate(points) if point["flag_rate"] > 0), None)
    if first_flag is not None:
        selected.add(first_flag)
    return [points[index] for index in sorted(selected)]


def _save_gameplay(
    path: Path,
    frames: list[Image.Image],
    *,
    level: str,
    step: int,
    mode: str,
    seed: int,
    fps: float,
    result: dict[str, Any],
) -> None:
    won = result.get("flag_rate") == 1.0
    outcome = "REACHED FLAG" if won else "DID NOT REACH FLAG"
    progress = result.get("mean_progress", 0.0)
    caption = (
        f"PPO | level {level} | {mode}\n"
        f"Checkpoint {step:,} | seed {seed}\n"
        f"{outcome} | progress {progress:.1%}"
    )

    def annotated(frame: Image.Image) -> Image.Image:
        image = Image.new("RGB", (frame.width, frame.height + 44), (17, 24, 39))
        image.paste(frame, (0, 44))
        ImageDraw.Draw(image).multiline_text((5, 3), caption, fill=(241, 245, 249), spacing=2)
        return image.quantize(colors=256, dither=Image.Dither.NONE)

    first = annotated(frames[0])
    first.save(
        path,
        save_all=True,
        append_images=(annotated(frame) for frame in frames[1:]),
        duration=gif_durations(len(frames), fps),
        loop=0,
        optimize=False,
    )


def _report(
    cfg: TrainConfig,
    checkpoint: Path,
    report: dict[str, Any],
    points: list[dict[str, float]],
) -> str:
    result = report["evaluation"]
    recording = report["recording"]
    mode = report["policy_mode"]
    evaluation_mode = report["evaluation_policy_mode"]
    steps = report["checkpoint_step"]
    count = int(result["episodes"])
    successes = round(result["flag_rate"] * count)
    env = cfg.env
    reward = make_reward_config(env.reward)
    rollout = cfg.n_envs * cfg.ppo.n_steps
    lines = [
        "# Learning to play with PPO",
        "",
        f"At **{steps:,} training interactions**, this checkpoint completed level "
        f"**{env.level} in {successes} of {count} evaluation episodes** "
        f"using **{evaluation_mode} actions**. Mean final progress was "
        f"**{result['mean_progress']:.1%}**.",
        "",
        f"Evaluation source: {report['evaluation_source']}. "
        f"The saved run has trained through **{report['training_step']:,} interactions**. "
        "The checkpoint shown here can be earlier because selection uses validation "
        "performance, not the held-out final assessment.",
        "",
        "This evaluates one policy on one fixed level. It does not establish performance "
        "on other levels. Episode seeds change the sampled action stream; they do not "
        "turn this fixed map into different levels.",
        "",
        "## What the algorithm is learning",
        "",
        f"The agent receives a **{env.obs_mode} observation** and chooses from the "
        f"**{env.action_set} action set**. Each action is held for up to "
        f"**{env.frame_skip} game frames**. It collects **{rollout:,} interactions** "
        f"per PPO update ({cfg.n_envs} environments × {cfg.ppo.n_steps} steps), "
        f"then revisits that rollout for **{cfg.ppo.n_epochs} optimization epochs**.",
        "",
        "The value network predicts future reward. PPO uses the difference between "
        "observed outcomes and those expectations to increase the probability of "
        "actions that worked better than expected. Its clipped objective limits how "
        "much those probabilities can change in one update. The agent receives no "
        "correct sequence of buttons.",
        "",
        f"Configured rewards: {reward.progress_weight * 16:g} per tile of horizontal "
        f"movement, {reward.time_penalty:g} per interaction, "
        f"{reward.death_penalty:g} for death and {reward.flag_bonus:g} for reaching "
        f"the flag. Leftward movement reverses the progress reward. Coin bonus: "
        f"{reward.coin_bonus:g}; score weight: {reward.score_weight:g}; "
        f"reward clipping: {reward.clip}.",
        "",
        "## Learning milestones",
        "",
    ]
    if points:
        periodic_mode = "greedy" if cfg.eval.deterministic else "sampled"
        lines.extend(
            [
                f"These are evaluations logged during this run, using {periodic_mode} "
                f"actions from evaluation seed {cfg.eval.seed:,}. The table retains "
                "the first and last evaluations and the first recorded completion; "
                "long histories are otherwise sampled at evenly spaced checkpoints.",
                "",
                "| Training interactions | Completed | Mean progress | Mean reward |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for point in _milestones(points):
            episodes = int(point["episodes"])
            completed = round(point["flag_rate"] * episodes)
            lines.append(
                f"| {int(point['step']):,} | {completed}/{episodes} | "
                f"{point['mean_progress']:.1%} | {point['mean_return']:.2f} |"
            )
        lines.extend(
            [
                "",
                "Rising progress means the agent gets farther on average; completion "
                "counts tell us whether it reaches the flag. Reward is the training "
                "signal, so a higher reward alone does not prove reliable completion. "
                "Small evaluation batches can fluctuate as PPO keeps exploring.",
            ]
        )
    else:
        lines.append("No periodic evaluation rows were found for this checkpoint's step range.")
    recording_outcome = (
        "reached the flag" if recording.get("flag_rate") == 1 else "did not reach the flag"
    )
    lines.extend(
        [
            "",
            "## Watch the checkpoint play",
            "",
            f"The recording shows **one fixed-seed episode**, seed "
            f"**{report['recording_seed']:,}**, with **{mode} actions**. It "
            f"**{recording_outcome}**, ending at **{recording['mean_progress']:.1%} "
            f"progress** after **{int(recording['mean_length']):,} decisions**. "
            "No retries or search for a successful take were performed.",
            "",
            "![PPO gameplay](gameplay.gif)",
            "",
            f"Assessment uses seed {report['evaluation_seed']:,}. The recording has a "
            f"{report['max_steps']:,}-decision cap. The clip plays at "
            "the environment's normal speed. A failure to reach the flag can include "
            "death, stalling, timeout, or this decision cap.",
            "",
            f"Checkpoint: `{checkpoint}`. SHA-256: `{report['checkpoint_sha256']}`. "
            "Exact configuration is stored in the checkpoint; machine-readable "
            "results and all logged evaluation points are in [evaluation.json](evaluation.json).",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Generate measured learning notes and a GIF for one PPO checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="defaults to checkpoints/selected.pt, then latest.pt"
    )
    parser.add_argument("--out", type=Path, help="defaults to run-dir/learning/step_N_MODE")
    parser.add_argument("--mode", choices=("sampled", "greedy"), default="greedy")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="run a new assessment instead of reusing saved results",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4_000_000)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_steps < 1 or args.seed < 0:
        parser.error("episodes/max-steps must be positive and seed must be nonnegative")
    torch.set_num_threads(1)
    selected = args.run_dir / "checkpoints/selected.pt"
    default_checkpoint = selected if selected.is_file() else args.run_dir / "checkpoints/latest.pt"
    checkpoint = (args.checkpoint or default_checkpoint).resolve()
    payload = load_checkpoint(checkpoint)
    if payload["algo_name"] != "ppo":
        parser.error("this helper documents PPO checkpoints only")
    algo, cfg = load_algorithm(checkpoint, device=args.device)
    if cfg.env.id != MARIO_ENV_ID or not isinstance(cfg.env.level, str):
        parser.error("this helper requires one fixed MarioPlay level")
    step = int(payload["global_step"])
    out = args.out or args.run_dir / "learning" / f"step_{step}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    options = {
        "seed": args.seed,
        "deterministic": args.mode == "greedy",
        "max_steps": args.max_steps,
    }
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assessment_path = args.run_dir / "final_assessment.json"
    assessment = (
        json.loads(assessment_path.read_text(encoding="utf-8")) if assessment_path.is_file() else {}
    )
    if assessment.get("checkpoint_sha256") != digest:
        assessment = {}
    print(f"Recording PPO at {step:,} interactions ({args.mode}, seed {args.seed}).")
    # Palette frames bound recording memory to one byte per pixel per decision.
    frames: list[Image.Image] = []

    def capture(frame: np.ndarray) -> None:
        frames.append(Image.fromarray(frame).quantize(colors=256, dither=Image.Dither.NONE))

    recording = evaluate(algo, cfg.env, episodes=1, frame_callback=capture, **options)
    evaluation_seed = args.seed
    evaluation_mode = args.mode
    if args.evaluate:
        result = evaluate(algo, cfg.env, episodes=args.episodes, **options)
        evaluation_source = "a new local evaluation requested with --evaluate"
    elif assessment:
        result = assessment["held_out"]["sampled"]
        evaluation_mode = "sampled"
        evaluation_seed = assessment["seed"]
        evaluation_source = "saved held-out final assessment, verified against the checkpoint hash"
    else:
        result = recording
        evaluation_source = (
            "the single recorded episode only; no matching final assessment is available"
        )
    env = make_env(cfg.env)
    try:
        fps = float(env.unwrapped.metadata["render_fps"])
    finally:
        env.close()
    _save_gameplay(
        out / "gameplay.gif",
        frames,
        level=cfg.env.level,
        step=step,
        mode=args.mode,
        seed=args.seed,
        fps=fps,
        result=recording,
    )
    progress_path = args.run_dir / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    )
    training_step = int(progress.get("global_step", step))
    points = _logged_evaluations(
        args.run_dir / "metrics.csv", training_step if assessment else step
    )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_step": step,
        "training_step": training_step,
        "level": cfg.env.level,
        "policy_mode": args.mode,
        "evaluation_seed": evaluation_seed,
        "evaluation_policy_mode": evaluation_mode,
        "evaluation_source": evaluation_source,
        "saved_assessment": assessment or None,
        "recording_seed": args.seed,
        "max_steps": args.max_steps,
        "evaluation": result,
        "recording": recording,
        "recording_selection": "one fixed-seed episode, with no retries",
        "logged_evaluations": points,
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_report(cfg, checkpoint, report, points), encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "evaluation": result, "recording": recording}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
