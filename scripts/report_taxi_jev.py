"""Summarize all Taxi training seeds and record fixed final-policy gameplay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

CONDITIONS = ("zeros", "rules", "jev")
LABELS = {"zeros": "PPO + zeros", "rules": "PPO + exact rules", "jev": "PPO + Jev"}
COLORS = {"zeros": "#2563eb", "rules": "#d97706", "jev": "#0f766e"}


def normalized_auc(records: list[dict[str, Any]]) -> float | None:
    """Trapezoidal success area divided by the observed milestone interval (excludes step zero)."""
    if len(records) < 2:
        return None
    x = np.array([row["global_step"] for row in records], dtype=np.float64)
    y = np.array([row["assessment"]["success_rate"] for row in records], dtype=np.float64)
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)) / (x[-1] - x[0]))


def confirmed_target(records: list[dict[str, Any]]) -> int | None:
    """Confirmation time is the third consecutive checkpoint at >=90%, never backdated."""
    streak = 0
    for row in records:
        streak = streak + 1 if row["assessment"]["success_rate"] >= 0.9 else 0
        if streak == 3:
            return int(row["global_step"])
    return None


def collect(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "experiment.json").read_text())
    conditions = {}
    for condition in CONDITIONS:
        runs = []
        for seed in manifest["seeds"]:
            directory = root / condition / f"seed_{seed}"
            path = directory / "milestones.json"
            if not path.exists():
                continue
            records = json.loads(path.read_text())
            if not records:
                continue
            final = records[-1]
            steps = [row["global_step"] for row in records]
            expected_steps = list(
                range(
                    manifest["milestone_steps"],
                    manifest["horizon"] + 1,
                    manifest["milestone_steps"],
                )
            )
            if steps != expected_steps[: len(steps)]:
                raise ValueError(f"invalid milestone sequence: {directory}")
            complete = steps == expected_steps and (directory / "result.json").exists()
            if complete:
                result = json.loads((directory / "result.json").read_text())
                final_checkpoint = directory / "checkpoints/final.pt"
                if (
                    result["status"] != "complete"
                    or result["global_step"] != manifest["horizon"]
                    or hashlib.sha256(final_checkpoint.read_bytes()).hexdigest()
                    != result["checkpoint_sha256"]
                ):
                    raise ValueError(f"invalid completed run: {directory}")
                for record in records:
                    if (
                        hashlib.sha256((directory / record["checkpoint"]).read_bytes()).hexdigest()
                        != record["checkpoint_sha256"]
                    ):
                        raise ValueError(f"milestone checkpoint hash mismatch: {directory}")
            runs.append(
                {
                    "seed": seed,
                    "complete": complete,
                    "records": records,
                    "normalized_success_auc": normalized_auc(records) if complete else None,
                    "final_success_rate": final["assessment"]["success_rate"] if complete else None,
                    "final_mean_return": final["assessment"]["mean_return"] if complete else None,
                    "final_mean_length": final["assessment"]["mean_length"] if complete else None,
                    "final_mean_illegal": final["assessment"]["mean_illegal_pickup_dropoff"]
                    if complete
                    else None,
                    "confirmed_90_percent_step": confirmed_target(records),
                    "final_step": final["global_step"],
                    "final_ppo_updates": final["ppo_updates"],
                }
            )
        conditions[condition] = runs
    complete = all(
        len(runs) == len(manifest["seeds"]) and all(row["complete"] for row in runs)
        for runs in conditions.values()
    )
    return {"complete": complete, "manifest": manifest, "conditions": conditions}


def plot_curves(root: Path, data: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    dimensions = (
        ("success_rate", "Delivery success", True),
        ("mean_return", "Mean episode return", False),
        ("mean_length", "Mean actions (failures included)", False),
        ("mean_illegal_pickup_dropoff", "Illegal pickup/dropoff per episode", False),
    )
    for axis, (metric, title, fraction) in zip(axes.flat, dimensions, strict=True):
        for condition, runs in data["conditions"].items():
            if not runs:
                continue
            common_steps = sorted(
                set.intersection(
                    *({record["global_step"] for record in row["records"]} for row in runs)
                )
            )
            values = []
            for row in runs:
                records = {record["global_step"]: record for record in row["records"]}
                x = [record["global_step"] / 1000 for record in row["records"]]
                y = [record["assessment"][metric] for record in row["records"]]
                axis.plot(x, y, color=COLORS[condition], alpha=0.22, linewidth=0.8)
                values.append([records[step]["assessment"][metric] for step in common_steps])
            y = np.array(values)
            x = np.array(common_steps) / 1000
            axis.fill_between(x, y.min(0), y.max(0), color=COLORS[condition], alpha=0.07)
            axis.plot(
                x,
                y.mean(0),
                color=COLORS[condition],
                linewidth=2.2,
                label=f"{LABELS[condition]} (n={len(runs)})",
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Training interactions (thousands)")
        axis.grid(alpha=0.15)
        if fraction:
            from matplotlib.ticker import PercentFormatter

            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.set_ylim(-0.03, 1.03)
            all_runs = [row for runs in data["conditions"].values() for row in runs]
            if data["complete"] and all(
                record["assessment"]["success_rate"] == 0
                for row in all_runs
                for record in row["records"]
            ):
                axis.text(
                    0.03,
                    0.15,
                    f"All {len(all_runs)} curves overlap: 0 deliveries",
                    transform=axis.transAxes,
                    fontsize=11,
                    fontweight="bold",
                )
    axes[0, 0].legend(loc="best", frameon=False)
    figure.suptitle(
        "Taxi: five paired training seeds, fixed 200,000-interaction budget\n"
        "Bold = seed mean; thin = each seed; shading = seed range, not confidence interval",
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(root / "learning-curves.png", dpi=160)
    figure.savefig(root / "learning-curves.svg")
    plt.close(figure)


def _caption(rgb: np.ndarray, label: str, step: int, note: str) -> Image.Image:
    image = Image.fromarray(rgb).convert("RGB")
    width = 300
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.NEAREST)
    result = Image.new("RGB", (width, height + 60), "#111827")
    result.paste(image, (0, 60))
    ImageDraw.Draw(result).multiline_text(
        (6, 5),
        f"{label} | training seed 0\nFinal 200k policy | action {step}\n{note}",
        fill="white",
        spacing=3,
    )
    return result


def record_gameplay(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Record predeclared states with final seed-0 policies; preserve failures."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    script = runpy.run_path(str(Path(__file__).with_name("train_taxi_jev.py")))
    from mario_play.envs.factory import make_env
    from mario_play.rl.utils import get_rng_state, set_rng_state

    rng = get_rng_state()
    outcomes = []
    try:
        for initial_state in manifest["gif_initial_states"]:
            clips = []
            for condition in CONDITIONS:
                checkpoint = root / condition / "seed_0/checkpoints/final.pt"
                algo, cfg = script["load_portable_algorithm"](checkpoint, root / "table.json")
                env = make_env(cfg.env, render_mode="rgb_array")
                frames = []
                total, illegal = 0.0, 0
                try:
                    obs, _ = env.reset(seed=0, options={"state": initial_state})
                    frames.append(
                        _caption(
                            env.render(), LABELS[condition], 0, f"Initial state {initial_state}"
                        )
                    )
                    for step in range(1, 201):
                        action = int(algo.predict(obs[None], deterministic=True)[0])
                        obs, reward, terminated, truncated, _ = env.step(action)
                        total += reward
                        illegal += action >= 4 and reward == -10
                        note = (
                            "DELIVERED"
                            if terminated
                            else "TIME LIMIT"
                            if truncated
                            else (f"Return {total:.0f} | illegal pickup/dropoff {illegal}")
                        )
                        frames.append(_caption(env.render(), LABELS[condition], step, note))
                        if terminated or truncated:
                            break
                    outcomes.append(
                        {
                            "condition": condition,
                            "training_seed": 0,
                            "initial_state": initial_state,
                            "checkpoint_sha256": script["digest"](checkpoint),
                            "steps": step,
                            "return": total,
                            "illegal_pickup_dropoff": illegal,
                            "success": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    clips.append(frames)
                finally:
                    env.close()
            combined = []
            for index in range(max(map(len, clips))):
                row = [clip[min(index, len(clip) - 1)] for clip in clips]
                frame = Image.new(
                    "RGB", (sum(image.width for image in row) + 16, row[0].height), "#111827"
                )
                x = 0
                for image in row:
                    frame.paste(image, (x, 0))
                    x += image.width + 8
                combined.append(frame.quantize(colors=256, dither=Image.Dither.NONE))
            combined[0].save(
                root / f"taxi-start-{initial_state}.gif",
                save_all=True,
                append_images=combined[1:],
                duration=150,
                loop=0,
                optimize=False,
            )
        (root / "gameplay.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    finally:
        set_rng_state(rng)
    return outcomes


def write_report(root: Path, data: dict[str, Any]) -> None:
    manifest = data["manifest"]
    total_steps = manifest["horizon"] * len(manifest["seeds"]) * len(CONDITIONS)
    all_zero = data["complete"] and all(
        record["assessment"]["success_rate"] == 0
        for runs in data["conditions"].values()
        for row in runs
        for record in row["records"]
    )
    interpretation = (
        "**The comparison is inconclusive about Jev's benefit:** no condition learned greedy "
        "delivery under this fixed budget. Every paired success difference is zero, but this "
        "floor effect does not establish equivalence or show that Jev generally cannot help. "
        "The finding applies to these four frozen features, Jev 1.13, and this PPO/Taxi setup."
        if all_zero
        else "Interpret paired seed outcomes within this fixed PPO/Taxi protocol."
    )
    lines = [
        "# Taxi: PPO with zero, exact-rule, and frozen Jev features",
        "",
        interpretation,
        "",
        "Five paired training seeds; no action masks. Every final policy is evaluated greedily "
        "from all 300 valid initial states. This is the existing task distribution, not "
        "held-out generalization. Training seeds are the replication unit.",
        "",
        f"Fixed budget: {manifest['horizon']:,} interactions per run and {total_steps:,} "
        "interactions across all conditions and seeds.",
        "",
        f"Status: **{'complete' if data['complete'] else 'incomplete'}**. "
        "AUC is normalized success area over the fixed 20k..200k interval, excluding 0..20k.",
        "",
        "![Learning curves](learning-curves.png)",
        "",
        "| Condition | Seeds completed | Mean success AUC | Final success mean [seed range] | "
        "Final return mean | Final illegal actions/episode |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, runs in data["conditions"].items():
        runs = [row for row in runs if row["complete"]]
        if not runs:
            lines.append(f"| {LABELS[condition]} | 0 | pending | pending | pending | pending |")
            continue
        final = np.array([row["final_success_rate"] for row in runs])
        aucs = [
            row["normalized_success_auc"]
            for row in runs
            if row["normalized_success_auc"] is not None
        ]
        auc = f"{np.mean(aucs):.3f}" if aucs else "pending"
        lines.append(
            f"| {LABELS[condition]} | {len(runs)} | {auc} | {final.mean():.1%} "
            f"[{final.min():.1%}, {final.max():.1%}] | "
            f"{np.mean([row['final_mean_return'] for row in runs]):.2f} | "
            f"{np.mean([row['final_mean_illegal'] for row in runs]):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Every training seed",
            "",
            "| Condition | Seed | Success AUC | Final successes | Final mean actions | "
            "Confirmed >=90% for 3 milestones |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for condition, runs in data["conditions"].items():
        for row in runs:
            if not row["complete"]:
                lines.append(
                    f"| {LABELS[condition]} | {row['seed']} | pending | pending | pending | "
                    f"incomplete at {row['final_step']:,} |"
                )
                continue
            auc = row["normalized_success_auc"]
            target = row["confirmed_90_percent_step"]
            lines.append(
                f"| {LABELS[condition]} | {row['seed']} | {auc:.3f}"
                if auc is not None
                else f"| {LABELS[condition]} | {row['seed']} | pending"
            )
            lines[-1] += (
                f" | {round(row['final_success_rate'] * 300)}/300 | "
                f"{row['final_mean_length']:.1f} | "
                f"{f'{target:,}' if target is not None else 'not observed'} |"
            )
    if data["complete"]:
        lines.extend(
            [
                "",
                "## Paired differences",
                "",
                "Positive means Jev did better on the specified metric; each difference "
                "pairs the same training seed. These five replicates do not establish "
                "a universal effect.",
                "",
            ]
        )
        jev = {row["seed"]: row for row in data["conditions"]["jev"]}
        for control in ("zeros", "rules"):
            controls = {row["seed"]: row for row in data["conditions"][control]}
            differences = [
                jev[seed]["normalized_success_auc"] - controls[seed]["normalized_success_auc"]
                for seed in sorted(jev)
            ]
            final = [
                jev[seed]["final_success_rate"] - controls[seed]["final_success_rate"]
                for seed in sorted(jev)
            ]
            lines.append(
                f"- Jev minus {control}: mean AUC difference **{np.mean(differences):+.3f}**; "
                f"per seed {', '.join(f'{value:+.3f}' for value in differences)}. "
                f"Mean final success difference **{np.mean(final) * 100:+.1f} percentage points**."
            )
    diagnostic_path = root / "initial_policy_diagnostic.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        lines.extend(
            [
                "",
                "## Post-hoc penalty-avoidance diagnostic",
                "",
                "Saved untrained checkpoints were evaluated after the experiment on the same "
                "300 starts. This diagnostic is excluded from the primary AUC. Initial and "
                "final delivery success were zero. The return and illegal-action changes "
                "support learning to avoid penalties, without learning delivery.",
                "",
                "| Condition | Initial return | Final return | Initial illegal actions/episode | "
                "Final illegal actions/episode |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in CONDITIONS:
            rows = [row for row in diagnostic["rows"] if row["condition"] == condition]
            means = [
                np.mean([row[stage][metric] for row in rows])
                for metric in ("mean_return", "mean_illegal_pickup_dropoff")
                for stage in ("initial_assessment", "final_assessment")
            ]
            lines.append(
                f"| {LABELS[condition]} | " + " | ".join(f"{value:.2f}" for value in means) + " |"
            )
        if (root / "penalty-learning.png").exists():
            lines.extend(
                ["", "![Post-hoc initial-to-final penalty learning](penalty-learning.png)"]
            )
    lines.extend(
        [
            "",
            "## Fixed gameplay",
            "",
            "Final seed-0 policies, with starts selected before training. Left: zeros; "
            "middle: exact rules; right: Jev. Shorter episodes hold their final frame.",
        ]
    )
    for state in data["manifest"]["gif_initial_states"]:
        if (root / f"taxi-start-{state}.gif").exists():
            lines.extend(["", f"![Initial state {state}](taxi-start-{state}.gif)"])
    lines.extend(
        [
            "",
            "All checkpoints, original table responses, exact configurations, per-start "
            "outcomes, source hashes and a source snapshot remain alongside this report. "
            "There were zero physical API requests during training/evaluation. Jev was "
            "neither fine-tuned nor used as an expert-action target.",
        ]
    )
    accuracy_path = root / "feature_accuracy.json"
    if accuracy_path.exists():
        accuracy = json.loads(accuracy_path.read_text())
        lines.extend(
            [
                "",
                "## Frozen feature diagnostics",
                "",
                "Feature diagnostics cover all 500 encoded states, including states outside the "
                "usual starting distribution. A probability threshold of 0.5 is used only for "
                "this diagnostic; PPO receives the continuous probabilities unchanged. "
                "Class imbalance makes overall accuracy insufficient. These diagnostics do not "
                "replace the paired learning comparisons.",
                "",
                "Full counts, positive prevalence, false positives, and false negatives: "
                "[feature_accuracy.json](feature_accuracy.json).",
                "",
                "| Feature | Positive states / 500 | True positives | False positives | "
                "False negatives | Precision | Recall |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, values in accuracy["features"].items():
            positive = values["positive_states"]
            tp = positive - values["false_negatives"]
            fp = values["false_positives"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / positive if positive else 0.0
            lines.append(
                f"| {name} | {positive} | {tp} | {fp} | {values['false_negatives']} | "
                f"{precision:.1%} | {recall:.1%} |"
            )
    usage = data["manifest"].get("precomputation", {})
    table_metadata = json.loads((root / "table.json").read_text())
    seconds = table_metadata.get("timing", {}).get("builder_wall_seconds")
    timing = f" Builder wall time: {seconds:.2f} seconds." if seconds is not None else ""
    lines.extend(
        [
            "",
            f"Frozen table precomputation: {usage.get('attempted_calls', 0):,} actual API "
            f"requests, {usage.get('input_tokens', 0):,} input tokens, "
            f"{usage.get('output_tokens', 0):,} output tokens, "
            f"{usage.get('unaccounted_calls', 0)} unaccounted requests.{timing}",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--skip-gifs", action="store_true")
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    data = collect(root)
    if not any(data["conditions"].values()):
        raise ValueError("no Taxi milestones exist yet")
    plot_curves(root, data)
    if data["complete"] and not args.skip_gifs:
        import torch

        torch.set_num_threads(1)
        record_gameplay(root, data["manifest"])
    write_report(root, data)
    (root / "summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(root / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
