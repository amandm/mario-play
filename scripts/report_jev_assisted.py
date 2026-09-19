"""Render a static learning comparison from downloaded Jev/PPO experiment artifacts.

    uv run --no-project --with matplotlib scripts/report_jev_assisted.py \
        --run-dir runs/colab/jev-assisted-v1

Only JSON artifacts are read. No models, credentials, network clients or training
code are loaded. Writes learning_curve.png and report.md inside the run directory.
The dependency-isolated command above leaves project dependencies unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CONDITIONS = (
    ("baseline", "PPO baseline · zero advice", "#475569"),
    ("jev_interval16", "Jev advice every 16 decisions", "#D97706"),
    ("jev_interval1", "Jev advice every decision", "#0F766E"),
)
SHORT_LABELS = {
    "baseline": "Baseline (0 advice)",
    "jev_interval16": "Jev / 16 decisions",
    "jev_interval1": "Jev / 1 decision",
}


def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def integer(value: Any) -> str:
    return "unavailable" if value is None else f"{int(value):,}"


def duration(value: Any) -> str:
    if value is None:
        return "unavailable"
    seconds = float(value)
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.1f} s"


def flags(result: dict[str, Any]) -> str:
    count = result.get("flags")
    episodes = result.get("episodes")
    if count is None and episodes is not None:
        count = round(float(result["flag_rate"]) * episodes)
    return f"{integer(count)}/{integer(episodes)}"


def criterion(record: dict[str, Any]) -> bool:
    sampled, greedy = record["sampled"], record["greedy"]
    return sampled["flag_rate"] >= 0.6 and greedy["flag_rate"] == 1.0


def update_count(record: dict[str, Any]) -> str:
    """Prefer checkpoint counters; label legacy rollout-derived estimates explicitly."""
    if record.get("ppo_updates") is not None:
        return integer(record["ppo_updates"])
    if record.get("global_step") is not None:
        return f"{integer(record['global_step'] // 1024)} (estimated)"
    return "unavailable"


def load_results(run_dir: Path) -> dict[str, Any]:
    progress = read_json(run_dir / "progress.json", {})
    if not progress:
        raise ValueError(f"experiment progress.json is missing or empty in {run_dir}")
    table = read_json(run_dir / "table.json", {})
    milestones, snapshots = {}, {}
    for name, _, _ in CONDITIONS:
        # Individual snapshots are fresher while the outer runner is inside a stage.
        snapshot = read_json(run_dir / name / "progress.json", {})
        if not snapshot:
            snapshot = progress.get("conditions", {}).get(name, {})
        snapshots[name] = snapshot
        records = read_json(run_dir / name / "milestones.json", [])
        seen: dict[int, dict[str, Any]] = {}
        for record in records:
            step = int(record["global_step"])
            if snapshot.get("global_step") is not None and step > snapshot["global_step"]:
                continue
            for mode in ("sampled", "greedy"):
                for key in ("flag_rate", "mean_progress"):
                    value = float(record[mode][key])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"invalid {mode}/{key} in {name} at {step}")
            seen[step] = record
        milestones[name] = dict(sorted(seen.items()))
    common_steps = sorted(set.intersection(*(set(rows) for rows in milestones.values())))
    return {
        "progress": progress,
        "table": table,
        "snapshots": snapshots,
        "milestones": milestones,
        "common_steps": common_steps,
        "continuation": read_json(run_dir / "continuation.json", {}),
    }


def plot_learning(data: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required; run: uv run --no-project --with matplotlib "
            "scripts/report_jev_assisted.py --run-dir PATH"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    steps = data["common_steps"]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10.5, 7.8))
        figure.subplots_adjust(top=0.82, bottom=0.16, left=0.105, right=0.97, hspace=0.17)
        figure.suptitle(
            "Jev advice cadence and PPO learning",
            x=0.105,
            y=0.972,
            ha="left",
            fontsize=20,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.105,
            0.925,
            "Level 1-1 · one training seed · identical initial 18-channel PPO networks",
            fontsize=11,
            color="#475569",
        )
        handles = []
        for name, label, color in CONDITIONS:
            records = [data["milestones"][name][step] for step in steps]
            x = [step / 1000 for step in steps]
            for axis, metric in zip(axes, ("flag_rate", "mean_progress"), strict=True):
                values = [row["sampled"][metric] for row in records]
                axis.plot(
                    x,
                    values,
                    color=color,
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    label=label,
                    zorder=3,
                )
                if metric == "flag_rate":
                    success = [
                        index
                        for index, row in enumerate(records)
                        if row["greedy"]["flag_rate"] == 1.0
                    ]
                    axis.scatter(
                        [x[index] for index in success],
                        [values[index] for index in success],
                        marker="*",
                        s=130,
                        color=color,
                        edgecolors="white",
                        linewidths=0.7,
                        zorder=5,
                    )
            handles.append(
                Line2D([0], [0], color=color, linewidth=2, marker="o", markersize=4, label=label)
            )
        axes[0].axhline(0.6, color="#64748B", linewidth=1, linestyle=(0, (4, 4)), zorder=1)
        axes[0].text(
            0.02,
            0.615,
            "Sampled criterion: 12/20; also requires greedy completion",
            transform=axes[0].get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=8.6,
            color="#64748B",
        )
        for axis in axes:
            axis.set_ylim(-0.04, 1.04)
            axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.grid(axis="y")
            axis.tick_params(axis="both", length=0, pad=7)
        resumed_at = data.get("continuation", {}).get("starting_steps_per_condition")
        if resumed_at is not None:
            for axis in axes:
                axis.axvline(resumed_at / 1000, color="#94A3B8", linestyle=":", linewidth=1)
            axes[1].text(
                resumed_at / 1000 + 12,
                0.04,
                "Continuation starts",
                fontsize=8,
                color="#64748B",
                rotation=90,
                va="bottom",
            )
        axes[0].set_ylabel("Sampled completion\n20 episodes")
        axes[1].set_ylabel("Sampled mean progress\n20 episodes")
        axes[1].set_xlabel("Training transitions per condition (thousands)", labelpad=10)
        axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        if not steps:
            for axis in axes:
                axis.text(
                    0.5,
                    0.45,
                    "Waiting for the first milestone shared by all three conditions",
                    transform=axis.transAxes,
                    ha="center",
                    color="#64748B",
                )
            axes[1].set_xlim(0, 100)
        elif len(steps) == 1:
            axes[1].set_xlim(max(0, steps[0] / 1000 - 50), steps[0] / 1000 + 50)
        else:
            axes[1].set_xlim(steps[0] / 1000 - 15, steps[-1] / 1000 + 15)
        figure.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.097, 0.898),
            frameon=False,
            ncol=3,
            fontsize=9.5,
            columnspacing=1.5,
            handlelength=1.8,
        )
        figure.text(
            0.105,
            0.066,
            "★ Greedy policy reached the flag at that checkpoint. Lines join raw evaluations; "
            "no smoothing or seed averaging.",
            fontsize=9,
            color="#475569",
        )
        figure.text(
            0.105,
            0.033,
            "Only equal-budget milestones shared by all conditions are plotted. "
            "Sampled seed 2,000,000; greedy seed 4,000,000.",
            fontsize=9,
            color="#475569",
        )
        figure.savefig(output, dpi=240, metadata={"Software": "mario-play report_jev_assisted.py"})
        plt.close(figure)


def write_report(data: dict[str, Any], run_dir: Path, output: Path) -> None:
    progress, table = data["progress"], data["table"]
    snapshots, milestones, steps = data["snapshots"], data["milestones"], data["common_steps"]
    cap = steps[-1] if steps else None
    attempts = table.get("usage", {}).get("attempted_calls")
    common_cost = progress.get("common_precomputation", {})
    if attempts is None:
        attempts = common_cost.get("physical_api_requests")
    usage = table.get("usage", common_cost.get("usage", {}))
    timing = table.get("timing", common_cost.get("timing", {}))
    coverage = table.get("coverage", {})
    lines = [
        "# Jev advice cadence and PPO learning",
        "",
        f"Run status: **{progress.get('status', 'unavailable')}**. "
        f"Latest matched evaluation budget: **{integer(cap)} training transitions per condition**. "
        f"Configured cap: {integer(progress.get('max_steps_per_condition'))} per condition.",
        "",
    ]
    continuation = data.get("continuation", {})
    if continuation:
        lines += [
            f"Continuation from **{integer(continuation['starting_steps_per_condition'])}** "
            "transitions per condition, using each latest checkpoint. "
            f"Learning-rate horizon remains "
            f"**{integer(continuation['fixed_learning_rate_horizon'])}**. "
            f"Restart semantics: {continuation['restart_semantics']}.",
            "",
        ]
    if cap is not None:
        results = [
            f"{SHORT_LABELS[name]}: **{flags(milestones[name][cap]['sampled'])}**"
            for name, _, _ in CONDITIONS
        ]
        lines += ["Sampled completions at that matched budget: " + "; ".join(results) + ".", ""]
    first_greedy = []
    for name, _, _ in CONDITIONS:
        successful = [
            step for step, row in milestones[name].items() if row["greedy"]["flag_rate"] == 1.0
        ]
        value = integer(min(successful)) if successful else "not observed"
        first_greedy.append(f"{SHORT_LABELS[name]}: **{value}**")
    lines += [
        "Secondary milestone, first greedy completion (training transitions): "
        + "; ".join(first_greedy)
        + ". A single greedy completion does not satisfy the combined target by itself.",
        "",
    ]
    lines += [
        "![Learning curves](learning_curve.png)",
        "",
        "The criterion is at least **12 flags in 20 sampled episodes and one greedy completion** "
        "at the same checkpoint. First crossing means the first evaluated checkpoint meeting both; "
        "it does not establish that performance remains above the threshold.",
        "",
        "| Condition | First criterion crossing | PPO updates at crossing | "
        "Sampled at matched budget | Greedy at matched budget | Mean progress | Selected step |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for name, _, _ in CONDITIONS:
        records = milestones[name]
        crossings = [step for step, record in records.items() if criterion(record)]
        crossing = min(crossings) if crossings else None
        record = records.get(cap)
        sampled = flags(record["sampled"]) if record else "unavailable"
        greedy = ("flag" if record["greedy"]["flag_rate"] == 1 else "no flag") if record else "—"
        mean = f"{record['sampled']['mean_progress']:.1%}" if record else "—"
        lines.append(
            f"| {SHORT_LABELS[name]} | {integer(crossing) if crossing else 'not observed'} | "
            f"{update_count(records[crossing]) if crossing else '—'} | "
            f"{sampled} | {greedy} | {mean} | "
            f"{integer(snapshots[name].get('selected_checkpoint_step'))} |"
        )
    lines += [
        "",
        "First crossings use all saved evaluations for each condition. "
        "Outcome columns compare the latest evaluation shared by all three conditions.",
        "",
        "| Condition | Transitions completed | PPO updates | Training advice refreshes | "
        "Transitions per refresh | Active wall time | Evaluation time within active time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _, _ in CONDITIONS:
        snapshot = snapshots[name]
        latest = next(reversed(milestones[name].values()), {})
        count = snapshot.get("global_step", latest.get("global_step"))
        refresh_count = snapshot.get("logical_advice_updates", latest.get("logical_advice_updates"))
        ratio = f"{count / refresh_count:.2f}" if count is not None and refresh_count else "—"
        lines.append(
            f"| {SHORT_LABELS[name]} | {integer(count)} | "
            f"{update_count({**latest, **snapshot})} | "
            f"{integer(refresh_count)} | {ratio} | "
            f"{duration(snapshot.get('active_seconds', latest.get('active_seconds')))} | "
            f"{duration(snapshot.get('evaluation_seconds', latest.get('evaluation_seconds')))} |"
        )
    lines += [
        "",
        "PPO updates use recorded counters when available. Legacy values marked estimated are "
        "floor(transitions / 1,024): eight environments × 128 steps per rollout. Each PPO update "
        "has four epochs and four minibatches. A process restart discards an incomplete rollout, "
        "so the estimate can exceed actual PPO updates after a restart. Ordinary stage boundaries "
        "preserve partial rollouts. Active wall time includes evaluation and checkpoint work, and "
        "excludes time while another condition is running.",
        "",
        f"Experiment wall time: **{duration(progress.get('wall_seconds'))}**. "
        f"Shared Jev table construction: **{duration(timing.get('builder_wall_seconds'))}**.",
        "",
        "## API requests and advice refreshes",
        "",
        f"The frozen table used **{integer(attempts)} actual API requests once**, "
        "shared across both "
        "assisted conditions. Training and evaluation use local lookups and make **zero new API "
        "requests**. A logical refresh obtains four advice features; it is not four API calls.",
        "",
        f"Table coverage: {integer(coverage.get('completed_cases'))}/"
        f"{integer(coverage.get('expected_cases'))} cases; "
        f"model `{table.get('model', 'unavailable')}`. "
        f"Recorded input tokens: {integer(usage.get('input_tokens'))}; output tokens: "
        f"{integer(usage.get('output_tokens'))}; calls with unaccounted usage: "
        f"{integer(usage.get('unaccounted_calls'))}.",
        "",
        "Advice refreshes on episode reset and every 16 or every 1 environment decisions. "
        "Reset refreshes explain why the measured transitions-per-refresh ratio can be below "
        "the nominal interval. The training counters exclude evaluation and playback refreshes; "
        "those are recorded separately in the episode artifacts. Sparse advice holds its previous "
        "four values while the original grid continues updating.",
        "",
        "## Method and interpretation",
        "",
        "All three policies originally started from scratch with training seed 0 and identical "
        "18-channel network weights. Observations use the same original 14-channel grid "
        "and four auxiliary planes. "
        "The control receives "
        "zeros; assisted policies receive four frozen Jev risk probabilities. PPO chooses every "
        "action. Action space, frame skip 4, stall limit 150, level 1-1, rewards, and PPO "
        "hyperparameters are shared. The learning-rate horizon remains 5,000,000 transitions "
        "even when this comparison stops sooner.",
        "",
        "Validation uses 20 sampled episodes starting at seed 2,000,000 and one greedy episode at "
        "seed 4,000,000, with a 6,000-decision cap. Every milestone repeats these fixed evaluation "
        "seeds. Completion estimates therefore move in five-percentage-point increments; these "
        "episodes are not independent training seeds. The plot shows raw evaluations, including "
        "regressions, and stops each comparison at budgets available for every condition.",
        "",
        "This experiment tests the cadence of a frozen advice feature stream. It cannot show "
        "that making more live API requests improves learning or establish live API latency/cost "
        "scaling. The auxiliary features also contain engineered geometry extraction and "
        "bucketing, so improvement over zeros cannot be attributed to Jev reasoning alone. "
        "Comparing the two assisted conditions holds that feature construction fixed while "
        "changing how recently the advice was refreshed.",
        "",
        "One training seed on one level is exploratory evidence. Earlier threshold crossing can "
        "be reported as an observation of this run, but it does not establish a reliable speedup, "
        "generalization, or a monotonic benefit from more frequent advice. The same validation "
        "episodes select example checkpoints. "
        + (
            "This run continues to its fixed budget regardless of evaluation scores. "
            if progress.get("fixed_budget")
            else "Validation can stop training once every condition meets the target. "
        )
        + "There is no independent held-out assessment in this comparison.",
        "",
    ]
    nonmonotonic = []
    for name, _, _ in CONDITIONS:
        rates = [row["sampled"]["flag_rate"] for row in milestones[name].values()]
        if any(after < before for before, after in zip(rates, rates[1:], strict=False)):
            nonmonotonic.append(SHORT_LABELS[name])
    if nonmonotonic:
        lines += [
            "Observed completion regressions between checkpoints: "
            + ", ".join(nonmonotonic)
            + ". Earliest crossing and final performance "
            "should therefore be read separately.",
            "",
        ]
    matched = sorted(
        (
            path
            for path in run_dir.glob("matched_*.gif")
            if path.stem.removeprefix("matched_").isdigit()
        ),
        key=lambda path: int(path.stem.removeprefix("matched_")),
    )
    if matched:
        latest_clip = matched[-1]
        clip_steps = int(latest_clip.stem.removeprefix("matched_"))
        lines += [
            "## Matched-budget policy clips",
            "",
            f"[Watch all three policies at {clip_steps:,} transitions]({latest_clip.name}). "
            "Each is the first greedy episode at seed 4,000,000. Simulator decisions are aligned "
            "at their original rate; finished episodes hold their actual final frame with an "
            "explicit EPISODE ENDED label.",
            "",
        ]
        for name, _, _ in CONDITIONS:
            individual = run_dir / name / latest_clip.name
            if individual.is_file():
                lines.append(f"- [{SHORT_LABELS[name]}]({name}/{latest_clip.name})")
        lines.append("")
    lines += ["## Selected policy clips", ""]
    clips = []
    for name, _, _ in CONDITIONS:
        if (run_dir / name / "selected.gif").is_file():
            selected = snapshots[name].get("selected_checkpoint_step")
            clips.append(
                f"- [{SHORT_LABELS[name]} · selected at {integer(selected)} transitions]"
                f"({name}/selected.gif)"
            )
    lines += clips or ["Selected clips have not been downloaded yet."]
    lines += [
        "",
        "These are the first fixed-seed greedy recordings (seed 4,000,000) of each selected "
        "checkpoint: earliest criterion crossing, otherwise best validation. Selected checkpoints "
        "can have different training budgets. "
        + (
            "Use the matched-budget clips above for a visual comparison at the same budget. "
            if matched
            else "Matched-budget clips have not been recorded yet. "
        )
        + "Use the learning curves above for the numerical comparison.",
        "",
        "Artifacts: [experiment progress](progress.json), [frozen table](table.json), "
        "[learning curve](learning_curve.png).",
        "",
    ]
    if (run_dir / "table.json").exists():
        table_sha = hashlib.sha256((run_dir / "table.json").read_bytes()).hexdigest()
        lines += [f"Frozen table SHA-256: `{table_sha}`.", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    data = load_results(run_dir)
    plot_learning(data, run_dir / "learning_curve.png")
    write_report(data, run_dir, run_dir / "report.md")
    print(
        json.dumps(
            {
                "report": str(run_dir / "report.md"),
                "chart": str(run_dir / "learning_curve.png"),
                "matched_steps": data["common_steps"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
