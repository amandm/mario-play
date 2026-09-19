"""Export the public research bundle from the original local run archives.

Run from the repository root with ``python docs/research/export_artifacts.py``.
The input runs are intentionally not rewritten. Pillow is required for GIF
encoding; all numerical outputs come directly from saved evaluations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MARIO = ROOT / "runs/colab/jev-assisted-v2"
TAXI = ROOT / "runs/taxi-jev-v1"
MARIO_ARMS = ("baseline", "jev_interval16", "jev_interval1")
TAXI_ARMS = ("zeros", "rules", "jev")
SOURCES: dict[str, str] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> Path:
    SOURCES[path.relative_to(ROOT).as_posix()] = sha256(path)
    return path


def read(path: Path):
    return json.loads(source(path).read_text())


def write_json(name: str, data) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pick(data: dict, keys: str) -> dict:
    return {key: data[key] for key in keys.split()}


def main() -> None:
    mario_rows, mario_episodes = [], []
    for condition in MARIO_ARMS:
        milestones = read(MARIO / condition / "milestones.json")
        assert [m["global_step"] for m in milestones] == list(range(100000, 2000001, 100000))
        for record in milestones:
            context = {
                "condition": condition,
                "training_seed": 0,
                "global_step": record["global_step"],
            }
            row = {
                **context,
                "checkpoint_sha256": record["checkpoint_sha256"],
                "ppo_updates": record.get("ppo_updates", record["global_step"] // 1024),
                "ppo_updates_kind": "recorded" if "ppo_updates" in record else "estimated",
                "training_feature_refreshes": record["logical_advice_updates"],
                "active_seconds": record["active_seconds"],
                "evaluation_seconds": record["evaluation_seconds"],
            }
            for mode in ("sampled", "greedy"):
                assessment = record[mode]
                assert assessment["episodes"] == (20 if mode == "sampled" else 1)
                fields = "episodes flags flag_rate mean_return std_return mean_length mean_progress"
                for key in fields.split():
                    row[f"{mode}_{key}"] = assessment[key]
                for episode in assessment["episode_results"]:
                    mario_episodes.append(
                        {
                            **context,
                            "policy_mode": mode,
                            **pick(
                                episode,
                                "seed return length flag_get progress terminated truncated "
                                "decision_cap_reached",
                            ),
                        }
                    )
            mario_rows.append(row)
    write_csv("mario-milestones.csv", mario_rows)
    write_csv("mario-episodes.csv", mario_episodes)
    assert len(mario_rows) == 60 and len(mario_episodes) == 1260

    taxi_rows, taxi_episodes, initializations = [], [], []
    for condition in TAXI_ARMS:
        for seed in range(5):
            run = TAXI / condition / f"seed_{seed}"
            result = read(run / "result.json")
            initializations.append(
                pick(result, "condition training_seed initial_model_sha256 checkpoint_sha256")
            )
            records = read(run / "milestones.json")
            assert [m["global_step"] for m in records] == list(range(20000, 200001, 20000))
            for record in records:
                context = {
                    "condition": condition,
                    "training_seed": seed,
                    "global_step": record["global_step"],
                }
                assessment = record["assessment"]
                assert assessment["episodes"] == 300
                taxi_rows.append(
                    {
                        **context,
                        **pick(record, "checkpoint_sha256 ppo_updates optimizer_steps"),
                        **pick(
                            assessment,
                            "episodes successes success_rate mean_return std_return mean_length "
                            "mean_illegal_pickup_dropoff logical_feature_refreshes "
                            "physical_api_requests",
                        ),
                    }
                )
                for episode in assessment["episode_results"]:
                    taxi_episodes.append(
                        {
                            **context,
                            **pick(
                                episode,
                                "initial_state final_state return steps illegal_pickup_dropoff "
                                "success terminated truncated",
                            ),
                        }
                    )
    write_csv("taxi-milestones.csv", taxi_rows)
    write_csv("taxi-episodes.csv", taxi_episodes)
    assert len(taxi_rows) == 150 and len(taxi_episodes) == 45000

    diagnostic_rows, initial_episodes = [], []
    for record in read(TAXI / "initial_policy_diagnostic.json")["rows"]:
        context = pick(record, "condition training_seed initial_checkpoint_sha256")
        row = dict(context)
        for phase in ("initial", "final"):
            assessment = record[f"{phase}_assessment"]
            for (
                key
            ) in "episodes successes success_rate mean_return mean_illegal_pickup_dropoff".split():
                row[f"{phase}_{key}"] = assessment[key]
        diagnostic_rows.append(row)
        for episode in record["initial_assessment"]["episode_results"]:
            initial_episodes.append(
                {
                    **context,
                    **pick(
                        episode,
                        "initial_state final_state return steps illegal_pickup_dropoff "
                        "success terminated truncated",
                    ),
                }
            )
    write_csv("taxi-initial-diagnostic.csv", diagnostic_rows)
    write_csv("taxi-initial-episodes.csv", initial_episodes)
    assert len(diagnostic_rows) == 15 and len(initial_episodes) == 4500

    for src, name in (
        (MARIO / "learning_curve.png", "mario-learning-curve.png"),
        (MARIO / "matched_2000000.gif", "mario-matched-2000000.gif"),
        (TAXI / "penalty-learning.png", "taxi-penalty-learning.png"),
        (TAXI / "learning-curves.png", "taxi-learning-curves.png"),
        (MARIO / "table.json", "mario-jev-table.json"),
        (TAXI / "table.json", "taxi-jev-table.json"),
    ):
        shutil.copyfile(source(src), OUT / name)
        assert sha256(src) == sha256(OUT / name)

    # Preserve every original Taxi frame, canvas dimension and duration. A
    # shared palette without dithering permits much smaller GIF compression.
    taxi_gif = source(TAXI / "taxi-start-461.gif")
    with Image.open(taxi_gif) as image:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(image.info["duration"])
    palette = frames[0].quantize(colors=128, dither=Image.Dither.NONE)
    encoded = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    encoded[0].save(
        OUT / "taxi-start-461.gif",
        save_all=True,
        append_images=encoded[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    with Image.open(OUT / "taxi-start-461.gif") as image:
        assert image.n_frames == len(frames)
        assert image.size == frames[0].size
        for index, duration in enumerate(durations):
            image.seek(index)
            assert image.info["duration"] == duration

    mario_experiment = read(MARIO / "experiment.json")
    taxi_experiment = read(TAXI / "experiment.json")
    mario_config = mario_experiment["conditions"][0]
    taxi_config = taxi_experiment["configs"][0]
    continuation = read(MARIO / "continuation.json")
    mario_playback = read(MARIO / "matched_2000000.json")
    settings = {
        "mario": {
            "environment": "MarioPlay-v0, original platform game; level 1-1",
            "training_seeds": [0],
            "conditions": {
                "baseline": "Four zero planes",
                "jev_interval16": (
                    "Four frozen Jev probabilities refreshed every 16 decisions and on reset"
                ),
                "jev_interval1": (
                    "Four frozen Jev probabilities refreshed every decision and on reset"
                ),
            },
            "observations": "14 grid planes plus four auxiliary planes; shape 18 x 15 x 16",
            "action_space": "7 simple actions; frame skip 4; frame stack 1; stall limit 150",
            "architecture": (
                "GridEncoder with hidden_size 256; identical initial weights across conditions"
            ),
            "initial_model_sha256": read(MARIO / "baseline/progress.json")["initial_model_sha256"],
            "total_transitions_per_condition": 2000000,
            "learning_rate_annealing_horizon": 5000000,
            "n_envs": 8,
            "ppo": mario_config["ppo"],
            "primary_target": (
                "At least 12/20 sampled completions plus greedy completion "
                "at the same evaluated checkpoint"
            ),
            "evaluation": (
                "Every 100000 transitions, 20 sampled episodes with seeds 2000000..2000019 "
                "and one greedy episode with seed 4000000; maximum 6000 decisions"
            ),
            "continuation": pick(
                continuation,
                "starting_steps_per_condition target_steps_per_condition restart_semantics "
                "expected_final_ppo_updates_per_condition",
            ),
            "api": {
                "model": "jev-1.13.0",
                "table_sha256": mario_experiment["table_sha256"],
                "precomputation_requests": 476,
                "training_evaluation_requests": 0,
            },
            "playback": {
                **pick(mario_playback, "global_step seed deterministic frames fps alignment"),
                "episodes": {
                    arm: {
                        "checkpoint_sha256": value["checkpoint_sha256"],
                        "decisions": value["result"]["episode_results"][0]["length"],
                        "completed": value["result"]["episode_results"][0]["flag_get"],
                    }
                    for arm, value in mario_playback["conditions"].items()
                },
            },
        },
        "taxi": {
            **pick(
                taxi_experiment,
                "environment gymnasium_version torch_version numpy_version horizon "
                "milestone_steps conditions seeds action_mask evaluation primary secondary "
                "gif_initial_states gif_training_seed",
            ),
            "observations": (
                "19 decoded state one-hot values + 40 fixed-map wall bits + eight landmark "
                "coordinates + four auxiliary values; total 71"
            ),
            "architecture": "64 x 64 MLP; paired identical initial weights within each seed",
            "n_envs": 8,
            "episode_limit": 200,
            "ppo": taxi_config["ppo"],
            "initializations_and_final_checkpoints": initializations,
            "api": {
                "model": "jev-1.13.0",
                "table_sha256": taxi_experiment["table_sha256"],
                "precomputation_requests": 500,
                "training_evaluation_requests": 0,
            },
            "playback": {
                "initial_state": 461,
                "training_seed": 0,
                "global_step": 200000,
                "frames": len(frames),
                "dimensions": list(frames[0].size),
                "total_duration_ms": sum(durations),
                "transformation": (
                    "128-color shared palette, no dithering; every source frame "
                    "and original duration preserved"
                ),
                "episodes": [x for x in read(TAXI / "gameplay.json") if x["initial_state"] == 461],
            },
        },
    }
    write_json("experiment-settings.json", settings)
    accuracy = read(TAXI / "feature_accuracy.json")
    write_json(
        "taxi-feature-diagnostics.json",
        pick(accuracy, "table_sha256 source features api_calls unaccounted_calls"),
    )
    mario_hashes = read(MARIO / "source_manifest.json")
    write_json(
        "source-hashes.json",
        {
            "description": (
                "Hashes of source files frozen for each training run. Reporting/documentation "
                "may have changed after training. These hashes do not imply that model "
                "checkpoint bytes are included in this public bundle."
            ),
            "mario_base_commit": mario_hashes["git_commit"],
            "mario_training_source": {
                k: v
                for k, v in mario_hashes["files_sha256"].items()
                if k.startswith(("src/", "scripts/", "configs/"))
                or k in ("pyproject.toml", "uv.lock")
            },
            "taxi_training_source": {
                k: v
                for k, v in taxi_experiment["source_hashes"].items()
                if k.startswith(("src/", "scripts/")) or k in ("pyproject.toml", "uv.lock")
            },
        },
    )
    write_json(
        "manifest.json",
        {
            "schema_version": 1,
            "description": (
                "Curated research artifact checksums. Local input names identify provenance; "
                "original run archives and checkpoint bytes are not bundled."
            ),
            "records": {
                "mario_milestones": 60,
                "mario_episodes": 1260,
                "taxi_milestones": 150,
                "taxi_episodes": 45000,
                "taxi_initial_diagnostic_rows": 15,
                "taxi_initial_episodes": 4500,
            },
            "source_artifact_sha256": SOURCES,
            "public_artifacts": {
                p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                for p in sorted(OUT.iterdir())
                if p.is_file() and p.name not in ("manifest.json", "README.md")
            },
        },
    )
    print(
        "Exported 60 Mario milestones, 150 Taxi milestones and all underlying evaluation episodes."
    )
    print(
        f"Taxi GIF: {len(frames)} frames, {sum(durations)} ms, "
        f"{(OUT / 'taxi-start-461.gif').stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
