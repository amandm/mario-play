"""Exercise actual training, interruption, checkpoint resume and safe reruns."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from mario_play.rl.checkpoint import load_checkpoint


def test_experiment_suite_resumes_and_rejects_changed_settings(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"
    main = runpy.run_path(str(script))["main"]
    trainer_type = main.__globals__["Trainer"]

    def stop_after_update(*args, **kwargs):
        trainer = trainer_type(*args, **kwargs)
        real_update = trainer.algo.update

        def update(*args, **kwargs):
            result = real_update(*args, **kwargs)
            trainer.request_stop()
            return result

        trainer.algo.update = update
        return trainer

    args = [
        "--run-root",
        str(tmp_path),
        "--device",
        "cpu",
        "--benchmark-steps",
        "16",
        "--steps",
        "32",
        "--seeds",
        "0",
        "1",
        "--checkpoint-interval",
        "16",
        "--eval-episodes",
        "2",
    ]
    for override in (
        "n_envs=2",
        "ppo.n_steps=4",
        "ppo.n_minibatches=1",
        "ppo.n_epochs=1",
        "network.hidden_size=16",
        "tensorboard=false",
        "env.level=flat",
        "env.max_episode_steps=8",
        "eval.episodes=1",
        "eval.interval=0",
        "log_interval=16",
    ):
        args.extend(["--set", override])

    monkeypatch.setitem(main.__globals__, "Trainer", stop_after_update)
    assert main(args) == 130
    summary = json.loads((tmp_path / "benchmark" / "summary.json").read_text())
    assert summary["status"] == "interrupted"
    assert summary["global_step"] == 8
    assert not (tmp_path / "seed_0").exists()

    monkeypatch.setitem(main.__globals__, "Trainer", trainer_type)
    assert main(args) == 0
    suite = json.loads((tmp_path / "summary.json").read_text())
    assert [run["run_name"] for run in suite["runs"]] == ["benchmark", "seed_0", "seed_1"]
    benchmark = suite["runs"][0]
    assert [session["start_step"] for session in benchmark["sessions"]] == [0, 8]
    assert benchmark["global_step"] == 16
    for summary in suite["runs"]:
        assert summary["status"] == "complete"
        assert summary["training_steps_per_second"] > 0
        assert summary["held_out"]["sampled"]["episodes"] == 2
        assert 0 <= summary["held_out"]["greedy"]["flag_rate"] <= 1
        assert summary["evaluation_settings"]["seed"] == 1_000_000
    checkpoint = tmp_path / "seed_0" / "checkpoints" / "latest.pt"
    assert load_checkpoint(checkpoint)["global_step"] == 32
    checkpoint_bytes = checkpoint.read_bytes()

    assert main([*args, "--skip-benchmark"]) == 0
    rerun = json.loads((tmp_path / "summary.json").read_text())
    assert suite["runs"] == rerun["runs"]
    assert checkpoint.read_bytes() == checkpoint_bytes

    with pytest.raises(ValueError, match="training settings differ"):
        main([*args, "--set", "ppo.lr=0.001"])
    assert checkpoint.read_bytes() == checkpoint_bytes


def test_source_metadata_works_without_git_and_detects_modified_upload(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"
    runner = runpy.run_path(str(script))
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("x = 1\n")
    local = runner["_source_metadata"](tmp_path)
    manifest = {"git_commit": "example-commit", "files_sha256": local["files_sha256"]}
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest))

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(runner["subprocess"], "run", missing_git)
    assert runner["_git"]("rev-parse", "HEAD") is None
    uploaded = runner["_source_metadata"](tmp_path)
    assert uploaded["matches_manifest"] is True
    assert uploaded["manifest_git_commit"] == "example-commit"
    assert uploaded["manifest_sha256"]
    assert uploaded["code_sha256"] == local["code_sha256"]

    source.write_text("x = 2\n")
    changed = runner["_source_metadata"](tmp_path)
    assert changed["matches_manifest"] is False
    assert changed["code_sha256"] != uploaded["code_sha256"]
