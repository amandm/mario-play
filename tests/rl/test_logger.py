"""Logger: CSV that stays valid as metric keys appear, resume/append, TensorBoard, console."""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mario_play.rl.logger import Logger


def read_csv(run_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def assert_rectangular(run_dir: Path) -> None:
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as fh:
        widths = {len(row) for row in csv.reader(fh)}
    assert len(widths) == 1, f"ragged csv: row widths {widths}"


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def test_creates_run_dir_and_writes_rows(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "exp1"
    logger = Logger(run_dir, use_tensorboard=False)
    logger.log({"rollout/return": 1.5, "rollout/length": 20}, step=100)
    logger.log({"rollout/return": 2.5, "rollout/length": 30}, step=200)
    logger.close()

    header, rows = read_csv(run_dir)
    assert header == ["step", "rollout/return", "rollout/length"]
    assert [row["step"] for row in rows] == ["100", "200"]
    assert [float(row["rollout/return"]) for row in rows] == [1.5, 2.5]
    assert [row["rollout/length"] for row in rows] == ["20", "30"]
    assert not (run_dir / "tb").exists()


def test_new_keys_rewrite_the_header_and_pad_old_rows(tmp_path) -> None:
    logger = Logger(tmp_path, use_tensorboard=False)
    logger.log({"rollout/return": 1.0}, step=10)
    logger.log({"rollout/return": 2.0, "loss/policy": 0.25}, step=20)
    logger.log({"eval/mean_return": 9.0}, step=30)  # a row may also omit older keys
    logger.log({"loss/policy": 0.125, "rollout/return": 3.0}, step=40)
    logger.close()

    assert_rectangular(tmp_path)
    header, rows = read_csv(tmp_path)
    assert header == ["step", "rollout/return", "loss/policy", "eval/mean_return"]
    assert rows == [
        {"step": "10", "rollout/return": "1", "loss/policy": "", "eval/mean_return": ""},
        {"step": "20", "rollout/return": "2", "loss/policy": "0.25", "eval/mean_return": ""},
        {"step": "30", "rollout/return": "", "loss/policy": "", "eval/mean_return": "9"},
        {"step": "40", "rollout/return": "3", "loss/policy": "0.125", "eval/mean_return": ""},
    ]
    # no leftovers of the header rewrites
    assert sorted(p.name for p in tmp_path.iterdir()) == ["log.txt", "metrics.csv"]


def test_rows_are_on_disk_before_close(tmp_path) -> None:
    logger = Logger(tmp_path, use_tensorboard=False)
    logger.log({"a": 1.0}, step=1)
    logger.log({"a": 2.0, "b": 3.0}, step=2)
    _, rows = read_csv(tmp_path)  # a crash right now must not lose anything
    assert [row["a"] for row in rows] == ["1", "2"]
    logger.close()


def test_resume_appends_and_keeps_growing_the_header(tmp_path) -> None:
    first = Logger(tmp_path, use_tensorboard=False)
    first.log({"a": 1.0}, step=10)
    first.log({"a": 2.0, "b": 5.0}, step=20)
    first.close()

    resumed = Logger(tmp_path, use_tensorboard=False, resume=True)
    resumed.log({"b": 6.0, "a": 3.0}, step=30)  # key order of the call does not matter
    resumed.log({"c": 7.0}, step=40)
    resumed.close()

    assert_rectangular(tmp_path)
    header, rows = read_csv(tmp_path)
    assert header == ["step", "a", "b", "c"]
    assert [row["step"] for row in rows] == ["10", "20", "30", "40"]
    assert rows[2] == {"step": "30", "a": "3", "b": "6", "c": ""}
    assert rows[3] == {"step": "40", "a": "", "b": "", "c": "7"}


def test_resume_step_drops_rows_logged_after_the_checkpoint(tmp_path) -> None:
    """A run that crashed at step 40 and resumes from the step-20 checkpoint re-logs 30 and 40."""
    first = Logger(tmp_path, use_tensorboard=False)
    for step in (10, 20, 30, 40):
        first.log({"a": float(step)}, step=step)
    first.close()

    resumed = Logger(tmp_path, use_tensorboard=False, resume=True, resume_step=20)
    resumed.log({"a": 31.0}, step=30)
    resumed.close()

    _, rows = read_csv(tmp_path)
    assert [(row["step"], row["a"]) for row in rows] == [("10", "10"), ("20", "20"), ("30", "31")]


def test_resume_without_an_existing_file_starts_fresh(tmp_path) -> None:
    logger = Logger(tmp_path / "new", use_tensorboard=False, resume=True)
    logger.log({"a": 1.0}, step=1)
    logger.close()
    header, rows = read_csv(tmp_path / "new")
    assert header == ["step", "a"] and len(rows) == 1


def test_resume_onto_an_empty_file_starts_fresh(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("", encoding="utf-8")
    logger = Logger(tmp_path, use_tensorboard=False, resume=True)
    logger.log({"a": 1.0}, step=1)
    logger.close()
    header, rows = read_csv(tmp_path)
    assert header == ["step", "a"] and len(rows) == 1


def test_without_resume_an_old_csv_is_replaced(tmp_path) -> None:
    old = Logger(tmp_path, use_tensorboard=False)
    old.log({"stale": 1.0}, step=1)
    old.close()
    fresh = Logger(tmp_path, use_tensorboard=False)
    fresh.log({"a": 2.0}, step=5)
    fresh.close()
    header, rows = read_csv(tmp_path)
    assert header == ["step", "a"]
    assert rows == [{"step": "5", "a": "2"}]


def test_value_formatting(tmp_path) -> None:
    logger = Logger(tmp_path, use_tensorboard=False)
    logger.log(
        {
            "np_float": np.float32(0.5),
            "np_int": np.int64(7),
            "third": 1 / 3,
            "nan": float("nan"),
            "missing": None,
            "flag": True,
        },
        step=np.int64(3),
    )
    logger.close()
    _, rows = read_csv(tmp_path)
    row = rows[0]
    assert row["step"] == "3"
    assert row["np_float"] == "0.5" and row["np_int"] == "7" and row["flag"] == "1"
    assert float(row["third"]) == pytest.approx(1 / 3, rel=1e-7)
    assert math.isnan(float(row["nan"]))
    assert row["missing"] == ""


def test_commas_and_quotes_in_keys_are_escaped(tmp_path) -> None:
    logger = Logger(tmp_path, use_tensorboard=False)
    logger.log({'weird "key", really': 1.0}, step=1)
    logger.log({"b": 2.0}, step=2)
    logger.close()
    assert_rectangular(tmp_path)
    header, _ = read_csv(tmp_path)
    assert header == ["step", 'weird "key", really', "b"]


def test_step_is_a_reserved_metric_name(tmp_path) -> None:
    logger = Logger(tmp_path, use_tensorboard=False)
    with pytest.raises(ValueError, match="step"):
        logger.log({"step": 1.0}, step=1)
    logger.close()


def test_close_is_idempotent_and_logging_afterwards_fails(tmp_path) -> None:
    logger = Logger(tmp_path, use_tensorboard=False)
    logger.log({"a": 1.0}, step=1)
    logger.close()
    logger.close()
    with pytest.raises(RuntimeError, match="closed"):
        logger.log({"a": 2.0}, step=2)


def test_context_manager_closes(tmp_path) -> None:
    with Logger(tmp_path, use_tensorboard=False) as logger:
        logger.log({"a": 1.0}, step=1)
    with pytest.raises(RuntimeError, match="closed"):
        logger.log({"a": 2.0}, step=2)


# --------------------------------------------------------------------------- #
# console
# --------------------------------------------------------------------------- #


def test_print_goes_to_stdout_and_the_run_log(tmp_path, capsys) -> None:
    logger = Logger(tmp_path, use_tensorboard=False)
    logger.print("step 100 | return 1.5")
    logger.close()
    assert capsys.readouterr().out == "step 100 | return 1.5\n"
    assert (tmp_path / "log.txt").read_text(encoding="utf-8") == "step 100 | return 1.5\n"

    resumed = Logger(tmp_path, use_tensorboard=False, resume=True)
    resumed.print("resumed")
    resumed.close()
    assert (tmp_path / "log.txt").read_text(encoding="utf-8").splitlines() == [
        "step 100 | return 1.5",
        "resumed",
    ]


# --------------------------------------------------------------------------- #
# TensorBoard
# --------------------------------------------------------------------------- #


def test_tensorboard_is_not_imported_when_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)  # any import now fails
    logger = Logger(tmp_path, use_tensorboard=False)
    logger.log({"a": 1.0}, step=1)
    logger.close()
    with pytest.raises(ImportError):
        Logger(tmp_path / "with_tb", use_tensorboard=True)


def test_tensorboard_scalars_are_written(tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logger = Logger(tmp_path, use_tensorboard=True)
    logger.log({"rollout/return": 1.5, "skipped": None}, step=100)
    logger.log({"rollout/return": 2.5, "loss/value": np.float32(0.75)}, step=200)
    logger.close()

    assert any(p.name.startswith("events.out.tfevents") for p in (tmp_path / "tb").iterdir())
    events = EventAccumulator(str(tmp_path / "tb"))
    events.Reload()
    assert set(events.Tags()["scalars"]) == {"rollout/return", "loss/value"}
    returns = events.Scalars("rollout/return")
    assert [(e.step, e.value) for e in returns] == [(100, 1.5), (200, 2.5)]
    _, rows = read_csv(tmp_path)
    assert len(rows) == 2  # the CSV is written alongside


def test_tensorboard_resume_purges_points_logged_after_the_checkpoint(tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    first = Logger(tmp_path, use_tensorboard=True)
    for step in (100, 200, 300):
        first.log({"rollout/return": float(step)}, step=step)
    first.close()

    # Event files are ordered by their name, which starts with the creation time in whole
    # seconds; a real resume happens later than the crash, so make that true here as well.
    created = int(time.time())
    while int(time.time()) == created:
        time.sleep(0.02)

    resumed = Logger(tmp_path, use_tensorboard=True, resume=True, resume_step=200)
    resumed.log({"rollout/return": -1.0}, step=300)
    resumed.close()

    events = EventAccumulator(str(tmp_path / "tb"))
    events.Reload()
    points = [(e.step, e.value) for e in events.Scalars("rollout/return")]
    assert points == [(100, 100.0), (200, 200.0), (300, -1.0)]
    _, rows = read_csv(tmp_path)
    assert [(r["step"], r["rollout/return"]) for r in rows] == [
        ("100", "100"),
        ("200", "200"),
        ("300", "-1"),
    ]
