"""The `mario-play` command line: parsing, user errors and every subcommand that needs no training.

Commands run in-process through `main(argv)`; training through the CLI is covered
by `tests/test_integration.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mario_play.cli import build_parser, main

REPO = Path(__file__).resolve().parents[1]
COMMANDS = ("play", "train", "eval", "watch", "record", "bench", "levels")


def assert_one_line_error(captured, *fragments: str) -> None:
    """A user error is a single `error: ...` line on stderr, never a traceback."""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1, captured.err
    assert lines[0].startswith("error: ")
    assert "Traceback" not in captured.err
    for fragment in fragments:
        assert fragment in lines[0]


# --- parser ---------------------------------------------------------------------------------------


def test_top_level_help_lists_every_command(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for command in COMMANDS:
        assert command in out


@pytest.mark.parametrize("command", COMMANDS)
def test_every_subcommand_has_help(command, capsys):
    assert main([command, "--help"]) == 0
    assert f"mario-play {command}" in capsys.readouterr().out


def test_no_command_is_a_usage_error(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_unknown_command_is_a_usage_error(capsys):
    assert main(["dance"]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_train_collects_trailing_overrides():
    args = build_parser().parse_args(
        ["train", "--config", "c.yaml", "--resume", "r", "ppo.lr=1e-4", "env.level=flat"]
    )
    assert args.config == "c.yaml"
    assert args.resume == "r"
    assert args.overrides == ["ppo.lr=1e-4", "env.level=flat"]


def test_eval_needs_exactly_one_policy_source(capsys):
    assert main(["eval"]) == 2
    assert main(["eval", "--agent", "random", "--checkpoint", "x.pt"]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_cli_and_levels_do_not_import_the_heavy_stack():
    code = (
        "import sys\n"
        "from mario_play.cli import main\n"
        "rc = main(['levels'])\n"
        "heavy = [m for m in ('torch', 'pygame', 'gymnasium') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "sys.exit(rc)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "1-1" in done.stdout


def test_python_dash_m_is_an_alias():
    done = subprocess.run(
        [sys.executable, "-m", "mario_play", "levels"], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert "flat" in done.stdout


# --- levels ---------------------------------------------------------------------------------------


def test_levels_lists_the_four_bundled_levels(capsys):
    assert main(["levels"]) == 0
    out = capsys.readouterr().out
    names = [line.split()[0] for line in out.strip().splitlines()[1:]]
    assert names == ["1-1", "1-2", "1-3", "flat"]


# --- eval -----------------------------------------------------------------------------------------


def test_eval_heuristic_reaches_the_flag_on_flat(capsys):
    assert main(["eval", "--agent", "heuristic", "--level", "flat", "--episodes", "1"]) == 0
    out = capsys.readouterr().out
    assert "heuristic" in out
    assert "flag_rate" in out
    assert "1.00" in out


def test_eval_json_is_machine_readable(capsys):
    rc = main(["eval", "--agent", "heuristic", "--level", "flat", "--episodes", "2", "--json"])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["policy"] == "heuristic"
    assert result["level"] == "flat"
    assert result["episodes"] == 2
    assert result["flag_rate"] == 1.0
    assert result["mean_progress"] == 1.0
    assert result["mean_return"] > 50
    assert result["std_return"] == pytest.approx(0.0)
    assert len(result["episode_results"]) == 2
    assert result["episode_results"][0]["flag_get"] is True


def test_eval_random_agent_is_seeded(capsys):
    argv = ["eval", "--agent", "random", "--level", "flat", "--max-steps", "40", "--json"]
    assert main([*argv, "--seed", "3"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main([*argv, "--seed", "3"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert first["mean_length"] == 40
    assert first["flag_rate"] == 0.0


def test_eval_search_agent_finishes_flat(capsys):
    assert main(["eval", "--agent", "search", "--level", "flat", "--episodes", "1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["flag_rate"] == 1.0


def test_unknown_level_is_a_one_line_error(capsys):
    assert main(["eval", "--agent", "random", "--level", "9-9"]) == 2
    assert_one_line_error(capsys.readouterr(), "9-9")


def test_missing_checkpoint_is_a_one_line_error(capsys, tmp_path):
    missing = tmp_path / "nope.pt"
    for command in (["eval"], ["watch"], ["record", "--out", str(tmp_path / "x.gif")]):
        assert main([*command, "--checkpoint", str(missing)]) == 2
        assert_one_line_error(capsys.readouterr(), "nope.pt")


def test_a_file_that_is_no_checkpoint_is_a_one_line_error(capsys, tmp_path):
    bogus = tmp_path / "bogus.pt"
    bogus.write_bytes(b"this is not a checkpoint")
    assert main(["eval", "--checkpoint", str(bogus)]) == 2
    assert_one_line_error(capsys.readouterr(), "bogus.pt")


def test_stochastic_needs_a_trained_policy(capsys):
    assert main(["eval", "--agent", "random", "--stochastic"]) == 2
    assert_one_line_error(capsys.readouterr(), "--stochastic", "--checkpoint")


def test_bad_episode_count_is_a_one_line_error(capsys):
    assert main(["eval", "--agent", "random", "--episodes", "0"]) == 2
    assert_one_line_error(capsys.readouterr(), "episodes")


# --- record ---------------------------------------------------------------------------------------


def gif_frames(path: Path) -> tuple[int, tuple[int, int], int]:
    """(number of frames, (width, height), total duration in ms) of a GIF."""
    with Image.open(path) as image:
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(int(image.info["duration"]))
        return image.n_frames, image.size, sum(durations)


def test_record_writes_a_full_resolution_gif(capsys, tmp_path):
    out = tmp_path / "clips" / "run.gif"
    assert main(["record", "--agent", "heuristic", "--level", "flat", "--out", str(out)]) == 0
    assert out.stat().st_size > 0
    n_frames, size, total_ms = gif_frames(out)
    assert size == (256, 240)  # the game frame, although the agent's env observes the grid
    assert n_frames > 60  # the run takes 79 env steps
    # 80 frames at the env's 15 fps: GIF delays are multiples of 10 ms, the total stays right
    assert total_ms == pytest.approx(80 * 1000 / 15, abs=10)
    assert str(out) in capsys.readouterr().out


def test_record_honours_scale_and_max_steps(tmp_path):
    out = tmp_path / "short.gif"
    argv = ["record", "--agent", "heuristic", "--level", "flat", "--out", str(out)]
    assert main([*argv, "--scale", "2", "--max-steps", "10", "--fps", "10"]) == 0
    n_frames, size, total_ms = gif_frames(out)
    assert size == (512, 480)
    assert n_frames == 11  # the reset frame + one per step
    assert total_ms == 1100


def test_record_rejects_unknown_formats_before_playing(capsys, tmp_path):
    out = tmp_path / "run.avi"
    assert main(["record", "--agent", "search", "--level", "1-3", "--out", str(out)]) == 2
    assert_one_line_error(capsys.readouterr(), ".avi", ".gif")
    assert not out.exists()


def test_record_mp4_without_imageio_is_a_one_line_error(capsys, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio", None)  # makes `import imageio` fail
    out = tmp_path / "run.mp4"
    assert main(["record", "--agent", "heuristic", "--level", "flat", "--out", str(out)]) == 2
    assert_one_line_error(capsys.readouterr(), "imageio")
    assert not out.exists()


def test_record_mp4_goes_through_imageio_when_installed(tmp_path, monkeypatch):
    calls: dict = {"frames": []}

    class FakeWriter:
        def append_data(self, frame: np.ndarray) -> None:
            calls["frames"].append(frame)

        def close(self) -> None:
            calls["closed"] = True

    def get_writer(path, **kwargs):
        calls["path"], calls["kwargs"] = path, kwargs
        return FakeWriter()

    fake = types.ModuleType("imageio")
    fake.get_writer = get_writer
    monkeypatch.setitem(sys.modules, "imageio", fake)

    out = tmp_path / "run.mp4"
    argv = ["record", "--agent", "heuristic", "--level", "flat", "--out", str(out)]
    assert main([*argv, "--max-steps", "5"]) == 0
    assert calls["path"] == str(out)
    assert calls["kwargs"]["fps"] == 15
    assert calls["closed"] is True
    assert len(calls["frames"]) == 6
    assert all(f.shape == (240, 256, 3) and f.dtype == np.uint8 for f in calls["frames"])


# --- watch / play ---------------------------------------------------------------------------------


class FakeWindow:
    """Records what the env's human mode does with its window; never waits."""

    instances: list[FakeWindow] = []
    quit_after: int | None = None

    def __init__(self, scale: int = 3, title: str = "") -> None:
        self.scale, self.title = scale, title
        self.frames = 0
        self.closed = False
        type(self).instances.append(self)

    def show(self, frame: np.ndarray) -> None:
        assert frame.shape == (240, 256, 3)
        self.frames += 1

    def poll(self) -> dict:
        limit = type(self).quit_after
        return {
            "quit": limit is not None and self.frames >= limit,
            "buttons": None,
            "restart": False,
            "pause": False,
        }

    def tick(self, fps: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_window(monkeypatch):
    FakeWindow.instances = []
    FakeWindow.quit_after = None
    module = types.ModuleType("mario_play.game.human")
    module.Window = FakeWindow
    monkeypatch.setitem(sys.modules, "mario_play.game.human", module)
    monkeypatch.setattr("mario_play.cli._WATCH_PAUSE_S", 0.0)
    return FakeWindow


def test_watch_shows_whole_episodes_in_the_human_window(fake_window, capsys):
    argv = ["watch", "--agent", "heuristic", "--level", "flat", "--episodes", "2", "--scale", "2"]
    assert main(argv) == 0
    (window,) = fake_window.instances
    assert window.scale == 2
    assert window.frames == 2 * 80  # per episode: the reset frame + 79 steps
    assert window.closed
    assert capsys.readouterr().out.count("flag") >= 2


def test_closing_the_window_ends_watch_cleanly(fake_window, capsys):
    fake_window.quit_after = 12
    assert main(["watch", "--agent", "random", "--level", "flat"]) == 0  # no episode limit
    assert fake_window.instances[0].frames == 12
    assert "Traceback" not in capsys.readouterr().err


def test_watch_with_the_real_window():
    # SDL's dummy video driver (tests/conftest.py): a real pygame window, paced in real time.
    argv = ["watch", "--agent", "search", "--level", "flat", "--episodes", "1"]
    assert main([*argv, "--max-steps", "4"]) == 0


def test_play_runs_the_human_game_loop():
    assert main(["play", "--level", "flat", "--scale", "1", "--max-frames", "5"]) == 0


def test_play_with_an_unknown_level_is_a_one_line_error(capsys):
    assert main(["play", "--level", "nowhere"]) == 2
    assert_one_line_error(capsys.readouterr(), "nowhere")


# --- bench ----------------------------------------------------------------------------------------


def bench_rows(out: str) -> dict[str, list[str]]:
    rows = [line.split("|") for line in out.splitlines() if "|" in line]
    return {cells[0].strip(): [c.strip() for c in cells[1:]] for cells in rows}


def test_bench_reports_single_and_vector_throughput(capsys):
    assert main(["bench", "--steps", "500", "--n-envs", "2", "--level", "flat"]) == 0
    out = capsys.readouterr().out
    assert "grid" in out
    rows = bench_rows(out)
    assert set(rows) >= {"setup", "single env", "2 sync envs"}
    for name in ("single env", "2 sync envs"):
        steps, _seconds, rate = rows[name][:3]
        assert int(steps.replace(",", "")) == 500
        assert float(rate.replace(",", "")) > 0


def test_bench_pixels(capsys):
    assert main(["bench", "--obs-mode", "pixels", "--steps", "100", "--n-envs", "2"]) == 0
    out = capsys.readouterr().out
    assert "(4, 84, 84)" in out


@pytest.mark.timeout(120)
def test_bench_subproc(capsys):
    assert main(["bench", "--steps", "200", "--n-envs", "2", "--vec", "subproc"]) == 0
    assert "2 subproc envs" in bench_rows(capsys.readouterr().out)


def test_bench_rejects_bad_numbers(capsys):
    assert main(["bench", "--steps", "0"]) == 2
    assert_one_line_error(capsys.readouterr(), "steps")


# --- train: user errors (real training lives in test_integration.py) ------------------------------


def test_train_needs_a_config_or_a_checkpoint(capsys):
    assert main(["train"]) == 2
    assert_one_line_error(capsys.readouterr(), "--config", "--resume")


def test_train_missing_config_file_is_a_one_line_error(capsys, tmp_path):
    assert main(["train", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert_one_line_error(capsys.readouterr(), "missing.yaml")


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ("ppo.no_such_key=1", "no_such_key"),
        ("n_envs=many", "n_envs"),
        ("just-a-word", "just-a-word"),
        ("env.level=9-9", "9-9"),
        ("algo=sarsa", "sarsa"),
        ("device=tpu", "tpu"),
    ],
)
def test_train_bad_override_is_a_one_line_error(override, fragment, capsys, tmp_path):
    config = REPO / "configs" / "ppo_grid.yaml"
    rc = main(["train", "--config", str(config), f"run_dir={tmp_path}", override])
    assert rc == 2
    assert_one_line_error(capsys.readouterr(), fragment)
    assert not any(tmp_path.iterdir())  # nothing was started


def test_train_resume_from_a_missing_checkpoint_is_a_one_line_error(capsys, tmp_path):
    assert main(["train", "--resume", str(tmp_path / "gone" / "latest.pt")]) == 2
    assert_one_line_error(capsys.readouterr(), "latest.pt")
