"""Offline checks for bounded, comparable recordings and failure preservation."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

from mario_play.rl.config import EnvConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_jev_ppo.py"


class FixedController:
    """Stand-in for inference: zero network traffic and a predictable failure."""

    def __init__(self, fail_after: int | None = None):
        self.last_decision = {"choice": "right", "probabilities": {"right": 1.0}}
        self.fail_after = fail_after
        self.calls = 0
        self.closed = False

    def reset(self):
        pass

    def act(self, obs):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("secret-provider-details-must-not-be-logged")
        self.calls += 1
        return 1

    def close(self):
        self.closed = True


def test_same_initial_observation_and_cap_with_no_overwrite(tmp_path):
    runner = runpy.run_path(str(SCRIPT))
    cfg = EnvConfig(level="flat", obs_mode="grid", action_set="simple", frame_skip=4)
    first = runner["record_episode"]("ppo", FixedController(), cfg, tmp_path, seed=77, max_steps=3)
    second = runner["record_episode"]("jev", FixedController(), cfg, tmp_path, seed=77, max_steps=3)
    assert first["initial_observation_sha256"] == second["initial_observation_sha256"]
    for result in (first, second):
        assert result["decisions"] == 3
        assert result["decision_cap_reached"] and result["truncated"]
        assert not result["terminated"]
        assert result["status"] == "complete"
    with pytest.raises(FileExistsError):
        runner["record_episode"]("jev", FixedController(), cfg, tmp_path, seed=77, max_steps=3)
    with Image.open(tmp_path / "jev" / "gameplay.gif") as image:
        assert image.n_frames == 4
    runner["write_report"](
        tmp_path,
        {
            "env": {"level": "flat", "frame_skip": 4},
            "seed": 77,
            "max_steps": 3,
            "checkpoint": "offline-test-only.pt",
            "checkpoint_step": 0,
            "checkpoint_sha256": "offline-test-only",
        },
    )
    with Image.open(tmp_path / "side-by-side.gif") as image:
        assert image.n_frames == 4
        assert image.width == 520
    assert "inference here" in (tmp_path / "report.md").read_text()


def test_error_keeps_partial_gif_and_excludes_exception_details(tmp_path):
    runner = runpy.run_path(str(SCRIPT))
    cfg = EnvConfig(level="flat", obs_mode="grid", action_set="simple", frame_skip=4)
    controller = FixedController(fail_after=1)
    result = runner["record_episode"]("jev", controller, cfg, tmp_path, seed=77, max_steps=3)
    assert result["status"] == "error"
    assert result["error_type"] == "RuntimeError"
    assert result["decisions"] == 1
    assert controller.closed
    assert len((tmp_path / "jev" / "decisions.jsonl").read_text().splitlines()) == 1
    text = (tmp_path / "jev" / "result.json").read_text()
    assert json.loads(text)["status"] == "error"
    assert "secret-provider-details" not in text
    with Image.open(tmp_path / "jev" / "gameplay.gif") as image:
        assert image.n_frames == 2


def test_dotenv_is_not_executed_and_environment_wins(tmp_path, monkeypatch):
    runner = runpy.run_path(str(SCRIPT))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OTHER=ignored\nexport TYPESAFE_API_KEY='literal$(not-a-command)' # comment\n")
    assert runner["load_api_key"](env) == "literal$(not-a-command)"
    monkeypatch.setenv("TYPESAFE_API_KEY", "environment-token")
    assert runner["load_api_key"](env) == "environment-token"
