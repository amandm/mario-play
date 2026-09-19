"""Shared helpers of the env tests.

Pytest runs with `--import-mode=importlib`, so test modules cannot import each
other: tiny hand-written levels and stub envs are handed out as fixtures instead.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

# The player starts on column 2; the flag is 8 tiles away: holding right wins in seconds.
SHORT_LEVEL = """\
; time=100
..........F.........
..........F.........
..S.......F.........
####################
####################
"""

# A 4-wide pit right in front of the start: holding right ends in the pit.
PIT_LEVEL = """\
; time=100
....................F...
....................F...
..S.................F...
####....################
####....################
"""

# A walker comes for the player: standing still ends with death by enemy.
ENEMY_LEVEL = """\
; time=100
....................F...
....................F...
..S.....g...........F...
########################
########################
"""

# Two time units = 48 frames: standing still ends with a timeout.
TIMEOUT_LEVEL = """\
; time=2
..........F.........
..........F.........
..S.......F.........
####################
####################
"""

# Feet-level coins on the way to the flag.
COIN_LEVEL = """\
; time=100
............F.......
............F.......
..S..ooo....F.......
####################
####################
"""

LEVEL_TEXTS = {
    "short": SHORT_LEVEL,
    "pit": PIT_LEVEL,
    "enemy": ENEMY_LEVEL,
    "timeout": TIMEOUT_LEVEL,
    "coins": COIN_LEVEL,
}


@pytest.fixture
def level_path(tmp_path: Path) -> Callable[[str], str]:
    """`level_path("pit")` -> path of a level file; the level's name is the given key."""

    def write(key: str) -> str:
        path = tmp_path / f"{key}.txt"
        if not path.exists():
            path.write_text(LEVEL_TEXTS[key], encoding="utf-8")
        return str(path)

    return write


class CounterEnv(gym.Env):
    """Stub env whose observation is filled with the number of steps since `reset`.

    `shape`/`dtype` choose the observation; `colors` (a list of RGB triples) instead
    makes frame `t` a flat `(H, W, 3)` image of colour `colors[t]`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        shape: tuple[int, ...] = (6, 8, 3),
        dtype: Any = np.uint8,
        low: float = 0,
        high: float = 255,
        colors: list[tuple[int, int, int]] | None = None,
    ) -> None:
        self.observation_space = gym.spaces.Box(low, high, shape=shape, dtype=dtype)
        self.action_space = gym.spaces.Discrete(2)
        self.colors = colors
        self.t = 0

    def _obs(self) -> np.ndarray:
        space = self.observation_space
        if self.colors is not None:
            obs = np.empty(space.shape, dtype=space.dtype)
            obs[:] = self.colors[self.t % len(self.colors)]
            return obs
        return np.full(space.shape, self.t, dtype=space.dtype)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action: int):
        self.t += 1
        return self._obs(), 1.0, False, False, {}


@pytest.fixture
def counter_env() -> Callable[..., CounterEnv]:
    """Factory of `CounterEnv` stubs (keyword arguments are forwarded)."""
    return CounterEnv
