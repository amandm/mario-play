"""Shared helpers of the agent tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from mario_play.envs.mario_env import MarioEnv


@pytest.fixture
def make_mario() -> Iterator[Callable[..., MarioEnv]]:
    """A factory for grid-observation `MarioEnv`s (cheap: no rendering) that closes them."""
    envs: list[MarioEnv] = []

    def factory(level: str = "flat", **kwargs: object) -> MarioEnv:
        kwargs.setdefault("obs_mode", "grid")
        env = MarioEnv(level=level, **kwargs)
        envs.append(env)
        return env

    yield factory
    for env in envs:
        env.close()
