"""`RandomAgent`: uniformly random actions - the floor every other agent must beat."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class RandomAgent:
    """Picks every action uniformly at random from the env's discrete action space.

    The agent draws from its own generator (never the env's or a global one), so an
    agent built with the same `seed` always produces the same action sequence.
    `reset` does not rewind the generator: successive episodes differ.
    """

    def __init__(self, env: gym.Env, seed: int | None = None) -> None:
        space = env.action_space
        if not isinstance(space, gym.spaces.Discrete):
            raise TypeError(f"RandomAgent needs a Discrete action space, got {space!r}")
        self.n_actions = int(space.n)
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Nothing to forget: the agent has no per-episode state."""

    def act(self, obs: Any = None) -> int:
        """A uniformly random action index; `obs` is ignored."""
        return int(self._rng.integers(self.n_actions))
