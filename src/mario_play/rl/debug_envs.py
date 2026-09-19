"""Tiny deterministic environments for testing the RL framework.

They live in the package (not in the test tree) because `SubprocVecEnv` workers are
started with the "spawn" method and must be able to import whatever they run.
This module depends on numpy and gymnasium only - never on torch.

Both envs are also registered with Gymnasium, so they can be built from an
`EnvConfig` through `make_env`; the `module:id` form makes `gym.make` import this
module first, which also works inside a freshly spawned worker::

    EnvConfig(id="mario_play.rl.debug_envs:Counting-v0", kwargs={"terminate_at": 5})
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry

COUNTING_ENV_ID = "Counting-v0"
SEEDED_NOISE_ENV_ID = "SeededNoise-v0"


class CountingEnv(gym.Env):
    """Observation is the step counter of the current episode; every step pays reward 1.

    `obs = [t]` (float32, shape `(1,)`) where `t` is 0 after `reset` and grows by one
    per `step`. The episode is `terminated` once `t >= terminate_at` and `truncated`
    once `t >= truncate_at` (either may be None = never). With `fail_at` set, the
    step that would reach `t == fail_at` raises `RuntimeError` instead, which is how
    the tests provoke a crash inside a vec-env worker.

    `info` reports `t` and the `action` just taken; the reset info reports `t = 0`
    and echoes the `seed` argument. Lifetime counters (`episode_index`,
    `total_steps`) are attributes rather than info entries, because infos must be a
    function of seed and actions alone to satisfy Gymnasium's env checker.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        terminate_at: int | None = 5,
        truncate_at: int | None = None,
        fail_at: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        for name, value in (
            ("terminate_at", terminate_at),
            ("truncate_at", truncate_at),
            ("fail_at", fail_at),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None, got {value}")
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.fail_at = fail_at
        self.render_mode = render_mode  # accepted because `make_env` always passes it
        high = float(np.finfo(np.float32).max)
        self.observation_space = gym.spaces.Box(0.0, high, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.t = 0
        self.total_steps = 0
        self.episode_index = -1

    def _obs(self) -> np.ndarray:
        return np.array([self.t], dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode at `t = 0`."""
        super().reset(seed=seed)
        self.t = 0
        self.episode_index += 1
        return self._obs(), {"t": 0, "seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the counter by one."""
        if self.fail_at is not None and self.t + 1 >= self.fail_at:
            raise RuntimeError(f"CountingEnv failed on purpose at t={self.t + 1}")
        self.t += 1
        self.total_steps += 1
        terminated = self.terminate_at is not None and self.t >= self.terminate_at
        truncated = self.truncate_at is not None and self.t >= self.truncate_at
        return self._obs(), 1.0, terminated, truncated, {"t": self.t, "action": int(action)}


class SeededNoiseEnv(gym.Env):
    """Every observation is fresh noise from `self.np_random`; used to test seeding.

    `obs` is `(obs_size,)` float32 uniform in [0, 1); the reward is `obs[0]` of the
    new observation plus the action; the episode terminates after `episode_length`
    steps. Two instances produce identical trajectories iff they were seeded alike,
    and a `reset()` without a seed continues the stream instead of restarting it.
    A large `obs_size` makes replies that do not fit into a pipe buffer.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, episode_length: int = 4, obs_size: int = 3, render_mode: str | None = None
    ) -> None:
        if episode_length < 1 or obs_size < 1:
            raise ValueError(
                f"episode_length and obs_size must be >= 1, got {episode_length} and {obs_size}"
            )
        self.episode_length = episode_length
        self.obs_size = obs_size
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.t = 0

    def _obs(self) -> np.ndarray:
        return self.np_random.random(self.obs_size, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode; reseeds only when `seed` is given."""
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Draw the next noise observation."""
        self.t += 1
        obs = self._obs()
        reward = float(obs[0]) + float(action)
        return obs, reward, self.t >= self.episode_length, False, {"t": self.t}


if COUNTING_ENV_ID not in registry:
    register(id=COUNTING_ENV_ID, entry_point="mario_play.rl.debug_envs:CountingEnv")
if SEEDED_NOISE_ENV_ID not in registry:
    register(id=SEEDED_NOISE_ENV_ID, entry_point="mario_play.rl.debug_envs:SeededNoiseEnv")
