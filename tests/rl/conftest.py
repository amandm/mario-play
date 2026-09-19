"""Shared fixtures for the trainer / evaluation tests.

Pytest runs with `--import-mode=importlib`, so test modules cannot import each
other; everything reusable is handed out through fixtures instead.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import register, registry

from mario_play.rl.config import (
    DQNConfig,
    EnvConfig,
    EvalConfig,
    NetworkConfig,
    PPOConfig,
    TrainConfig,
)

FLAG_RUN_ENV_ID = "FlagRun-v0"
FLAG_RUN_LENGTH = 4


class FlagRunEnv(gym.Env):
    """Four-step stand-in for the Mario env: its terminal info carries `flag_get` and `progress`.

    Every step with action 1 advances one quarter of the way; the episode always
    lasts `FLAG_RUN_LENGTH` steps and the flag is reached iff every action was 1.
    The reward is the distance covered by the step.
    """

    metadata = {"render_modes": []}

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._t = 0
        self._advanced = 0

    def _obs(self) -> np.ndarray:
        return np.array(
            [self._t / FLAG_RUN_LENGTH, self._advanced / FLAG_RUN_LENGTH], dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._t = 0
        self._advanced = 0
        return self._obs(), {"progress": 0.0, "flag_get": False}

    def step(self, action: int):
        self._t += 1
        self._advanced += int(action == 1)
        terminated = self._t >= FLAG_RUN_LENGTH
        info = {
            "progress": self._advanced / FLAG_RUN_LENGTH,
            "flag_get": terminated and self._advanced == FLAG_RUN_LENGTH,
        }
        return self._obs(), float(action == 1), terminated, False, info


if FLAG_RUN_ENV_ID not in registry:
    register(id=FLAG_RUN_ENV_ID, entry_point=FlagRunEnv)


class ConstantPolicy:
    """The slice of the `Algorithm` interface that `evaluate` needs: always the same action."""

    def __init__(self, action: int) -> None:
        self.action = action
        self.batch_shapes: list[tuple[int, ...]] = []
        self.deterministic_flags: list[bool] = []

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        self.batch_shapes.append(tuple(np.shape(obs)))
        self.deterministic_flags.append(deterministic)
        return np.full(len(obs), self.action, dtype=np.int64)


@pytest.fixture
def flag_run_env_id() -> str:
    """Id of a registered env whose infos look like the Mario env's (`flag_get`, `progress`)."""
    return FLAG_RUN_ENV_ID


@pytest.fixture
def constant_policy() -> type[ConstantPolicy]:
    return ConstantPolicy


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> Callable[..., TrainConfig]:
    """Factory for small CPU training configs that write below `tmp_path`.

    Keyword arguments replace top-level `TrainConfig` fields; `ppo`, `dqn`, `eval`
    and `env` may be given as dicts of field overrides on top of the tiny defaults.
    """

    def build(algo: str = "ppo", **overrides: Any) -> TrainConfig:
        ppo = {"n_steps": 16, "n_epochs": 2, "n_minibatches": 2, "lr": 1e-3}
        dqn = {
            "buffer_size": 2_000,
            "learning_starts": 200,
            "batch_size": 16,
            "train_freq": 2,
            "target_update_interval": 100,
            "eps_decay_steps": 500,
            "lr": 1e-3,
        }
        evaluation = {"interval": 400, "episodes": 2, "seed": 123}
        env = {"id": "CartPole-v1"}
        ppo.update(overrides.pop("ppo", {}))
        dqn.update(overrides.pop("dqn", {}))
        evaluation.update(overrides.pop("eval", {}))
        env.update(overrides.pop("env", {}))
        fields: dict[str, Any] = {
            "algo": algo,
            "total_timesteps": 1_000,
            "n_envs": 2,
            "seed": 1,
            "device": "cpu",
            "run_dir": str(tmp_path / "runs"),
            "run_name": "run",
            "log_interval": 200,
            "checkpoint_interval": 400,
            "tensorboard": False,
            "env": EnvConfig(**env),
            "network": NetworkConfig(hidden_size=16, mlp_hidden=[16, 16]),
            "ppo": PPOConfig(**ppo),
            "dqn": DQNConfig(**dqn),
            "eval": EvalConfig(**evaluation),
        }
        fields.update(overrides)
        return TrainConfig(**fields)

    return build
