"""The `Algorithm` interface every learner implements.

One trainer loop drives on-policy and off-policy algorithms alike::

    obs = venv.reset(seed)
    while global_step < total:
        actions, extras = algo.select_actions(obs, global_step)
        step = venv.step(actions)
        algo.observe(obs, actions, extras, step)
        global_step += n_envs
        if algo.ready_to_update(global_step):
            metrics = algo.update(global_step, progress=global_step / total)
        obs = step.obs

PPO buffers transitions in `observe` and is ready once its rollout is full; DQN
pushes to replay in `observe` and is ready every `train_freq` vector steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from mario_play.rl.config import TrainConfig
from mario_play.rl.types import VecStep


class Algorithm(ABC):
    """Base class for learners over a `Box` observation and `Discrete` action space."""

    name: str = "base"

    def __init__(
        self,
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        cfg: TrainConfig,
        device: torch.device,
        n_envs: int,
    ) -> None:
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"observation space must be Box, got {type(obs_space).__name__}")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(f"action space must be Discrete, got {type(action_space).__name__}")
        self.obs_space = obs_space
        self.action_space = action_space
        self.n_actions = int(action_space.n)
        self.cfg = cfg
        self.device = device
        self.n_envs = n_envs

    def obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Move a batch of observations to the device, keeping the space's dtype.

        Networks do their own scaling (uint8 images are divided by 255 inside the encoder).
        """
        return torch.as_tensor(np.asarray(obs, dtype=self.obs_space.dtype), device=self.device)

    @abstractmethod
    def select_actions(
        self, obs: np.ndarray, global_step: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Training-time (exploratory) actions for a batch `(n_envs, *obs_shape)`.

        Returns `(actions, extras)`: `actions` is an int64 array of shape `(n_envs,)`;
        `extras` carries whatever `observe` needs later (e.g. log-probs and values).
        """

    @abstractmethod
    def observe(
        self, obs: np.ndarray, actions: np.ndarray, extras: dict[str, Any], step: VecStep
    ) -> None:
        """Record the transitions `obs --actions--> step` produced by `select_actions`."""

    @abstractmethod
    def ready_to_update(self, global_step: int) -> bool:
        """Whether `update` should run now."""

    @abstractmethod
    def update(self, global_step: int, progress: float) -> dict[str, float]:
        """Run one learning update and return scalar metrics.

        `progress` in [0, 1] is the fraction of training completed (for schedules).
        """

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Evaluation-time actions for a batch `(n, *obs_shape)` -> int64 `(n,)`. Side-effect free."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Everything needed to resume: models, optimizers, counters.

        Only tensors and plain Python containers/primitives, so checkpoints load
        with `torch.load(weights_only=True)`.
        """

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from `state_dict()` output."""
