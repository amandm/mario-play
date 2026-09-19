"""Shared data types of the RL framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VecStep:
    """Result of one `VecEnv.step` over `n` environments with same-step auto-reset.

    For an env whose episode ended on this step, `obs` already holds the first
    observation of the *next* episode, while `final_obs` holds the true successor
    observation of the transition that just happened and `infos` holds the
    terminal info dict. For all other envs `final_obs[i]` equals `obs[i]`.
    """

    obs: np.ndarray  # (n, *obs_shape)
    rewards: np.ndarray  # (n,) float32
    terminated: np.ndarray  # (n,) bool - the MDP ended (death, flag); never bootstrap
    truncated: np.ndarray  # (n,) bool - cut off (time limit, stall); bootstrap from final_obs
    final_obs: np.ndarray  # (n, *obs_shape)
    infos: list[dict[str, Any]]  # length n

    @property
    def dones(self) -> np.ndarray:
        """`terminated | truncated`, shape (n,) bool."""
        return np.logical_or(self.terminated, self.truncated)
