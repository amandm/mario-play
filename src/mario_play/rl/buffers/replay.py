"""Uniform experience replay for off-policy learners (DQN).

Transitions live in pre-allocated numpy arrays on the CPU, in the observation's
native dtype: uint8 frames stay uint8 (a float copy would be four times larger)
and only the sampled minibatch is moved to the training device. Both `obs` and
`next_obs` are stored. That doubles the footprint compared to the "successor is
the next slot" trick, but it keeps the true successor (`VecStep.final_obs`) of
episode-ending transitions with no special cases around auto-reset or the ring
boundary.

Memory is therefore dominated by `2 * capacity * prod(obs_shape) * itemsize`::

    (4, 84, 84) uint8 pixels,  100k transitions  ->  5.6 GB
    (4, 84, 84) uint8 pixels,   50k transitions  ->  2.8 GB
    (14, 15, 16) float32 grid, 100k transitions  ->  2.7 GB
    (4,) float32 (CartPole),   100k transitions  ->  4.5 MB

so size `dqn.buffer_size` to the machine (`ReplayBuffer.estimate_nbytes` gives the
exact figure). The arrays are zero-initialised, which the OS backs lazily: resident
memory grows as the buffer fills, and a buffer that does not fit in RAM only starts
swapping late in the run. A warning is issued up front when that is foreseeable.

The buffer is not part of a checkpoint (see `mario_play.rl.algos.dqn`).
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

# Warn when the buffer alone would occupy more than this share of physical memory.
_MEMORY_WARNING_FRACTION = 0.5


def _physical_memory_bytes() -> int | None:
    """Total physical RAM, or None where the platform does not tell (e.g. Windows)."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def _storage_dtype(obs_dtype: Any) -> np.dtype:
    """Dtype observations are stored in: the native one, except that float64 becomes float32.

    Networks compute in float32 anyway (MPS has no float64), so doubles would only
    cost memory.
    """
    dtype = np.dtype(obs_dtype)
    return np.dtype(np.float32) if dtype == np.float64 else dtype


class ReplayBuffer:
    """Fixed-capacity ring buffer of `(obs, action, reward, next_obs, terminated)` transitions.

    `add_batch` appends one transition per environment and overwrites the oldest
    data once full; `sample` draws uniformly (with replacement) from what is stored
    and returns tensors on `device`. `terminated` must be the MDP's own termination
    flag only: a truncated episode still bootstraps from `next_obs`.

    The storage arrays `obs`, `next_obs`, `actions`, `rewards` and `terminated` are
    public for inspection; only the first `len(buffer)` slots hold data.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        obs_dtype: Any,
        device: torch.device | str,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"replay capacity must be a positive integer, got {capacity!r}")
        self.capacity = int(capacity)
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.obs_dtype = _storage_dtype(obs_dtype)
        self.device = torch.device(device)

        needed = self.estimate_nbytes(self.capacity, self.obs_shape, self.obs_dtype)
        available = _physical_memory_bytes()
        if available is not None and needed > _MEMORY_WARNING_FRACTION * available:
            warnings.warn(
                f"the replay buffer needs {needed / 1e9:.1f} GB once full ({self.capacity:,} "
                f"transitions of {self.obs_shape} {self.obs_dtype} observations, stored as obs "
                f"and next_obs) but this machine has {available / 1e9:.1f} GB of RAM; "
                "lower dqn.buffer_size to avoid swapping",
                stacklevel=2,
            )

        self.obs = np.zeros((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.bool_)
        self._pos = 0
        self._size = 0

    # ------------------------------------------------------------------ #
    # bookkeeping
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Number of transitions currently stored (at most `capacity`)."""
        return self._size

    @property
    def full(self) -> bool:
        """Whether new transitions overwrite old ones."""
        return self._size == self.capacity

    @property
    def position(self) -> int:
        """Slot the next transition is written to."""
        return self._pos

    @property
    def nbytes(self) -> int:
        """Bytes of storage this buffer occupies once every slot has been written."""
        arrays = (self.obs, self.next_obs, self.actions, self.rewards, self.terminated)
        return sum(array.nbytes for array in arrays)

    @staticmethod
    def estimate_nbytes(capacity: int, obs_shape: Sequence[int], obs_dtype: Any) -> int:
        """Storage needed by a buffer with these settings, without allocating it."""
        obs_bytes = math.prod(int(dim) for dim in obs_shape) * _storage_dtype(obs_dtype).itemsize
        scalar_bytes = (
            np.dtype(np.int64).itemsize
            + np.dtype(np.float32).itemsize
            + np.dtype(np.bool_).itemsize
        )
        return int(capacity) * (2 * obs_bytes + scalar_bytes)

    def clear(self) -> None:
        """Forget every stored transition (the memory stays allocated)."""
        self._pos = 0
        self._size = 0

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #

    def _check_obs(self, name: str, value: Any, n: int | None) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim != len(self.obs_shape) + 1 or array.shape[1:] != self.obs_shape:
            raise ValueError(
                f"{name} must be a batch of shape (n, {', '.join(map(str, self.obs_shape))}), "
                f"got {array.shape}"
            )
        if n is not None and array.shape[0] != n:
            raise ValueError(f"{name} holds {array.shape[0]} transitions but obs holds {n}")
        if not np.can_cast(array.dtype, self.obs_dtype, casting="same_kind"):
            raise TypeError(
                f"{name} has dtype {array.dtype}, which cannot be stored in a {self.obs_dtype} "
                "replay buffer without losing information; build the buffer with the "
                "observation space's dtype"
            )
        return array

    @staticmethod
    def _check_vector(name: str, value: Any, dtype: Any, n: int) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},) to match obs, got {array.shape}")
        return array.astype(dtype, copy=False)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        """Store `n` transitions (one per environment); the oldest are overwritten when full.

        `obs` and `next_obs` are `(n, *obs_shape)`, the rest `(n,)`. Pass
        `VecStep.final_obs` as `next_obs` and `VecStep.terminated` (not `dones`) as
        `terminated`. Inputs are validated before anything is written and copied
        on write, so the caller may reuse its arrays.
        """
        obs = self._check_obs("obs", obs, None)
        n = obs.shape[0]
        if n == 0:
            raise ValueError("add_batch needs at least one transition")
        next_obs = self._check_obs("next_obs", next_obs, n)
        actions = self._check_vector("actions", actions, np.int64, n)
        rewards = self._check_vector("rewards", rewards, np.float32, n)
        terminated = self._check_vector("terminated", terminated, np.bool_, n)

        if n > self.capacity:  # only the newest `capacity` transitions can survive anyway
            keep = slice(n - self.capacity, n)
            obs, next_obs = obs[keep], next_obs[keep]
            actions, rewards, terminated = actions[keep], rewards[keep], terminated[keep]
            n = self.capacity

        slots = (self._pos + np.arange(n)) % self.capacity
        self.obs[slots] = obs
        self.next_obs[slots] = next_obs
        self.actions[slots] = actions
        self.rewards[slots] = rewards
        self.terminated[slots] = terminated
        self._pos = int((self._pos + n) % self.capacity)
        self._size = min(self._size + n, self.capacity)

    # ------------------------------------------------------------------ #
    # reading
    # ------------------------------------------------------------------ #

    def sample(self, batch_size: int, generator: np.random.Generator) -> dict[str, Tensor]:
        """Draw `batch_size` stored transitions uniformly, with replacement, onto `device`.

        Returns `obs` and `next_obs` `(B, *obs_shape)` in the storage dtype (networks
        scale uint8 themselves), `actions` `(B,)` int64, `rewards` `(B,)` float32 and
        `terminated` `(B,)` float32 (1.0 where the MDP ended). The tensors are copies.
        All randomness comes from `generator`, which makes sampling reproducible and
        checkpointable by whoever owns it.
        """
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        slots = generator.integers(0, self._size, size=int(batch_size))
        return {
            "obs": self._to_device(self.obs[slots]),
            "actions": self._to_device(self.actions[slots]),
            "rewards": self._to_device(self.rewards[slots]),
            "next_obs": self._to_device(self.next_obs[slots]),
            "terminated": self._to_device(self.terminated[slots].astype(np.float32)),
        }

    def _to_device(self, array: np.ndarray) -> Tensor:
        # Fancy indexing already produced a fresh array, so from_numpy shares nothing with storage.
        return torch.from_numpy(array).to(self.device)
