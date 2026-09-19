"""On-policy rollout storage for PPO with GAE(lambda) advantage estimation.

Conventions (they differ from CleanRL's `dones[t]` = "obs_t starts an episode"):

* Row `t` holds one transition per env: `obs_t`, `action_t`, `log_prob_t`,
  `value_t = V(obs_t)`, `reward_t` and `done_t`, where `done_t` is the flag of
  *that* transition (`terminated | truncated`): the episode ended by taking
  `action_t` in `obs_t`.
* GAE therefore masks with the done flag of the same row::

      delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
      A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

  `V(s_{t+1})` is `values[t + 1]`, or `last_values` for the final row. After a
  done the next row belongs to a new episode, so neither its value nor its
  advantage may leak backwards - which is exactly what the mask does.
* Truncation (time limit, stall) is not the buffer's business: the caller folds
  `gamma * V(final_obs)` into the stored reward, after which a truncated step is
  handled like any other episode end.

Everything is stored on `device`; observations keep their dtype (uint8 frames
stay uint8, float64 is narrowed to float32), all other floats are float32.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

ArrayLike = Any  # numpy array, torch tensor or nested sequence


class RolloutBuffer:
    """Fixed-size storage for `n_steps` transitions of `n_envs` parallel environments."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: Sequence[int],
        obs_dtype: np.dtype | type,
        device: torch.device | str,
    ) -> None:
        if int(n_steps) < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if int(n_envs) < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.device = torch.device(device)

        np_dtype = np.dtype(obs_dtype)
        if np_dtype == np.float64:  # MPS has no float64 and the networks compute in float32
            np_dtype = np.dtype(np.float32)
        self._obs_np_dtype = np_dtype
        obs_torch_dtype = torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype

        rows = (self.n_steps, self.n_envs)
        self.obs = torch.zeros((*rows, *self.obs_shape), dtype=obs_torch_dtype, device=self.device)
        self.actions = torch.zeros(rows, dtype=torch.int64, device=self.device)
        self.log_probs = torch.zeros(rows, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(rows, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(rows, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(rows, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(rows, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(rows, dtype=torch.float32, device=self.device)
        self.pos = 0
        self._advantages_ready = False

    # ------------------------------------------------------------------ #
    # filling
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Number of rows (vector steps) stored so far."""
        return self.pos

    @property
    def full(self) -> bool:
        """True once `n_steps` rows have been added."""
        return self.pos >= self.n_steps

    @property
    def batch_size(self) -> int:
        """Samples in a full rollout: `n_steps * n_envs`."""
        return self.n_steps * self.n_envs

    def reset(self) -> None:
        """Forget the stored rollout (storage is reused, not reallocated)."""
        self.pos = 0
        self._advantages_ready = False

    def add(
        self,
        obs: ArrayLike,
        actions: ArrayLike,
        log_probs: ArrayLike,
        values: ArrayLike,
        rewards: ArrayLike,
        dones: ArrayLike,
    ) -> None:
        """Append one vector step; every argument has a leading `n_envs` axis.

        Accepts numpy arrays or tensors (on any device) and copies them, so callers
        may reuse their arrays. `dones` are the flags of *this* transition.
        """
        if self.full:
            raise RuntimeError(
                f"rollout buffer is full ({self.n_steps} steps): run the update / call reset() "
                "before adding more transitions"
            )
        per_env = (self.n_envs,)
        staged = {
            "obs": (
                self.obs,
                self._as_tensor(obs, self._obs_np_dtype),
                (*per_env, *self.obs_shape),
            ),
            "actions": (self.actions, self._as_tensor(actions), per_env),
            "log_probs": (self.log_probs, self._as_tensor(log_probs), per_env),
            "values": (self.values, self._as_tensor(values), per_env),
            "rewards": (self.rewards, self._as_tensor(rewards), per_env),
            "dones": (self.dones, self._as_tensor(dones), per_env),
        }
        # Validate everything before writing anything: copy_ would broadcast a (1,) silently.
        for name, (_, value, expected) in staged.items():
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        for storage, value, _ in staged.values():
            storage[self.pos].copy_(value)  # converts dtype and device
        self.pos += 1
        self._advantages_ready = False

    @staticmethod
    def _as_tensor(value: ArrayLike, np_dtype: np.dtype | None = None) -> Tensor:
        """Detached tensor view of `value` (no float64: numpy defaults would leak it in)."""
        if isinstance(value, Tensor):
            return value.detach()
        array = np.asarray(value, dtype=np_dtype)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        if not array.flags.writeable:  # torch.from_numpy warns about read-only memory
            array = array.copy()
        return torch.from_numpy(np.ascontiguousarray(array))

    # ------------------------------------------------------------------ #
    # advantages
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_returns_and_advantages(
        self, last_values: ArrayLike, last_dones: ArrayLike, gamma: float, gae_lambda: float
    ) -> None:
        """Fill `advantages` (GAE) and `returns = advantages + values` for a full buffer.

        `last_values` `(n_envs,)` is `V` of the observation that follows the final
        row. `last_dones` `(n_envs,)` are the done flags of that final transition;
        where set, `last_values` belongs to a fresh episode and is masked out. With
        this buffer's convention they must equal the flags stored with the final
        row - they are taken explicitly (mirroring the `(obs, done)` pair a training
        loop carries) and checked, so a caller with a different convention fails
        loudly instead of bootstrapping across an episode boundary.
        """
        if not self.full:
            raise RuntimeError(
                f"rollout buffer is not full ({self.pos}/{self.n_steps} steps): "
                "advantages need a complete rollout"
            )
        per_env = (self.n_envs,)
        last_values_t = self._as_tensor(last_values).to(self.device, torch.float32)
        last_dones_t = self._as_tensor(last_dones).to(self.device, torch.float32)
        if tuple(last_values_t.shape) != per_env:
            raise ValueError(
                f"last_values must have shape {per_env}, got {tuple(last_values_t.shape)}"
            )
        if tuple(last_dones_t.shape) != per_env:
            raise ValueError(
                f"last_dones must have shape {per_env}, got {tuple(last_dones_t.shape)}"
            )
        if not torch.equal(last_dones_t != 0, self.dones[-1] != 0):
            raise ValueError(
                "last_dones must be the done flags of the final stored transition "
                "(dones[t] belongs to the transition stored at t, not to the next observation)"
            )

        gae = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
        next_values = last_values_t
        for t in reversed(range(self.n_steps)):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            self.advantages[t] = gae
            next_values = self.values[t]
        self.returns.copy_(self.advantages + self.values)
        self._advantages_ready = True

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #

    def minibatches(
        self, n_minibatches: int, generator: torch.Generator | None = None
    ) -> Iterator[dict[str, Tensor]]:
        """Yield one epoch: a fresh random partition of all samples into `n_minibatches` parts.

        Every one of the `n_steps * n_envs` samples appears in exactly one minibatch;
        sizes differ by at most one when the batch does not divide evenly. Each
        minibatch is a dict with `obs` `(B, *obs_shape)` and `actions`, `log_probs`,
        `values`, `advantages`, `returns` of shape `(B,)`. The permutation is drawn
        from `generator` (a CPU generator) or, by default, from torch's global RNG,
        which is what makes training reproducible from a checkpointed RNG state.
        """
        if not self.full or not self._advantages_ready:
            raise RuntimeError(
                "call compute_returns_and_advantages() on a full buffer before sampling minibatches"
            )
        if not 1 <= int(n_minibatches) <= self.batch_size:
            raise ValueError(
                f"n_minibatches must be between 1 and the batch size {self.batch_size} "
                f"(n_steps * n_envs), got {n_minibatches}"
            )
        flat = {
            "obs": self.obs.reshape(self.batch_size, *self.obs_shape),
            "actions": self.actions.reshape(-1),
            "log_probs": self.log_probs.reshape(-1),
            "values": self.values.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
        }
        permutation = torch.randperm(self.batch_size, generator=generator).to(self.device)
        for indices in torch.tensor_split(permutation, int(n_minibatches)):
            yield {key: value[indices] for key, value in flat.items()}
