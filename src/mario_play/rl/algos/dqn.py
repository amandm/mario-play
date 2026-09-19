"""Double DQN (van Hasselt et al., 2016) with uniform replay, behind the `Algorithm` interface.

How it plugs into the trainer loop (`select_actions -> step -> observe -> update`):

* `select_actions` is epsilon-greedy, decided independently per environment, with
  epsilon falling linearly from `eps_start` to `eps_end` over the first
  `eps_decay_steps` env transitions (`global_step`).
* `observe` pushes one transition per environment into replay. The successor is
  `VecStep.final_obs` (the true next observation, not the auto-reset one) and the
  done flag is `VecStep.terminated` only, so a *truncated* episode (time limit,
  stall) still bootstraps from its last observation.
* `ready_to_update` is true every `train_freq` vector steps once `learning_starts`
  env transitions have been collected; each `update` runs `gradient_steps`
  gradient steps of Huber loss on `r + gamma * (1 - terminated) * Q_target(s', a*)`
  with `a* = argmax_a Q_online(s', a)` (or the target network's own max when
  `double_q` is off), Adam and gradient-norm clipping.
* Target network: with `tau == 1.0` a hard copy at the first update at or after
  every multiple of `target_update_interval` env transitions (no drift when the
  interval is not a multiple of `n_envs * train_freq`); with `tau < 1.0` Polyak
  averaging `target <- tau * online + (1 - tau) * target` after every gradient
  step, and `target_update_interval` is ignored.

Units: `learning_starts`, `target_update_interval` and `eps_decay_steps` count env
transitions summed over all envs (`global_step`); `train_freq` counts vector steps.

Memory: replay keeps observations on the CPU in their native dtype, as `obs` and
`next_obs`. For `(4, 84, 84)` uint8 pixels that is 56 kB per transition - 5.6 GB at
the default `dqn.buffer_size` of 100k - so pixel configs should set
`dqn.buffer_size` to what the machine can hold (see `mario_play.rl.buffers.replay`).

Checkpoints hold both networks, the optimizer, the counters and the exploration
RNG, but *not* the replay contents (gigabytes of frames). After a resume the buffer
refills: updates pause until it again holds as many transitions as a fresh run has
when it starts learning (`min(learning_starts, buffer_size)`), rather than fitting
the restored network to the first few, highly correlated, new samples.
"""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mario_play.rl.algos.base import Algorithm
from mario_play.rl.buffers.replay import ReplayBuffer
from mario_play.rl.config import DQNConfig, TrainConfig
from mario_play.rl.networks import QNetwork
from mario_play.rl.types import VecStep
from mario_play.rl.utils import linear_schedule

_BIT_GENERATOR = "PCG64"  # what np.random.default_rng uses
_STATE_KEYS = (
    "q_net",
    "target_net",
    "optimizer",
    "vector_steps",
    "n_updates",
    "last_target_sync",
    "rng",
)


def _validate(hp: DQNConfig) -> None:
    """Reject hyperparameters that would crash later or silently never train."""

    def require(ok: bool, field: str, rule: str) -> None:
        if not ok:
            raise ValueError(f"dqn.{field} {rule}, got {getattr(hp, field)!r}")

    require(hp.lr > 0, "lr", "must be positive")
    require(hp.buffer_size >= 1, "buffer_size", "must be at least 1")
    require(hp.batch_size >= 1, "batch_size", "must be at least 1")
    if hp.buffer_size < hp.batch_size:
        raise ValueError(
            f"dqn.buffer_size ({hp.buffer_size}) must be at least dqn.batch_size "
            f"({hp.batch_size}), otherwise replay never holds a full batch"
        )
    require(hp.learning_starts >= 0, "learning_starts", "must not be negative")
    require(hp.train_freq >= 1, "train_freq", "must be at least 1")
    require(hp.gradient_steps >= 1, "gradient_steps", "must be at least 1")
    require(hp.target_update_interval >= 1, "target_update_interval", "must be at least 1")
    require(0.0 < hp.tau <= 1.0, "tau", "must be in (0, 1]")
    require(0.0 <= hp.gamma <= 1.0, "gamma", "must be in [0, 1]")
    require(0.0 <= hp.eps_start <= 1.0, "eps_start", "must be in [0, 1]")
    require(0.0 <= hp.eps_end <= 1.0, "eps_end", "must be in [0, 1]")


def _rng_to_plain(rng: np.random.Generator) -> dict[str, Any]:
    """The generator's state as str/int only, so it survives `torch.load(weights_only=True)`."""
    state = rng.bit_generator.state
    if state["bit_generator"] != _BIT_GENERATOR:  # pragma: no cover - numpy's default
        raise RuntimeError(f"unexpected numpy bit generator {state['bit_generator']!r}")
    return {
        "bit_generator": _BIT_GENERATOR,
        "state": int(state["state"]["state"]),  # 128-bit Python ints
        "inc": int(state["state"]["inc"]),
        "has_uint32": int(state["has_uint32"]),
        "uinteger": int(state["uinteger"]),
    }


def _check_plain_rng(plain: dict[str, Any]) -> None:
    if plain.get("bit_generator") != _BIT_GENERATOR:
        raise ValueError(
            f"cannot restore the exploration RNG from bit generator "
            f"{plain.get('bit_generator')!r}; expected {_BIT_GENERATOR}"
        )


def _restore_rng(rng: np.random.Generator, plain: dict[str, Any]) -> None:
    _check_plain_rng(plain)
    rng.bit_generator.state = {
        "bit_generator": _BIT_GENERATOR,
        "state": {"state": int(plain["state"]), "inc": int(plain["inc"])},
        "has_uint32": int(plain["has_uint32"]),
        "uinteger": int(plain["uinteger"]),
    }


class DQN(Algorithm):
    """Double DQN learner; see the module docstring for the exact semantics.

    Public attributes: `q_net` (online network), `target_net`, `optimizer`,
    `buffer` (`ReplayBuffer`), `rng` (numpy `Generator` seeded from `cfg.seed`; it
    drives both exploration and replay sampling and is checkpointed) and
    `n_updates` (gradient steps taken so far).
    """

    name = "dqn"

    def __init__(
        self,
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        cfg: TrainConfig,
        device: torch.device,
        n_envs: int,
    ) -> None:
        super().__init__(obs_space, action_space, cfg, device, n_envs)
        self.hp: DQNConfig = cfg.dqn
        _validate(self.hp)

        self.q_net = QNetwork(obs_space, self.n_actions, cfg.network, self.hp.dueling).to(device)
        # A copy instead of a second construction: identical weights from the start, and no
        # second (slow, RNG-consuming) orthogonal initialisation.
        self.target_net = copy.deepcopy(self.q_net).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.hp.lr)

        self.buffer = ReplayBuffer(self.hp.buffer_size, obs_space.shape, obs_space.dtype, device)
        # A fresh run starts learning with this many transitions in replay; a resumed run
        # (whose buffer starts empty) waits for the same amount. See `ready_to_update`.
        self._min_replay_size = max(
            self.hp.batch_size, min(self.hp.learning_starts, self.hp.buffer_size)
        )

        seed = int(cfg.seed) % 2**32  # numpy rejects negative seeds
        self.rng = np.random.default_rng(seed)
        # Stochastic `predict` draws from its own stream so that evaluating in the middle of
        # training never perturbs exploration or replay sampling.
        self._predict_rng = np.random.default_rng([seed, 1])

        self.n_updates = 0
        self._vector_steps = 0
        self._last_target_sync = 0  # global_step of the latest hard sync

    # ------------------------------------------------------------------ #
    # acting
    # ------------------------------------------------------------------ #

    def epsilon(self, global_step: int) -> float:
        """Exploration rate after `global_step` env transitions (linear decay, then constant)."""
        hp = self.hp
        return linear_schedule(hp.eps_start, hp.eps_end, hp.eps_decay_steps, global_step)

    def _greedy_actions(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            q_values = self.q_net(self.obs_to_tensor(obs))
        return q_values.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)

    def _epsilon_greedy(
        self, obs: np.ndarray, epsilon: float, rng: np.random.Generator
    ) -> np.ndarray:
        n = len(obs)
        # Both draws always happen, so the RNG stream does not depend on the outcomes.
        explore = rng.random(n) < epsilon
        random_actions = rng.integers(0, self.n_actions, size=n, dtype=np.int64)
        if explore.all():  # typical early in training: skip the forward pass
            return random_actions
        return np.where(explore, random_actions, self._greedy_actions(obs))

    def select_actions(
        self, obs: np.ndarray, global_step: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Epsilon-greedy actions `(n_envs,)` int64; every env explores independently."""
        return self._epsilon_greedy(obs, self.epsilon(global_step), self.rng), {}

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Greedy actions, or epsilon-greedy with the final rate `eps_end` when not deterministic.

        Never touches the training RNG, the networks or replay.
        """
        if deterministic:
            return self._greedy_actions(obs)
        return self._epsilon_greedy(obs, self.hp.eps_end, self._predict_rng)

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #

    def observe(
        self, obs: np.ndarray, actions: np.ndarray, extras: dict[str, Any], step: VecStep
    ) -> None:
        """Store `(obs, action, reward, final_obs, terminated)` for every environment."""
        self.buffer.add_batch(obs, actions, step.rewards, step.final_obs, step.terminated)
        self._vector_steps += 1

    def ready_to_update(self, global_step: int) -> bool:
        """True every `train_freq` vector steps once learning has started and replay is warm.

        In a fresh run replay holds `min(global_step, buffer_size)` transitions, so the
        replay condition is implied by `global_step >= learning_starts` plus a full
        batch; it only bites after a resume, while the empty buffer refills.
        """
        return (
            global_step >= self.hp.learning_starts
            and len(self.buffer) >= self._min_replay_size
            and self._vector_steps % self.hp.train_freq == 0
        )

    @torch.no_grad()
    def compute_targets(self, batch: dict[str, Tensor]) -> Tensor:
        """TD targets `(B,)` for a replay batch: `r + gamma * (1 - terminated) * Q_target(s', a*)`.

        Double DQN picks `a*` with the online network and evaluates it with the target
        network; with `double_q` off both come from the target network.
        """
        next_q_target = self.target_net(batch["next_obs"])
        if self.hp.double_q:
            next_actions = self.q_net(batch["next_obs"]).argmax(dim=1, keepdim=True)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
        else:
            next_q = next_q_target.max(dim=1).values
        return batch["rewards"] + self.hp.gamma * (1.0 - batch["terminated"]) * next_q

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Huber loss between `Q_online(s, a)` and the TD targets; also returns `Q_online(s, a)`."""
        targets = self.compute_targets(batch)
        q_taken = self.q_net(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        return F.smooth_l1_loss(q_taken, targets), q_taken

    def update(self, global_step: int, progress: float) -> dict[str, float]:
        """Run `gradient_steps` gradient steps and maintain the target network.

        Metrics (means over the gradient steps): `loss`, `q_mean` (Q of the taken
        actions), `grad_norm` (before clipping), plus `epsilon`, `buffer_size` and
        `n_updates`. `progress` is unused: the learning rate is constant.
        """
        hp = self.hp
        max_norm = hp.max_grad_norm if hp.max_grad_norm > 0 else float("inf")  # <= 0: no clipping
        stats = []
        for _ in range(hp.gradient_steps):
            batch = self.buffer.sample(hp.batch_size, self.rng)
            loss, q_taken = self.compute_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm)
            self.optimizer.step()
            self.n_updates += 1
            if hp.tau < 1.0:
                self._polyak_update(hp.tau)
            stats.append(torch.stack([loss.detach(), q_taken.detach().mean(), grad_norm]))

        interval = hp.target_update_interval
        if hp.tau >= 1.0 and global_step // interval > self._last_target_sync // interval:
            self.sync_target()
            self._last_target_sync = int(global_step)

        # One device-to-host transfer for all metrics.
        loss_mean, q_mean, grad_norm_mean = torch.stack(stats).mean(dim=0).tolist()
        return {
            "loss": loss_mean,
            "q_mean": q_mean,
            "grad_norm": grad_norm_mean,
            "epsilon": self.epsilon(global_step),
            "buffer_size": float(len(self.buffer)),
            "n_updates": float(self.n_updates),
        }

    def sync_target(self) -> None:
        """Hard update: copy the online network into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    @torch.no_grad()
    def _polyak_update(self, tau: float) -> None:
        online, target = self.q_net, self.target_net
        for target_param, param in zip(target.parameters(), online.parameters(), strict=True):
            target_param.lerp_(param, tau)
        for target_buffer, buffer in zip(target.buffers(), online.buffers(), strict=True):
            target_buffer.copy_(buffer)

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        """Snapshot of networks, optimizer, counters and the exploration RNG (no replay data).

        The result shares no memory with the live learner, so it stays valid while
        training continues, and holds only tensors and plain Python values.
        """
        return {
            "q_net": copy.deepcopy(self.q_net.state_dict()),
            "target_net": copy.deepcopy(self.target_net.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "vector_steps": self._vector_steps,
            "n_updates": self.n_updates,
            "last_target_sync": self._last_target_sync,
            "rng": _rng_to_plain(self.rng),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a `state_dict()` snapshot; replay starts empty and refills.

        The learning rate is taken from the current config rather than from the
        checkpoint, so `dqn.lr` can be overridden when resuming.
        """
        missing = [key for key in _STATE_KEYS if key not in state]
        if missing:
            raise ValueError(f"not a DQN state dict: missing keys {missing}")
        _check_plain_rng(state["rng"])  # fail before anything has been modified

        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.hp.lr
        self._vector_steps = int(state["vector_steps"])
        self.n_updates = int(state["n_updates"])
        self._last_target_sync = int(state["last_target_sync"])
        _restore_rng(self.rng, state["rng"])
