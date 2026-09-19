"""Proximal Policy Optimization (clip variant) for discrete actions.

Follows the CleanRL reference / "37 implementation details of PPO": orthogonal
init (in `ActorCritic`), Adam with eps 1e-5, GAE(lambda), per-minibatch advantage
normalisation, clipped surrogate objective, optional clipped value loss, entropy
bonus, global gradient-norm clipping, linear LR annealing and optional early
stopping on the approximate KL.

How it plugs into the trainer loop (`mario_play.rl.algos.base`):

* `select_actions` samples from the current policy and hands the log-probs and
  values to `observe` through `extras`.
* `observe` stores the transition. `done = terminated | truncated` is the flag of
  that transition. Where an episode was *truncated but not terminated* the state
  still had value, so `gamma * V(final_obs)` is folded into the stored reward
  (one batched value forward over just those envs); after that, GAE can treat
  every episode end alike. A terminal state is worth nothing and gets no bonus.
* `update` runs once the rollout holds `n_steps` vector steps. It bootstraps from
  the observation returned by the last `observe` (`step.obs`) - masked where that
  last transition ended its episode, because `step.obs` is then the first
  observation of a new episode.

The minibatch permutation and action sampling use torch's global RNG, so a
trainer that checkpoints the global RNG state resumes bit-for-bit; PPO itself
only needs model, optimizer and its update counter. The learning rate is a pure
function of `progress` and therefore needs no state either.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn

from mario_play.rl.algos.base import Algorithm
from mario_play.rl.buffers.rollout import RolloutBuffer
from mario_play.rl.config import PPOConfig, TrainConfig
from mario_play.rl.networks import ActorCritic
from mario_play.rl.types import VecStep

ADAM_EPS = 1e-5  # PPO convention (torch's default is 1e-8)


class PPOLossTerms(NamedTuple):
    """Scalar tensors for one minibatch; only the two losses carry gradients."""

    policy_loss: Tensor
    value_loss: Tensor
    approx_kl: Tensor
    clip_frac: Tensor


def ppo_loss_terms(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    *,
    clip_coef: float,
    clip_vloss: bool,
    norm_adv: bool,
) -> PPOLossTerms:
    """PPO-clip policy and value losses for one minibatch (all inputs shaped `(B,)`).

    * policy: `mean(max(-A * r, -A * clip(r, 1 - c, 1 + c)))` with `r = pi_new / pi_old`;
      `A` is normalised over the minibatch when `norm_adv` (skipped for `B == 1`,
      whose standard deviation is undefined).
    * value: `0.5 * mean((V - R)^2)`; with `clip_vloss` the element-wise maximum of
      that and the same error for `V_old + clip(V - V_old, -c, c)`.
    * `approx_kl` is the low-variance estimator `mean((r - 1) - log r)` of
      KL(pi_old || pi_new); `clip_frac` is the fraction of samples with `|r - 1| > c`.
    """
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean()

    if norm_adv and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = torch.max(unclipped, clipped).mean()

    value_error = (new_values - returns).pow(2)
    if clip_vloss:
        clipped_values = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
        value_error = torch.max(value_error, (clipped_values - returns).pow(2))
    value_loss = 0.5 * value_error.mean()
    return PPOLossTerms(policy_loss, value_loss, approx_kl, clip_frac)


def explained_variance(values: Tensor, returns: Tensor) -> float:
    """`1 - Var(returns - values) / Var(returns)`: 1 is a perfect critic, <= 0 a useless one.

    Undefined for constant returns; 0.0 is reported then so that logs stay finite.
    """
    var_returns = returns.var(correction=0)
    if var_returns.item() <= 0.0:
        return 0.0
    return float(1.0 - (returns - values).var(correction=0) / var_returns)


def _validate(hp: PPOConfig, n_envs: int) -> None:
    batch_size = hp.n_steps * n_envs
    checks = [
        (n_envs >= 1, f"n_envs must be >= 1, got {n_envs}"),
        (hp.n_steps >= 1, f"ppo.n_steps must be >= 1, got {hp.n_steps}"),
        (hp.n_epochs >= 1, f"ppo.n_epochs must be >= 1, got {hp.n_epochs}"),
        (
            1 <= hp.n_minibatches <= max(batch_size, 1),
            f"ppo.n_minibatches must be between 1 and the batch size n_steps * n_envs = "
            f"{batch_size}, got {hp.n_minibatches}",
        ),
        (hp.lr > 0.0, f"ppo.lr must be > 0, got {hp.lr}"),
        (0.0 <= hp.gamma <= 1.0, f"ppo.gamma must be in [0, 1], got {hp.gamma}"),
        (0.0 <= hp.gae_lambda <= 1.0, f"ppo.gae_lambda must be in [0, 1], got {hp.gae_lambda}"),
        (hp.clip_coef > 0.0, f"ppo.clip_coef must be > 0, got {hp.clip_coef}"),
        (hp.max_grad_norm > 0.0, f"ppo.max_grad_norm must be > 0, got {hp.max_grad_norm}"),
        (
            hp.target_kl is None or hp.target_kl > 0.0,
            f"ppo.target_kl must be > 0 or null, got {hp.target_kl}",
        ),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


class PPO(Algorithm):
    """On-policy actor-critic learner; see the module docstring for the data flow."""

    name = "ppo"

    def __init__(
        self,
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        cfg: TrainConfig,
        device: torch.device,
        n_envs: int,
    ) -> None:
        super().__init__(obs_space, action_space, cfg, device, n_envs)
        self.hp: PPOConfig = cfg.ppo
        _validate(self.hp, n_envs)
        self.model = ActorCritic(obs_space, self.n_actions, cfg.network).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp.lr, eps=ADAM_EPS)
        self.buffer = RolloutBuffer(
            self.hp.n_steps, n_envs, obs_space.shape, obs_space.dtype, device
        )
        self.n_updates = 0
        # Successor of the rollout's final transition, captured when the buffer fills up.
        self._last_obs: Tensor | None = None
        self._last_dones: Tensor | None = None

    # ------------------------------------------------------------------ #
    # acting
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select_actions(
        self, obs: np.ndarray, global_step: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sample actions from the policy.

        `extras` holds `log_prob` and `value` as float32 tensors `(n_envs,)` on the
        training device (kept there to spare a device round trip; `observe` also
        accepts numpy arrays).
        """
        actions, log_probs, _, values = self.model.get_action_and_value(self.obs_to_tensor(obs))
        return actions.cpu().numpy().astype(np.int64, copy=False), {
            "log_prob": log_probs,
            "value": values,
        }

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Policy argmax (or a sample when not `deterministic`) for a batch of observations.

        Never touches the rollout; the greedy path draws no random numbers.
        """
        actions, _, _, _ = self.model.get_action_and_value(
            self.obs_to_tensor(obs), deterministic=deterministic
        )
        return actions.cpu().numpy().astype(np.int64, copy=False)

    # ------------------------------------------------------------------ #
    # collecting
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def observe(
        self, obs: np.ndarray, actions: np.ndarray, extras: dict[str, Any], step: VecStep
    ) -> None:
        """Store the transitions `obs --actions--> step`; `extras` comes from `select_actions`."""
        missing = [key for key in ("log_prob", "value") if key not in extras]
        if missing:
            raise KeyError(
                f"PPO.observe needs the extras returned by select_actions; missing {missing}"
            )
        # torch.tensor copies: the bootstrap below must never leak into the caller's step.rewards.
        rewards = torch.tensor(step.rewards, dtype=torch.float32, device=self.device)
        bootstrap = np.logical_and(step.truncated, np.logical_not(step.terminated))
        if bootstrap.any():
            indices = np.flatnonzero(bootstrap)
            final_values = self.model.get_value(self.obs_to_tensor(step.final_obs[indices]))
            rewards[torch.as_tensor(indices, device=self.device)] += (
                self.hp.gamma * final_values.to(torch.float32)
            )

        dones = step.dones
        self.buffer.add(obs, actions, extras["log_prob"], extras["value"], rewards, dones)
        if self.buffer.full:
            # Cloned because a vec env may reuse its observation array on the next step.
            self._last_obs = self.obs_to_tensor(step.obs).clone()
            self._last_dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

    def ready_to_update(self, global_step: int) -> bool:
        """True once the rollout holds `n_steps` vector steps."""
        return self.buffer.full

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #

    def learning_rate(self, progress: float) -> float:
        """`lr * (1 - progress)` when annealing (progress clamped to [0, 1]), else `lr`."""
        if not self.hp.anneal_lr:
            return float(self.hp.lr)
        return float(self.hp.lr) * (1.0 - min(max(float(progress), 0.0), 1.0))

    def update(self, global_step: int, progress: float) -> dict[str, float]:
        """Run the PPO epochs over the full rollout, then empty it.

        Metrics: `policy_loss`, `value_loss`, `entropy`, `loss`, `clip_frac` and
        `grad_norm` (before clipping) are means over every minibatch of the update;
        `approx_kl` is the mean over the last epoch that ran - the number compared
        with `target_kl`; `epochs` counts the epochs that ran; `explained_variance`
        rates the critic that collected the rollout; `lr` is the rate that was used.
        """
        if not self.buffer.full or self._last_obs is None or self._last_dones is None:
            raise RuntimeError(
                f"PPO.update needs a full rollout ({len(self.buffer)}/{self.hp.n_steps} steps "
                "collected); call it only when ready_to_update() is true"
            )
        hp = self.hp
        lr = self.learning_rate(progress)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        with torch.no_grad():
            last_values = self.model.get_value(self._last_obs)
        self.buffer.compute_returns_and_advantages(
            last_values, self._last_dones, hp.gamma, hp.gae_lambda
        )
        critic_quality = explained_variance(
            self.buffer.values.reshape(-1), self.buffer.returns.reshape(-1)
        )

        # Metric tensors stay on the device until the end: one sync per update, not per minibatch.
        history: dict[str, list[Tensor]] = {
            key: []
            for key in ("policy_loss", "value_loss", "entropy", "loss", "clip_frac", "grad_norm")
        }
        epoch_kl = torch.zeros((), device=self.device)
        epochs_run = 0
        for _ in range(hp.n_epochs):
            kls = []
            for batch in self.buffer.minibatches(hp.n_minibatches):
                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    batch["obs"], batch["actions"]
                )
                terms = ppo_loss_terms(
                    new_log_probs,
                    batch["log_probs"],
                    batch["advantages"],
                    new_values,
                    batch["values"],
                    batch["returns"],
                    clip_coef=hp.clip_coef,
                    clip_vloss=hp.clip_vloss,
                    norm_adv=hp.norm_adv,
                )
                entropy_mean = entropy.mean()
                loss = (
                    terms.policy_loss - hp.ent_coef * entropy_mean + hp.vf_coef * terms.value_loss
                )
                if not torch.isfinite(loss):
                    # Stepping on a NaN would destroy the weights; stop while they are still good.
                    raise FloatingPointError(
                        f"PPO loss became non-finite at update {self.n_updates + 1} "
                        f"(policy {terms.policy_loss.item()}, value {terms.value_loss.item()}, "
                        f"entropy {entropy_mean.item()}); lower ppo.lr or check the rewards"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), hp.max_grad_norm)
                self.optimizer.step()

                kls.append(terms.approx_kl)
                history["policy_loss"].append(terms.policy_loss.detach())
                history["value_loss"].append(terms.value_loss.detach())
                history["entropy"].append(entropy_mean.detach())
                history["loss"].append(loss.detach())
                history["clip_frac"].append(terms.clip_frac)
                history["grad_norm"].append(grad_norm.detach())
            epochs_run += 1
            epoch_kl = torch.stack(kls).mean()
            if hp.target_kl is not None and epoch_kl.item() > hp.target_kl:
                break

        self.buffer.reset()
        self._last_obs = None
        self._last_dones = None
        self.n_updates += 1

        metrics = {key: torch.stack(values).mean().item() for key, values in history.items()}
        metrics["approx_kl"] = epoch_kl.item()
        metrics["explained_variance"] = critic_quality
        metrics["epochs"] = float(epochs_run)
        metrics["lr"] = lr
        return metrics

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        """Model, optimizer and update counter (tensors and plain Python values only).

        As with `nn.Module.state_dict`, tensors are live references: save them right
        away or `copy.deepcopy` the result to keep an in-memory snapshot. The rollout
        is deliberately left out - it is on-policy data, recollected after a resume.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "n_updates": int(self.n_updates),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore `state_dict()` output; any partially collected rollout is discarded."""
        missing = [key for key in ("model", "optimizer", "n_updates") if key not in state]
        if missing:
            raise KeyError(f"PPO state is missing {missing}; got keys {sorted(state)}")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.n_updates = int(state["n_updates"])
        self.buffer.reset()
        self._last_obs = None
        self._last_dones = None
