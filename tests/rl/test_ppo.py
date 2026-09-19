"""PPO: interface contract, loss terms, done/truncation semantics, schedules, resume, learning."""

from __future__ import annotations

import copy
import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from mario_play.envs.factory import make_env
from mario_play.rl.algos import get_algorithm
from mario_play.rl.algos.base import Algorithm
from mario_play.rl.algos.ppo import PPO, ppo_loss_terms
from mario_play.rl.config import EnvConfig, NetworkConfig, PPOConfig, TrainConfig
from mario_play.rl.types import VecStep

CPU = torch.device("cpu")
VECTOR_SPACE = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
SCALAR_SPACE = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
TWO_ACTIONS = gym.spaces.Discrete(2)
METRIC_KEYS = {
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "explained_variance",
    "lr",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def make_algo(
    n_envs: int = 2,
    obs_space: gym.spaces.Box = VECTOR_SPACE,
    action_space: gym.spaces.Discrete = TWO_ACTIONS,
    seed: int = 0,
    **ppo_overrides,
) -> PPO:
    """A tiny CPU PPO; `ppo_overrides` replace fields of a small default PPOConfig."""
    ppo_kwargs = {"n_steps": 8, "n_epochs": 2, "n_minibatches": 2, **ppo_overrides}
    cfg = TrainConfig(
        algo="ppo",
        n_envs=n_envs,
        network=NetworkConfig(hidden_size=32, mlp_hidden=[32, 32]),
        ppo=PPOConfig(**ppo_kwargs),
    )
    torch.manual_seed(seed)
    return PPO(obs_space, action_space, cfg, CPU, n_envs)


def make_step(
    obs,
    rewards,
    terminated=None,
    truncated=None,
    final_obs=None,
) -> VecStep:
    obs = np.asarray(obs, dtype=np.float32)
    n = len(obs)
    return VecStep(
        obs=obs,
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.zeros(n, bool) if terminated is None else np.asarray(terminated, bool),
        truncated=np.zeros(n, bool) if truncated is None else np.asarray(truncated, bool),
        final_obs=obs.copy() if final_obs is None else np.asarray(final_obs, dtype=np.float32),
        infos=[{} for _ in range(n)],
    )


def random_obs(rng: np.random.Generator, n_envs: int, space: gym.spaces.Box) -> np.ndarray:
    if space.dtype == np.uint8:
        return rng.integers(0, 256, size=(n_envs, *space.shape), dtype=np.uint8)
    return rng.normal(size=(n_envs, *space.shape)).astype(space.dtype)


def random_vec_step(rng: np.random.Generator, n_envs: int, space: gym.spaces.Box) -> VecStep:
    obs = random_obs(rng, n_envs, space)
    terminated = rng.random(n_envs) < 0.15
    truncated = rng.random(n_envs) < 0.1
    done = terminated | truncated
    final_obs = obs.copy()
    final_obs[done] = random_obs(rng, int(done.sum()), space)
    return VecStep(
        obs=obs,
        rewards=rng.normal(size=n_envs).astype(np.float32),
        terminated=terminated,
        truncated=truncated,
        final_obs=final_obs,
        infos=[{} for _ in range(n_envs)],
    )


def collect_random_rollout(algo: PPO, rng: np.random.Generator) -> None:
    """Drive the trainer loop against a synthetic random MDP until the rollout is full."""
    obs = random_obs(rng, algo.n_envs, algo.obs_space)
    global_step = 0
    while not algo.ready_to_update(global_step):
        actions, extras = algo.select_actions(obs, global_step)
        step = random_vec_step(rng, algo.n_envs, algo.obs_space)
        algo.observe(obs, actions, extras, step)
        global_step += algo.n_envs
        obs = step.obs


def rollout_and_update(algo: PPO, seed: int, progress: float = 0.0) -> dict[str, float]:
    collect_random_rollout(algo, np.random.default_rng(seed))
    return algo.update(global_step=algo.n_envs * algo.hp.n_steps, progress=progress)


def flat_params(algo: PPO) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).clone() for p in algo.model.parameters()])


def adam_steps(algo: PPO) -> int:
    """Number of optimizer steps taken so far, read from Adam's own state."""
    steps = {int(state["step"]) for state in algo.optimizer.state.values()}
    assert len(steps) <= 1
    return steps.pop() if steps else 0


class MiniVecEnv:
    """Minimal same-step auto-reset vector env (stand-in for the real `SyncVecEnv`)."""

    def __init__(self, envs: list[gym.Env]) -> None:
        self.envs = envs
        self.n_envs = len(envs)

    def reset(self, seed: int) -> np.ndarray:
        return np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(self.envs)])

    def step(self, actions: np.ndarray) -> VecStep:
        obs, final_obs, rewards, terminated, truncated, infos = [], [], [], [], [], []
        for env, action in zip(self.envs, actions, strict=True):
            next_obs, reward, term, trunc, info = env.step(int(action))
            final_obs.append(next_obs)
            if term or trunc:
                next_obs, _ = env.reset()
            obs.append(next_obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
        return VecStep(
            obs=np.stack(obs),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
            final_obs=np.stack(final_obs),
            infos=infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


# --------------------------------------------------------------------------- #
# construction and acting
# --------------------------------------------------------------------------- #


def test_registry_resolves_ppo():
    assert get_algorithm("ppo") is PPO
    assert issubclass(PPO, Algorithm)
    assert PPO.name == "ppo"


def test_adam_uses_the_ppo_epsilon_and_configured_lr():
    algo = make_algo(lr=3e-4)
    assert isinstance(algo.optimizer, torch.optim.Adam)
    (group,) = algo.optimizer.param_groups
    assert group["eps"] == pytest.approx(1e-5)
    assert group["lr"] == pytest.approx(3e-4)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"n_steps": 0}, "n_steps"),
        ({"n_epochs": 0}, "n_epochs"),
        ({"n_minibatches": 0}, "n_minibatches"),
        ({"n_steps": 4, "n_minibatches": 9}, "n_minibatches"),  # batch is 4 * 2 = 8
        ({"gamma": 1.5}, "gamma"),
        ({"gae_lambda": -0.1}, "gae_lambda"),
        ({"clip_coef": 0.0}, "clip_coef"),
        ({"lr": -1.0}, "lr"),
    ],
)
def test_invalid_hyperparameters_are_rejected_at_construction(overrides: dict, match: str):
    with pytest.raises(ValueError, match=match):
        make_algo(**overrides)


def test_select_actions_contract():
    algo = make_algo(n_envs=5)
    obs = random_obs(np.random.default_rng(0), 5, VECTOR_SPACE)
    actions, extras = algo.select_actions(obs, global_step=0)

    assert isinstance(actions, np.ndarray)
    assert actions.shape == (5,) and actions.dtype == np.int64
    assert set(extras) >= {"log_prob", "value"}
    log_prob, value = torch.as_tensor(extras["log_prob"]), torch.as_tensor(extras["value"])
    assert log_prob.shape == (5,) and value.shape == (5,)
    assert log_prob.dtype == torch.float32 and value.dtype == torch.float32
    assert not log_prob.requires_grad and not value.requires_grad

    with torch.no_grad():
        logits, expected_value = algo.model(torch.as_tensor(obs))
        expected_log_prob = Categorical(logits=logits).log_prob(torch.as_tensor(actions))
    torch.testing.assert_close(log_prob, expected_log_prob)
    torch.testing.assert_close(value, expected_value)


def test_select_actions_samples_instead_of_taking_the_argmax():
    algo = make_algo(n_envs=512)
    obs = random_obs(np.random.default_rng(0), 512, VECTOR_SPACE)
    actions, _ = algo.select_actions(obs, global_step=0)
    # The 0.01-gain policy head makes the initial policy near-uniform.
    assert 0.35 < actions.mean() < 0.65
    assert not np.array_equal(actions, algo.predict(obs, deterministic=True))


def test_select_actions_does_not_store_anything():
    algo = make_algo()
    algo.select_actions(random_obs(np.random.default_rng(0), 2, VECTOR_SPACE), 0)
    assert len(algo.buffer) == 0


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #


def test_predict_deterministic_is_the_argmax_and_has_no_side_effects():
    algo = make_algo(n_envs=2)
    collect = np.random.default_rng(0)
    obs = random_obs(collect, 2, VECTOR_SPACE)
    actions, extras = algo.select_actions(obs, 0)
    algo.observe(obs, actions, extras, random_vec_step(collect, 2, VECTOR_SPACE))
    stored = len(algo.buffer)

    batch = random_obs(np.random.default_rng(1), 64, VECTOR_SPACE)  # batch size != n_envs is fine
    rng_before = torch.get_rng_state()
    predicted = algo.predict(batch, deterministic=True)

    assert torch.equal(rng_before, torch.get_rng_state())  # greedy prediction draws no randomness
    assert len(algo.buffer) == stored
    assert predicted.shape == (64,) and predicted.dtype == np.int64
    with torch.no_grad():
        logits, _ = algo.model(torch.as_tensor(batch))
    np.testing.assert_array_equal(predicted, logits.argmax(dim=-1).numpy())
    np.testing.assert_array_equal(predicted, algo.predict(batch))  # deterministic by default


def test_predict_stochastic_samples_from_the_policy():
    algo = make_algo()
    with torch.no_grad():  # make the policy clearly non-uniform: P(action 1) = sigmoid(2)
        algo.model.policy_head.weight.zero_()
        algo.model.policy_head.bias.copy_(torch.tensor([0.0, 2.0]))
    batch = np.zeros((4000, 4), np.float32)
    torch.manual_seed(0)
    sampled = algo.predict(batch, deterministic=False)
    assert sampled.dtype == np.int64
    assert abs(sampled.mean() - 1 / (1 + math.exp(-2.0))) < 0.03
    assert np.all(algo.predict(batch, deterministic=True) == 1)


# --------------------------------------------------------------------------- #
# loss terms (pure function, hand-computed numbers)
# --------------------------------------------------------------------------- #


def loss_terms(ratios, advantages, **kwargs):
    ratios = torch.tensor(ratios)
    n = len(ratios)
    defaults = {
        "new_values": torch.zeros(n),
        "old_values": torch.zeros(n),
        "returns": torch.zeros(n),
        "clip_coef": 0.2,
        "clip_vloss": False,
        "norm_adv": False,
    }
    defaults.update(kwargs)
    return ppo_loss_terms(
        new_log_probs=ratios.log(),
        old_log_probs=torch.zeros(n),
        advantages=torch.tensor(advantages),
        **defaults,
    )


def test_clipped_surrogate_matches_hand_computed_cases():
    """Per sample, loss = max(-A * r, -A * clip(r, 0.8, 1.2)):

    r=1.5, A=+1 -> max(-1.5, -1.2) = -1.2   (gain capped)
    r=1.5, A=-1 -> max(+1.5, +1.2) = +1.5   (penalty not capped)
    r=0.5, A=+1 -> max(-0.5, -0.8) = -0.5
    r=0.5, A=-1 -> max(+0.5, +0.8) = +0.8
    r=1.0, A=+2 -> -2
    mean = (-1.2 + 1.5 - 0.5 + 0.8 - 2) / 5 = -0.28; 4 of 5 ratios are outside [0.8, 1.2].
    """
    terms = loss_terms([1.5, 1.5, 0.5, 0.5, 1.0], [1.0, -1.0, 1.0, -1.0, 2.0])
    assert terms.policy_loss.item() == pytest.approx(-0.28, abs=1e-6)
    assert terms.clip_frac.item() == pytest.approx(0.8)


def test_approx_kl_is_the_k3_estimator():
    """mean((r - 1) - log r); the ratios do not average to 1, so it differs from mean(-log r)."""
    ratios = [1.5, 0.8, 1.0]
    expected = ((0.5 - math.log(1.5)) + (-0.2 - math.log(0.8)) + 0.0) / 3
    naive = -(math.log(1.5) + math.log(0.8)) / 3
    assert abs(expected - naive) > 0.05
    terms = loss_terms(ratios, [0.0, 0.0, 0.0])
    assert terms.approx_kl.item() == pytest.approx(expected, abs=1e-6)
    # Every term (r - 1) - log r is >= 0, so the estimate can never go negative.
    assert loss_terms([0.5, 0.6], [0.0, 0.0]).approx_kl.item() > 0
    assert not terms.approx_kl.requires_grad and not terms.clip_frac.requires_grad


def test_value_loss_with_and_without_clipping():
    """old V = 1, new V = 2, return = 3, clip 0.2.

    unclipped: 0.5 * (2 - 3)^2 = 0.5
    clipped:   V_clip = 1 + clip(2 - 1, +-0.2) = 1.2 -> 0.5 * max(1, (1.2 - 3)^2 = 3.24) = 1.62
    Second sample: new V = 1.1 stays inside the clip range -> 0.5 * (1.1 - 3)^2 = 1.805 either way.
    """
    kwargs = {
        "new_values": torch.tensor([2.0, 1.1]),
        "old_values": torch.tensor([1.0, 1.0]),
        "returns": torch.tensor([3.0, 3.0]),
    }
    plain = loss_terms([1.0, 1.0], [0.0, 0.0], clip_vloss=False, **kwargs)
    clipped = loss_terms([1.0, 1.0], [0.0, 0.0], clip_vloss=True, **kwargs)
    assert plain.value_loss.item() == pytest.approx((0.5 + 1.805) / 2, abs=1e-6)
    assert clipped.value_loss.item() == pytest.approx((1.62 + 1.805) / 2, abs=1e-6)


def test_advantage_normalisation_is_per_minibatch():
    # A = [1, 2, 3]: mean 2, unbiased std 1 -> normalised [-1, 0, 1]; with r = 1 the loss is 0
    normalised = loss_terms([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], norm_adv=True)
    raw = loss_terms([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], norm_adv=False)
    assert normalised.policy_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert raw.policy_loss.item() == pytest.approx(-2.0, abs=1e-6)
    # r = [1, 1, 1.1] (inside the clip range): -(−1*1 + 0*1 + 1*1.1)/3 = -0.1/3
    tilted = loss_terms([1.0, 1.0, 1.1], [1.0, 2.0, 3.0], norm_adv=True)
    assert tilted.policy_loss.item() == pytest.approx(-0.1 / 3, abs=1e-6)


def test_single_sample_minibatch_is_not_normalised_into_nan():
    terms = loss_terms([1.0], [3.0], norm_adv=True)
    assert math.isfinite(terms.policy_loss.item())


def test_loss_gradients_flow_to_log_probs_and_values():
    new_log_probs = torch.zeros(4, requires_grad=True)
    new_values = torch.zeros(4, requires_grad=True)
    terms = ppo_loss_terms(
        new_log_probs=new_log_probs,
        old_log_probs=torch.zeros(4),
        advantages=torch.tensor([1.0, -1.0, 2.0, 0.5]),
        new_values=new_values,
        old_values=torch.zeros(4),
        returns=torch.ones(4),
        clip_coef=0.2,
        clip_vloss=True,
        norm_adv=False,
    )
    (terms.policy_loss + terms.value_loss).backward()
    assert new_log_probs.grad.abs().sum() > 0
    assert new_values.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# rollout bookkeeping
# --------------------------------------------------------------------------- #


def test_ready_to_update_only_when_the_rollout_is_full():
    algo = make_algo(n_envs=3, n_steps=5)
    rng = np.random.default_rng(0)
    obs = random_obs(rng, 3, VECTOR_SPACE)
    for t in range(5):
        assert not algo.ready_to_update(global_step=3 * t)
        actions, extras = algo.select_actions(obs, 3 * t)
        step = random_vec_step(rng, 3, VECTOR_SPACE)
        algo.observe(obs, actions, extras, step)
        obs = step.obs
    assert algo.ready_to_update(global_step=15)
    assert len(algo.buffer) == 5


def test_update_needs_a_full_rollout():
    algo = make_algo()
    with pytest.raises(RuntimeError, match="full"):
        algo.update(global_step=0, progress=0.0)


def test_observing_past_a_full_rollout_raises_instead_of_dropping_data():
    algo = make_algo(n_steps=2)
    rng = np.random.default_rng(0)
    collect_random_rollout(algo, rng)
    obs = random_obs(rng, 2, VECTOR_SPACE)
    actions, extras = algo.select_actions(obs, 0)
    with pytest.raises(RuntimeError, match="full"):
        algo.observe(obs, actions, extras, random_vec_step(rng, 2, VECTOR_SPACE))


def test_update_resets_the_buffer_and_counts_updates():
    algo = make_algo()
    assert algo.n_updates == 0
    rollout_and_update(algo, seed=0)
    assert algo.n_updates == 1
    assert len(algo.buffer) == 0 and not algo.ready_to_update(0)
    rollout_and_update(algo, seed=1)  # the buffer is reusable straight away
    assert algo.n_updates == 2


def test_observe_stores_the_transition_and_leaves_its_inputs_untouched():
    algo = make_algo(n_envs=2, obs_space=SCALAR_SPACE, gamma=0.9)
    obs = np.array([[1.0], [2.0]], np.float32)
    actions = np.array([1, 0])
    extras = {"log_prob": np.array([-0.5, -0.7], np.float32), "value": np.array([0.1, 0.2])}
    step = make_step(
        obs=[[3.0], [0.0]],
        rewards=[1.5, -2.0],
        truncated=[False, True],
        final_obs=[[3.0], [9.0]],
    )
    snapshot = copy.deepcopy(step)

    algo.observe(obs, actions, extras, step)

    np.testing.assert_array_equal(step.rewards, snapshot.rewards)  # bootstrapping must not leak
    np.testing.assert_array_equal(step.final_obs, snapshot.final_obs)
    np.testing.assert_array_equal(extras["value"], [0.1, 0.2])
    buffer = algo.buffer
    torch.testing.assert_close(buffer.obs[0], torch.tensor([[1.0], [2.0]]))
    torch.testing.assert_close(buffer.actions[0], torch.tensor([1, 0]))
    torch.testing.assert_close(buffer.log_probs[0], torch.tensor([-0.5, -0.7]))
    torch.testing.assert_close(buffer.values[0], torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(buffer.dones[0], torch.tensor([0.0, 1.0]))
    assert buffer.rewards[0, 0].item() == pytest.approx(1.5)


def test_observe_without_the_extras_of_select_actions_fails_clearly():
    algo = make_algo()
    rng = np.random.default_rng(0)
    obs = random_obs(rng, 2, VECTOR_SPACE)
    with pytest.raises(KeyError, match="select_actions"):
        algo.observe(obs, np.zeros(2, np.int64), {}, random_vec_step(rng, 2, VECTOR_SPACE))
    assert len(algo.buffer) == 0


def test_observe_accepts_exactly_what_select_actions_returned():
    algo = make_algo(n_envs=3)
    rng = np.random.default_rng(0)
    obs = random_obs(rng, 3, VECTOR_SPACE)
    actions, extras = algo.select_actions(obs, 0)
    algo.observe(obs, actions, extras, random_vec_step(rng, 3, VECTOR_SPACE))
    torch.testing.assert_close(algo.buffer.actions[0], torch.as_tensor(actions))
    torch.testing.assert_close(algo.buffer.values[0], torch.as_tensor(extras["value"]))
    torch.testing.assert_close(algo.buffer.log_probs[0], torch.as_tensor(extras["log_prob"]))


# --------------------------------------------------------------------------- #
# done / truncation semantics with a known value function V(obs) = 2 * obs
# --------------------------------------------------------------------------- #


class ValueSpy:
    """Replaces `model.get_value` by V(obs) = 2 * obs[:, 0] and records every batch it sees."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        self.calls.append(obs.detach().cpu().numpy().copy())
        return 2.0 * obs[:, 0].float()


def observe_with_known_values(algo: PPO, obs, step: VecStep) -> None:
    obs = np.asarray(obs, dtype=np.float32)
    n = len(obs)
    extras = {"log_prob": np.full(n, -math.log(2.0), np.float32), "value": 2.0 * obs[:, 0]}
    algo.observe(obs, np.zeros(n, np.int64), extras, step)


def test_truncation_bootstraps_only_the_truncated_envs_with_one_batched_forward(monkeypatch):
    algo = make_algo(n_envs=4, obs_space=SCALAR_SPACE, n_steps=2, gamma=0.5)
    spy = ValueSpy()
    monkeypatch.setattr(algo.model, "get_value", spy)

    # env 0 runs on, env 1 is truncated, env 2 terminated, env 3 terminated *and* truncated.
    step = make_step(
        obs=[[1.0], [0.0], [0.0], [0.0]],
        rewards=[1.0, 1.0, 1.0, 1.0],
        terminated=[False, False, True, True],
        truncated=[False, True, False, True],
        final_obs=[[1.0], [7.0], [8.0], [9.0]],
    )
    observe_with_known_values(algo, [[0.0], [6.0], [7.0], [8.0]], step)

    assert len(spy.calls) == 1  # one forward pass ...
    np.testing.assert_array_equal(spy.calls[0], [[7.0]])  # ... over the truncated env only
    # r + gamma * V(final_obs) = 1 + 0.5 * 14 = 8 for env 1; a terminal state is worth nothing.
    torch.testing.assert_close(algo.buffer.rewards[0], torch.tensor([1.0, 8.0, 1.0, 1.0]))
    torch.testing.assert_close(algo.buffer.dones[0], torch.tensor([0.0, 1.0, 1.0, 1.0]))


def test_no_value_forward_when_nothing_was_truncated(monkeypatch):
    algo = make_algo(n_envs=2, obs_space=SCALAR_SPACE, n_steps=4)
    spy = ValueSpy()
    monkeypatch.setattr(algo.model, "get_value", spy)
    step = make_step(obs=[[1.0], [0.0]], rewards=[1.0, 1.0], terminated=[False, True])
    observe_with_known_values(algo, [[0.0], [5.0]], step)
    assert spy.calls == []


def test_update_bootstraps_from_the_last_observed_obs_and_dones(monkeypatch):
    """Hand-computed GAE through the PPO plumbing; gamma = lambda = 0.5, V(obs) = 2 * obs.

    env 0: obs 1 -> 2 -> (truncated, true successor 3, restarts at 0) -> 1
        stored r = [1, 1 + 0.5 * V(3) = 4, 1], V = [2, 4, 0], done = [0, 1, 0], V(last obs 1) = 2
        t=2: delta = 1 + 0.5*2 - 0 = 2   A = 2
        t=1: delta = 4 - 4         = 0   A = 0            (done: nothing carried over)
        t=0: delta = 1 + 0.5*4 - 2 = 1   A = 1 + 0.25*0 = 1
    env 1: obs 5 -> 6 -> (terminated, restarts at 9) -> (terminated again, restarts at 9)
        r = [2, 2, 3], V = [10, 12, 18], done = [0, 1, 1], V(last obs 9) = 18 must be masked
        t=2: delta = 3 - 18          = -15  A = -15
        t=1: delta = 2 - 12          = -10  A = -10
        t=0: delta = 2 + 0.5*12 - 10 = -2   A = -2 + 0.25 * -10 = -4.5
    """
    algo = make_algo(n_envs=2, obs_space=SCALAR_SPACE, n_steps=3, gamma=0.5, gae_lambda=0.5)
    spy = ValueSpy()
    monkeypatch.setattr(algo.model, "get_value", spy)

    observe_with_known_values(
        algo, [[1.0], [5.0]], make_step(obs=[[2.0], [6.0]], rewards=[1.0, 2.0])
    )
    observe_with_known_values(
        algo,
        [[2.0], [6.0]],
        make_step(
            obs=[[0.0], [9.0]],
            rewards=[1.0, 2.0],
            terminated=[False, True],
            truncated=[True, False],
            final_obs=[[3.0], [7.0]],
        ),
    )
    last = make_step(
        obs=[[1.0], [9.0]], rewards=[1.0, 3.0], terminated=[False, True], final_obs=[[1.0], [4.0]]
    )
    observe_with_known_values(algo, [[0.0], [9.0]], last)
    last.obs[:] = -100.0  # vec envs may reuse their arrays: PPO must have kept its own copy

    captured: dict[str, torch.Tensor] = {}
    original = algo.buffer.compute_returns_and_advantages

    def capture(last_values, last_dones, gamma, gae_lambda):
        original(last_values, last_dones, gamma, gae_lambda)
        captured["last_values"] = torch.as_tensor(last_values).clone()
        captured["advantages"] = algo.buffer.advantages.clone()
        captured["returns"] = algo.buffer.returns.clone()

    monkeypatch.setattr(algo.buffer, "compute_returns_and_advantages", capture)
    spy.calls.clear()
    algo.update(global_step=6, progress=0.0)

    np.testing.assert_array_equal(spy.calls[0], [[1.0], [9.0]])
    torch.testing.assert_close(captured["last_values"], torch.tensor([2.0, 18.0]))
    torch.testing.assert_close(
        captured["advantages"], torch.tensor([[1.0, -4.5], [0.0, -10.0], [2.0, -15.0]])
    )
    torch.testing.assert_close(
        captured["returns"], torch.tensor([[3.0, 5.5], [4.0, 2.0], [2.0, 3.0]])
    )


# --------------------------------------------------------------------------- #
# update
# --------------------------------------------------------------------------- #


def test_update_returns_all_metrics_with_finite_values():
    algo = make_algo(n_envs=4, n_steps=16, n_epochs=3, n_minibatches=4)
    metrics = rollout_and_update(algo, seed=0)
    assert set(metrics) >= METRIC_KEYS
    for key, value in metrics.items():
        assert isinstance(value, float), key
        assert math.isfinite(value), key
    assert metrics["entropy"] == pytest.approx(math.log(2), abs=0.05)  # near-uniform at init
    assert 0.0 <= metrics["clip_frac"] <= 1.0
    assert metrics["approx_kl"] >= -1e-6
    assert metrics["explained_variance"] <= 1.0


def test_update_changes_policy_and_value_parameters():
    algo = make_algo()
    before = {name: p.detach().clone() for name, p in algo.model.named_parameters()}
    rollout_and_update(algo, seed=0)
    for name, param in algo.model.named_parameters():
        assert not torch.equal(before[name], param), f"{name} did not change"


def test_first_pass_over_fresh_data_has_ratio_one(monkeypatch):
    """One epoch, one minibatch: the policy is still the behaviour policy, so r = 1 exactly."""
    algo = make_algo(n_envs=4, n_steps=16, n_epochs=1, n_minibatches=1, clip_vloss=True)
    collect_random_rollout(algo, np.random.default_rng(0))

    seen: dict[str, torch.Tensor] = {}
    original = algo.buffer.compute_returns_and_advantages

    def capture(*args):
        original(*args)
        seen["advantages"] = algo.buffer.advantages.clone()

    monkeypatch.setattr(algo.buffer, "compute_returns_and_advantages", capture)
    metrics = algo.update(global_step=64, progress=0.0)

    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["clip_frac"] == 0.0
    # r = 1 and zero-mean normalised advantages: the surrogate is -mean(A_norm) = 0 ...
    assert metrics["policy_loss"] == pytest.approx(0.0, abs=1e-5)
    # ... and new V = old V, so clipping is inactive and the loss is 0.5 * mean((V - R)^2).
    expected_value_loss = 0.5 * seen["advantages"].pow(2).mean().item()
    assert metrics["value_loss"] == pytest.approx(expected_value_loss, rel=1e-4)


def test_later_epochs_move_the_policy_away_from_the_behaviour_policy():
    algo = make_algo(n_envs=4, n_steps=16, n_epochs=4, n_minibatches=4, lr=1e-2)
    metrics = rollout_and_update(algo, seed=0)
    assert metrics["approx_kl"] > 1e-6


def test_each_update_runs_epochs_times_minibatches_gradient_steps():
    algo = make_algo(n_epochs=3, n_minibatches=2)
    metrics = rollout_and_update(algo, seed=0)
    assert adam_steps(algo) == 6
    assert metrics["epochs"] == 3.0


def test_target_kl_stops_the_epochs_early():
    kwargs = {"n_envs": 4, "n_steps": 16, "n_epochs": 8, "n_minibatches": 4, "lr": 1e-2}
    unlimited = make_algo(**kwargs, target_kl=None)
    rollout_and_update(unlimited, seed=0)
    assert adam_steps(unlimited) == 32

    limited = make_algo(**kwargs, target_kl=1e-7)
    metrics = rollout_and_update(limited, seed=0)
    assert adam_steps(limited) == 4  # the first epoch always completes, then KL > target
    assert metrics["epochs"] == 1.0
    assert metrics["approx_kl"] > 1e-7

    generous = make_algo(**kwargs, target_kl=1e6)
    rollout_and_update(generous, seed=0)
    assert adam_steps(generous) == 32


def test_gradients_are_clipped_to_max_grad_norm():
    # Adam rescales gradients, but with a norm this small its epsilon dominates: no visible step.
    clipped = make_algo(max_grad_norm=1e-12)
    before = flat_params(clipped)
    metrics = rollout_and_update(clipped, seed=0)
    assert (flat_params(clipped) - before).abs().max() < 1e-8
    assert metrics["grad_norm"] > 1e-6  # the reported norm is the one measured before clipping

    free = make_algo(max_grad_norm=0.5)
    before = flat_params(free)
    rollout_and_update(free, seed=0)
    assert (flat_params(free) - before).abs().max() > 1e-5


def test_non_finite_loss_raises_before_it_can_destroy_the_weights():
    algo = make_algo(n_steps=4)
    rng = np.random.default_rng(0)
    obs = random_obs(rng, 2, VECTOR_SPACE)
    for t in range(4):
        actions, extras = algo.select_actions(obs, 0)
        step = random_vec_step(rng, 2, VECTOR_SPACE)
        if t == 1:
            step.rewards[0] = np.inf  # e.g. a reward bug in an env
        algo.observe(obs, actions, extras, step)
        obs = step.obs
    before = flat_params(algo)

    with pytest.raises(FloatingPointError, match="non-finite"):
        algo.update(global_step=8, progress=0.0)

    assert torch.isfinite(flat_params(algo)).all()
    torch.testing.assert_close(flat_params(algo), before, rtol=0, atol=0)


@pytest.mark.parametrize("flags", [{"clip_vloss": False}, {"norm_adv": False}, {"ent_coef": 0.0}])
def test_update_runs_with_optional_pieces_switched_off(flags: dict):
    metrics = rollout_and_update(make_algo(**flags), seed=0)
    assert all(math.isfinite(value) for value in metrics.values())


def test_entropy_bonus_enters_the_loss(monkeypatch):
    """A huge entropy coefficient must dominate and push the policy towards uniform."""
    algo = make_algo(n_envs=4, n_steps=16, n_epochs=10, ent_coef=100.0, lr=1e-2, max_grad_norm=1e9)
    with torch.no_grad():
        algo.model.policy_head.bias.copy_(torch.tensor([0.0, 2.0]))
    obs = torch.zeros(1, 4)

    def entropy() -> float:
        with torch.no_grad():
            return algo.model.get_action_and_value(obs)[2].item()

    before = entropy()
    rollout_and_update(algo, seed=0)
    assert entropy() > before


# --------------------------------------------------------------------------- #
# learning-rate schedule
# --------------------------------------------------------------------------- #


def test_lr_anneals_linearly_with_progress():
    algo = make_algo(lr=1e-3, anneal_lr=True)
    metrics = rollout_and_update(algo, seed=0, progress=0.25)
    assert metrics["lr"] == pytest.approx(0.75e-3)
    assert algo.optimizer.param_groups[0]["lr"] == pytest.approx(0.75e-3)
    metrics = rollout_and_update(algo, seed=1, progress=1.0)
    assert metrics["lr"] == pytest.approx(0.0, abs=1e-12)


def test_lr_is_constant_without_annealing():
    algo = make_algo(lr=1e-3, anneal_lr=False)
    assert rollout_and_update(algo, seed=0, progress=0.9)["lr"] == pytest.approx(1e-3)
    assert algo.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


@pytest.mark.parametrize("progress,expected", [(-0.5, 1e-3), (1.7, 0.0)])
def test_progress_outside_the_unit_interval_is_clamped(progress: float, expected: float):
    algo = make_algo(lr=1e-3, anneal_lr=True)
    metrics = rollout_and_update(algo, seed=0, progress=progress)
    assert metrics["lr"] == pytest.approx(expected, abs=1e-12)
    assert all(math.isfinite(value) for value in metrics.values())


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #


def test_state_dict_holds_model_optimizer_and_update_counter():
    algo = make_algo()
    rollout_and_update(algo, seed=0)
    state = algo.state_dict()
    assert set(state) == {"model", "optimizer", "n_updates"}
    assert state["n_updates"] == 1 and isinstance(state["n_updates"], int)


def test_state_dict_round_trip_reproduces_predict_and_the_next_update(tmp_path):
    algo = make_algo(seed=1, n_envs=3, n_steps=8, n_epochs=2, n_minibatches=3)
    rollout_and_update(algo, seed=0, progress=0.1)  # gives Adam some state worth restoring
    path = tmp_path / "ppo.pt"
    torch.save(algo.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)  # as checkpoints are loaded

    clone = make_algo(seed=2, n_envs=3, n_steps=8, n_epochs=2, n_minibatches=3)
    assert not torch.equal(flat_params(clone), flat_params(algo))
    clone.load_state_dict(state)

    assert clone.n_updates == 1
    torch.testing.assert_close(flat_params(clone), flat_params(algo), rtol=0, atol=0)
    batch = random_obs(np.random.default_rng(5), 32, VECTOR_SPACE)
    np.testing.assert_array_equal(clone.predict(batch), algo.predict(batch))
    torch.manual_seed(3)
    sampled = algo.predict(batch, deterministic=False)
    torch.manual_seed(3)
    np.testing.assert_array_equal(clone.predict(batch, deterministic=False), sampled)

    results = []
    for learner in (algo, clone):
        torch.manual_seed(123)  # same RNG stream: same sampled actions, same minibatch order
        results.append(rollout_and_update(learner, seed=7, progress=0.2))
    assert results[0] == results[1]
    torch.testing.assert_close(flat_params(clone), flat_params(algo), rtol=0, atol=0)
    assert clone.n_updates == algo.n_updates == 2


def test_load_state_dict_starts_with_an_empty_rollout():
    source = make_algo()
    rollout_and_update(source, seed=0)
    state = copy.deepcopy(source.state_dict())

    target = make_algo(seed=5, n_steps=8)
    rng = np.random.default_rng(1)
    obs = random_obs(rng, 2, VECTOR_SPACE)
    for _ in range(3):  # a half-collected rollout from the old policy must not survive the load
        actions, extras = target.select_actions(obs, 0)
        step = random_vec_step(rng, 2, VECTOR_SPACE)
        target.observe(obs, actions, extras, step)
        obs = step.obs
    assert len(target.buffer) == 3

    target.load_state_dict(state)
    assert len(target.buffer) == 0
    assert not target.ready_to_update(0)
    rollout_and_update(target, seed=2)  # and a fresh full rollout works
    assert target.n_updates == 2


def test_lr_schedule_needs_no_state_to_resume():
    """The LR is a pure function of `progress`, whatever LR the checkpointed optimizer had."""
    source = make_algo(lr=1e-3)
    rollout_and_update(source, seed=0, progress=0.5)
    resumed = make_algo(lr=1e-3, seed=9)
    resumed.load_state_dict(copy.deepcopy(source.state_dict()))
    assert rollout_and_update(resumed, seed=1, progress=0.75)["lr"] == pytest.approx(0.25e-3)


# --------------------------------------------------------------------------- #
# other observation types
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "space",
    [
        pytest.param(gym.spaces.Box(0, 255, (2, 36, 36), np.uint8), id="pixels"),
        pytest.param(gym.spaces.Box(-1.0, 1.0, (14, 15, 16), np.float32), id="grid"),
    ],
)
def test_full_cycle_on_image_like_observations(space: gym.spaces.Box):
    algo = make_algo(
        n_envs=2, obs_space=space, action_space=gym.spaces.Discrete(7), n_steps=4, n_epochs=1
    )
    metrics = rollout_and_update(algo, seed=0)
    assert algo.buffer.obs.dtype == (torch.uint8 if space.dtype == np.uint8 else torch.float32)
    assert all(math.isfinite(value) for value in metrics.values())
    actions = algo.predict(random_obs(np.random.default_rng(1), 3, space))
    assert actions.shape == (3,) and 0 <= actions.min() and actions.max() < 7


# --------------------------------------------------------------------------- #
# learning sanity (fast, CPU)
# --------------------------------------------------------------------------- #


def test_learns_a_two_armed_bandit():
    """One-step episodes, reward = action: the policy must come to prefer action 1."""
    n_envs = 8
    algo = make_algo(
        n_envs=n_envs, obs_space=SCALAR_SPACE, n_steps=8, n_epochs=4, n_minibatches=4, lr=3e-3
    )
    obs = np.ones((n_envs, 1), np.float32)
    global_step = 0
    for _ in range(30 * 8):
        actions, extras = algo.select_actions(obs, global_step)
        step = make_step(obs=obs, rewards=actions.astype(np.float32), terminated=np.ones(n_envs))
        algo.observe(obs, actions, extras, step)
        global_step += n_envs
        if algo.ready_to_update(global_step):
            algo.update(global_step, progress=0.0)
    with torch.no_grad():
        logits, value = algo.model(torch.ones(1, 1))
    assert torch.softmax(logits, dim=-1)[0, 1].item() > 0.9
    assert value.item() == pytest.approx(1.0, abs=0.25)  # V = E[r] -> 1 under the learned policy
    assert algo.predict(obs)[0] == 1


def test_value_function_respects_termination_but_bootstraps_through_truncation():
    """Reward 1 per step, 4-step episodes, gamma 0.5.

    If the episode *terminates*, V(s_t) = 1 + 0.5 + ... over the remaining steps, so the
    last state is worth exactly 1. If it is merely *truncated*, the true value is the
    infinite-horizon 1 / (1 - gamma) = 2 everywhere - which the critic can only learn when
    truncated transitions bootstrap from `final_obs`.
    """
    values = {}
    for kind in ("terminated", "truncated"):
        n_envs, horizon = 4, 4
        algo = make_algo(
            n_envs=n_envs, obs_space=SCALAR_SPACE, n_steps=16, n_epochs=4, n_minibatches=4,
            gamma=0.5, gae_lambda=0.95, lr=1e-2, anneal_lr=False, seed=0,
        )  # fmt: skip
        t = np.zeros(n_envs, np.int64)
        global_step = 0
        for _ in range(40 * 16):
            obs = (t[:, None] / horizon).astype(np.float32)
            actions, extras = algo.select_actions(obs, global_step)
            t = t + 1
            done = t >= horizon
            final_obs = (t[:, None] / horizon).astype(np.float32)
            t = np.where(done, 0, t)
            step = make_step(
                obs=(t[:, None] / horizon),
                rewards=np.ones(n_envs),
                terminated=done if kind == "terminated" else None,
                truncated=done if kind == "truncated" else None,
                final_obs=final_obs,
            )
            algo.observe(obs, actions, extras, step)
            global_step += n_envs
            if algo.ready_to_update(global_step):
                algo.update(global_step, progress=0.0)
        with torch.no_grad():
            states = torch.arange(horizon, dtype=torch.float32).reshape(-1, 1) / horizon
            values[kind] = algo.model.get_value(states).numpy()

    # Seeds 0-5 land within 0.06 of these targets.
    np.testing.assert_allclose(values["terminated"], [1.875, 1.75, 1.5, 1.0], atol=0.1)
    np.testing.assert_allclose(values["truncated"], [2.0, 2.0, 2.0, 2.0], atol=0.1)


def test_cartpole_return_improves_within_a_few_thousand_steps():
    n_envs = 4
    venv = MiniVecEnv([make_env(EnvConfig(id="CartPole-v1"), seed=i) for i in range(n_envs)])
    cfg = TrainConfig(
        algo="ppo",
        n_envs=n_envs,
        network=NetworkConfig(mlp_hidden=[32, 32]),
        ppo=PPOConfig(
            lr=2.5e-3, n_steps=64, n_epochs=6, n_minibatches=4, ent_coef=0.0, anneal_lr=False
        ),
    )
    torch.manual_seed(0)
    algo = PPO(venv.envs[0].observation_space, venv.envs[0].action_space, cfg, CPU, n_envs)

    total = 6_400
    returns: list[float] = []
    obs = venv.reset(seed=0)
    global_step = 0
    while global_step < total:
        actions, extras = algo.select_actions(obs, global_step)
        step = venv.step(actions)
        algo.observe(obs, actions, extras, step)
        global_step += n_envs
        returns += [float(info["episode"]["r"]) for info in step.infos if "episode" in info]
        if algo.ready_to_update(global_step):
            metrics = algo.update(global_step, progress=global_step / total)
            assert all(math.isfinite(value) for value in metrics.values())
        obs = step.obs
    venv.close()

    # A random CartPole policy survives ~22 steps; seeds 0-7 end between 150 and 230 here.
    early, late = np.mean(returns[:20]), np.mean(returns[-20:])
    assert late > 2 * early, f"no learning: first episodes {early:.1f}, last episodes {late:.1f}"
    assert late > 80, f"no learning: first episodes {early:.1f}, last episodes {late:.1f}"
