"""Double DQN: exploration, replay bookkeeping, TD targets, target-network sync, checkpoints."""

import copy
import io
import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from mario_play.rl.algos import get_algorithm
from mario_play.rl.algos.base import Algorithm
from mario_play.rl.algos.dqn import DQN
from mario_play.rl.checkpoint import load_checkpoint, save_checkpoint
from mario_play.rl.config import DQNConfig, NetworkConfig, TrainConfig, config_to_dict
from mario_play.rl.networks import QNetwork
from mario_play.rl.types import VecStep

CPU = torch.device("cpu")
N_STATES = 3
ONE_HOT_SPACE = gym.spaces.Box(0.0, 1.0, (N_STATES,), np.float32)
VECTOR_SPACE = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
IMAGE_SPACE = gym.spaces.Box(0, 255, (4, 36, 36), np.uint8)

METRIC_KEYS = {"loss", "q_mean", "epsilon", "buffer_size"}

# Small, fast defaults: learn from the first full batch, update on every vector step.
FAST = dict(
    lr=1e-3,
    buffer_size=256,
    learning_starts=0,
    batch_size=8,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1_000_000,
    eps_decay_steps=100,
)


def make_cfg(seed=0, network=None, **dqn):
    return TrainConfig(
        algo="dqn",
        seed=seed,
        network=network or NetworkConfig(hidden_size=16, mlp_hidden=[16]),
        dqn=DQNConfig(**{**FAST, **dqn}),
    )


def make_algo(space=VECTOR_SPACE, n_actions=3, n_envs=2, seed=0, network=None, **dqn):
    torch.manual_seed(seed)
    cfg = make_cfg(seed=seed, network=network, **dqn)
    return DQN(space, gym.spaces.Discrete(n_actions), cfg, CPU, n_envs)


def make_tabular_algo(online_q, target_q, **dqn):
    """A DQN whose networks are lookup tables: Q(one_hot(s), a) = table[s][a]."""
    online_q = torch.tensor(online_q, dtype=torch.float32)
    algo = make_algo(
        space=gym.spaces.Box(0.0, 1.0, (online_q.shape[0],), np.float32),
        n_actions=online_q.shape[1],
        network=NetworkConfig(encoder="mlp", mlp_hidden=[]),
        **dqn,
    )
    with torch.no_grad():
        for net, table in ((algo.q_net, online_q), (algo.target_net, torch.tensor(target_q))):
            net.q_head.weight.copy_(table.to(torch.float32).T)
            net.q_head.bias.zero_()
    return algo


def one_hot(states, n=N_STATES):
    return np.eye(n, dtype=np.float32)[np.asarray(states)]


def random_obs(space, n, rng):
    if space.dtype == np.uint8:
        return rng.integers(0, 256, size=(n, *space.shape), dtype=np.uint8)
    return rng.standard_normal((n, *space.shape)).astype(np.float32)


def make_step(obs, rewards=None, terminated=None, truncated=None, final_obs=None):
    n = len(obs)
    return VecStep(
        obs=obs,
        rewards=np.zeros(n, np.float32) if rewards is None else np.asarray(rewards, np.float32),
        terminated=np.zeros(n, bool) if terminated is None else np.asarray(terminated, bool),
        truncated=np.zeros(n, bool) if truncated is None else np.asarray(truncated, bool),
        final_obs=obs if final_obs is None else final_obs,
        infos=[{} for _ in range(n)],
    )


def fill(algo, vector_steps, seed=0):
    """Drive `observe` with random transitions; returns the number of env transitions added."""
    rng = np.random.default_rng(seed)
    n = algo.n_envs
    for _ in range(vector_steps):
        obs = random_obs(algo.obs_space, n, rng)
        actions = rng.integers(0, algo.n_actions, size=n)
        step = make_step(
            random_obs(algo.obs_space, n, rng),
            rewards=rng.standard_normal(n),
            terminated=rng.random(n) < 0.2,
        )
        algo.observe(obs, actions, {}, step)
    return vector_steps * n


def params(net):
    return [p.detach().clone() for p in net.parameters()]


def assert_params_equal(a, b):
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def params_differ(a, b):
    return any(not torch.equal(x, y) for x, y in zip(a, b, strict=True))


def batch_of(states, actions, rewards, next_states, terminated, n=N_STATES):
    return {
        "obs": torch.from_numpy(one_hot(states, n)),
        "actions": torch.tensor(actions, dtype=torch.int64),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "next_obs": torch.from_numpy(one_hot(next_states, n)),
        "terminated": torch.tensor(terminated, dtype=torch.float32),
    }


# Q tables for the hand-computed examples; rows are states, columns actions.
ONLINE_Q = [[1.0, 2.0], [5.0, 3.0], [0.0, 4.0]]  # greedy actions: s0 -> a1, s1 -> a0, s2 -> a1
TARGET_Q = [[0.5, 0.7], [1.0, 9.0], [2.0, -1.0]]  # the target net disagrees about s1 and s2


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_is_registered_and_implements_the_algorithm_interface():
    assert get_algorithm("dqn") is DQN
    assert issubclass(DQN, Algorithm)
    assert DQN.name == "dqn"


def test_builds_online_and_frozen_target_networks_with_identical_weights():
    algo = make_algo()
    assert isinstance(algo.q_net, QNetwork) and isinstance(algo.target_net, QNetwork)
    assert algo.q_net is not algo.target_net
    assert_params_equal(params(algo.q_net), params(algo.target_net))
    assert all(p.requires_grad for p in algo.q_net.parameters())
    assert not any(p.requires_grad for p in algo.target_net.parameters())
    assert algo.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_dueling_flag_reaches_both_networks():
    algo = make_algo(dueling=True)
    assert algo.q_net.dueling and algo.target_net.dueling
    fill(algo, 8)
    assert math.isfinite(algo.update(16, 0.0)["loss"])


def test_replay_buffer_follows_the_config_and_the_observation_space():
    algo = make_algo(space=IMAGE_SPACE, buffer_size=32, network=NetworkConfig(hidden_size=8))
    assert algo.buffer.capacity == 32
    assert algo.buffer.obs.dtype == np.uint8
    assert algo.buffer.obs.shape == (32, *IMAGE_SPACE.shape)
    assert len(algo.buffer) == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"buffer_size": 0},
        {"batch_size": 0},
        {"buffer_size": 4, "batch_size": 8},
        {"train_freq": 0},
        {"gradient_steps": 0},
        {"target_update_interval": 0},
        {"tau": 0.0},
        {"tau": 1.5},
        {"gamma": 1.1},
        {"gamma": -0.1},
        {"eps_start": 1.2},
        {"eps_end": -0.1},
        {"learning_starts": -1},
        {"lr": 0.0},
    ],
    ids=lambda bad: ",".join(f"{k}={v}" for k, v in bad.items()),
)
def test_invalid_hyperparameters_are_rejected(bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        make_algo(**bad)


# --------------------------------------------------------------------------- #
# exploration
# --------------------------------------------------------------------------- #


def test_epsilon_is_linear_in_global_step_and_clamped():
    algo = make_algo(eps_start=1.0, eps_end=0.1, eps_decay_steps=1000)
    assert algo.epsilon(0) == pytest.approx(1.0)
    assert algo.epsilon(250) == pytest.approx(0.775)
    assert algo.epsilon(500) == pytest.approx(0.55)
    assert algo.epsilon(1000) == pytest.approx(0.1)
    assert algo.epsilon(10**9) == pytest.approx(0.1)
    assert algo.epsilon(-5) == pytest.approx(1.0)


def test_predict_is_greedy_and_select_actions_is_greedy_once_epsilon_is_zero():
    algo = make_algo(n_actions=5, n_envs=64, eps_start=0.0, eps_end=0.0)
    obs = random_obs(VECTOR_SPACE, 64, np.random.default_rng(0))
    with torch.no_grad():
        greedy = algo.q_net(torch.from_numpy(obs)).argmax(dim=1).numpy()
    assert len(set(greedy.tolist())) > 1  # otherwise the comparison proves nothing

    predicted = algo.predict(obs, deterministic=True)
    assert predicted.dtype == np.int64 and predicted.shape == (64,)
    np.testing.assert_array_equal(predicted, greedy)
    np.testing.assert_array_equal(algo.predict(obs), greedy)  # deterministic is the default

    actions, extras = algo.select_actions(obs, global_step=0)
    assert actions.dtype == np.int64 and actions.shape == (64,)
    assert isinstance(extras, dict)
    np.testing.assert_array_equal(actions, greedy)


def test_fully_random_exploration_covers_the_action_space_uniformly():
    algo = make_algo(n_actions=4, n_envs=4000, eps_start=1.0, eps_end=1.0)
    obs = np.zeros((4000, *VECTOR_SPACE.shape), np.float32)  # greedy would repeat one action
    actions, _ = algo.select_actions(obs, global_step=0)
    assert actions.min() >= 0 and actions.max() < 4
    counts = np.bincount(actions, minlength=4)
    assert np.all(np.abs(counts - 1000) < 150)


def test_exploration_is_decided_per_environment():
    # eps = 0.5 half way through the decay: each env flips its own coin.
    algo = make_algo(n_actions=4, n_envs=4000, eps_start=1.0, eps_end=0.0, eps_decay_steps=100)
    obs = random_obs(VECTOR_SPACE, 4000, np.random.default_rng(1))
    greedy = algo.predict(obs)
    actions, _ = algo.select_actions(obs, global_step=50)
    deviating = np.mean(actions != greedy)
    # P(random draw differs from greedy) = eps * (1 - 1/n_actions) = 0.375
    assert deviating == pytest.approx(0.375, abs=0.04)


def test_exploration_follows_the_schedule_through_global_step():
    algo = make_algo(n_actions=4, n_envs=2000, eps_start=1.0, eps_end=0.0, eps_decay_steps=100)
    obs = random_obs(VECTOR_SPACE, 2000, np.random.default_rng(2))
    greedy = algo.predict(obs)
    early = np.mean(algo.select_actions(obs, global_step=0)[0] != greedy)
    late = np.mean(algo.select_actions(obs, global_step=100)[0] != greedy)
    assert early == pytest.approx(0.75, abs=0.05)
    assert late == 0.0


def test_exploration_is_seeded_from_the_config_not_from_global_state():
    obs = random_obs(VECTOR_SPACE, 32, np.random.default_rng(3))

    def rollout(seed):
        algo = make_algo(n_envs=32, seed=seed, eps_start=1.0, eps_end=1.0)
        np.random.seed(123)  # the legacy global RNG must not matter
        return np.stack([algo.select_actions(obs, t)[0] for t in range(5)])

    np.testing.assert_array_equal(rollout(7), rollout(7))
    assert not np.array_equal(rollout(7), rollout(8))


def test_predict_has_no_side_effects_on_training_state():
    algo = make_algo(n_envs=16, eps_start=0.5, eps_end=0.5)
    obs = random_obs(VECTOR_SPACE, 16, np.random.default_rng(4))
    rng_before = copy.deepcopy(algo.rng.bit_generator.state)
    weights_before = params(algo.q_net)
    for deterministic in (True, False):
        algo.predict(obs, deterministic=deterministic)
    assert algo.rng.bit_generator.state == rng_before
    assert_params_equal(params(algo.q_net), weights_before)
    assert len(algo.buffer) == 0


def test_stochastic_predict_explores_with_the_final_epsilon():
    obs = np.zeros((2000, *VECTOR_SPACE.shape), np.float32)
    explorer = make_algo(n_actions=4, eps_start=1.0, eps_end=1.0)
    assert len(set(explorer.predict(obs, deterministic=False).tolist())) == 4
    exploiter = make_algo(n_actions=4, eps_start=1.0, eps_end=0.0)
    np.testing.assert_array_equal(
        exploiter.predict(obs, deterministic=False), exploiter.predict(obs, deterministic=True)
    )


# --------------------------------------------------------------------------- #
# observe / replay bookkeeping
# --------------------------------------------------------------------------- #


def test_observe_stores_final_obs_as_successor_and_only_terminated_as_done():
    algo = make_algo(space=ONE_HOT_SPACE, n_actions=2, n_envs=3)
    obs = one_hot([0, 1, 2])
    reset_obs = one_hot([0, 0, 0])  # env 1 and env 2 finished: `obs` is already the new episode
    final_obs = one_hot([1, 2, 1])
    step = make_step(
        np.stack([final_obs[0], reset_obs[1], reset_obs[2]]),
        rewards=[0.5, -1.0, 2.0],
        terminated=[False, True, False],
        truncated=[False, False, True],
        final_obs=final_obs,
    )
    algo.observe(obs, np.array([1, 0, 1]), {}, step)

    assert len(algo.buffer) == 3
    np.testing.assert_array_equal(algo.buffer.obs[:3], obs)
    np.testing.assert_array_equal(algo.buffer.next_obs[:3], final_obs)
    assert algo.buffer.actions[:3].tolist() == [1, 0, 1]
    assert algo.buffer.rewards[:3].tolist() == [0.5, -1.0, 2.0]
    assert algo.buffer.terminated[:3].tolist() == [False, True, False]  # truncation is not done


def test_no_update_before_learning_starts():
    algo = make_algo(n_envs=2, learning_starts=20, batch_size=4)
    global_step = 0
    ready_at = []
    for _ in range(14):
        global_step += fill(algo, 1, seed=global_step)
        if algo.ready_to_update(global_step):
            ready_at.append(global_step)
    assert ready_at == [20, 22, 24, 26, 28]


def test_learning_starts_counts_env_transitions_not_buffer_fill():
    # A buffer smaller than learning_starts is full long before learning may begin.
    algo = make_algo(n_envs=2, learning_starts=20, batch_size=4, buffer_size=8)
    global_step = 0
    ready_at = []
    for _ in range(12):
        global_step += fill(algo, 1, seed=global_step)
        if algo.ready_to_update(global_step):
            ready_at.append(global_step)
    assert algo.buffer.full
    assert ready_at == [20, 22, 24]


def test_no_update_until_the_buffer_holds_a_full_batch():
    algo = make_algo(n_envs=2, learning_starts=0, batch_size=8)
    global_step = 0
    ready_at = []
    for _ in range(6):
        global_step += fill(algo, 1, seed=global_step)
        if algo.ready_to_update(global_step):
            ready_at.append(global_step)
    assert ready_at == [8, 10, 12]


def test_updates_happen_every_train_freq_vector_steps():
    algo = make_algo(n_envs=2, learning_starts=0, batch_size=2, train_freq=4)
    ready = []
    for vector_step in range(1, 13):
        fill(algo, 1, seed=vector_step)
        ready.append(algo.ready_to_update(2 * vector_step))
        assert algo.ready_to_update(2 * vector_step) == ready[-1]  # asking twice changes nothing
    assert [i + 1 for i, r in enumerate(ready) if r] == [4, 8, 12]


# --------------------------------------------------------------------------- #
# TD targets and loss
# --------------------------------------------------------------------------- #

TRANSITIONS = dict(
    states=[0, 1, 2],
    actions=[1, 0, 1],
    rewards=[1.0, 0.5, 2.0],
    next_states=[1, 2, 0],
    terminated=[0.0, 0.0, 1.0],
)


def test_double_dqn_target_uses_online_argmax_and_target_value():
    algo = make_tabular_algo(ONLINE_Q, TARGET_Q, gamma=0.9, double_q=True)
    targets = algo.compute_targets(batch_of(**TRANSITIONS))
    # s' = s1: online argmax a0 -> target Q 1.0;  s' = s2: online argmax a1 -> target Q -1.0
    expected = [1.0 + 0.9 * 1.0, 0.5 + 0.9 * -1.0, 2.0]
    torch.testing.assert_close(targets, torch.tensor(expected))
    assert not targets.requires_grad


def test_vanilla_dqn_target_uses_the_target_networks_max():
    algo = make_tabular_algo(ONLINE_Q, TARGET_Q, gamma=0.9, double_q=False)
    targets = algo.compute_targets(batch_of(**TRANSITIONS))
    expected = [1.0 + 0.9 * 9.0, 0.5 + 0.9 * 2.0, 2.0]
    torch.testing.assert_close(targets, torch.tensor(expected))


def test_loss_is_the_huber_loss_between_q_of_taken_actions_and_targets():
    algo = make_tabular_algo(ONLINE_Q, TARGET_Q, gamma=0.9, double_q=True)
    loss, q_taken = algo.compute_loss(batch_of(**TRANSITIONS))
    torch.testing.assert_close(q_taken, torch.tensor([2.0, 5.0, 4.0]))
    # TD errors: 2.0 - 1.9 = 0.1 (quadratic), 5.0 + 0.4 = 5.4 and 4.0 - 2.0 = 2.0 (linear)
    expected = (0.5 * 0.1**2 + (5.4 - 0.5) + (2.0 - 0.5)) / 3
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    assert loss.requires_grad


def test_truncated_transitions_bootstrap_and_terminated_ones_do_not():
    algo = make_tabular_algo(ONLINE_Q, TARGET_Q, gamma=0.9, n_envs=2, buffer_size=8)
    obs = one_hot([0, 0])
    reset_obs = one_hot([0, 0])
    final_obs = one_hot([1, 1])
    step = make_step(
        reset_obs,
        rewards=[1.0, 1.0],
        terminated=[False, True],
        truncated=[True, False],
        final_obs=final_obs,
    )
    algo.observe(obs, np.array([1, 1]), {}, step)

    batch = algo.buffer.sample(64, np.random.default_rng(0))
    assert set(batch["terminated"].tolist()) == {0.0, 1.0}
    targets = algo.compute_targets(batch)
    # truncated env: r + gamma * Q_target(final_obs = s1, argmax_online = a0) = 1 + 0.9 * 1.0
    # terminated env: r only
    expected = torch.where(batch["terminated"] == 1.0, 1.0, 1.9)
    torch.testing.assert_close(targets, expected)


# --------------------------------------------------------------------------- #
# update
# --------------------------------------------------------------------------- #


def test_update_returns_finite_metrics_and_changes_only_the_online_network():
    algo = make_algo(gradient_steps=3)
    global_step = fill(algo, 10)
    online_before, target_before = params(algo.q_net), params(algo.target_net)

    metrics = algo.update(global_step, progress=0.1)

    assert METRIC_KEYS <= set(metrics)
    assert all(isinstance(v, float) and math.isfinite(v) for v in metrics.values())
    assert metrics["buffer_size"] == 20.0
    assert metrics["epsilon"] == pytest.approx(algo.epsilon(global_step))
    assert metrics["loss"] >= 0.0
    assert algo.n_updates == 3
    assert params_differ(params(algo.q_net), online_before)
    assert_params_equal(params(algo.target_net), target_before)


def test_update_on_an_empty_buffer_raises():
    with pytest.raises(ValueError, match="empty"):
        make_algo().update(0, 0.0)


def test_hard_target_sync_happens_at_the_interval():
    algo = make_algo(n_envs=2, target_update_interval=10, lr=1e-2)
    initial_target = params(algo.target_net)
    global_step = 0
    synced_at = []
    for _ in range(12):
        global_step += fill(algo, 1, seed=global_step)
        if not algo.ready_to_update(global_step):
            continue
        target_before = params(algo.target_net)
        algo.update(global_step, 0.0)
        if params_differ(params(algo.target_net), target_before):
            synced_at.append(global_step)
            assert_params_equal(params(algo.target_net), params(algo.q_net))
        else:
            assert params_differ(params(algo.target_net), params(algo.q_net))
    assert synced_at == [10, 20]
    assert params_differ(params(algo.target_net), initial_target)


def test_hard_sync_interval_does_not_drift_when_updates_are_sparse():
    # Updates run every 3 vector steps * 2 envs = 6 transitions (6, 12, 18, 24, 30, 36) with a
    # sync interval of 10: the target syncs at the first update at or after each multiple of
    # 10. Measuring from the previous sync instead would give 12, 24, 36 (a period of 12).
    algo = make_algo(n_envs=2, train_freq=3, batch_size=2, target_update_interval=10, lr=1e-2)
    global_step = 0
    synced_at = []
    for _ in range(18):
        global_step += fill(algo, 1, seed=global_step)
        if algo.ready_to_update(global_step):
            target_before = params(algo.target_net)
            algo.update(global_step, 0.0)
            if params_differ(params(algo.target_net), target_before):
                synced_at.append(global_step)
    assert synced_at == [12, 24, 30]


def test_polyak_averaging_replaces_the_hard_sync_when_tau_is_below_one():
    algo = make_algo(tau=0.25, target_update_interval=1, lr=1e-2)
    global_step = fill(algo, 10)
    target_before = params(algo.target_net)

    algo.update(global_step, 0.0)

    online_after = params(algo.q_net)
    assert params_differ(online_after, target_before)
    for new, old, online in zip(params(algo.target_net), target_before, online_after, strict=True):
        torch.testing.assert_close(new, 0.75 * old + 0.25 * online)


def test_polyak_runs_after_every_gradient_step():
    algo = make_algo(tau=0.5, gradient_steps=2, lr=1e-2)
    twin = make_algo(tau=0.5, gradient_steps=1, lr=1e-2)
    global_step = fill(algo, 10)
    fill(twin, 10)
    algo.update(global_step, 0.0)
    twin.update(global_step, 0.0)
    twin.update(global_step, 0.0)
    assert_params_equal(params(algo.target_net), params(twin.target_net))
    assert_params_equal(params(algo.q_net), params(twin.q_net))


def test_gradients_are_clipped_to_max_grad_norm():
    algo = make_algo(max_grad_norm=1e-3)
    global_step = fill(algo, 10)
    metrics = algo.update(global_step, 0.0)
    grads = [p.grad for p in algo.q_net.parameters() if p.grad is not None]
    assert grads
    total_norm = torch.sqrt(sum((g**2).sum() for g in grads)).item()
    assert total_norm <= 1e-3 * 1.001
    assert metrics["grad_norm"] > 1e-3  # the reported norm is the one before clipping


def test_loss_decreases_when_fitting_a_fixed_batch():
    # buffer == batch: every gradient step sees the same transitions; the target net is frozen.
    algo = make_algo(buffer_size=16, batch_size=16, lr=1e-2, gradient_steps=1)
    global_step = fill(algo, 8)
    assert len(algo.buffer) == 16
    losses = [algo.update(global_step, 0.0)["loss"] for _ in range(500)]
    # 16 transitions are easily memorised: the loss collapses (observed ratio < 1e-3).
    assert np.mean(losses[-10:]) < 0.01 * np.mean(losses[:10])


def test_update_works_on_uint8_image_observations():
    algo = make_algo(space=IMAGE_SPACE, buffer_size=32, network=NetworkConfig(hidden_size=16))
    global_step = fill(algo, 8)
    obs = random_obs(IMAGE_SPACE, 2, np.random.default_rng(0))
    actions, _ = algo.select_actions(obs, global_step)
    assert actions.shape == (2,)
    metrics = algo.update(global_step, 0.0)
    assert math.isfinite(metrics["loss"])
    assert algo.buffer.obs.dtype == np.uint8


FLOAT64_SPACE = gym.spaces.Box(-1.0, 1.0, (4,), np.float64)


def test_float64_observations_become_float32_tensors():
    """MPS has no float64: the acting path narrows just like the replay buffer does."""
    algo = make_algo(space=FLOAT64_SPACE, eps_start=0.0, eps_end=0.0)
    obs = np.random.default_rng(0).standard_normal((2, 4))
    assert obs.dtype == np.float64
    assert algo.obs_to_tensor(obs).dtype == torch.float32
    assert algo.obs_to_tensor(obs.astype(np.float32)).dtype == torch.float32
    assert algo.buffer.obs.dtype == np.float32
    assert algo.select_actions(obs, 0)[0].shape == (2,)
    assert algo.predict(obs).shape == (2,)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_float64_observation_space_acts_and_learns_on_mps():
    torch.manual_seed(0)
    cfg = make_cfg(eps_start=0.0, eps_end=0.0)
    algo = DQN(FLOAT64_SPACE, gym.spaces.Discrete(3), cfg, torch.device("mps"), 2)
    obs = np.random.default_rng(0).standard_normal((2, 4))
    actions, _ = algo.select_actions(obs, 0)  # epsilon 0: the greedy forward pass runs
    assert actions.shape == (2,) and algo.predict(obs).shape == (2,)
    global_step = fill(algo, 8)
    assert math.isfinite(algo.update(global_step, 0.0)["loss"])


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #


# 6 vector steps of 4 envs: updates at 8, 12, ..., 24 and one target sync (at 16), so the
# final state has online != target and non-trivial Adam moments.
RESUME = dict(n_envs=4, eps_start=0.5, eps_end=0.5, target_update_interval=16)


def trained_algo(**dqn):
    algo = make_algo(**{**RESUME, **dqn})
    global_step = 0
    rng = np.random.default_rng(0)
    for _ in range(6):
        algo.select_actions(random_obs(VECTOR_SPACE, 4, rng), global_step)
        global_step += fill(algo, 1, seed=global_step)
        if algo.ready_to_update(global_step):
            algo.update(global_step, 0.0)
    return algo, global_step


def adam_moments(algo):
    state = algo.optimizer.state_dict()["state"]
    assert state, "the optimizer has not stepped yet"
    return [
        entry[key].detach().clone().float()
        for _, entry in sorted(state.items())
        for key in ("step", "exp_avg", "exp_avg_sq")
    ]


def weights_only_round_trip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def test_state_dict_holds_models_optimizer_counters_and_rng_but_no_replay_data():
    algo, _ = trained_algo()
    state = algo.state_dict()
    assert {"q_net", "target_net", "optimizer", "rng"} <= set(state)
    assert state["vector_steps"] == 6
    assert state["n_updates"] == algo.n_updates > 0
    assert not any("buffer" in key or "replay" in key for key in state)
    n_tensors = sum(v.numel() for v in state["q_net"].values())
    assert n_tensors == sum(p.numel() for p in algo.q_net.parameters())


def test_state_dict_survives_a_weights_only_checkpoint(tmp_path):
    algo, global_step = trained_algo()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        algo_name="dqn",
        algo_state=algo.state_dict(),
        config_dict=config_to_dict(algo.cfg),
        global_step=global_step,
        best_eval=None,
        rng_state={},
    )
    restored = make_algo(seed=99, **RESUME)
    restored.load_state_dict(load_checkpoint(path)["algo_state"])
    assert_params_equal(params(restored.q_net), params(algo.q_net))
    assert_params_equal(params(restored.target_net), params(algo.target_net))


def test_state_dict_round_trip_restores_models_optimizer_counters_and_exploration():
    algo, global_step = trained_algo()
    state = weights_only_round_trip(algo.state_dict())

    restored = make_algo(seed=99, **RESUME)
    assert params_differ(params(restored.q_net), params(algo.q_net))
    restored.load_state_dict(state)

    assert_params_equal(params(restored.q_net), params(algo.q_net))
    assert_params_equal(params(restored.target_net), params(algo.target_net))
    assert params_differ(params(restored.q_net), params(restored.target_net))
    assert_params_equal(adam_moments(restored), adam_moments(algo))
    assert restored.n_updates == algo.n_updates
    assert restored.state_dict()["vector_steps"] == algo.state_dict()["vector_steps"]
    assert restored.state_dict()["last_target_sync"] == algo.state_dict()["last_target_sync"]

    obs = random_obs(VECTOR_SPACE, 4, np.random.default_rng(5))
    np.testing.assert_array_equal(restored.predict(obs), algo.predict(obs))
    for t in range(10):  # same exploration stream from here on
        np.testing.assert_array_equal(
            restored.select_actions(obs, global_step + t)[0],
            algo.select_actions(obs, global_step + t)[0],
        )


def test_restored_algorithm_performs_the_identical_next_update():
    algo, global_step = trained_algo()
    restored = make_algo(seed=99, **RESUME)
    restored.load_state_dict(weights_only_round_trip(algo.state_dict()))
    # Replay data is not part of a checkpoint; give both the same transitions again.
    algo.buffer.clear()
    for twin in (algo, restored):
        fill(twin, 4, seed=42)

    for offset in (4, 8, 12):  # 28, 32, 36: crosses the target sync at 32
        ours = algo.update(global_step + offset, 0.0)
        theirs = restored.update(global_step + offset, 0.0)
        assert ours == theirs
    assert_params_equal(params(restored.q_net), params(algo.q_net))
    assert_params_equal(params(restored.target_net), params(algo.target_net))


def test_state_dict_is_a_snapshot_not_a_view_of_live_training_state():
    algo, global_step = trained_algo()
    state = algo.state_dict()
    frozen = copy.deepcopy(state)
    algo.select_actions(random_obs(VECTOR_SPACE, 4, np.random.default_rng(6)), global_step)
    fill(algo, 2)
    algo.update(global_step + 8, 0.0)
    assert state["rng"] == frozen["rng"]
    assert state["n_updates"] == frozen["n_updates"]
    for key, value in state["q_net"].items():
        assert torch.equal(value, frozen["q_net"][key])
    assert algo.n_updates == frozen["n_updates"] + 1


def test_resume_waits_for_the_replay_buffer_to_refill():
    # A fresh run reaches learning_starts with that many transitions in replay. After a
    # resume the buffer is empty, so updates pause until the same amount has been collected
    # instead of fitting the restored network to the first handful of correlated samples.
    algo = make_algo(n_envs=2, learning_starts=12, batch_size=4)
    global_step = fill(algo, 10)
    assert algo.ready_to_update(global_step)

    resumed = make_algo(n_envs=2, learning_starts=12, batch_size=4)
    resumed.load_state_dict(weights_only_round_trip(algo.state_dict()))
    ready_at = []
    for _ in range(8):
        global_step += fill(resumed, 1, seed=global_step)
        if resumed.ready_to_update(global_step):
            ready_at.append(len(resumed.buffer))
    assert ready_at == [12, 14, 16]


def test_learning_rate_comes_from_the_current_config_on_resume():
    algo, _ = trained_algo(lr=1e-3)
    resumed = make_algo(lr=5e-4, **RESUME)
    resumed.load_state_dict(weights_only_round_trip(algo.state_dict()))
    assert [g["lr"] for g in resumed.optimizer.param_groups] == [pytest.approx(5e-4)]
    assert resumed.optimizer.state_dict()["state"]  # Adam moments were restored


def test_load_state_dict_rejects_a_foreign_rng_format():
    algo, _ = trained_algo()
    state = algo.state_dict()
    state["rng"] = {**state["rng"], "bit_generator": "MT19937"}
    with pytest.raises(ValueError, match="bit generator"):
        make_algo(n_envs=4).load_state_dict(state)


# --------------------------------------------------------------------------- #
# end-to-end sanity
# --------------------------------------------------------------------------- #


class Corridor:
    """Vectorised 5-cell corridor with same-step auto-reset, mimicking `VecEnv.step`.

    Action 1 moves right, action 0 left (bumping into the wall). Entering cell 4 pays +1 and
    terminates. Episodes start in a random cell 0..3 and are truncated after only 3 steps, so
    the goal is out of reach from cell 0 within one episode: its value can only be learned by
    bootstrapping through truncated transitions.
    """

    n_cells = 5
    max_steps = 3

    def __init__(self, n_envs, seed=0):
        self.rng = np.random.default_rng(seed)
        self.pos = self.rng.integers(0, self.n_cells - 1, size=n_envs)
        self.t = np.zeros(n_envs, dtype=np.int64)

    def observe(self):
        return one_hot(self.pos, self.n_cells)

    def step(self, actions):
        self.pos = np.clip(self.pos + np.where(actions == 1, 1, -1), 0, self.n_cells - 1)
        self.t += 1
        terminated = self.pos == self.n_cells - 1
        truncated = ~terminated & (self.t >= self.max_steps)
        final_obs = self.observe()
        done = terminated | truncated
        self.pos[done] = self.rng.integers(0, self.n_cells - 1, size=int(done.sum()))
        self.t[done] = 0
        return VecStep(
            obs=self.observe(),
            rewards=terminated.astype(np.float32),
            terminated=terminated,
            truncated=truncated,
            final_obs=final_obs,
            infos=[{} for _ in actions],
        )


def test_learns_the_optimal_values_of_a_small_corridor():
    n_envs, gamma, total = 2, 0.9, 3000
    algo = make_algo(
        space=gym.spaces.Box(0.0, 1.0, (Corridor.n_cells,), np.float32),
        n_actions=2,
        n_envs=n_envs,
        network=NetworkConfig(encoder="mlp", mlp_hidden=[32]),
        lr=5e-3,
        gamma=gamma,
        buffer_size=2000,
        learning_starts=100,
        batch_size=32,
        target_update_interval=100,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay_steps=1500,
    )
    env = Corridor(n_envs)
    obs = env.observe()
    global_step = 0
    while global_step < total:  # the trainer loop of `mario_play.rl.algos.base`
        actions, extras = algo.select_actions(obs, global_step)
        step = env.step(actions)
        algo.observe(obs, actions, extras, step)
        global_step += n_envs
        if algo.ready_to_update(global_step):
            algo.update(global_step, global_step / total)
        obs = step.obs

    states = one_hot(range(4), Corridor.n_cells)
    np.testing.assert_array_equal(algo.predict(states), [1, 1, 1, 1])  # always walk right
    with torch.no_grad():
        q_right = algo.q_net(torch.from_numpy(states))[:, 1].numpy()
    # Time-limit truncation must not leak into the values: Q*(s, right) = gamma^(3 - s).
    optimal = np.array([gamma**3, gamma**2, gamma, 1.0])
    np.testing.assert_allclose(q_right, optimal, atol=0.02)
