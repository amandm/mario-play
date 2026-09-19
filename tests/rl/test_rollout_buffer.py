"""RolloutBuffer: storage, GAE(lambda) against hand-computed numbers, minibatch partitioning."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mario_play.rl.buffers.rollout import RolloutBuffer

CPU = torch.device("cpu")
MINIBATCH_KEYS = {"obs", "actions", "log_probs", "values", "advantages", "returns"}


def fill(
    buffer: RolloutBuffer,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
) -> None:
    """Fill `buffer` from `(n_steps, n_envs)` arrays; obs/actions/log-probs encode the sample id."""
    n_steps, n_envs = rewards.shape
    for t in range(n_steps):
        ids = np.arange(n_envs) + t * n_envs
        obs = np.broadcast_to(
            ids.reshape(n_envs, *([1] * len(buffer.obs_shape))), (n_envs, *buffer.obs_shape)
        )
        buffer.add(
            obs=obs.astype(buffer.obs.numpy().dtype),
            actions=ids.astype(np.int64),
            log_probs=(-0.5 * ids).astype(np.float32),
            values=values[t].astype(np.float32),
            rewards=rewards[t].astype(np.float32),
            dones=dones[t].astype(bool),
        )


def make_filled(
    n_steps: int = 5, n_envs: int = 3, obs_shape: tuple[int, ...] = (2,), seed: int = 0
) -> RolloutBuffer:
    rng = np.random.default_rng(seed)
    buffer = RolloutBuffer(n_steps, n_envs, obs_shape, np.float32, CPU)
    dones = rng.random((n_steps, n_envs)) < 0.3
    fill(buffer, rng.normal(size=(n_steps, n_envs)), rng.normal(size=(n_steps, n_envs)), dones)
    buffer.compute_returns_and_advantages(
        torch.as_tensor(rng.normal(size=n_envs), dtype=torch.float32), dones[-1], 0.99, 0.95
    )
    return buffer


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #


def test_starts_empty_and_fills_after_n_steps():
    buffer = RolloutBuffer(3, 2, (4,), np.float32, CPU)
    assert not buffer.full
    assert len(buffer) == 0
    for t in range(3):
        assert not buffer.full
        buffer.add(
            np.full((2, 4), t, np.float32),
            np.array([t, t + 1]),
            np.zeros(2, np.float32),
            np.zeros(2, np.float32),
            np.ones(2, np.float32),
            np.zeros(2, bool),
        )
        assert len(buffer) == t + 1
    assert buffer.full
    assert buffer.obs.shape == (3, 2, 4)
    assert buffer.actions.dtype == torch.int64
    torch.testing.assert_close(buffer.actions, torch.tensor([[0, 1], [1, 2], [2, 3]]))
    torch.testing.assert_close(buffer.obs[2], torch.full((2, 4), 2.0))


def test_storage_dtypes_are_float32_and_obs_keeps_uint8():
    buffer = RolloutBuffer(2, 2, (1, 4, 4), np.uint8, CPU)
    assert buffer.obs.dtype == torch.uint8
    for name in ("log_probs", "values", "rewards", "dones", "advantages", "returns"):
        assert getattr(buffer, name).dtype == torch.float32, name
        assert getattr(buffer, name).shape == (2, 2), name
    frame = np.full((2, 1, 4, 4), 255, np.uint8)
    buffer.add(frame, np.zeros(2, np.int64), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    assert buffer.obs[0].max().item() == 255


def test_float64_observations_are_stored_as_float32():
    buffer = RolloutBuffer(1, 2, (3,), np.float64, CPU)
    assert buffer.obs.dtype == torch.float32
    buffer.add(
        np.full((2, 3), 0.25, np.float64),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        [0, 0],
    )
    torch.testing.assert_close(buffer.obs[0], torch.full((2, 3), 0.25))


def test_add_accepts_tensors_and_copies_its_inputs():
    buffer = RolloutBuffer(1, 2, (3,), np.float32, CPU)
    obs = np.ones((2, 3), np.float32)
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 0.25], requires_grad=True)
    buffer.add(
        obs, torch.tensor([1, 0]), torch.tensor([-0.1, -0.2]), values, rewards, torch.tensor([1, 0])
    )
    obs[:] = 7.0
    rewards[:] = 9.0
    torch.testing.assert_close(buffer.obs[0], torch.ones(2, 3))
    torch.testing.assert_close(buffer.rewards[0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(buffer.dones[0], torch.tensor([1.0, 0.0]))
    assert not buffer.values.requires_grad


def test_add_when_full_raises():
    buffer = make_filled()
    with pytest.raises(RuntimeError, match="full"):
        buffer.add(
            np.zeros((3, 2), np.float32),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        )


@pytest.mark.parametrize("field", ["obs", "actions", "log_probs", "values", "rewards", "dones"])
def test_add_rejects_wrong_shapes(field: str):
    """Shapes are checked explicitly: `copy_` would silently broadcast a (1,) into (n_envs,)."""
    buffer = RolloutBuffer(2, 3, (2,), np.float32, CPU)
    args = {
        "obs": np.zeros((3, 2), np.float32),
        "actions": np.zeros(3, np.int64),
        "log_probs": np.zeros(3, np.float32),
        "values": np.zeros(3, np.float32),
        "rewards": np.zeros(3, np.float32),
        "dones": np.zeros(3, bool),
    }
    args[field] = args[field][:1]
    with pytest.raises(ValueError, match=field):
        buffer.add(**args)
    assert len(buffer) == 0


def test_constructor_validates_sizes():
    with pytest.raises(ValueError, match="n_steps"):
        RolloutBuffer(0, 2, (4,), np.float32, CPU)
    with pytest.raises(ValueError, match="n_envs"):
        RolloutBuffer(2, 0, (4,), np.float32, CPU)


def test_reset_empties_the_buffer_and_invalidates_advantages():
    buffer = make_filled()
    buffer.reset()
    assert not buffer.full
    assert len(buffer) == 0
    with pytest.raises(RuntimeError):
        next(iter(buffer.minibatches(1)))


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #


def test_gae_matches_hand_computed_example_with_mid_rollout_and_final_done():
    """gamma = lambda = 0.5, so gamma * lambda = 0.25.

    env 0: r = [1, 2, 3, 4], V = [0.5, 1, 1.5, 2], done at t=1, bootstrap value 4
        t=3: delta = 4 + 0.5*4   - 2   = 4     A = 4
        t=2: delta = 3 + 0.5*2   - 1.5 = 2.5   A = 2.5 + 0.25*4 = 3.5
        t=1: delta = 2           - 1   = 1     A = 1              (done: no bootstrap, no carry)
        t=0: delta = 1 + 0.5*1   - 0.5 = 1     A = 1 + 0.25*1 = 1.25
    env 1: r = [0, -1, 2, 1], V = [1, 2, -1, 0.5], done at t=3, "bootstrap" value 10 is the
    value of a fresh episode's first obs and must be masked out
        t=3: delta = 1           - 0.5 = 0.5   A = 0.5
        t=2: delta = 2 + 0.5*0.5 + 1   = 3.25  A = 3.25 + 0.25*0.5 = 3.375
        t=1: delta = -1 - 0.5    - 2   = -3.5  A = -3.5 + 0.25*3.375 = -2.65625
        t=0: delta = 0 + 0.5*2   - 1   = 0     A = 0.25 * -2.65625 = -0.6640625
    """
    rewards = np.array([[1.0, 0.0], [2.0, -1.0], [3.0, 2.0], [4.0, 1.0]])
    values = np.array([[0.5, 1.0], [1.0, 2.0], [1.5, -1.0], [2.0, 0.5]])
    dones = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], dtype=bool)
    buffer = RolloutBuffer(4, 2, (1,), np.float32, CPU)
    fill(buffer, rewards, values, dones)

    buffer.compute_returns_and_advantages(
        last_values=torch.tensor([4.0, 10.0]),
        last_dones=np.array([False, True]),
        gamma=0.5,
        gae_lambda=0.5,
    )

    expected_adv = torch.tensor([[1.25, -0.6640625], [1.0, -2.65625], [3.5, 3.375], [4.0, 0.5]])
    expected_ret = torch.tensor([[1.75, 0.3359375], [2.0, -0.65625], [5.0, 2.375], [6.0, 1.0]])
    torch.testing.assert_close(buffer.advantages, expected_adv, atol=1e-6, rtol=0)
    torch.testing.assert_close(buffer.returns, expected_ret, atol=1e-6, rtol=0)


def reference_gae(rewards, values, dones, last_values, gamma, lam):
    """Forward-sum definition: A_t = sum_l (gamma*lam)^l delta_{t+l}, cut at the episode end."""
    n_steps, n_envs = rewards.shape
    next_values = np.concatenate([values[1:], last_values[None]], axis=0)
    deltas = rewards + gamma * next_values * (1.0 - dones) - values
    adv = np.zeros_like(rewards)
    for env in range(n_envs):
        for t in range(n_steps):
            weight = 1.0
            for k in range(t, n_steps):
                adv[t, env] += weight * deltas[k, env]
                if dones[k, env]:
                    break
                weight *= gamma * lam
    return adv


@pytest.mark.parametrize("gamma,lam", [(0.99, 0.95), (0.9, 0.0), (1.0, 1.0)])
def test_gae_matches_forward_sum_reference_on_random_rollouts(gamma: float, lam: float):
    rng = np.random.default_rng(3)
    n_steps, n_envs = 12, 4
    rewards = rng.normal(size=(n_steps, n_envs))
    values = rng.normal(size=(n_steps, n_envs))
    dones = rng.random((n_steps, n_envs)) < 0.25
    last_values = rng.normal(size=n_envs)
    buffer = RolloutBuffer(n_steps, n_envs, (1,), np.float32, CPU)
    fill(buffer, rewards, values, dones)

    buffer.compute_returns_and_advantages(last_values, dones[-1], gamma, lam)

    expected = reference_gae(rewards, values, dones.astype(np.float64), last_values, gamma, lam)
    np.testing.assert_allclose(buffer.advantages.numpy(), expected, atol=1e-5)
    np.testing.assert_allclose(buffer.returns.numpy(), expected + values, atol=1e-5)


def test_lambda_one_gives_discounted_monte_carlo_returns():
    rng = np.random.default_rng(11)
    n_steps, n_envs, gamma = 10, 3, 0.9
    rewards = rng.normal(size=(n_steps, n_envs))
    values = rng.normal(size=(n_steps, n_envs))
    dones = np.zeros((n_steps, n_envs), dtype=bool)
    dones[3, 0] = dones[7, 0] = True  # two episode ends, open tail (bootstrapped)
    dones[9, 1] = True  # ends exactly on the last step (bootstrap masked); env 2 never ends
    last_values = np.array([2.0, 50.0, -1.0])
    buffer = RolloutBuffer(n_steps, n_envs, (1,), np.float32, CPU)
    fill(buffer, rewards, values, dones)

    buffer.compute_returns_and_advantages(last_values, dones[-1], gamma, gae_lambda=1.0)

    mc = np.zeros_like(rewards)
    for env in range(n_envs):
        for t in range(n_steps):
            discount, total, ended = 1.0, 0.0, False
            for k in range(t, n_steps):
                total += discount * rewards[k, env]
                discount *= gamma
                if dones[k, env]:
                    ended = True
                    break
            mc[t, env] = total if ended else total + discount * last_values[env]
    np.testing.assert_allclose(buffer.returns.numpy(), mc, atol=1e-5)
    np.testing.assert_allclose(buffer.advantages.numpy(), mc - values, atol=1e-5)


def test_lambda_zero_gives_one_step_td_errors():
    rewards = np.array([[1.0], [1.0], [1.0]])
    values = np.array([[3.0], [2.0], [1.0]])
    dones = np.array([[0], [0], [1]], dtype=bool)
    buffer = RolloutBuffer(3, 1, (1,), np.float32, CPU)
    fill(buffer, rewards, values, dones)
    buffer.compute_returns_and_advantages(np.array([100.0]), dones[-1], 0.5, 0.0)
    torch.testing.assert_close(buffer.advantages, torch.tensor([[-1.0], [-0.5], [0.0]]))


def test_compute_requires_a_full_buffer_and_consistent_last_dones():
    buffer = RolloutBuffer(2, 2, (1,), np.float32, CPU)
    with pytest.raises(RuntimeError, match="full"):
        buffer.compute_returns_and_advantages(np.zeros(2), np.zeros(2, bool), 0.99, 0.95)
    fill(buffer, np.ones((2, 2)), np.zeros((2, 2)), np.array([[0, 0], [0, 1]], dtype=bool))
    with pytest.raises(ValueError, match="last_dones"):
        buffer.compute_returns_and_advantages(np.zeros(2), np.array([True, False]), 0.99, 0.95)
    with pytest.raises(ValueError, match="last_values"):
        buffer.compute_returns_and_advantages(np.zeros(3), np.array([False, True]), 0.99, 0.95)


def test_compute_does_not_track_gradients():
    buffer = RolloutBuffer(2, 1, (1,), np.float32, CPU)
    fill(buffer, np.ones((2, 1)), np.zeros((2, 1)), np.zeros((2, 1), dtype=bool))
    last_values = torch.tensor([1.0], requires_grad=True)
    buffer.compute_returns_and_advantages(last_values, np.array([False]), 0.99, 0.95)
    assert not buffer.advantages.requires_grad
    assert not buffer.returns.requires_grad


# --------------------------------------------------------------------------- #
# minibatches
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_minibatches", [1, 3, 4, 15])
def test_minibatches_partition_every_sample_exactly_once(n_minibatches: int):
    buffer = make_filled(n_steps=5, n_envs=3)  # 15 samples
    batches = list(buffer.minibatches(n_minibatches))
    assert len(batches) == n_minibatches
    sizes = [len(batch["actions"]) for batch in batches]
    assert sum(sizes) == 15
    assert max(sizes) - min(sizes) <= 1
    ids = torch.cat([batch["actions"] for batch in batches])
    assert sorted(ids.tolist()) == list(range(15))


def test_minibatch_rows_stay_aligned_across_fields():
    buffer = make_filled(n_steps=5, n_envs=3, obs_shape=(2, 2))
    flat_adv = buffer.advantages.reshape(-1)
    flat_ret = buffer.returns.reshape(-1)
    flat_val = buffer.values.reshape(-1)
    for batch in buffer.minibatches(4):
        assert set(batch) == MINIBATCH_KEYS
        ids = batch["actions"]
        size = len(ids)
        assert batch["obs"].shape == (size, 2, 2)
        for key in MINIBATCH_KEYS - {"obs"}:
            assert batch[key].shape == (size,), key
        torch.testing.assert_close(batch["obs"][:, 0, 0], ids.float())
        torch.testing.assert_close(batch["log_probs"], -0.5 * ids.float())
        torch.testing.assert_close(batch["advantages"], flat_adv[ids])
        torch.testing.assert_close(batch["returns"], flat_ret[ids])
        torch.testing.assert_close(batch["values"], flat_val[ids])


def test_minibatches_are_shuffled_anew_each_epoch():
    buffer = make_filled(n_steps=16, n_envs=4)
    torch.manual_seed(0)
    first = torch.cat([batch["actions"] for batch in buffer.minibatches(4)])
    second = torch.cat([batch["actions"] for batch in buffer.minibatches(4)])
    assert first.tolist() != list(range(64))
    assert first.tolist() != second.tolist()


def test_minibatch_order_is_reproducible_with_a_generator_or_the_global_seed():
    buffer = make_filled(n_steps=8, n_envs=4)

    def order(**kwargs) -> list[int]:
        return torch.cat([b["actions"] for b in buffer.minibatches(4, **kwargs)]).tolist()

    gen_a = torch.Generator().manual_seed(5)
    gen_b = torch.Generator().manual_seed(5)
    assert order(generator=gen_a) == order(generator=gen_b)

    torch.manual_seed(9)
    global_a = order()
    torch.manual_seed(9)
    assert order() == global_a


def test_explicit_generator_leaves_the_global_rng_untouched():
    buffer = make_filled()
    before = torch.get_rng_state()
    list(buffer.minibatches(3, generator=torch.Generator().manual_seed(1)))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("n_minibatches", [0, -1, 16])
def test_minibatches_rejects_bad_counts(n_minibatches: int):
    buffer = make_filled(n_steps=5, n_envs=3)
    with pytest.raises(ValueError, match="n_minibatches"):
        list(buffer.minibatches(n_minibatches))


def test_minibatches_require_computed_advantages():
    buffer = RolloutBuffer(2, 2, (1,), np.float32, CPU)
    fill(buffer, np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))
    with pytest.raises(RuntimeError, match="compute_returns_and_advantages"):
        list(buffer.minibatches(2))
