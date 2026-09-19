"""ReplayBuffer: ring storage, dtype preservation, uniform sampling and input validation."""

import numpy as np
import pytest
import torch

from mario_play.rl.buffers import replay
from mario_play.rl.buffers.replay import ReplayBuffer

CPU = torch.device("cpu")
OBS_SHAPE = (3,)


def transitions(ids, obs_shape=OBS_SHAPE, dtype=np.float32):
    """Transitions that are recognisable by their id `k`.

    obs is filled with `k`, next_obs with `k + 0.5` (float) or `k + 100` (integer dtypes),
    action is `k`, reward `10 * k`, and odd ids are terminal.
    """
    ids = np.asarray(ids, dtype=np.int64)
    shape = (len(ids), *obs_shape)
    grid = ids.reshape(-1, *([1] * len(obs_shape)))
    successor_offset = 0.5 if np.issubdtype(dtype, np.floating) else 100
    obs = np.broadcast_to(grid, shape).astype(dtype)
    next_obs = np.broadcast_to(grid + successor_offset, shape).astype(dtype)
    rewards = (10.0 * ids).astype(np.float32)
    terminated = ids % 2 == 1
    return obs, ids.copy(), rewards, next_obs, terminated


def make_buffer(capacity=5, obs_shape=OBS_SHAPE, dtype=np.float32):
    return ReplayBuffer(capacity, obs_shape, dtype, CPU)


def stored_ids(buffer):
    """Ids currently held, in storage order (slot 0 first)."""
    return buffer.actions[: len(buffer)].tolist()


# --------------------------------------------------------------------------- #
# length and ring order
# --------------------------------------------------------------------------- #


def test_starts_empty_and_length_grows_up_to_capacity():
    buffer = make_buffer(capacity=5)
    assert len(buffer) == 0
    assert not buffer.full
    buffer.add_batch(*transitions([0, 1]))
    assert len(buffer) == 2
    buffer.add_batch(*transitions([2, 3]))
    assert len(buffer) == 4
    assert not buffer.full
    buffer.add_batch(*transitions([4, 5]))
    assert len(buffer) == 5
    assert buffer.full
    buffer.add_batch(*transitions([6, 7]))
    assert len(buffer) == 5
    assert buffer.capacity == 5


def test_ring_overwrites_the_oldest_transitions_first():
    buffer = make_buffer(capacity=5)
    for start in (0, 2, 4):  # the third batch wraps: id 4 -> slot 4, id 5 -> slot 0
        buffer.add_batch(*transitions([start, start + 1]))
    assert stored_ids(buffer) == [5, 1, 2, 3, 4]
    assert buffer.position == 1

    buffer.add_batch(*transitions([6, 7]))
    assert stored_ids(buffer) == [5, 6, 7, 3, 4]
    assert buffer.position == 3


def test_every_field_of_a_transition_lands_in_the_same_slot():
    buffer = make_buffer(capacity=5)
    for start in (0, 2, 4, 6):
        buffer.add_batch(*transitions([start, start + 1]))
    for slot, k in enumerate(stored_ids(buffer)):
        np.testing.assert_array_equal(buffer.obs[slot], np.full(OBS_SHAPE, k, np.float32))
        np.testing.assert_array_equal(buffer.next_obs[slot], np.full(OBS_SHAPE, k + 0.5))
        assert buffer.rewards[slot] == 10.0 * k
        assert bool(buffer.terminated[slot]) == (k % 2 == 1)


def test_a_batch_larger_than_the_capacity_keeps_only_its_newest_transitions():
    buffer = make_buffer(capacity=4)
    buffer.add_batch(*transitions([0]))
    buffer.add_batch(*transitions(range(1, 8)))  # 7 transitions into 4 slots
    assert len(buffer) == 4
    assert sorted(stored_ids(buffer)) == [4, 5, 6, 7]
    buffer.add_batch(*transitions([8]))  # the next write must replace the oldest survivor (4)
    assert sorted(stored_ids(buffer)) == [5, 6, 7, 8]


def test_single_transition_batches_and_capacity_one():
    buffer = make_buffer(capacity=1)
    for k in range(3):
        buffer.add_batch(*transitions([k]))
        assert stored_ids(buffer) == [k]
    batch = buffer.sample(4, np.random.default_rng(0))
    assert batch["actions"].tolist() == [2, 2, 2, 2]


def test_clear_empties_the_buffer():
    buffer = make_buffer(capacity=5)
    buffer.add_batch(*transitions([0, 1, 2]))
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.position == 0
    with pytest.raises(ValueError, match="empty"):
        buffer.sample(1, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# dtypes and memory
# --------------------------------------------------------------------------- #


def test_uint8_frames_are_stored_and_sampled_as_uint8():
    shape = (4, 8, 8)
    buffer = make_buffer(capacity=6, obs_shape=shape, dtype=np.uint8)
    buffer.add_batch(*transitions([1, 2, 3], obs_shape=shape, dtype=np.uint8))
    assert buffer.obs.dtype == np.uint8
    assert buffer.next_obs.dtype == np.uint8
    assert buffer.obs.shape == (6, *shape)

    batch = buffer.sample(5, np.random.default_rng(0))
    assert batch["obs"].dtype == torch.uint8
    assert batch["next_obs"].dtype == torch.uint8
    assert batch["obs"].shape == (5, *shape)
    # values survive untouched (no scaling, no float round trip)
    ids = batch["actions"]
    assert torch.equal(batch["obs"][:, 0, 0, 0].long(), ids)
    assert torch.equal(batch["next_obs"][:, 0, 0, 0].long(), ids + 100)


def test_float32_observations_stay_float32():
    buffer = make_buffer(dtype=np.float32)
    buffer.add_batch(*transitions([0, 1]))
    assert buffer.obs.dtype == np.float32
    assert buffer.sample(2, np.random.default_rng(0))["obs"].dtype == torch.float32


def test_float64_spaces_are_stored_as_float32():
    # All tensors are float32 (MPS has no float64); storing doubles would also waste memory.
    buffer = make_buffer(dtype=np.float64)
    assert buffer.obs_dtype == np.float32
    buffer.add_batch(*transitions([0, 1], dtype=np.float64))
    assert buffer.obs.dtype == np.float32
    assert buffer.sample(2, np.random.default_rng(0))["obs"].dtype == torch.float32


def test_lossy_observation_casts_are_rejected():
    buffer = make_buffer(obs_shape=(2, 2), dtype=np.uint8)
    obs, actions, rewards, next_obs, terminated = transitions([1], obs_shape=(2, 2))
    with pytest.raises(TypeError, match="uint8"):
        buffer.add_batch(obs, actions, rewards, next_obs, terminated)  # float32 into uint8
    with pytest.raises(TypeError, match="uint8"):
        buffer.add_batch(obs.astype(np.uint8), actions, rewards, next_obs, terminated)
    assert len(buffer) == 0


def test_nbytes_matches_the_documented_pixel_budget():
    # 100k transitions of (4, 84, 84) uint8 frames: ~2.8 GB for obs plus the same for next_obs.
    estimate = ReplayBuffer.estimate_nbytes(100_000, (4, 84, 84), np.uint8)
    frames = 2 * 100_000 * 4 * 84 * 84
    per_transition_scalars = 8 + 4 + 1  # int64 action, float32 reward, bool terminated
    assert estimate == frames + 100_000 * per_transition_scalars
    assert 5.6e9 < estimate < 5.7e9

    small = make_buffer(capacity=7, obs_shape=(2, 3), dtype=np.float32)
    assert small.nbytes == ReplayBuffer.estimate_nbytes(7, (2, 3), np.float32)
    assert small.nbytes == sum(
        a.nbytes
        for a in (small.obs, small.next_obs, small.actions, small.rewards, small.terminated)
    )


def test_warns_when_the_buffer_cannot_fit_in_physical_memory(monkeypatch):
    monkeypatch.setattr(replay, "_physical_memory_bytes", lambda: 1000)
    with pytest.warns(UserWarning, match="buffer_size"):
        ReplayBuffer(100, (10,), np.float32, CPU)  # 8 kB of observations vs 1 kB of "RAM"


def test_no_memory_warning_for_a_buffer_that_fits(monkeypatch, recwarn):
    monkeypatch.setattr(replay, "_physical_memory_bytes", lambda: 10**9)
    ReplayBuffer(100, (10,), np.float32, CPU)
    monkeypatch.setattr(replay, "_physical_memory_bytes", lambda: None)  # unknown: stay quiet
    ReplayBuffer(100, (10,), np.float32, CPU)
    assert not [w for w in recwarn if "buffer_size" in str(w.message)]


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #


def test_sample_returns_tensors_with_the_documented_keys_shapes_dtypes_and_device():
    buffer = make_buffer(capacity=10)
    buffer.add_batch(*transitions(range(6)))
    batch = buffer.sample(4, np.random.default_rng(0))

    assert set(batch) == {"obs", "actions", "rewards", "next_obs", "terminated"}
    assert batch["obs"].shape == (4, *OBS_SHAPE)
    assert batch["next_obs"].shape == (4, *OBS_SHAPE)
    for key in ("actions", "rewards", "terminated"):
        assert batch[key].shape == (4,)
    assert batch["actions"].dtype == torch.int64
    assert batch["rewards"].dtype == torch.float32
    assert batch["terminated"].dtype == torch.float32
    assert all(tensor.device == CPU for tensor in batch.values())


def test_sampled_rows_are_whole_transitions():
    buffer = make_buffer(capacity=8)
    for start in range(0, 12, 3):  # wraps around
        buffer.add_batch(*transitions(range(start, start + 3)))
    batch = buffer.sample(64, np.random.default_rng(1))
    ids = batch["actions"].float()
    assert torch.equal(batch["obs"], ids[:, None].expand(-1, OBS_SHAPE[0]))
    assert torch.equal(batch["next_obs"], (ids + 0.5)[:, None].expand(-1, OBS_SHAPE[0]))
    assert torch.equal(batch["rewards"], 10.0 * ids)
    assert torch.equal(batch["terminated"], (batch["actions"] % 2 == 1).float())


def test_sampling_never_returns_unwritten_slots_and_covers_every_stored_transition():
    buffer = make_buffer(capacity=100)
    buffer.add_batch(*transitions(range(1, 8)))  # ids 1..7; untouched slots would read as id 0
    batch = buffer.sample(2000, np.random.default_rng(2))
    ids, counts = np.unique(batch["actions"].numpy(), return_counts=True)
    assert ids.tolist() == list(range(1, 8))
    expected = 2000 / 7
    assert np.all(np.abs(counts - expected) < 0.35 * expected)  # uniform, within sampling noise


def test_sampling_is_uniform_over_a_full_wrapped_buffer():
    buffer = make_buffer(capacity=5)
    buffer.add_batch(*transitions(range(8)))
    ids = np.unique(buffer.sample(500, np.random.default_rng(3))["actions"].numpy())
    assert ids.tolist() == [3, 4, 5, 6, 7]


def test_sampling_is_reproducible_for_a_given_generator_state():
    buffer = make_buffer(capacity=50)
    buffer.add_batch(*transitions(range(40)))
    first = buffer.sample(16, np.random.default_rng(7))
    second = buffer.sample(16, np.random.default_rng(7))
    other = buffer.sample(16, np.random.default_rng(8))
    assert torch.equal(first["actions"], second["actions"])
    assert not torch.equal(first["actions"], other["actions"])

    generator = np.random.default_rng(7)  # consecutive draws from one generator differ
    a, b = buffer.sample(16, generator), buffer.sample(16, generator)
    assert not torch.equal(a["actions"], b["actions"])


def test_batch_may_exceed_the_number_of_stored_transitions():
    buffer = make_buffer(capacity=10)
    buffer.add_batch(*transitions([3, 4]))
    batch = buffer.sample(32, np.random.default_rng(0))
    assert batch["obs"].shape == (32, *OBS_SHAPE)
    assert set(batch["actions"].tolist()) == {3, 4}


def test_buffer_copies_its_inputs_and_samples_are_independent_of_storage():
    buffer = make_buffer(capacity=4)
    obs, actions, rewards, next_obs, terminated = transitions([1, 2])
    obs, next_obs = obs.copy(), next_obs.copy()  # vec envs reuse their observation arrays
    buffer.add_batch(obs, actions, rewards, next_obs, terminated)
    obs[:] = -1.0
    next_obs[:] = -1.0
    assert buffer.obs[:2].min() == 1.0
    assert buffer.next_obs[:2].min() == 1.5

    batch = buffer.sample(8, np.random.default_rng(0))
    batch["obs"].zero_()
    assert buffer.obs[:2].min() == 1.0


def test_accepts_lists_and_other_numeric_dtypes_for_the_scalar_fields():
    buffer = make_buffer(capacity=4)
    obs, _, _, next_obs, _ = transitions([1, 2])
    buffer.add_batch(obs, [1, 0], [0.5, 2], next_obs, [True, False])
    buffer.add_batch(obs, np.array([1, 0], np.int32), np.array([1.0, 2.0]), next_obs, [0, 1])
    assert buffer.actions.dtype == np.int64
    assert buffer.rewards.dtype == np.float32
    assert buffer.rewards[:4].tolist() == [0.5, 2.0, 1.0, 2.0]
    assert buffer.terminated[:4].tolist() == [True, False, False, True]


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("capacity", [0, -3])
def test_capacity_must_be_positive(capacity):
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity, OBS_SHAPE, np.float32, CPU)


def test_sampling_an_empty_buffer_raises():
    with pytest.raises(ValueError, match="empty"):
        make_buffer().sample(4, np.random.default_rng(0))


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(batch_size):
    buffer = make_buffer()
    buffer.add_batch(*transitions([0]))
    with pytest.raises(ValueError, match="batch_size"):
        buffer.sample(batch_size, np.random.default_rng(0))


def test_wrong_observation_shape_is_rejected():
    buffer = make_buffer()
    obs, actions, rewards, next_obs, terminated = transitions([0, 1], obs_shape=(4,))
    with pytest.raises(ValueError, match="obs"):
        buffer.add_batch(obs, actions, rewards, next_obs, terminated)
    good_obs = transitions([0, 1])[0]
    with pytest.raises(ValueError, match="next_obs"):
        buffer.add_batch(good_obs, actions, rewards, next_obs, terminated)
    with pytest.raises(ValueError, match="obs"):  # a single unbatched observation
        buffer.add_batch(good_obs[0], actions[:1], rewards[:1], good_obs[0], terminated[:1])
    assert len(buffer) == 0


@pytest.mark.parametrize("field", ["actions", "rewards", "terminated"])
def test_mismatched_batch_lengths_are_rejected(field):
    buffer = make_buffer()
    names = ("obs", "actions", "rewards", "next_obs", "terminated")
    batch = dict(zip(names, transitions([0, 1, 2]), strict=True))
    batch[field] = batch[field][:2]
    with pytest.raises(ValueError, match=field):
        buffer.add_batch(**batch)
    assert len(buffer) == 0


def test_empty_batches_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        make_buffer().add_batch(*transitions([]))
