"""Observation wrappers (spec 4): `GrayscaleResize` and `FrameStack` spaces and outputs."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from mario_play.envs.mario_env import MarioEnv
from mario_play.envs.wrappers import FrameStack, GrayscaleResize

RIGHT = 1
WHITE, BLACK, RED, GREEN, BLUE = (
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
)


# ---------------------------------------------------------------------- GrayscaleResize


def test_default_is_84x84_grayscale():
    env = GrayscaleResize(MarioEnv("flat", obs_mode="pixels"))
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (84, 84) and space.dtype == np.uint8
    assert space.low.min() == 0 and space.high.max() == 255
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84) and obs.dtype == np.uint8 and space.contains(obs)
    obs, *_ = env.step(RIGHT)
    assert obs.shape == (84, 84) and obs.dtype == np.uint8 and space.contains(obs)
    assert obs.min() < obs.max()  # a picture, not a blank


@pytest.mark.parametrize(
    ("size", "grayscale", "shape"),
    [
        ((84, 84), True, (84, 84)),
        ((60, 80), True, (60, 80)),  # size is (H, W)
        (None, True, (240, 256)),
        ((84, 84), False, (3, 84, 84)),
        ((60, 80), False, (3, 60, 80)),
        (None, False, (3, 240, 256)),
        ([42, 42], True, (42, 42)),  # a list straight from a YAML config
    ],
)
def test_output_shapes(size, grayscale, shape):
    env = GrayscaleResize(MarioEnv("flat", obs_mode="pixels"), size=size, grayscale=grayscale)
    assert env.observation_space.shape == shape
    obs, _ = env.reset(seed=0)
    assert obs.shape == shape and obs.dtype == np.uint8
    assert env.observation_space.contains(obs)
    assert obs.flags.c_contiguous and obs.flags.writeable


def test_grayscale_uses_luma_weights(counter_env):
    colors = [WHITE, BLACK, RED, GREEN, BLUE]
    env = GrayscaleResize(counter_env(shape=(6, 8, 3), colors=colors), size=None)
    obs, _ = env.reset()
    seen = [int(obs[0, 0])]
    assert (obs == obs[0, 0]).all()
    for _ in range(4):
        obs, *_ = env.step(0)
        assert (obs == obs[0, 0]).all()
        seen.append(int(obs[0, 0]))
    assert seen[0] == 255 and seen[1] == 0
    for got, want in zip(seen[2:], (0.299 * 255, 0.587 * 255, 0.114 * 255), strict=True):
        assert abs(got - want) <= 1.5


def test_colour_mode_is_channel_first_rgb(counter_env):
    env = GrayscaleResize(
        counter_env(shape=(6, 8, 3), colors=[(10, 20, 30)]), size=None, grayscale=False
    )
    obs, _ = env.reset()
    assert obs.shape == (3, 6, 8)
    assert (obs[0] == 10).all() and (obs[1] == 20).all() and (obs[2] == 30).all()


def test_resize_keeps_the_picture_layout(counter_env):
    class HalfAndHalf(counter_env):  # top half white, bottom half black
        def _obs(self):
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            obs[: obs.shape[0] // 2] = 255
            return obs

    env = GrayscaleResize(HalfAndHalf(shape=(40, 80, 3)), size=(10, 20))
    obs, _ = env.reset()
    assert obs.shape == (10, 20)
    assert (obs[:4] == 255).all() and (obs[6:] == 0).all()

    colour = GrayscaleResize(HalfAndHalf(shape=(40, 80, 3)), size=(10, 20), grayscale=False)
    obs, _ = colour.reset()
    assert (obs[:, :4] == 255).all() and (obs[:, 6:] == 0).all()


def test_without_resize_the_frame_is_only_converted():
    env = MarioEnv("flat", obs_mode="pixels", hud=False)
    wrapped = GrayscaleResize(env, size=None, grayscale=False)
    obs, _ = wrapped.reset(seed=0)
    np.testing.assert_array_equal(obs, env.render_frame().transpose(2, 0, 1))


def test_outputs_are_fresh_arrays():
    env = GrayscaleResize(MarioEnv("flat", obs_mode="pixels"))
    first, _ = env.reset(seed=0)
    snapshot = first.copy()
    second, *_ = env.step(RIGHT)
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, snapshot)


@pytest.mark.parametrize("size", [(84,), (0, 84), (84, -1), (84, 84, 3), "big"])
def test_bad_size_raises_value_error(size):
    with pytest.raises(ValueError):
        GrayscaleResize(MarioEnv("flat", obs_mode="pixels"), size=size)


def test_needs_an_rgb_image_observation(counter_env):
    with pytest.raises(ValueError):
        GrayscaleResize(MarioEnv("flat", obs_mode="grid"))
    with pytest.raises(ValueError):
        GrayscaleResize(counter_env(shape=(6, 8)))
    with pytest.raises(ValueError):
        GrayscaleResize(counter_env(shape=(6, 8, 3), dtype=np.float32, high=1.0))


# --------------------------------------------------------------------------- FrameStack


def test_stacks_2d_frames_on_a_new_leading_axis(counter_env):
    env = FrameStack(counter_env(shape=(5, 6)), 4)
    space = env.observation_space
    assert space.shape == (4, 5, 6) and space.dtype == np.uint8
    assert space.low.min() == 0 and space.high.max() == 255

    obs, _ = env.reset()
    assert obs.shape == (4, 5, 6) and obs.dtype == np.uint8
    assert [int(f[0, 0]) for f in obs] == [0, 0, 0, 0]  # the first frame, repeated
    history = []
    for _ in range(6):
        obs, reward, terminated, truncated, _ = env.step(0)
        assert reward == 1.0 and not terminated and not truncated
        assert all((frame == frame[0, 0]).all() for frame in obs)
        history.append([int(f[0, 0]) for f in obs])
    assert history == [
        [0, 0, 0, 1],
        [0, 0, 1, 2],
        [0, 1, 2, 3],
        [1, 2, 3, 4],  # oldest first, newest last
        [2, 3, 4, 5],
        [3, 4, 5, 6],
    ]


def test_concatenates_channel_first_frames_along_axis_0(counter_env):
    env = FrameStack(counter_env(shape=(3, 5, 6)), 2)
    assert env.observation_space.shape == (6, 5, 6)
    obs, _ = env.reset()
    assert obs.shape == (6, 5, 6) and (obs == 0).all()
    env.step(0)
    obs, *_ = env.step(0)
    assert [int(c[0, 0]) for c in obs] == [1, 1, 1, 2, 2, 2]


def test_reset_clears_the_history(counter_env):
    env = FrameStack(counter_env(shape=(5, 6)), 3)
    env.reset()
    for _ in range(5):
        env.step(0)
    obs, _ = env.reset()
    assert (obs == 0).all()
    obs, *_ = env.step(0)
    assert [int(f[0, 0]) for f in obs] == [0, 0, 1]


def test_a_stack_of_one_still_adds_the_channel_axis(counter_env):
    env = FrameStack(counter_env(shape=(5, 6)), 1)
    assert env.observation_space.shape == (1, 5, 6)
    obs, _ = env.reset()
    assert obs.shape == (1, 5, 6)
    unchanged = FrameStack(counter_env(shape=(3, 5, 6)), 1)
    assert unchanged.observation_space.shape == (3, 5, 6)


def test_float_bounds_and_dtype_are_preserved():
    env = FrameStack(MarioEnv("flat", obs_mode="grid"), 2)
    space = env.observation_space
    assert space.shape == (28, 15, 16) and space.dtype == np.float32
    assert (space.low == -1.0).all() and (space.high == 1.0).all()
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32 and space.contains(obs)
    np.testing.assert_array_equal(obs[:14], obs[14:])
    obs, *_ = env.step(RIGHT)
    assert space.contains(obs)
    assert not np.array_equal(obs[:14], obs[14:])  # the player moved between the two frames
    np.testing.assert_array_equal(obs[14:], env.unwrapped.observe())


def test_per_element_bounds_are_tiled(counter_env):
    inner = counter_env(shape=(2, 3))
    inner.observation_space = gym.spaces.Box(
        low=np.arange(6, dtype=np.float32).reshape(2, 3) - 10,
        high=np.arange(6, dtype=np.float32).reshape(2, 3),
        dtype=np.float32,
    )
    space = FrameStack(inner, 3).observation_space
    assert space.shape == (3, 2, 3)
    for k in range(3):
        np.testing.assert_array_equal(space.high[k], inner.observation_space.high)
        np.testing.assert_array_equal(space.low[k], inner.observation_space.low)


def test_stacked_observations_are_fresh_arrays(counter_env):
    env = FrameStack(counter_env(shape=(5, 6)), 3)
    obs, _ = env.reset()
    kept = [(obs, obs.copy())]
    for _ in range(5):
        obs, *_ = env.step(0)
        kept.append((obs, obs.copy()))
    for live, snapshot in kept:
        np.testing.assert_array_equal(live, snapshot)
    kept[-1][0][:] = 99  # scribbling on a returned stack must not corrupt the next one
    obs, *_ = env.step(0)
    assert [int(f[0, 0]) for f in obs] == [4, 5, 6]


@pytest.mark.parametrize("k", [0, -2, 1.5, "4"])
def test_bad_stack_size_raises_value_error(counter_env, k):
    with pytest.raises(ValueError):
        FrameStack(counter_env(shape=(5, 6)), k)


def test_unsupported_observation_rank_raises_value_error(counter_env):
    with pytest.raises(ValueError):
        FrameStack(counter_env(shape=(4,)), 2)
    with pytest.raises(ValueError):
        FrameStack(counter_env(shape=(2, 3, 4, 5)), 2)
    discrete = counter_env(shape=(5, 6))
    discrete.observation_space = gym.spaces.Discrete(3)
    with pytest.raises(ValueError):
        FrameStack(discrete, 2)


def test_the_pixel_pipeline_composes():
    env = FrameStack(GrayscaleResize(MarioEnv("flat", obs_mode="pixels")), 4)
    assert env.observation_space.shape == (4, 84, 84)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert all(np.array_equal(obs[0], frame) for frame in obs[1:])
    for _ in range(3):
        obs, *_ = env.step(3)
    assert env.observation_space.contains(obs)
    assert not np.array_equal(obs[0], obs[3])
