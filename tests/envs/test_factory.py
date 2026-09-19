"""`make_env` (contract code) against the real env and wrappers: spaces, wrappers, stats."""

from __future__ import annotations

import pickle
from functools import partial

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers import RecordEpisodeStatistics

from mario_play.envs.factory import make_env
from mario_play.envs.mario_env import MarioEnv
from mario_play.rl.config import EnvConfig

NOOP, RIGHT = 0, 1


def test_grid_env():
    env = make_env(EnvConfig(level="flat", obs_mode="grid"), seed=0)
    assert isinstance(env, RecordEpisodeStatistics)
    assert isinstance(env.unwrapped, MarioEnv)
    assert env.observation_space.shape == (14, 15, 16)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.n == 7
    obs, info = env.reset(seed=0)
    assert obs.shape == (14, 15, 16) and obs.dtype == np.float32 and info["level"] == "flat"
    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert env.observation_space.contains(obs) and isinstance(reward, float)
    env.close()


def test_default_config_is_grid_on_1_1():
    env = make_env(EnvConfig())
    assert env.observation_space.shape == (14, 15, 16)
    assert env.reset(seed=0)[1]["level"] == "1-1"
    env.close()


def test_grid_env_with_frame_stack():
    env = make_env(EnvConfig(level="flat", obs_mode="grid", frame_stack=3))
    assert env.observation_space.shape == (42, 15, 16)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (42, 15, 16) and obs.dtype == np.float32
    env.close()


def test_pixel_env_is_4x84x84_uint8():
    env = make_env(EnvConfig(level="flat", obs_mode="pixels", frame_stack=4), seed=0)
    assert env.observation_space.shape == (4, 84, 84)
    assert env.observation_space.dtype == np.uint8
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    obs, *_ = env.step(RIGHT)
    assert obs.shape == (4, 84, 84) and env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize(
    ("overrides", "shape"),
    [
        ({"frame_stack": 1}, (1, 84, 84)),  # pixels are always channel-first
        ({"frame_stack": 2, "grayscale": False}, (6, 84, 84)),
        ({"frame_stack": 2, "resize": None}, (2, 240, 256)),
        ({"frame_stack": 1, "resize": [60, 80], "grayscale": False}, (3, 60, 80)),
    ],
)
def test_pixel_env_variants(overrides, shape):
    env = make_env(EnvConfig(level="flat", obs_mode="pixels", **overrides))
    assert env.observation_space.shape == shape
    obs, _ = env.reset(seed=0)
    assert obs.shape == shape and obs.dtype == np.uint8
    env.close()


def test_other_ids_go_through_gym_make():
    env = make_env(EnvConfig(id="CartPole-v1"), seed=0)
    assert isinstance(env, RecordEpisodeStatistics)
    assert env.observation_space.shape == (4,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)
    env.close()


def test_env_options_reach_the_mario_env(level_path):
    cfg = EnvConfig(
        level=[level_path("short"), level_path("pit")],
        obs_mode="grid",
        action_set="complex",
        frame_skip=2,
        stall_steps=9,
        reward={"flag_bonus": 5.0, "clip": 1.0},
    )
    env = make_env(cfg, render_mode="rgb_array")
    base = env.unwrapped
    assert base.action_space.n == 10 and base.frame_skip == 2
    assert base.reward_config.flag_bonus == 5.0 and base.reward_config.clip == 1.0
    assert base.reward_config.death_penalty == -15.0
    assert base.render_mode == "rgb_array"
    levels = {env.reset(seed=s)[1]["level"] for s in range(12)}
    assert levels == {"short", "pit"}
    assert env.render().shape == (240, 256, 3)
    env.close()


def test_unknown_reward_key_in_the_config_raises():
    with pytest.raises(ValueError, match="bogus"):
        make_env(EnvConfig(level="flat", reward={"bogus": 1.0}))


def test_episode_statistics_at_the_end_of_an_episode(level_path):
    env = make_env(EnvConfig(level=level_path("pit"), obs_mode="grid"))
    env.reset(seed=0)
    rewards = []
    terminated = False
    while not terminated:
        _, reward, terminated, truncated, info = env.step(RIGHT)
        rewards.append(reward)
        assert not truncated
        if not terminated:
            assert "episode" not in info
    assert set(info["episode"]) >= {"r", "l", "t"}
    assert float(info["episode"]["r"]) == pytest.approx(sum(rewards), abs=1e-5)
    assert int(info["episode"]["l"]) == len(rewards)
    assert info["death_cause"] == "pit" and info["flag_get"] is False
    env.close()


def test_max_episode_steps_truncates_with_statistics():
    env = make_env(EnvConfig(level="flat", obs_mode="pixels", max_episode_steps=6))
    env.reset(seed=0)
    for step in range(1, 7):
        obs, _, terminated, truncated, info = env.step(NOOP)
        assert not terminated
        assert truncated == (step == 6)
    assert int(info["episode"]["l"]) == 6
    assert obs.shape == (1, 84, 84)  # default frame_stack of the config is 1
    env.close()


def test_stall_truncation_carries_statistics():
    env = make_env(EnvConfig(level="flat", obs_mode="grid", stall_steps=4))
    env.reset(seed=0)
    for _ in range(3):
        assert not env.step(NOOP)[3]
    _, _, terminated, truncated, info = env.step(NOOP)
    assert truncated and not terminated and int(info["episode"]["l"]) == 4
    env.close()


def test_seed_seeds_the_action_space():
    first = make_env(EnvConfig(level="flat"), seed=42)
    second = make_env(EnvConfig(level="flat"), seed=42)
    assert [first.action_space.sample() for _ in range(20)] == [
        second.action_space.sample() for _ in range(20)
    ]


def test_factory_partial_is_picklable_for_subprocess_workers():
    fn = pickle.loads(pickle.dumps(partial(make_env, EnvConfig(level="flat"), seed=3)))
    env = fn()
    assert isinstance(env.unwrapped, MarioEnv)
    env.close()


def test_whole_episode_through_the_factory_reaches_the_flag(level_path):
    env = make_env(EnvConfig(level=level_path("short"), obs_mode="pixels", frame_stack=4))
    env.reset(seed=0)
    done = False
    total = 0.0
    while not done:
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        total += reward
        done = terminated or truncated
    assert info["flag_get"] is True and info["progress"] == 1.0
    assert float(info["episode"]["r"]) == pytest.approx(total, abs=1e-4)
    assert isinstance(env.unwrapped, gym.Env)
    env.close()
