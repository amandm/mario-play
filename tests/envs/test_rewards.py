"""Reward shaping (spec 4): every component on its own, clipping, and exact accounting
of whole episodes against the game state."""

from __future__ import annotations

import dataclasses
import math

import pytest

from mario_play.envs.mario_env import MarioEnv
from mario_play.envs.rewards import RewardConfig, compute_reward, make_reward_config
from mario_play.game.engine import StepEvents

RIGHT = 1  # index in every action set
RUN_RIGHT = 3
NOOP = 0

ZERO = dict(
    progress_weight=0.0,
    time_penalty=0.0,
    death_penalty=0.0,
    flag_bonus=0.0,
    coin_bonus=0.0,
    score_weight=0.0,
)


def only(**overrides: float) -> dict:
    """Reward overrides with every component switched off except the given ones."""
    return {**ZERO, **overrides}


def play(env: MarioEnv, action: int, seed: int = 0, max_steps: int = 2_000):
    """Repeat `action` to the end of the episode: (rewards, last info, terminated, truncated)."""
    env.reset(seed=seed)
    rewards: list[float] = []
    for _ in range(max_steps):
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            return rewards, info, terminated, truncated
    raise AssertionError("episode did not end")


# --------------------------------------------------------------------------------- config


def test_defaults_match_the_spec():
    cfg = RewardConfig()
    assert cfg.progress_weight == 1 / 16
    assert cfg.time_penalty == -0.01
    assert cfg.death_penalty == -15.0
    assert cfg.flag_bonus == 50.0
    assert cfg.coin_bonus == 0.0
    assert cfg.score_weight == 0.0
    assert cfg.clip is None
    assert dataclasses.is_dataclass(cfg)


def test_make_reward_config_accepts_none_dict_and_instance():
    assert make_reward_config(None) == RewardConfig()
    assert make_reward_config({}) == RewardConfig()
    cfg = make_reward_config({"flag_bonus": 5, "clip": 1})
    assert cfg.flag_bonus == 5.0 and cfg.clip == 1.0
    assert cfg.death_penalty == -15.0  # untouched fields keep their defaults
    mine = RewardConfig(coin_bonus=2.0)
    copy = make_reward_config(mine)
    assert copy == mine and copy is not mine


def test_unknown_reward_key_raises_value_error_naming_it():
    with pytest.raises(ValueError, match="flag_bonuz"):
        make_reward_config({"flag_bonuz": 1.0})


def test_values_are_coerced_to_float_and_bad_values_rejected():
    # YAML 1.1 reads `1e-2` as a string: the config must still end up numeric.
    cfg = make_reward_config({"time_penalty": "-1e-2", "clip": None})
    assert cfg.time_penalty == -0.01 and cfg.clip is None
    assert isinstance(make_reward_config({"flag_bonus": 5}).flag_bonus, float)
    for bad in ({"flag_bonus": "lots"}, {"flag_bonus": None}, {"coin_bonus": True}):
        with pytest.raises(ValueError):
            make_reward_config(bad)
    for bad_clip in (0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            make_reward_config({"clip": bad_clip})
    with pytest.raises(ValueError):
        make_reward_config({"progress_weight": float("inf")})
    with pytest.raises(TypeError):
        make_reward_config("big")  # type: ignore[arg-type]


# ------------------------------------------------------------------------ compute_reward


def test_progress_is_one_per_tile_moved_right_and_negative_to_the_left():
    cfg = RewardConfig(time_penalty=0.0)
    assert compute_reward(cfg, 16.0, StepEvents()) == pytest.approx(1.0)
    assert compute_reward(cfg, -8.0, StepEvents()) == pytest.approx(-0.5)
    assert compute_reward(cfg, 0.0, StepEvents()) == 0.0


def test_time_penalty_is_paid_once_per_call():
    assert compute_reward(RewardConfig(), 0.0, StepEvents()) == pytest.approx(-0.01)


def test_death_flag_coin_and_score_components():
    died = StepEvents(died=True, death_cause="pit")
    won = StepEvents(won=True, score_delta=1500)
    loot = StepEvents(coins=3, score_delta=600)
    assert compute_reward(RewardConfig(**only(death_penalty=-15.0)), 0.0, died) == -15.0
    assert compute_reward(RewardConfig(**only(flag_bonus=50.0)), 0.0, won) == 50.0
    assert compute_reward(RewardConfig(**only(coin_bonus=0.5)), 0.0, loot) == 1.5
    assert compute_reward(RewardConfig(**only(score_weight=0.01)), 0.0, loot) == pytest.approx(6.0)
    # Defaults ignore coins and score altogether.
    assert compute_reward(RewardConfig(time_penalty=0.0), 0.0, loot) == 0.0


def test_components_add_up():
    cfg = RewardConfig(coin_bonus=1.0, score_weight=0.001)
    events = StepEvents(coins=2, won=True, score_delta=400)
    expected = 8.0 / 16 - 0.01 + 50.0 + 2.0 + 0.4
    assert compute_reward(cfg, 8.0, events) == pytest.approx(expected)


def test_clip_is_symmetric_and_applies_to_the_total():
    cfg = RewardConfig(clip=1.0)
    assert compute_reward(cfg, 0.0, StepEvents(won=True)) == 1.0
    assert compute_reward(cfg, 0.0, StepEvents(died=True)) == -1.0
    assert compute_reward(cfg, 4.0, StepEvents()) == pytest.approx(0.25 - 0.01)


def test_reward_is_a_python_float():
    assert type(compute_reward(RewardConfig(), 3.0, StepEvents(coins=1))) is float


# --------------------------------------------------------------- episode accounting (env)


def test_progress_sums_to_the_distance_covered(level_path):
    env = MarioEnv(level_path("short"), obs_mode="grid", reward=only(progress_weight=1.0))
    env.reset(seed=0)
    x_start = env.game.player.x
    rewards, info, terminated, _ = play(env, RUN_RIGHT)
    assert terminated and info["flag_get"]
    assert math.fsum(rewards) == pytest.approx(env.game.player.x - x_start, abs=1e-9)
    assert info["x_pos"] == env.game.player.x


def test_time_penalty_sums_to_the_number_of_env_steps(level_path):
    env = MarioEnv(level_path("short"), obs_mode="grid", reward=only(time_penalty=-0.01))
    rewards, _, _, _ = play(env, RUN_RIGHT)
    assert rewards == [-0.01] * len(rewards)


def test_death_penalty_is_paid_exactly_once_on_the_terminal_step(level_path):
    env = MarioEnv(level_path("pit"), obs_mode="grid", reward=only(death_penalty=-15.0))
    rewards, info, terminated, _ = play(env, RIGHT)
    assert terminated and info["death_cause"] == "pit" and not info["flag_get"]
    assert rewards[-1] == -15.0
    assert all(r == 0.0 for r in rewards[:-1])


def test_flag_bonus_is_paid_exactly_once_on_the_terminal_step(level_path):
    env = MarioEnv(level_path("short"), obs_mode="grid", reward=only(flag_bonus=50.0))
    rewards, info, terminated, _ = play(env, RIGHT)
    assert terminated and info["flag_get"] and info["death_cause"] is None
    assert rewards[-1] == 50.0
    assert all(r == 0.0 for r in rewards[:-1])


def test_coin_bonus_sums_to_the_coins_collected(level_path):
    env = MarioEnv(level_path("coins"), obs_mode="grid", reward=only(coin_bonus=1.0))
    rewards, info, _, _ = play(env, RIGHT)
    assert info["coins"] == 3
    assert math.fsum(rewards) == 3.0


def test_score_weight_sums_to_the_final_score_including_the_flag(level_path):
    env = MarioEnv(level_path("coins"), obs_mode="grid", reward=only(score_weight=1.0))
    rewards, info, _, _ = play(env, RIGHT)
    assert info["flag_get"]
    assert info["score"] == 3 * 200 + 1000 + 10 * info["time_left"]
    assert math.fsum(rewards) == info["score"]


def test_default_reward_of_a_whole_episode_is_fully_accounted_for(level_path):
    env = MarioEnv(level_path("pit"), obs_mode="grid")
    env.reset(seed=0)
    x_start = env.game.player.x
    rewards, info, _, _ = play(env, RIGHT)
    expected = (env.game.player.x - x_start) / 16 - 0.01 * len(rewards) - 15.0
    assert math.fsum(rewards) == pytest.approx(expected, abs=1e-9)


def test_clip_bounds_every_env_step_reward(level_path):
    env = MarioEnv(level_path("short"), obs_mode="grid", reward={"clip": 0.05})
    rewards, info, _, _ = play(env, RUN_RIGHT)
    assert info["flag_get"]
    assert rewards[-1] == 0.05  # the flag bonus, clipped
    assert max(abs(r) for r in rewards) <= 0.05
    # Running covers more than 0.05 tiles per env step, so clipping really happened on the way.
    assert rewards.count(0.05) > 1


def test_frame_skip_sums_the_per_frame_rewards(level_path):
    shaping = only(progress_weight=1 / 16, coin_bonus=1.0, flag_bonus=50.0, score_weight=0.01)
    skipping = MarioEnv(level_path("coins"), obs_mode="grid", frame_skip=4, reward=shaping)
    single = MarioEnv(level_path("coins"), obs_mode="grid", frame_skip=1, reward=shaping)
    skipping.reset(seed=0)
    single.reset(seed=0)
    done = False
    while not done:
        _, reward, done, _, _ = skipping.step(RIGHT)
        frames: list[float] = []
        while len(frames) < 4 and not single.game.over:
            frames.append(single.step(RIGHT)[1])
        assert reward == pytest.approx(math.fsum(frames), abs=1e-9)
    assert single.game.over and skipping.game.frame == single.game.frame


def test_time_penalty_is_per_env_step_not_per_frame(level_path):
    env = MarioEnv(level_path("short"), obs_mode="grid", frame_skip=8, reward=only(time_penalty=-1))
    env.reset(seed=0)
    assert env.step(NOOP)[1] == -1.0
