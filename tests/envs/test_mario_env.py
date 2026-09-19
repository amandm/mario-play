"""`MarioEnv` (spec 4): Gymnasium API, seeding, termination, truncation, info, rendering."""

from __future__ import annotations

import sys
import types
import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import mario_play.envs  # noqa: F401 - registers MarioPlay-v0
import mario_play.envs.mario_env as mario_env_module
from mario_play.envs.actions import ACTION_SETS
from mario_play.envs.mario_env import MarioEnv
from mario_play.envs.rewards import RewardConfig
from mario_play.game.engine import Game

NOOP, RIGHT, RIGHT_JUMP, RUN_RIGHT = 0, 1, 2, 3
INFO_KEYS = {
    "x_pos",
    "max_x",
    "progress",
    "coins",
    "score",
    "time_left",
    "flag_get",
    "death_cause",
    "level",
}


def run_until_done(env: gym.Env, action: int, max_steps: int = 2_000):
    """Repeat `action`; returns (steps, last obs, last info, terminated, truncated)."""
    for step in range(1, max_steps + 1):
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return step, obs, info, terminated, truncated
    raise AssertionError("episode did not end")


# ------------------------------------------------------------------------ construction


def test_defaults_follow_the_spec():
    env = MarioEnv()
    assert env.metadata["render_modes"] == ["human", "rgb_array"]
    assert MarioEnv.metadata["render_fps"] == 15
    assert env.observation_space.shape == (240, 256, 3)
    assert env.observation_space.dtype == np.uint8
    assert env.action_space.n == 7
    assert env.frame_skip == 4
    assert env.action_buttons == ACTION_SETS["simple"]
    assert env.reward_config == RewardConfig()
    assert env.render_mode is None
    assert isinstance(env.game, Game) and env.game.level.name == "1-1"


@pytest.mark.parametrize(("action_set", "n"), [("right_only", 5), ("simple", 7), ("complex", 10)])
def test_action_space_follows_the_action_set(action_set, n):
    env = MarioEnv("flat", obs_mode="grid", action_set=action_set)
    assert isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == n
    assert env.action_buttons == ACTION_SETS[action_set]


def test_grid_observation_space():
    env = MarioEnv("flat", obs_mode="grid")
    space = env.observation_space
    assert space.shape == (14, 15, 16) and space.dtype == np.float32
    assert space.low.min() == -1.0 and space.high.max() == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"obs_mode": "ram"},
        {"action_set": "everything"},
        {"frame_skip": 0},
        {"frame_skip": 2.5},
        {"stall_steps": 0},
        {"render_mode": "ansi"},
        {"level": "no-such-level"},
        {"level": []},
        {"level": ["flat", "no-such-level"]},
        {"reward": {"bogus": 1.0}},
    ],
)
def test_bad_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        MarioEnv(**{"level": "flat", **kwargs})


def test_reward_accepts_config_dict_and_none():
    assert MarioEnv("flat", reward=None).reward_config == RewardConfig()
    assert MarioEnv("flat", reward={"flag_bonus": 1}).reward_config.flag_bonus == 1.0
    assert MarioEnv("flat", reward=RewardConfig(clip=1.0)).reward_config.clip == 1.0


def test_levels_may_be_paths_and_level_names_lists_them(level_path):
    from pathlib import Path

    env = MarioEnv([Path(level_path("short")), level_path("pit"), "flat"], obs_mode="grid")
    assert env.level_names == ["short", "pit", "flat"]
    assert env.game.level.name == "short"  # `game` exists before the first reset
    single = MarioEnv(Path(level_path("pit")), obs_mode="grid")
    assert single.level_names == ["pit"] and single.reset(seed=0)[1]["level"] == "pit"


def test_the_env_never_mutates_the_levels_it_was_given(level_path):
    env = MarioEnv(level_path("coins"), obs_mode="grid")
    for _ in range(2):  # the coins are back in every new episode
        env.reset(seed=0)
        run_until_done(env, RIGHT)
        assert env.game.coins == 3


def test_observe_returns_the_current_observation():
    for obs_mode in ("grid", "pixels"):
        env = MarioEnv("flat", obs_mode=obs_mode)
        obs, _ = env.reset(seed=0)
        np.testing.assert_array_equal(env.observe(), obs)
        obs, *_ = env.step(RUN_RIGHT)
        again = env.observe()
        np.testing.assert_array_equal(again, obs)
        assert again is not obs and env.observation_space.contains(again)


def test_registered_id_builds_the_env():
    env = gym.make("MarioPlay-v0", level="flat", obs_mode="grid")
    assert isinstance(env.unwrapped, MarioEnv)
    obs, info = env.reset(seed=0)
    assert obs.shape == (14, 15, 16) and info["level"] == "flat"
    env.close()


# ------------------------------------------------------------------------------ API


def checker_complaints(env: gym.Env, **kwargs) -> list[str]:
    """Run Gymnasium's `check_env` (which raises on violations) and return what it warned about."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, **kwargs)
    return [str(w.message) for w in caught if "WARN" in str(w.message)]


@pytest.mark.parametrize("obs_mode", ["grid", "pixels"])
def test_check_env_passes(obs_mode):
    env = MarioEnv("1-1", obs_mode=obs_mode)
    assert checker_complaints(env, skip_render_check=True) == []
    env.close()


@pytest.mark.parametrize("obs_mode", ["grid", "pixels"])
def test_check_env_passes_for_the_registered_env_including_render_modes(obs_mode):
    pytest.importorskip("mario_play.game.human")  # the checker also opens the human mode
    env = gym.make("MarioPlay-v0", level="1-1", obs_mode=obs_mode, render_mode="rgb_array")
    assert checker_complaints(env.unwrapped) == []
    env.close()


def test_step_before_reset_raises():
    env = MarioEnv("flat", obs_mode="grid")
    with pytest.raises(gym.error.ResetNeeded):
        env.step(NOOP)


@pytest.mark.parametrize("action", [-1, 7, 2.5, "right"])
def test_invalid_action_raises_value_error(action):
    env = MarioEnv("flat", obs_mode="grid")
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


def test_numpy_integer_actions_are_accepted():
    env = MarioEnv("flat", obs_mode="grid")
    env.reset(seed=0)
    env.step(np.int64(RIGHT))
    env.step(np.array(RIGHT))
    env.step(np.array([RIGHT]))
    assert env.game.player.x > 34.0


def test_step_returns_plain_python_scalars():
    env = MarioEnv("flat", obs_mode="grid")
    _, info = env.reset(seed=0)
    assert set(info) == INFO_KEYS
    _, reward, terminated, truncated, info = env.step(RIGHT)
    assert type(reward) is float and type(terminated) is bool and type(truncated) is bool
    assert set(info) == INFO_KEYS
    assert type(info["x_pos"]) is float and type(info["max_x"]) is float
    assert type(info["progress"]) is float
    assert type(info["coins"]) is int and type(info["score"]) is int
    assert type(info["time_left"]) is int and type(info["flag_get"]) is bool
    assert info["death_cause"] is None and info["level"] == "flat"


def test_action_buttons_drive_the_game():
    env = MarioEnv("flat", obs_mode="grid", action_set="simple")
    env.reset(seed=0)
    env.step(RIGHT_JUMP)
    player = env.game.player
    assert player.vx > 0 and player.vy < 0 and not player.on_ground
    assert env.game.frame == 4


# -------------------------------------------------------------------------- seeding


@pytest.mark.parametrize(("obs_mode", "steps"), [("grid", 400), ("pixels", 80)])
def test_same_seed_and_actions_give_the_same_trajectory(obs_mode, steps):
    def rollout(seed: int):
        env = MarioEnv(["1-1", "1-2", "1-3"], obs_mode=obs_mode, action_set="complex")
        rng = np.random.default_rng(99)
        obs, info = env.reset(seed=seed)
        trace = [(obs, 0.0, False, info["level"])]
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(int(rng.integers(10)))
            trace.append((obs, reward, terminated, info["level"]))
            if terminated or truncated:
                obs, info = env.reset()
                trace.append((obs, 0.0, False, info["level"]))
        return trace

    first, second = rollout(5), rollout(5)
    assert len(first) == len(second)
    for (obs_a, *rest_a), (obs_b, *rest_b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(obs_a, obs_b)
        assert rest_a == rest_b


def test_level_sampling_is_seeded_and_covers_the_list():
    names = ["flat", "1-1", "1-2", "1-3"]

    def levels(seed: int) -> list[str]:
        env = MarioEnv(names, obs_mode="grid")
        drawn = [env.reset(seed=seed)[1]["level"]]
        drawn += [env.reset()[1]["level"] for _ in range(39)]
        assert env.game.level.name == drawn[-1]
        return drawn

    first = levels(3)
    assert first == levels(3)
    assert set(first) == set(names)
    assert first != levels(4)


def test_reseeding_restarts_the_level_sequence():
    env = MarioEnv(["flat", "1-1", "1-2", "1-3"], obs_mode="grid")
    first = [env.reset(seed=11)[1]["level"]] + [env.reset()[1]["level"] for _ in range(9)]
    again = [env.reset(seed=11)[1]["level"]] + [env.reset()[1]["level"] for _ in range(9)]
    assert first == again


def test_single_level_given_as_a_list():
    env = MarioEnv(["flat"], obs_mode="grid")
    assert all(env.reset(seed=s)[1]["level"] == "flat" for s in range(3))


def test_reset_options_can_force_a_level():
    env = MarioEnv(["flat", "1-1"], obs_mode="grid")
    for _ in range(5):
        assert env.reset(seed=0, options={"level": "1-2"})[1]["level"] == "1-2"
    with pytest.raises(ValueError):
        env.reset(options={"level": "no-such-level"})


def test_reset_starts_a_fresh_episode():
    env = MarioEnv("flat", obs_mode="grid")
    first, _ = env.reset(seed=0)
    for _ in range(30):
        env.step(RUN_RIGHT)
    assert env.game.frame == 120
    again, info = env.reset()
    np.testing.assert_array_equal(first, again)
    assert env.game.frame == 0 and info["max_x"] == info["x_pos"] == 34.0
    assert info["progress"] == 0.0


def test_game_rng_is_seeded_from_the_env_seed():
    def draws(seed: int) -> list[float]:
        env = MarioEnv("flat", obs_mode="grid")
        env.reset(seed=seed)
        return [env.game.rng.random() for _ in range(3)]

    assert draws(1) == draws(1)
    assert draws(1) != draws(2)


# ------------------------------------------------------------- termination and info


def test_pit_death_terminates(level_path):
    env = MarioEnv(level_path("pit"), obs_mode="grid")
    env.reset(seed=0)
    _, _, info, terminated, truncated = run_until_done(env, RIGHT)
    assert terminated and not truncated
    assert info["death_cause"] == "pit" and info["flag_get"] is False
    assert info["level"] == "pit"
    assert 0.0 < info["progress"] < 1.0
    assert env.game.over and not env.game.won


def test_enemy_death_terminates(level_path):
    env = MarioEnv(level_path("enemy"), obs_mode="grid")
    env.reset(seed=0)
    _, _, info, terminated, _ = run_until_done(env, NOOP)
    assert terminated and info["death_cause"] == "enemy" and not info["flag_get"]
    assert info["progress"] == 0.0


def test_timeout_death_terminates(level_path):
    env = MarioEnv(level_path("timeout"), obs_mode="grid")
    env.reset(seed=0)
    steps, _, info, terminated, _ = run_until_done(env, NOOP)
    assert terminated and info["death_cause"] == "timeout" and info["time_left"] == 0
    assert steps == 12  # 2 time units x 24 frames / frame_skip 4


def test_flag_terminates_with_full_progress(level_path):
    env = MarioEnv(level_path("short"), obs_mode="grid")
    env.reset(seed=0)
    _, _, info, terminated, truncated = run_until_done(env, RIGHT)
    assert terminated and not truncated
    assert info["flag_get"] is True and info["death_cause"] is None
    assert info["progress"] == 1.0
    assert info["score"] == 1000 + 10 * info["time_left"]
    assert env.game.won


def test_progress_and_max_x_never_decrease(level_path):
    env = MarioEnv(level_path("short"), obs_mode="grid")
    _, info = env.reset(seed=0)
    start_x = info["x_pos"]
    history = [info]
    for action in [RIGHT] * 10 + [6] * 15 + [RIGHT] * 5:  # 6 = left in the simple set
        history.append(env.step(action)[4])
    xs = [i["x_pos"] for i in history]
    assert min(xs[10:26]) < max(xs[:11])  # it really walked back
    for before, after in zip(history, history[1:], strict=False):
        assert after["max_x"] >= before["max_x"] and after["progress"] >= before["progress"]
        assert after["max_x"] >= after["x_pos"]
    assert history[-1]["max_x"] == max(xs)
    flag_x = 10 * 16 - env.game.player.w
    assert history[-1]["progress"] == pytest.approx((max(xs) - start_x) / (flag_x - start_x))


def test_step_after_the_end_is_a_harmless_no_op(level_path):
    env = MarioEnv(level_path("pit"), obs_mode="grid")
    env.reset(seed=0)
    _, last_obs, last_info, _, _ = run_until_done(env, RIGHT)
    frame = env.game.frame
    with pytest.warns(UserWarning, match="reset"):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert reward == 0.0 and terminated and not truncated
    assert env.game.frame == frame and info == last_info
    np.testing.assert_array_equal(obs, last_obs)


def test_frame_skip_stops_early_when_the_game_ends(level_path):
    env = MarioEnv(level_path("timeout"), obs_mode="grid", frame_skip=1_000)
    env.reset(seed=0)
    calls = 0
    real_step = env.game.step

    def counting_step(buttons):
        nonlocal calls
        calls += 1
        return real_step(buttons)

    env.game.step = counting_step
    _, _, terminated, _, info = env.step(NOOP)
    assert terminated and info["death_cause"] == "timeout"
    assert calls == 48 and env.game.frame == 48


def test_frame_skip_repeats_the_action():
    env = MarioEnv("flat", obs_mode="grid", frame_skip=7)
    env.reset(seed=0)
    env.step(RIGHT)
    assert env.game.frame == 7


# ----------------------------------------------------------------------- truncation


def test_stall_truncates_after_n_steps_without_progress():
    env = MarioEnv("flat", obs_mode="grid", stall_steps=5)
    env.reset(seed=0)
    steps, _, info, terminated, truncated = run_until_done(env, NOOP)
    assert steps == 5 and truncated and not terminated
    assert info["death_cause"] is None and not info["flag_get"]


def test_progress_resets_the_stall_counter():
    env = MarioEnv("flat", obs_mode="grid", stall_steps=5)
    env.reset(seed=0)
    for _ in range(4):
        assert env.step(NOOP)[3] is False
    assert env.step(RIGHT)[3] is False  # a new max x: the count starts over
    for _ in range(20):
        _, _, _, truncated, _ = env.step(RIGHT)
        assert not truncated
    # Going left sets no new max x (once the momentum to the right is used up).
    steps, _, _, terminated, truncated = run_until_done(env, 6)
    assert truncated and not terminated and 5 <= steps <= 8


def test_stall_counter_restarts_with_every_episode():
    env = MarioEnv("flat", obs_mode="grid", stall_steps=3)
    for _ in range(2):
        env.reset(seed=0)
        steps, _, _, _, truncated = run_until_done(env, NOOP)
        assert truncated and steps == 3


def test_no_stall_truncation_by_default():
    env = MarioEnv("flat", obs_mode="grid")
    env.reset(seed=0)
    assert not any(env.step(NOOP)[3] for _ in range(300))


def test_death_wins_over_stall_truncation(level_path):
    env = MarioEnv(level_path("timeout"), obs_mode="grid", stall_steps=12)
    env.reset(seed=0)
    steps, _, _, terminated, truncated = run_until_done(env, NOOP)
    assert steps == 12 and terminated and not truncated


# ------------------------------------------------------------------------ rendering


class _CountingRenderer(mario_env_module.Renderer):
    created = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).created += 1
        super().__init__(*args, **kwargs)


@pytest.fixture
def renderer_spy(monkeypatch):
    _CountingRenderer.created = 0
    monkeypatch.setattr(mario_env_module, "Renderer", _CountingRenderer)
    return _CountingRenderer


def test_grid_mode_builds_no_renderer_until_render_is_called(renderer_spy):
    env = MarioEnv("flat", obs_mode="grid", render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(10):
        env.step(RIGHT)
    assert renderer_spy.created == 0
    frame = env.render()
    assert renderer_spy.created == 1
    assert frame.shape == (240, 256, 3) and frame.dtype == np.uint8
    env.render()
    assert renderer_spy.created == 1  # built once, then reused


def test_pixel_mode_builds_one_renderer(renderer_spy):
    env = MarioEnv("flat", obs_mode="pixels", render_mode="rgb_array")
    assert renderer_spy.created == 0  # not even here before the first observation
    env.reset(seed=0)
    env.step(RIGHT)
    env.render()
    assert renderer_spy.created == 1


def test_rgb_array_render_shows_the_current_state():
    env = MarioEnv("flat", obs_mode="pixels", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    first = env.render()
    np.testing.assert_array_equal(first, obs)
    assert first is not obs and first is not env.render()
    for _ in range(10):
        obs, *_ = env.step(RUN_RIGHT)
    np.testing.assert_array_equal(env.render(), obs)
    assert not np.array_equal(first, obs)


def test_render_without_a_render_mode_returns_none():
    env = MarioEnv("flat", obs_mode="grid")
    env.reset(seed=0)
    with pytest.warns(UserWarning, match="render_mode"):
        assert env.render() is None


def test_hud_flag_reaches_the_renderer():
    with_hud = MarioEnv("flat", hud=True)
    without = MarioEnv("flat", hud=False)
    a, _ = with_hud.reset(seed=0)
    b, _ = without.reset(seed=0)
    assert not np.array_equal(a[:28], b[:28])
    np.testing.assert_array_equal(a[28:], b[28:])


def test_pixel_observations_are_fresh_arrays():
    env = MarioEnv("flat", obs_mode="pixels")
    obs, _ = env.reset(seed=0)
    kept = [(obs, obs.copy())]
    for _ in range(12):
        obs, *_ = env.step(RUN_RIGHT)
        assert obs.flags.writeable and obs.shape == (240, 256, 3) and obs.dtype == np.uint8
        kept.append((obs, obs.copy()))
    for live, snapshot in kept:
        np.testing.assert_array_equal(live, snapshot)  # nothing was overwritten later
    assert len({id(live) for live, _ in kept}) == len(kept)
    assert not any(np.shares_memory(kept[0][0], live) for live, _ in kept[1:])


class _FakeWindow:
    instances: list[_FakeWindow] = []

    def __init__(self, scale: int = 3, title: str = "mario-play") -> None:
        self.scale, self.title = scale, title
        self.frames: list[np.ndarray] = []
        self.ticks: list[int] = []
        self.closed = 0
        self.quit = False
        self.script: list[dict] = []  # answers of the next polls, before the default one
        self.polls = 0
        type(self).instances.append(self)

    def show(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def poll(self) -> dict:
        self.polls += 1
        answer = {"quit": self.quit, "buttons": None, "restart": False, "pause": False}
        if self.script:
            answer.update(self.script.pop(0))
        return answer

    def tick(self, fps: int) -> None:
        self.ticks.append(fps)

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def fake_window(monkeypatch):
    """Stand-in for `mario_play.game.human` so the human mode is testable without pygame."""
    _FakeWindow.instances = []
    module = types.ModuleType("mario_play.game.human")
    module.Window = _FakeWindow
    monkeypatch.setitem(sys.modules, "mario_play.game.human", module)
    return _FakeWindow


def test_human_mode_shows_every_reset_and_step_in_a_window(fake_window):
    env = MarioEnv("flat", obs_mode="grid", render_mode="human")
    assert fake_window.instances == []  # the window opens with the first frame, not before
    env.reset(seed=0)
    env.step(RIGHT)
    env.step(RIGHT)
    (window,) = fake_window.instances
    assert len(window.frames) == 3
    assert all(f.shape == (240, 256, 3) and f.dtype == np.uint8 for f in window.frames)
    assert window.ticks == [15, 15, 15]  # 60 fps / frame_skip 4
    assert env.render() is None
    env.close()
    env.close()
    assert window.closed == 1


def test_human_mode_frame_rate_follows_the_frame_skip(fake_window):
    env = MarioEnv("flat", obs_mode="grid", frame_skip=2, render_mode="human")
    env.reset(seed=0)
    assert fake_window.instances[0].ticks == [30]
    assert env.metadata["render_fps"] == 30 and MarioEnv.metadata["render_fps"] == 15
    env.close()


def test_closing_the_window_interrupts_the_run(fake_window):
    env = MarioEnv("flat", obs_mode="grid", render_mode="human")
    env.reset(seed=0)
    window = fake_window.instances[0]
    window.quit = True
    with pytest.raises(KeyboardInterrupt):
        env.step(RIGHT)
    assert window.closed == 1
    env.close()
    assert window.closed == 1


def test_pause_key_holds_the_env_until_pressed_again(fake_window):
    env = MarioEnv("flat", obs_mode="grid", render_mode="human")
    env.reset(seed=0)
    window = fake_window.instances[0]
    window.script = [{"pause": True}, {}, {}, {"pause": True}]
    polls, ticks, frame = window.polls, len(window.ticks), env.game.frame
    env.step(RIGHT)
    assert window.polls - polls == 4  # paused, two idle polls, resumed
    assert len(window.ticks) - ticks == 4  # the wait is paced, not a busy loop
    assert env.game.frame == frame + 4 and len(window.frames) == 2
    env.close()


def test_quitting_while_paused_interrupts_the_run(fake_window):
    env = MarioEnv("flat", obs_mode="grid", render_mode="human")
    env.reset(seed=0)
    window = fake_window.instances[0]
    window.script = [{"pause": True}, {}, {"quit": True}]
    with pytest.raises(KeyboardInterrupt):
        env.step(RIGHT)
    assert window.closed == 1


def test_human_mode_with_the_real_window():
    pytest.importorskip("mario_play.game.human")
    env = MarioEnv("flat", obs_mode="pixels", render_mode="human")
    try:
        env.reset(seed=0)
        for _ in range(3):
            env.step(RIGHT)
        assert env.render() is None
    finally:
        env.close()
        env.close()


def test_close_without_a_window_is_fine():
    env = MarioEnv("flat", obs_mode="grid")
    env.close()
    env.reset(seed=0)
    env.close()
