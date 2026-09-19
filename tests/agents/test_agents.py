"""Baseline agents: random, heuristic, search - and the proof that every level can be won."""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import pytest

from mario_play.agents.heuristic_agent import HeuristicAgent
from mario_play.agents.random_agent import RandomAgent
from mario_play.agents.runner import run_episode
from mario_play.agents.search_agent import SearchAgent, advance, default_plan_library
from mario_play.envs.actions import get_action_set
from mario_play.envs.factory import make_env
from mario_play.game.engine import Buttons, Game
from mario_play.game.level import Level, list_levels
from mario_play.rl.config import EnvConfig

SUMMARY_KEYS = {"return", "length", "flag_get", "progress", "death_cause"}


def _full_state(game: Game) -> tuple:
    """Everything that defines a game state, for "was it touched?" comparisons."""
    entities = tuple(tuple(sorted(e.__dict__.items())) for e in game.entities)
    return (
        tuple(sorted(game.player.__dict__.items())),
        entities,
        game.level.tiles.tobytes(),
        game.frame,
        game.time_left,
        game.score,
        game.coins,
        game.camera_x,
        game.over,
        game.won,
        game.death_cause,
        game.rng.getstate(),
    )


# --- run_episode -----------------------------------------------------------------------------


class _ScriptedAgent:
    def __init__(self, action: int) -> None:
        self.action = action
        self.resets = 0
        self.seen: list[np.ndarray] = []

    def reset(self) -> None:
        self.resets += 1

    def act(self, obs: np.ndarray) -> int:
        self.seen.append(obs)
        return self.action


def test_run_episode_summary_matches_the_env(make_mario):
    env = make_mario("flat")
    agent = _ScriptedAgent(action=3)  # right+run finishes `flat`
    summary = run_episode(env, agent, seed=1)
    assert set(summary) == SUMMARY_KEYS
    assert summary["flag_get"] is True
    assert summary["progress"] == 1.0
    assert summary["death_cause"] is None
    assert summary["length"] == len(agent.seen) > 10
    assert summary["return"] > 0
    assert agent.resets == 1
    assert agent.seen[0].shape == env.observation_space.shape

    # The same episode by hand gives the same numbers.
    env.reset(seed=1)
    total, steps, done = 0.0, 0, False
    while not done:
        _, reward, terminated, truncated, _ = env.step(3)
        total += reward
        steps += 1
        done = terminated or truncated
    assert steps == summary["length"]
    assert total == pytest.approx(summary["return"])


def test_run_episode_stops_at_max_steps_and_reports_death(make_mario):
    env = make_mario("flat")
    summary = run_episode(env, _ScriptedAgent(action=0), max_steps=7)
    assert summary["length"] == 7
    assert summary["flag_get"] is False
    assert summary["progress"] == 0.0

    summary = run_episode(env, _ScriptedAgent(action=0))  # standing still: the clock runs out
    assert summary["death_cause"] == "timeout"
    assert summary["flag_get"] is False
    with pytest.raises(ValueError):
        run_episode(env, _ScriptedAgent(action=0), max_steps=-1)


def test_run_episode_works_through_the_factory_wrappers_and_truncation():
    env = make_env(EnvConfig(level="flat", obs_mode="grid", max_episode_steps=5))
    try:
        summary = run_episode(env, HeuristicAgent(env), seed=0)
    finally:
        env.close()
    assert summary["length"] == 5  # TimeLimit truncation ends the episode
    assert 0.0 < summary["progress"] < 1.0


def test_run_episode_on_an_env_without_mario_info():
    env = gym.make("CartPole-v1")
    try:
        summary = run_episode(env, RandomAgent(env, seed=0), seed=0)
    finally:
        env.close()
    assert summary["flag_get"] is False
    assert summary["progress"] == 0.0
    assert summary["death_cause"] is None
    assert summary["length"] == summary["return"] > 0


# --- RandomAgent -----------------------------------------------------------------------------


def test_random_agent_runs_an_episode(make_mario):
    env = make_mario("1-1")
    summary = run_episode(env, RandomAgent(env, seed=0), max_steps=300, seed=0)
    assert set(summary) == SUMMARY_KEYS
    assert 1 <= summary["length"] <= 300
    assert 0.0 <= summary["progress"] <= 1.0


def test_random_agent_is_seeded_and_covers_the_action_space(make_mario):
    env = make_mario("flat")
    first = [RandomAgent(env, seed=5).act(None) for _ in range(1)]
    a, b, c = RandomAgent(env, seed=5), RandomAgent(env, seed=5), RandomAgent(env, seed=6)
    seq_a = [a.act(None) for _ in range(300)]
    seq_b = [b.act(None) for _ in range(300)]
    seq_c = [c.act(None) for _ in range(300)]
    assert seq_a == seq_b != seq_c
    assert first[0] == seq_a[0]
    assert set(seq_a) == set(range(env.action_space.n))
    assert all(type(action) is int for action in seq_a)
    a.reset()  # no per-episode state, and the stream is not rewound
    assert [a.act(None) for _ in range(300)] != seq_a


def test_random_agent_does_not_touch_other_rngs(make_mario):
    env = make_mario("flat")
    env.reset(seed=0)
    env.action_space.seed(0)
    expected = env.action_space.sample()
    env.action_space.seed(0)
    state = np.random.get_state()[1].copy()
    agent = RandomAgent(env, seed=1)
    for _ in range(10):
        agent.act(None)
    assert env.action_space.sample() == expected
    assert (np.random.get_state()[1] == state).all()


def test_random_agent_needs_a_discrete_action_space():
    env = gym.make("Pendulum-v1")
    try:
        with pytest.raises(TypeError):
            RandomAgent(env)
    finally:
        env.close()


# --- HeuristicAgent --------------------------------------------------------------------------


def test_heuristic_agent_finishes_flat(make_mario):
    env = make_mario("flat")
    summary = run_episode(env, HeuristicAgent(env), seed=0)
    assert summary["flag_get"] is True
    assert summary["progress"] == 1.0
    assert summary["return"] > 0


@pytest.mark.parametrize("action_set", ["right_only", "simple", "complex"])
def test_heuristic_agent_works_with_every_action_set(make_mario, action_set):
    env = make_mario("flat", action_set=action_set)
    assert run_episode(env, HeuristicAgent(env))["flag_get"] is True


def test_heuristic_agent_is_deterministic_and_beats_standing_still(make_mario):
    env = make_mario("1-1")
    first = run_episode(env, HeuristicAgent(env), seed=0)
    second = run_episode(env, HeuristicAgent(env), seed=0)
    assert first == second
    assert first["progress"] > 0.25  # gets past the first enemies, pipes and the first pit


def _level(rows: list[str]) -> Level:
    return Level.from_string("\n".join(rows), name="test")


def test_heuristic_senses_walls_pits_and_enemies():
    clear = Game(_level(["................F", ".S..............F", "#################"]))
    assert not HeuristicAgent.wall_ahead(clear)
    assert not HeuristicAgent.pit_ahead(clear)
    assert not HeuristicAgent.enemy_ahead(clear)

    wall = Game(_level(["................F", ".SX.............F", "#################"]))
    assert HeuristicAgent.wall_ahead(wall)
    assert not HeuristicAgent.pit_ahead(wall)

    pit = Game(_level(["................F", ".S..............F", "##..#############"]))
    assert HeuristicAgent.pit_ahead(pit)
    assert not HeuristicAgent.wall_ahead(pit)

    enemy = Game(_level(["................F", ".S.g............F", "#################"]))
    assert HeuristicAgent.enemy_ahead(enemy)
    far = Game(_level(["................F", ".S........g.....F", "#################"]))
    assert not HeuristicAgent.enemy_ahead(far)


def test_heuristic_agent_jumps_a_wall_and_rearms_the_jump(make_mario, tmp_path):
    path = tmp_path / "walls.txt"
    path.write_text(
        "\n".join(
            [
                "..................................................F....",
                "..................................................F....",
                "........X.......XX................................F....",
                ".S......X.......XX.........g......................F....",
                "#####################################..################",
            ]
        )
    )
    env = make_mario(str(path))
    summary = run_episode(env, HeuristicAgent(env))
    assert summary["flag_get"] is True


def test_agents_reject_envs_without_a_game():
    env = gym.make("CartPole-v1")
    try:
        with pytest.raises(TypeError):
            HeuristicAgent(env)
        with pytest.raises(TypeError):
            SearchAgent(env)
    finally:
        env.close()


# --- SearchAgent -----------------------------------------------------------------------------


def _win_with_search(env, **kwargs) -> tuple[dict, float]:
    agent = SearchAgent(env, **kwargs)
    start = time.perf_counter()
    summary = run_episode(env, agent, seed=0)
    return summary, time.perf_counter() - start


def test_search_agent_finishes_flat(make_mario):
    env = make_mario("flat")
    summary, _ = _win_with_search(env)
    assert summary["flag_get"] is True
    assert summary["death_cause"] is None


def test_search_agent_finishes_1_1(make_mario):
    env = make_mario("1-1")
    summary, seconds = _win_with_search(env)
    print(f"SearchAgent 1-1: {summary['length']} steps in {seconds:.1f} s")
    assert summary["flag_get"] is True, summary


@pytest.mark.slow
@pytest.mark.parametrize("level", ["1-2", "1-3"])
def test_search_agent_finishes_the_hard_levels(make_mario, level):
    env = make_mario(level)
    summary, seconds = _win_with_search(env)
    print(f"SearchAgent {level}: {summary['length']} steps in {seconds:.1f} s")
    assert summary["flag_get"] is True, summary


def test_every_bundled_level_has_a_completion_test():
    assert set(list_levels()) == {"flat", "1-1", "1-2", "1-3"}


@pytest.mark.slow
@pytest.mark.parametrize("action_set", ["right_only", "complex"])
def test_search_agent_finishes_1_1_with_other_action_sets(make_mario, action_set):
    env = make_mario("1-1", action_set=action_set)
    summary, _ = _win_with_search(env)
    assert summary["flag_get"] is True, summary


@pytest.mark.slow
@pytest.mark.parametrize("frame_skip", [2, 6])
def test_search_agent_finishes_1_1_with_other_frame_skips(make_mario, frame_skip):
    env = make_mario("1-1", frame_skip=frame_skip)
    summary, _ = _win_with_search(env)
    assert summary["flag_get"] is True, summary


def test_search_agent_works_on_the_wrapped_pixel_env():
    env = make_env(EnvConfig(level="flat"))
    try:
        summary = run_episode(env, SearchAgent(env), seed=3)
    finally:
        env.close()
    assert summary["flag_get"] is True


def test_advance_is_exactly_one_env_step(make_mario):
    """A clone stepped with `advance` stays in lockstep with the real env, to the last frame."""
    env = make_mario("1-1")
    env.reset(seed=0)
    twin = env.game.clone()
    rng = np.random.default_rng(0)
    terminated = False
    steps = 0
    while not terminated and steps < 400:
        action = int(rng.integers(env.action_space.n))
        _, _, terminated, _, _ = env.step(action)
        advance(twin, env.action_buttons[action], env.frame_skip)
        steps += 1
        assert _full_state(twin) == _full_state(env.game)
    assert terminated  # random play dies early: the partial last step was compared too
    assert twin.over


def test_search_prediction_equals_what_the_env_does(make_mario):
    """Every committed action finds the live game in exactly the state the plan expected."""
    env = make_mario("1-1")
    agent = SearchAgent(env)
    env.reset(seed=0)
    agent.reset()
    steps = 0
    terminated = False
    while not terminated and steps < 150:  # enemies, pipes and the first pit
        decisions = agent.replans
        action = agent.act(None)
        if agent.replans > decisions:  # a new plan: replay it on a copy, next to the real env
            twin = env.game.clone()
            planned = [action, *(a for a, _ in agent._queue)]
            assert 1 <= len(planned) <= agent.horizon
            for i, planned_action in enumerate(planned):
                if i:
                    assert agent.act(None) == planned_action
                    assert agent.replans == decisions + 1  # no surprise: the plan still holds
                _, _, terminated, _, _ = env.step(planned_action)
                advance(twin, env.action_buttons[planned_action], env.frame_skip)
                assert _full_state(twin) == _full_state(env.game)
                steps += 1
                if terminated:
                    break
    assert steps >= 150 and not env.game.over
    assert env.game.player.x > 40 * 16


def test_search_agent_never_mutates_the_live_game(make_mario):
    env = make_mario("1-2")
    agent = SearchAgent(env)
    env.reset(seed=0)
    for _ in range(40):
        game = env.game
        before = _full_state(game)
        action = agent.act(None)
        assert env.game is game
        assert _full_state(game) == before
        env.step(action)


def test_simulate_only_touches_the_game_it_is_given(make_mario):
    env = make_mario("1-1")
    agent = SearchAgent(env)
    env.reset(seed=0)
    before = _full_state(env.game)
    twin = env.game.clone()
    score = agent.simulate(twin, agent.plans[0])
    assert _full_state(env.game) == before
    assert twin.frame > env.game.frame
    assert score > env.game.player.x  # running right gets somewhere


def test_search_agent_is_deterministic(make_mario):
    def actions() -> list[int]:
        env = make_mario("1-1")
        agent = SearchAgent(env)
        env.reset(seed=0)
        agent.reset()
        taken = []
        for _ in range(60):
            taken.append(agent.act(None))
            env.step(taken[-1])
        return taken

    first = actions()
    assert first == actions()
    assert all(type(action) is int for action in first)


def test_search_agent_replans_when_the_game_is_not_where_it_expected(make_mario):
    env = make_mario("1-1")
    agent = SearchAgent(env, commit=6)
    env.reset(seed=0)
    env.step(agent.act(None))
    assert agent.replans == 1
    env.step(agent.act(None))
    assert agent.replans == 1  # still following the plan
    env.step(6)  # somebody else moved the player: left
    agent.act(None)
    assert agent.replans == 2

    env.reset(seed=0)
    agent.reset()
    assert agent.replans == 0
    agent.act(None)
    assert agent.replans == 1


def test_search_agent_notices_an_env_reset_without_agent_reset(make_mario):
    env = make_mario(["flat", "1-1"])
    agent = SearchAgent(env)
    env.reset(seed=0)
    for _ in range(3):
        env.step(agent.act(None))
    env.reset(seed=0, options={"level": "1-1"})
    env.step(agent.act(None))
    assert agent.replans == 2


def _pit_level(tmp_path, pit: int) -> str:
    path = tmp_path / f"pit{pit}.txt"
    floor = "#" * 12 + "." * pit + "#" * (40 - 12 - pit)
    path.write_text("\n".join(["." * 34 + "F" + "." * 5, ".S" + "." * 32 + "F" + "." * 5, floor]))
    return str(path)


def test_search_scores_flag_over_progress_over_death(make_mario, tmp_path):
    env = make_mario(_pit_level(tmp_path, 4))
    agent = SearchAgent(env)
    env.reset(seed=0)
    run = agent.plans[0]
    assert env.action_buttons[run[0]] == Buttons(right=True, run=True)
    assert len(set(run)) == 1
    start_x = env.game.player.x
    stay = agent.simulate(env.game.clone(), (0,) * agent.horizon)
    assert stay == pytest.approx(start_x * 1.25)  # final x + 0.25 * average x

    # Running on falls into the pit: optimistically worth the ground before the edge.
    doomed = agent.simulate(env.game.clone(), run)
    assert stay < doomed < 12 * 16 * 1.25
    # In mid-air over the pit nothing is left to stand on: a plain death, below everything.
    game = env.game.clone()
    while game.player.on_ground:
        advance(game, env.action_buttons[run[0]], env.frame_skip)
    assert not game.over
    dead = agent.simulate(game.clone(), run)
    assert dead < -1e8 < stay
    # Among deaths, the later one is better.
    advance(game, env.action_buttons[run[0]], env.frame_skip)
    sooner = agent.simulate(game.clone(), run)
    assert sooner < dead

    near = tmp_path / "near.txt"
    near.write_text("\n".join(["......F.........", ".S....F.........", "#" * 16]))
    env2 = make_mario(str(near))
    agent2 = SearchAgent(env2)
    env2.reset(seed=0)
    win = agent2.simulate(env2.game.clone(), agent2.plans[0])
    slow_win = agent2.simulate(env2.game.clone(), (0,) * 5 + agent2.plans[0][5:])
    assert win > slow_win > 1e8 > doomed


def test_search_agent_never_dies_at_a_pit_it_can_cross(make_mario, tmp_path):
    env = make_mario(_pit_level(tmp_path, 4))
    summary = run_episode(env, SearchAgent(env))
    assert summary["flag_get"] is True


def test_search_agent_makes_room_for_a_run_up(make_mario, tmp_path):
    """Standing at the edge of a wide pit with no speed: only going back first helps."""
    env = make_mario(_pit_level(tmp_path, 7))
    agent = SearchAgent(env)
    env.reset(seed=0)
    env.game.player.x = float(12 * 16 - env.game.player.w - 1)  # test setup: put it at the edge
    env.step(0)
    assert env.game.player.vx == 0.0 and env.game.player.on_ground
    edge_x = env.game.player.x
    went_back = False
    terminated = False
    steps = 0
    while not terminated and steps < 400:
        _, _, terminated, _, info = env.step(agent.act(None))
        went_back = went_back or env.game.player.x < edge_x - 8
        steps += 1
    assert info["flag_get"] is True
    assert went_back


def test_search_agent_survives_as_long_as_it_can_when_doomed(make_mario, tmp_path):
    env = make_mario(_pit_level(tmp_path, 14))  # nobody jumps that
    agent = SearchAgent(env)
    summary = run_episode(env, agent, max_steps=80)
    assert summary["flag_get"] is False
    assert summary["death_cause"] is None  # it waits at the edge rather than jumping to death
    assert summary["length"] == 80
    assert agent.detours >= 1


def test_search_agent_parameters(make_mario):
    env = make_mario("flat")
    agent = SearchAgent(env)
    assert (agent.horizon, agent.commit, agent.settle_steps) == (30, 4, 24)
    assert all(len(plan) == agent.horizon for plan in agent.plans)
    assert len(set(agent.plans)) == len(agent.plans) > 20
    n_actions = env.action_space.n
    assert all(0 <= action < n_actions for plan in agent.plans for action in plan)

    # Durations are frames at heart: another frame_skip keeps the look-ahead time.
    fast = SearchAgent(make_mario("flat", frame_skip=2))
    assert (fast.horizon, fast.commit, fast.settle_steps) == (60, 8, 48)

    custom = SearchAgent(env, horizon=10, commit=50, plans=[[("right", 3), ("noop", 1)]])
    assert custom.commit == 10
    assert custom.plans == [(1, 1, 1) + (0,) * 7]
    for bad in ({"horizon": 0}, {"commit": 0}, {"settle_steps": -1}, {"horizon": 2.5}):
        with pytest.raises(ValueError):
            SearchAgent(env, **bad)
    with pytest.raises(ValueError):
        SearchAgent(env, plans=[[("up", 3)]])


def test_plan_library_adapts_to_the_action_set(make_mario):
    names = {name for plan in default_plan_library() for name, _ in plan}
    assert {"right+run", "right+run+jump", "left", "noop", "jump"} <= names
    small = SearchAgent(make_mario("flat", action_set="right_only"))
    full = SearchAgent(make_mario("flat", action_set="simple"))
    assert 0 < len(small.plans) < len(full.plans)
    assert len(get_action_set("right_only")) == 5
    assert all(action < 5 for plan in small.plans for action in plan)
