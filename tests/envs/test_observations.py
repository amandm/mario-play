"""Grid observation (spec 4): plane contents, egocentric window, scalar planes, speed."""

from __future__ import annotations

import time

import numpy as np
import pytest

from mario_play.envs.mario_env import MarioEnv
from mario_play.envs.observations import (
    GRID_PLANES,
    GRID_SHAPE,
    PLAYER_COLUMN,
    grid_observation,
    grid_observation_space,
    pixel_observation_space,
)
from mario_play.game.engine import Buttons, Game
from mario_play.game.entities import Mushroom
from mario_play.game.level import Level

P = {name: index for index, name in enumerate(GRID_PLANES)}

# 8 rows, padded to 15 on top: the text rows are level rows 7..14.
#          0         1         2
#          0123456789012345678901234567
OBS_LEVEL = """\
; time=100
......................F.....
......................F.....
.....B?M..o...........F.....
......................F.....
......................F.....
..S......g...k....[]..F.....
############################
############################
"""
FLAG_COL = 22
START_COL = 2
SHIFT = PLAYER_COLUMN - START_COL  # window column = level column + SHIFT at the start


@pytest.fixture
def game() -> Game:
    return Game(Level.from_string(OBS_LEVEL, name="obs"))


def test_plane_names_and_shape():
    assert GRID_PLANES == (
        "solid",
        "breakable",
        "coin",
        "enemy",
        "moving_shell",
        "mushroom",
        "goal",
        "player",
        "vx",
        "vy",
        "on_ground",
        "big",
        "x_offset",
        "time",
    )
    assert GRID_SHAPE == (14, 15, 16)
    assert PLAYER_COLUMN == 4


def test_spaces():
    grid = grid_observation_space()
    assert grid.shape == (14, 15, 16) and grid.dtype == np.float32
    assert grid.low.min() == -1.0 and grid.high.max() == 1.0
    pixels = pixel_observation_space()
    assert pixels.shape == (240, 256, 3) and pixels.dtype == np.uint8
    assert pixels.low.min() == 0 and pixels.high.max() == 255


def test_shape_dtype_and_fresh_array_per_call(game):
    first = grid_observation(game)
    second = grid_observation(game)
    assert first.shape == (14, 15, 16) and first.dtype == np.float32
    assert first is not second and not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, second)
    first[:] = 0.5  # writable, and writing does not leak into later observations
    np.testing.assert_array_equal(grid_observation(game), second)
    assert grid_observation_space().contains(second)


def test_solid_under_the_player_on_flat():
    obs = grid_observation(Game("flat"))
    solid = obs[P["solid"]]
    assert solid[13, PLAYER_COLUMN] == 1.0 and solid[14, PLAYER_COLUMN] == 1.0
    assert solid[12, PLAYER_COLUMN] == 0.0
    np.testing.assert_array_equal(solid[13:, 2:], 1.0)  # ground everywhere in the level
    assert solid[:13, 2:].sum() == 0.0  # and nothing else in sight at the start


def test_player_body_sits_in_window_column_4(game):
    body = grid_observation(game)[P["player"]]
    assert body[12, PLAYER_COLUMN] == 1.0
    assert body.sum() == 1.0  # small player, 12 px wide at x offset 2: a single cell


def test_player_stays_in_column_4_while_the_world_scrolls(game):
    game.entities.clear()  # nobody in the way: this is about scrolling only
    coin_cols = []
    for _ in range(40):
        for _ in range(4):
            game.step(Buttons(right=True))
        obs = grid_observation(game)
        assert obs[P["player"]][:, PLAYER_COLUMN].any()
        assert not obs[P["player"]][:, :PLAYER_COLUMN].any()
        level_col = int(game.player.x // 16)
        coin = np.flatnonzero(obs[P["coin"]][9])
        if coin.size:
            assert coin.tolist() == [10 - level_col + PLAYER_COLUMN]
            coin_cols.append(int(coin[0]))
    assert coin_cols == sorted(coin_cols, reverse=True) and len(set(coin_cols)) > 2


def test_big_player_covers_two_rows(game):
    game.player.grow()
    body = grid_observation(game)[P["player"]]
    assert body[11, PLAYER_COLUMN] == 1.0 and body[12, PLAYER_COLUMN] == 1.0
    assert body.sum() == 2.0


def test_player_straddling_two_columns_marks_both(game):
    game.player.x = 5 * 16 + 10.0  # 12 px wide: reaches into the next column
    body = grid_observation(game)[P["player"]]
    assert body[12, PLAYER_COLUMN] == 1.0 and body[12, PLAYER_COLUMN + 1] == 1.0
    assert body.sum() == 2.0


def test_tile_planes(game):
    obs = grid_observation(game)
    solid, breakable, coin = obs[P["solid"]], obs[P["breakable"]], obs[P["coin"]]
    for level_col in (5, 6, 7):  # brick, ?-coin, ?-mushroom
        assert breakable[9, level_col + SHIFT] == 1.0
        assert solid[9, level_col + SHIFT] == 1.0
    assert breakable.sum() == 3.0
    assert coin[9, 10 + SHIFT] == 1.0 and coin.sum() == 1.0
    assert solid[9, 10 + SHIFT] == 0.0  # a coin is not an obstacle
    assert obs[P["goal"]].sum() == 0.0  # the flag is out of sight


def test_used_blocks_are_solid_but_no_longer_breakable(game):
    from mario_play.game.tiles import Tile

    game.level.tiles[9, 6] = Tile.USED
    obs = grid_observation(game)
    assert obs[P["solid"]][9, 6 + SHIFT] == 1.0
    assert obs[P["breakable"]][9, 6 + SHIFT] == 0.0


def test_observation_follows_tile_changes(game):
    from mario_play.game.tiles import Tile

    game.level.tiles[9, 10] = Tile.EMPTY  # coin collected
    game.level.tiles[9, 5] = Tile.EMPTY  # brick broken
    obs = grid_observation(game)
    assert obs[P["coin"]].sum() == 0.0
    assert obs[P["solid"]][9, 5 + SHIFT] == 0.0 and obs[P["breakable"]][9, 5 + SHIFT] == 0.0


def test_enemy_plane_shows_walker_and_turtle(game):
    kinds = sorted(e.kind for e in game.entities)
    assert kinds == ["turtle", "walker"]  # both spawns are awake at the start
    obs = grid_observation(game)
    enemy = obs[P["enemy"]]
    assert enemy[12, 9 + SHIFT] == 1.0  # walker: one cell
    assert enemy[11, 13 + SHIFT] == 1.0 and enemy[12, 13 + SHIFT] == 1.0  # turtle: 22 px tall
    assert enemy.sum() == 3.0
    assert obs[P["moving_shell"]].sum() == 0.0 and obs[P["mushroom"]].sum() == 0.0


def test_shell_states(game):
    turtle = next(e for e in game.entities if e.kind == "turtle")
    turtle.to_shell()
    obs = grid_observation(game)
    assert obs[P["enemy"]][12, 13 + SHIFT] == 1.0 and obs[P["enemy"]][11, 13 + SHIFT] == 0.0
    assert obs[P["moving_shell"]].sum() == 0.0
    turtle.kick(1)
    obs = grid_observation(game)
    assert obs[P["enemy"]][12, 13 + SHIFT] == 1.0
    assert obs[P["moving_shell"]][12, 13 + SHIFT] == 1.0 and obs[P["moving_shell"]].sum() == 1.0


def test_squished_walker_is_no_longer_an_enemy(game):
    walker = next(e for e in game.entities if e.kind == "walker")
    walker.squish()
    assert grid_observation(game)[P["enemy"]][12, 9 + SHIFT] == 0.0


def test_mushroom_plane(game):
    game.entities.append(Mushroom(8 * 16 + 1.0, 13 * 16 - 14.0))
    obs = grid_observation(game)
    assert obs[P["mushroom"]][12, 8 + SHIFT] == 1.0 and obs[P["mushroom"]].sum() == 1.0
    assert obs[P["enemy"]][12, 8 + SHIFT] == 0.0


def test_entities_outside_the_window_are_ignored(game):
    game.entities.append(Mushroom(-100.0, 100.0))
    game.entities.append(Mushroom(27 * 16.0, 100.0))
    game.entities.append(Mushroom(5 * 16.0, -200.0))
    game.entities.append(Mushroom(5 * 16.0, 400.0))
    assert grid_observation(game)[P["mushroom"]].sum() == 0.0


def test_goal_plane_near_the_flag(game):
    game.player.x = (FLAG_COL - 6) * 16 + 3.0
    goal = grid_observation(game)[P["goal"]]
    np.testing.assert_array_equal(goal[7:13, PLAYER_COLUMN + 6], 1.0)  # FLAG_TOP + 5 pole tiles
    assert goal.sum() == 6.0


def test_columns_left_of_the_level_read_as_solid(game):
    solid = grid_observation(game)[P["solid"]]
    np.testing.assert_array_equal(solid[:, :SHIFT], 1.0)
    assert solid[:13, SHIFT].sum() == 0.0  # level column 0 is just air above the ground


def test_columns_right_of_the_level_read_as_solid(game):
    width = game.level.width_tiles
    game.player.x = float(game.level.width_px - game.player.w)  # flush with the right edge
    obs = grid_observation(game)
    solid = obs[P["solid"]]
    inside = width - int(game.player.x // 16)  # level columns from the player's to the last
    np.testing.assert_array_equal(solid[:, PLAYER_COLUMN + inside :], 1.0)
    assert solid[:13, PLAYER_COLUMN : PLAYER_COLUMN + inside].sum() == 0.0
    assert obs[P["breakable"] : P["player"]][:, :, PLAYER_COLUMN + inside :].sum() == 0.0


def test_scalar_planes_are_broadcast_and_scaled(game):
    p = game.player
    p.vx, p.vy, p.on_ground, p.big = 1.25, -2.5, False, False
    obs = grid_observation(game)
    for name, value in (("vx", 0.5), ("vy", -0.5), ("on_ground", 0.0), ("big", 0.0), ("time", 1.0)):
        np.testing.assert_array_equal(obs[P[name]], np.float32(value))
    np.testing.assert_array_equal(obs[P["x_offset"]], np.float32(2 / 16))  # x = 34

    p.grow()
    p.on_ground = True
    p.x = 5 * 16 + 12.0
    game.time_left = 25
    obs = grid_observation(game)
    np.testing.assert_array_equal(obs[P["on_ground"]], 1.0)
    np.testing.assert_array_equal(obs[P["big"]], 1.0)
    np.testing.assert_array_equal(obs[P["x_offset"]], np.float32(0.75))
    np.testing.assert_array_equal(obs[P["time"]], np.float32(0.25))


def test_velocity_planes_are_clipped_to_the_unit_range(game):
    game.player.vx, game.player.vy = 9.0, -5.4
    obs = grid_observation(game)
    np.testing.assert_array_equal(obs[P["vx"]], 1.0)
    np.testing.assert_array_equal(obs[P["vy"]], -1.0)
    game.player.vx, game.player.vy = -9.0, 99.0
    obs = grid_observation(game)
    np.testing.assert_array_equal(obs[P["vx"]], -1.0)
    np.testing.assert_array_equal(obs[P["vy"]], 1.0)


@pytest.mark.parametrize("level", ["1-1", "1-2", "1-3"])
def test_random_play_stays_within_bounds(level):
    env = MarioEnv(level, obs_mode="grid", action_set="complex")
    space = env.observation_space
    rng = np.random.default_rng(7)
    obs, _ = env.reset(seed=7)
    seen_enemy = False
    for _ in range(600):
        obs, _, terminated, _, _ = env.step(int(rng.integers(env.action_space.n)))
        assert obs.dtype == np.float32 and obs.shape == (14, 15, 16)
        assert space.contains(obs), "grid observation left [-1, 1]"
        assert np.isin(obs[:8], (0.0, 1.0)).all()
        seen_enemy |= bool(obs[P["enemy"]].any())
        if terminated:
            obs, _ = env.reset()
    assert seen_enemy


def test_dead_player_below_the_level_is_still_observable(level_path):
    env = MarioEnv(level_path("pit"), obs_mode="grid")
    env.reset(seed=0)
    terminated = False
    while not terminated:
        obs, _, terminated, _, _ = env.step(1)
    assert env.observation_space.contains(obs)
    assert obs[P["player"]].sum() == 0.0  # fell out of the bottom of the window


def test_grid_env_throughput():
    """Loose floor (the real figure is more than 10x higher): the grid path must stay cheap."""
    env = MarioEnv("1-1", obs_mode="grid", frame_skip=4)
    rng = np.random.default_rng(0)
    actions = rng.integers(env.action_space.n, size=4_000).tolist()
    env.reset(seed=0)
    best = 0.0
    for _ in range(3):  # best of three: other processes may be hogging the CPU
        start = time.perf_counter()
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
        best = max(best, len(actions) / (time.perf_counter() - start))
        if best >= 1_500:
            break
    assert best >= 1_500, f"grid env runs at {best:.0f} env steps/s; expected >= 1500"
