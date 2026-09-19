"""Movement invariants of spec section 3.1: jump arcs, speeds, collisions, screen edges."""

from __future__ import annotations

import pytest

from mario_play.game.constants import (
    GRAVITY,
    JUMP_IMPULSE,
    JUMP_IMPULSE_FAST,
    MAX_FALL,
    PLAYER_BIG_H,
    PLAYER_SMALL_H,
    PLAYER_W,
    RUN_MAX,
    TILE,
    VIEW_W,
    WALK_MAX,
)
from mario_play.game.engine import Buttons, Game
from mario_play.game.entities import Walker
from mario_play.game.level import Level
from mario_play.game.physics import move_x, move_y, overlaps

NOOP = Buttons()
RIGHT = Buttons(right=True)
LEFT = Buttons(left=True)
JUMP = Buttons(jump=True)
RUN_RIGHT = Buttons(right=True, run=True)
RUN_RIGHT_JUMP = Buttons(right=True, run=True, jump=True)

FLOOR_Y = 13 * TILE  # top of the ground


def make_game(
    width: int = 64,
    put: dict[tuple[int, int], str] | None = None,
    pits: tuple[tuple[int, int], ...] = (),
    start: tuple[int, int] = (2, 12),
) -> Game:
    """A flat test level: ground rows 13-14, flag near the right end, `put[(col, row)] = char`."""
    grid = [["."] * width for _ in range(15)]
    for row in (13, 14):
        grid[row] = ["#"] * width
    for first, last in pits:
        for col in range(first, last + 1):
            grid[13][col] = grid[14][col] = "."
    for row in range(2, 13):
        grid[row][width - 10] = "F"
    grid[start[1]][start[0]] = "S"
    for (col, row), char in (put or {}).items():
        grid[row][col] = char
    return Game(Level.from_string("\n".join("".join(row) for row in grid), name="test"))


def hold(game: Game, buttons: Buttons, frames: int) -> None:
    for _ in range(frames):
        game.step(buttons)


def jump_apex(game: Game, hold_frames: int, base: Buttons = NOOP) -> float:
    """Height in px gained by a jump whose button is held for `hold_frames` frames."""
    held = Buttons(left=base.left, right=base.right, run=base.run, jump=True)
    start_y = game.player.y
    top = start_y
    for frame in range(200):
        game.step(held if frame < hold_frames else base)
        top = min(top, game.player.y)
        if frame > 2 and game.player.on_ground:
            break
    return start_y - top


def settle(game: Game) -> None:
    hold(game, NOOP, 3)
    assert game.player.on_ground


# --- resting state -----------------------------------------------------------------------


def test_player_starts_standing_on_the_start_tile() -> None:
    game = make_game()
    p = game.player

    assert (p.w, p.h) == (PLAYER_W, PLAYER_SMALL_H)
    assert p.x == 2 * TILE + 2
    assert p.y + p.h == FLOOR_Y
    settle(game)
    assert p.y + p.h == FLOOR_Y and p.vy == 0.0 and p.vx == 0.0


# --- jump invariants ----------------------------------------------------------------------


def test_standing_full_hold_jump_apex_is_between_3_5_and_4_5_tiles() -> None:
    game = make_game()
    settle(game)

    apex = jump_apex(game, hold_frames=200)

    assert 3.5 * TILE <= apex <= 4.5 * TILE


def test_tap_jump_apex_is_at_most_2_tiles() -> None:
    game = make_game()
    settle(game)

    apex = jump_apex(game, hold_frames=1)

    assert 0.5 * TILE < apex <= 2 * TILE


def test_jump_height_grows_with_hold_time() -> None:
    apexes = []
    for frames in (1, 4, 8, 16, 40):
        game = make_game()
        settle(game)
        apexes.append(jump_apex(game, hold_frames=frames))

    assert apexes == sorted(apexes)
    assert len(set(apexes)) == len(apexes)


def test_running_full_hold_jump_covers_at_least_6_tiles() -> None:
    game = make_game(width=80)
    p = game.player
    while p.vx < RUN_MAX:
        game.step(RUN_RIGHT)
    assert p.on_ground

    takeoff_x = p.x
    game.step(RUN_RIGHT_JUMP)
    assert p.vy == pytest.approx(JUMP_IMPULSE_FAST + 0.2)
    while not p.on_ground:
        game.step(RUN_RIGHT_JUMP)

    assert p.x - takeoff_x >= 6 * TILE


def test_fast_takeoff_jumps_higher_than_a_standing_jump() -> None:
    standing = make_game()
    settle(standing)
    running = make_game(width=80)
    while running.player.vx < RUN_MAX:
        running.step(RUN_RIGHT)

    low = jump_apex(standing, hold_frames=200)
    high = jump_apex(running, hold_frames=200, base=RUN_RIGHT)

    assert high > low
    assert high <= 4.5 * TILE


def test_jump_is_edge_triggered() -> None:
    game = make_game()
    p = game.player
    settle(game)

    game.step(JUMP)
    assert p.vy < 0 and not p.on_ground
    airborne_frames = 0
    # Keep holding: the player lands and must stay on the ground.
    for _ in range(200):
        game.step(JUMP)
        airborne_frames += not p.on_ground
    assert p.on_ground
    assert airborne_frames < 80  # exactly one jump arc

    game.step(NOOP)
    game.step(JUMP)
    assert p.vy < 0 and not p.on_ground


def test_jump_pressed_in_the_air_fires_on_landing() -> None:
    game = make_game()
    p = game.player
    settle(game)
    game.step(JUMP)
    hold(game, NOOP, 10)  # released since the last jump...
    assert not p.on_ground

    landed_then_jumped = False
    was_on_ground = False
    for _ in range(100):  # ...then pressed again and held through the landing
        game.step(JUMP)
        if was_on_ground and p.vy < 0:
            landed_then_jumped = True
            break
        was_on_ground = p.on_ground

    assert landed_then_jumped


def test_no_jump_in_mid_air() -> None:
    game = make_game(put={(c, 6): "X" for c in range(0, 6)}, start=(2, 5))
    p = game.player
    settle(game)
    hold(game, RIGHT, 60)  # walk off the platform
    assert not p.on_ground and p.vy > 0

    vy_before = p.vy
    game.step(Buttons(right=True, jump=True))

    assert p.vy >= vy_before


def test_jump_impulse_constant_is_applied() -> None:
    game = make_game()
    settle(game)

    game.step(JUMP)

    assert game.player.vy == pytest.approx(JUMP_IMPULSE + 0.2)


# --- horizontal movement ---------------------------------------------------------------


def test_walk_and_run_top_speeds() -> None:
    game = make_game(width=120)
    hold(game, RIGHT, 120)
    assert game.player.vx == pytest.approx(WALK_MAX)
    assert game.player.facing == 1

    hold(game, RUN_RIGHT, 120)
    assert game.player.vx == pytest.approx(RUN_MAX)

    hold(game, RIGHT, 120)  # letting go of run slows back down to a walk
    assert game.player.vx == pytest.approx(WALK_MAX)


def test_walking_left_mirrors_walking_right() -> None:
    game = make_game(start=(12, 12))
    hold(game, LEFT, 40)

    assert game.player.vx == pytest.approx(-WALK_MAX)
    assert game.player.facing == -1


def test_per_frame_displacement_never_exceeds_run_speed() -> None:
    game = make_game(width=120)
    last_x = game.player.x
    for _ in range(300):
        game.step(RUN_RIGHT)
        assert game.player.x - last_x <= RUN_MAX + 1e-9
        last_x = game.player.x


def test_releasing_the_direction_stops_the_player_and_skidding_is_faster() -> None:
    def frames_to_stop(buttons: Buttons) -> int:
        game = make_game(width=120)
        hold(game, RIGHT, 60)
        for frame in range(1, 200):
            game.step(buttons)
            if game.player.vx <= 0:
                return frame
        raise AssertionError("never stopped")

    release, skid = frames_to_stop(NOOP), frames_to_stop(LEFT)

    assert skid < release <= 25


def test_left_and_right_together_cancel_out() -> None:
    game = make_game()
    hold(game, Buttons(left=True, right=True), 30)

    assert game.player.vx == 0.0
    assert game.player.x == 2 * TILE + 2


def test_air_control_is_weaker_than_ground_control() -> None:
    ground = make_game()
    settle(ground)
    hold(ground, RIGHT, 10)

    air = make_game()
    settle(air)
    air.step(JUMP)
    hold(air, Buttons(right=True, jump=True), 9)

    assert 0 < air.player.vx < ground.player.vx


# --- gravity ---------------------------------------------------------------------------------


def test_fall_speed_is_capped_at_terminal_velocity() -> None:
    game = make_game(put={(c, 3): "X" for c in range(0, 4)}, start=(2, 2))
    p = game.player
    speeds: list[float] = []
    for _ in range(300):  # walk off the high platform and fall to the ground
        game.step(RIGHT)
        if not p.on_ground:
            speeds.append(p.vy)
        elif speeds:
            break

    assert speeds[0] == GRAVITY
    assert speeds[1] - speeds[0] == pytest.approx(GRAVITY)
    assert max(speeds) == MAX_FALL
    assert speeds == sorted(speeds)


# --- collisions ----------------------------------------------------------------------------


def test_player_stops_flush_against_a_wall() -> None:
    game = make_game(put={(10, 12): "X", (10, 11): "X", (10, 10): "X", (10, 9): "X", (10, 8): "X"})
    p = game.player

    hold(game, RUN_RIGHT, 200)

    assert p.x == 10 * TILE - PLAYER_W
    assert p.vx == 0.0
    hold(game, LEFT, 3)
    assert p.x < 10 * TILE - PLAYER_W


def test_player_lands_flush_on_the_floor_and_on_blocks() -> None:
    game = make_game(put={(5, 12): "X", (6, 12): "X"})
    p = game.player
    p.x, p.y, p.on_ground = 5 * TILE + 4.0, 100.0, False

    while not p.on_ground:
        game.step(NOOP)

    assert p.y + p.h == 12 * TILE
    assert p.vy == 0.0


def test_player_head_stops_flush_under_a_ceiling() -> None:
    game = make_game(put={(c, 10): "X" for c in range(0, 8)})
    p = game.player
    settle(game)

    tops = []
    for _ in range(30):
        game.step(JUMP)
        tops.append(p.y)

    assert min(tops) == 11 * TILE
    hold(game, NOOP, 30)
    assert p.on_ground and p.y + p.h == FLOOR_Y


def test_no_tunnelling_through_a_thin_floor_at_terminal_velocity() -> None:
    game = make_game(put={(c, 8): "X" for c in range(0, 10)}, pits=((0, 9),))
    p = game.player
    for offset in (0.5, 1.0, 2.5, 4.4):
        p.x, p.y, p.vy = 40.0, 8 * TILE - p.h - offset, MAX_FALL

        game.step(NOOP)

        assert p.y + p.h == 8 * TILE
        assert p.on_ground and p.vy == 0.0 and not game.over


@pytest.mark.parametrize("speed", [RUN_MAX, 7.9, 15.9, 40.0, 1000.0])
@pytest.mark.parametrize("direction", [1, -1])
def test_no_tunnelling_through_a_thin_wall_at_any_speed(speed: float, direction: int) -> None:
    wall = {(20, r): "X" for r in range(0, 13)}
    game = make_game(put=wall, start=(18, 12) if direction > 0 else (22, 12))
    p = game.player
    settle(game)
    p.x = 20 * TILE - PLAYER_W - 1.0 if direction > 0 else 21 * TILE + 1.0
    p.vx = speed * direction

    game.step(RIGHT if direction > 0 else LEFT)

    assert p.x == (20 * TILE - PLAYER_W if direction > 0 else 21 * TILE)
    assert p.vx == 0.0


@pytest.mark.parametrize("distance", [4.5, 7.9, 8.0, 15.9, 16.0, 40.0, 1000.0])
def test_move_helpers_never_tunnel_whatever_the_distance(distance: float) -> None:
    put = {(c, 8): "X" for c in range(0, 30)} | {(20, r): "X" for r in range(0, 8)}
    level = make_game(put=put).level

    faller = Walker(40.0, 8 * TILE - 14.0 - 3.0)
    assert move_y(level, faller, distance) == 1
    assert faller.y + faller.h == 8 * TILE

    riser = Walker(40.0, 9 * TILE + 3.0)
    assert move_y(level, riser, -distance) == 2
    assert riser.y == 9 * TILE

    runner = Walker(20 * TILE - 14.0 - 3.0, 8 * TILE - 14.0)
    assert move_x(level, runner, distance) is True
    assert runner.x == 20 * TILE - 14.0

    back = Walker(21 * TILE + 3.0, 8 * TILE - 14.0)
    assert move_x(level, back, -distance) is True
    assert back.x == 21 * TILE


def test_no_tunnelling_upwards_through_a_thin_ceiling() -> None:
    game = make_game(put={(c, 9): "X" for c in range(0, 8)})
    p = game.player
    settle(game)
    p.vy, p.on_ground = -60.0, False

    game.step(NOOP)

    assert p.y == 10 * TILE
    assert p.vy == 0.0


def test_big_player_hitbox_collides_with_a_two_tile_gap() -> None:
    game = make_game(put={(c, 10): "X" for c in range(6, 12)})
    p = game.player
    p.grow()
    assert (p.w, p.h) == (PLAYER_W, PLAYER_BIG_H)
    assert p.y + p.h == FLOOR_Y

    hold(game, RUN_RIGHT, 100)  # a 32 px gap fits the 30 px player

    assert p.x > 12 * TILE
    assert p.y + p.h == FLOOR_Y


def test_move_helpers_report_collisions_and_leave_the_entity_flush() -> None:
    game = make_game(put={(10, 12): "X"})
    level = game.level
    walker = Walker(9 * TILE - 4.0, FLOOR_Y - 14.0)

    assert move_x(level, walker, 1.0) is False
    assert walker.x == 9 * TILE - 3.0
    assert move_x(level, walker, 6.0) is True
    assert walker.x == 10 * TILE - walker.w
    assert move_y(level, walker, 3.0) == 1
    assert walker.y + walker.h == FLOOR_Y
    assert move_y(level, walker, -5.0) == 0
    assert walker.y + walker.h == FLOOR_Y - 5.0


def test_level_edges_are_walls_but_the_sky_and_the_abyss_are_open() -> None:
    game = make_game()
    level = game.level
    walker = Walker(3.0, 100.0)

    assert move_x(level, walker, -7.0) is True
    assert walker.x == 0.0
    walker.x = level.width_px - walker.w - 2.0
    assert move_x(level, walker, 7.0) is True
    assert walker.x == level.width_px - walker.w

    assert move_y(level, walker, -500.0) == 0
    assert walker.y == -400.0
    walker.x, walker.y = 100.0, 300.0
    assert move_y(level, walker, 50.0) == 0


def test_overlaps_is_strict() -> None:
    a = Walker(0.0, 0.0)
    b = Walker(14.0, 0.0)

    assert not overlaps(a, b)  # touching edges do not overlap
    b.x = 13.9
    assert overlaps(a, b)
    b.y = 14.0
    assert not overlaps(a, b)


# --- screen and level edges --------------------------------------------------------------


def test_left_screen_edge_blocks_the_player() -> None:
    game = make_game(width=80)
    p = game.player
    hold(game, RUN_RIGHT, 150)
    camera = game.camera_x
    assert camera > 0

    for _ in range(200):
        game.step(Buttons(left=True, run=True))
        assert p.x >= game.camera_x

    assert game.camera_x >= camera  # it kept following while the player skidded to a halt
    assert p.x == game.camera_x
    assert p.vx == 0.0


def test_right_level_edge_blocks_the_player() -> None:
    game = make_game(width=40)
    p = game.player
    p.x = game.level.width_px - 40.0  # beyond the flag

    hold(game, RUN_RIGHT, 100)

    assert not game.over
    assert p.x == game.level.width_px - PLAYER_W


def test_camera_follows_the_player_and_stops_at_the_level_end() -> None:
    game = make_game(width=40)
    p = game.player
    assert game.camera_x == 0.0

    hold(game, RUN_RIGHT, 100)
    assert game.camera_x == pytest.approx(p.x - 112)

    p.x = game.level.width_px - 40.0
    game.step(NOOP)
    assert game.camera_x == game.level.width_px - VIEW_W
