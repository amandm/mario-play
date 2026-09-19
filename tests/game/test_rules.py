"""Game rules of spec section 3.4: tiles, items, enemies, death, victory, camera, score."""

from __future__ import annotations

import pytest

from mario_play.game.constants import (
    ENEMY_SPEED,
    INVULN_FRAMES,
    LEVEL_H_PX,
    MUSHROOM_H,
    MUSHROOM_SPEED,
    PLAYER_BIG_H,
    PLAYER_SMALL_H,
    SCORE_BRICK,
    SCORE_COIN,
    SCORE_FLAG,
    SCORE_FLAG_PER_TIME,
    SCORE_MUSHROOM,
    SCORE_SHELL_KILL,
    SCORE_STOMP,
    SHELL_H,
    SHELL_SPEED,
    SPAWN_ACTIVATION_DISTANCE,
    SQUISH_FRAMES,
    STOMP_BOUNCE,
    STOMP_BOUNCE_HELD,
    TILE,
    TURTLE_H,
    WALKER_H,
    WALKER_W,
)
from mario_play.game.engine import Buttons, Game, StepEvents
from mario_play.game.entities import Mushroom, Turtle, Walker
from mario_play.game.level import Level
from mario_play.game.tiles import IS_SOLID, Tile

NOOP = Buttons()
RIGHT = Buttons(right=True)
LEFT = Buttons(left=True)
JUMP = Buttons(jump=True)
RUN_RIGHT = Buttons(right=True, run=True)

FLOOR_Y = 13 * TILE
WALKER_Y = float(FLOOR_Y - WALKER_H)
TURTLE_Y = float(FLOOR_Y - TURTLE_H)


def make_game(
    width: int = 64,
    put: dict[tuple[int, int], str] | None = None,
    pits: tuple[tuple[int, int], ...] = (),
    start: tuple[int, int] = (2, 12),
    time: int | None = None,
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
    header = f"; time={time}\n" if time else ""
    return Game(Level.from_string(header + "\n".join("".join(row) for row in grid), name="test"))


def hold(game: Game, buttons: Buttons, frames: int) -> list[StepEvents]:
    return [game.step(buttons) for _ in range(frames)]


def step_until(game: Game, buttons: Buttons, condition, limit: int = 600) -> StepEvents:
    """Step until `condition(events)` is true; returns the events of that frame."""
    for _ in range(limit):
        events = game.step(buttons)
        if condition(events):
            return events
    raise AssertionError(f"condition not reached within {limit} frames")


def player_in_solid(game: Game) -> bool:
    p = game.player
    cols = range(int(p.x // TILE), int((p.x + p.w - 1e-9) // TILE) + 1)
    rows = range(int(p.y // TILE), int((p.y + p.h - 1e-9) // TILE) + 1)
    return any(game.level.solid_at(c, r) for c in cols for r in rows)


def drop_player_on(game: Game, target, height: float = 20.0) -> None:
    p = game.player
    p.x, p.y, p.vx, p.vy = target.x + 1.0, target.y - p.h - height, 0.0, 0.0
    p.on_ground = False


# --- tiles ---------------------------------------------------------------------------------


def test_coin_tiles_are_collected_on_overlap() -> None:
    game = make_game(put={(5, 12): "o", (6, 12): "o", (6, 8): "o"})

    events = step_until(game, RIGHT, lambda e: e.coins > 0)

    assert events.coins == 1 and events.score_delta == SCORE_COIN
    assert game.level.tiles[12, 5] == Tile.EMPTY
    hold(game, RIGHT, 30)
    assert game.coins == 2 and game.score == 2 * SCORE_COIN
    assert game.level.tiles[12, 6] == Tile.EMPTY
    assert game.level.tiles[8, 6] == Tile.COIN  # out of reach, untouched


def test_question_block_gives_one_coin_and_becomes_used() -> None:
    game = make_game(put={(2, 9): "?"})

    events = step_until(game, JUMP, lambda e: e.coins > 0, limit=30)

    assert events.coins == 1 and events.score_delta == SCORE_COIN
    assert game.level.tiles[9, 2] == Tile.USED
    assert game.player.y == 10 * TILE  # head flush under the block
    assert game.player.vy >= 0

    hold(game, NOOP, 40)
    hold(game, JUMP, 30)  # bumping a used block gives nothing
    assert game.coins == 1 and game.score == SCORE_COIN


def test_mushroom_block_spawns_a_mushroom_that_makes_the_player_big() -> None:
    game = make_game(put={(2, 9): "M", (8, 12): "X", (8, 11): "X"})
    p = game.player

    step_until(game, JUMP, lambda e: game.level.tiles[9, 2] == Tile.USED, limit=30)

    mushrooms = [e for e in game.entities if e.kind == "mushroom"]
    assert len(mushrooms) == 1
    shroom = mushrooms[0]
    assert isinstance(shroom, Mushroom)
    # It pops out centred on top of the block and has already slid one frame to the right.
    assert (shroom.x, shroom.y) == (2 * TILE + 1 + MUSHROOM_SPEED, 9 * TILE - MUSHROOM_H)
    assert game.coins == 0 and game.score == 0 and not p.big

    # It slides off the block, bounces off the wall at column 8 and comes back.
    events = step_until(game, NOOP, lambda e: e.powerups > 0, limit=400)

    assert events.powerups == 1 and events.score_delta == SCORE_MUSHROOM
    assert p.big and p.h == PLAYER_BIG_H
    assert p.y + p.h == FLOOR_Y  # feet stay put
    assert shroom not in game.entities and not shroom.alive
    assert game.score == SCORE_MUSHROOM


def test_mushroom_walks_right_first_and_falls_off_the_block() -> None:
    game = make_game(put={(2, 9): "M"})
    step_until(game, JUMP, lambda e: game.level.tiles[9, 2] == Tile.USED, limit=30)
    shroom = game.entities[0]

    hold(game, NOOP, 5)
    assert shroom.vx > 0 and shroom.y == 9 * TILE - MUSHROOM_H
    hold(game, NOOP, 60)
    assert shroom.y == FLOOR_Y - MUSHROOM_H and shroom.vx > 0


def test_second_mushroom_only_scores() -> None:
    game = make_game()
    p = game.player
    p.grow()
    game.entities.append(Mushroom(p.x + 4.0, p.y + 10.0))

    events = game.step(NOOP)

    assert events.powerups == 1 and game.score == SCORE_MUSHROOM
    assert p.big and p.h == PLAYER_BIG_H and p.y + p.h == FLOOR_Y
    assert game.entities == []


def test_mushroom_is_not_collected_where_the_big_player_would_not_fit() -> None:
    game = make_game(put={(c, 11): "X" for c in range(4, 10)})  # a one-tile-high tunnel
    p = game.player
    p.x = 6 * TILE + 2.0
    shroom = Mushroom(p.x + 6.0, FLOOR_Y - MUSHROOM_H, facing=-1)
    game.entities.append(shroom)

    hold(game, NOOP, 5)

    assert not p.big and shroom.alive
    assert not player_in_solid(game)


def test_small_player_cannot_break_bricks() -> None:
    game = make_game(put={(2, 9): "B"})

    events = hold(game, JUMP, 30)

    assert game.level.tiles[9, 2] == Tile.BRICK
    assert game.score == 0 and all(e.bricks == 0 for e in events)


def test_big_player_breaks_bricks() -> None:
    game = make_game(put={(2, 9): "B"})
    game.player.grow()

    events = step_until(game, JUMP, lambda e: e.bricks > 0, limit=30)

    assert events.bricks == 1 and events.score_delta == SCORE_BRICK
    assert game.level.tiles[9, 2] == Tile.EMPTY
    assert game.player.vy >= 0  # the brick still stops the jump
    assert game.score == SCORE_BRICK


@pytest.mark.parametrize(
    ("player_x", "expect_question_used"),
    [(5 * TILE - 5.0, True), (5 * TILE - 8.0, False)],
)
def test_head_bump_picks_the_tile_nearest_the_player_centre(
    player_x: float, expect_question_used: bool
) -> None:
    game = make_game(put={(4, 9): "B", (5, 9): "?"})
    game.player.x = player_x

    hold(game, JUMP, 30)

    assert game.level.tiles[9, 4] == Tile.BRICK
    assert bool(game.level.tiles[9, 5] == Tile.USED) is expect_question_used
    assert game.coins == int(expect_question_used)


def test_head_bump_ignores_empty_neighbours() -> None:
    game = make_game(put={(5, 9): "?"})
    game.player.x = 5 * TILE - 10.0  # centre is under column 4, which is empty

    hold(game, JUMP, 30)

    assert game.level.tiles[9, 5] == Tile.USED and game.coins == 1


# --- walkers -------------------------------------------------------------------------------


@pytest.mark.parametrize(("buttons", "bounce"), [(NOOP, STOMP_BOUNCE), (JUMP, STOMP_BOUNCE_HELD)])
def test_stomp_squishes_a_walker_and_bounces_the_player(buttons: Buttons, bounce: float) -> None:
    game = make_game()
    p = game.player
    walker = Walker(100.0, WALKER_Y)
    game.entities.append(walker)
    drop_player_on(game, walker)

    events = step_until(game, buttons, lambda e: e.stomps > 0, limit=30)

    assert events.stomps == 1 and events.score_delta == SCORE_STOMP
    assert not events.hurt and not events.died
    assert walker.squished_frames == SQUISH_FRAMES
    assert p.vy == bounce
    assert game.score == SCORE_STOMP

    hold(game, NOOP, SQUISH_FRAMES - 1)
    assert walker in game.entities and walker.vx == 0.0
    game.step(NOOP)
    assert walker not in game.entities and not walker.alive
    assert not game.over


def test_walker_squished_in_mid_air_falls_to_the_ground() -> None:
    game = make_game()
    walker = Walker(100.0, WALKER_Y - 60.0)
    walker.squish()
    game.entities.append(walker)

    hold(game, NOOP, SQUISH_FRAMES - 1)

    assert walker.y == WALKER_Y and walker.x == 100.0


def test_squished_walker_is_harmless() -> None:
    game = make_game()
    walker = Walker(60.0, WALKER_Y)
    walker.squish()
    game.entities.append(walker)

    hold(game, RIGHT, SQUISH_FRAMES - 2)  # walk right through the corpse

    assert not game.over
    assert game.player.x + game.player.w > walker.x


def test_side_contact_kills_a_small_player() -> None:
    game = make_game(put={(8, 12): "g"})

    events = step_until(game, RIGHT, lambda e: e.died, limit=200)

    assert events.death_cause == "enemy" and not events.won
    assert game.over and not game.won and game.death_cause == "enemy"
    assert game.player.dead
    assert game.snapshot()["death_cause"] == "enemy"


def test_side_contact_shrinks_a_big_player_with_invulnerability() -> None:
    game = make_game(put={(8, 12): "g"})
    p = game.player
    p.grow()

    events = step_until(game, RIGHT, lambda e: e.hurt, limit=200)

    assert not events.died and not game.over
    assert not p.big and p.h == PLAYER_SMALL_H
    assert p.y + p.h == FLOOR_Y
    assert p.invuln_frames == INVULN_FRAMES

    later = hold(game, RIGHT, 40)  # walks through the walker unharmed
    assert not game.over and not any(e.hurt or e.died for e in later)
    hold(game, NOOP, INVULN_FRAMES - 40)
    assert p.invuln_frames == 0


def test_invulnerability_lasts_exactly_120_frames() -> None:
    def hurt_then_touch_again_after(frames: int) -> Game:
        game = make_game()
        p = game.player
        p.grow()
        game.entities.append(Walker(p.x + 8.0, WALKER_Y, facing=1))
        assert game.step(NOOP).hurt
        game.entities.clear()
        hold(game, NOOP, frames - 1)
        game.entities.append(Walker(p.x - 4.0, WALKER_Y, facing=1))
        game.step(NOOP)
        return game

    assert not hurt_then_touch_again_after(INVULN_FRAMES).over
    late = hurt_then_touch_again_after(INVULN_FRAMES + 1)
    assert late.over and late.death_cause == "enemy"


def test_falling_next_to_an_enemy_is_not_a_stomp() -> None:
    game = make_game()
    p = game.player
    walker = Walker(100.0, WALKER_Y)
    game.entities.append(walker)
    # Moving down, but the feet are already below the walker's vertical midpoint.
    p.x, p.y, p.vy = walker.x - p.w - 0.2, walker.y + 10.0 - p.h, 1.0
    p.on_ground = False

    events = step_until(game, NOOP, lambda e: e.died or e.stomps, limit=10)

    assert events.died and events.stomps == 0


def test_enemies_reverse_at_walls() -> None:
    game = make_game(put={(5, 12): "X", (10, 12): "X", (8, 12): "g"})
    walker = game.entities[0]
    assert walker.facing == -1

    xs, facings = [], []
    for _ in range(400):
        game.step(NOOP)
        xs.append(walker.x)
        facings.append(walker.facing)
        assert walker.vx == walker.facing * ENEMY_SPEED

    assert min(xs) == 6 * TILE
    assert max(xs) == 10 * TILE - WALKER_W
    assert {-1, 1} == set(facings)


def test_enemies_reverse_when_they_bump_into_each_other() -> None:
    game = make_game()
    a = Walker(200.0, WALKER_Y, facing=1)
    b = Walker(240.0, WALKER_Y, facing=-1)
    game.entities += [a, b]

    gaps = []
    for _ in range(120):
        game.step(NOOP)
        gaps.append(b.x - a.x)

    assert min(gaps) >= WALKER_W - 1.0  # they barely touch
    assert (a.facing, b.facing) == (-1, 1)
    assert a.vx < 0 < b.vx
    assert gaps[-1] > 40.0


def test_enemies_fall_off_ledges_and_keep_walking() -> None:
    game = make_game(put={(c, 9): "X" for c in range(6, 10)})
    walker = Walker(7.0 * TILE, 9 * TILE - WALKER_H)
    game.entities.append(walker)

    hold(game, NOOP, 120)

    assert walker.y == WALKER_Y
    assert walker.x < 6 * TILE and walker.vx == -ENEMY_SPEED


def test_enemies_that_fall_into_a_pit_are_removed() -> None:
    game = make_game(pits=((5, 7),), put={(10, 12): "g"})
    walker = game.entities[0]

    hold(game, NOOP, 250)

    assert walker not in game.entities and not walker.alive
    assert walker.y > LEVEL_H_PX - 1
    assert not game.over


# --- turtles and shells -----------------------------------------------------------------------


def test_turtle_shell_chain() -> None:
    game = make_game(put={(8, 12): "k", (16, 12): "g", (24, 12): "X", (24, 11): "X"})
    p = game.player
    turtle, walker = game.entities
    assert isinstance(turtle, Turtle) and turtle.state == "walk"
    assert (turtle.w, turtle.h) == (14, TURTLE_H) and turtle.y == TURTLE_Y

    # 1. Stomp: the turtle becomes a resting shell with its feet where they were.
    drop_player_on(game, turtle)
    events = step_until(game, NOOP, lambda e: e.stomps > 0, limit=30)
    assert events.score_delta == SCORE_STOMP and p.vy == STOMP_BOUNCE
    assert turtle.state == "shell" and turtle.vx == 0.0
    assert turtle.h == SHELL_H and turtle.y + turtle.h == FLOOR_Y

    # 2. Walking into the resting shell kicks it away from the player.
    hold(game, NOOP, 5)
    p.x, p.y, p.vx, p.vy = turtle.x - 30.0, float(FLOOR_Y - p.h), 0.0, 0.0
    step_until(game, RIGHT, lambda e: turtle.state == "shell_moving", limit=60)
    assert turtle.vx == SHELL_SPEED and not game.over
    hold(game, RIGHT, 10)  # the shell runs away from the player: no harm
    assert not game.over

    # 3. The moving shell kills the walker in its way.
    events = step_until(game, NOOP, lambda e: walker not in game.entities, limit=120)
    assert not walker.alive
    assert events.score_delta == SCORE_SHELL_KILL and events.stomps == 0

    # 4. It bounces off the wall and comes back.
    step_until(game, NOOP, lambda e: turtle.vx == -SHELL_SPEED, limit=120)
    assert turtle.state == "shell_moving"

    # 5. Stomping the moving shell stops it.
    p.x, p.y, p.vy = turtle.x - 2 * SHELL_SPEED, turtle.y - p.h - 1.0, 2.0
    p.on_ground = False
    events = game.step(NOOP)
    assert events.stomps == 1
    assert turtle.state == "shell" and turtle.vx == 0.0
    assert p.vy == STOMP_BOUNCE and not game.over


def test_shell_is_kicked_away_from_the_player_on_either_side() -> None:
    game = make_game()
    p = game.player
    shell = Turtle(p.x - 20.0, TURTLE_Y)
    shell.to_shell()
    game.entities.append(shell)

    step_until(game, LEFT, lambda e: shell.state == "shell_moving", limit=60)

    assert shell.vx == -SHELL_SPEED


def test_shell_coming_back_hurts_the_player() -> None:
    game = make_game(put={(12, 12): "X", (12, 11): "X"})
    shell = Turtle(80.0, TURTLE_Y)
    shell.to_shell()
    game.entities.append(shell)

    step_until(game, RIGHT, lambda e: shell.state == "shell_moving", limit=60)
    events = step_until(game, NOOP, lambda e: e.died, limit=200)

    assert events.death_cause == "enemy"
    assert shell.vx < 0


@pytest.mark.parametrize(("direction", "deadly"), [(1, False), (-1, True)])
def test_moving_shell_only_hurts_with_its_front(direction: int, deadly: bool) -> None:
    game = make_game()
    p = game.player
    shell = Turtle(p.x + 6.0, TURTLE_Y)  # overlapping, its centre to the right of the player's
    shell.to_shell()
    shell.kick(direction)
    shell.contact_cooldown = 0
    game.entities.append(shell)

    hold(game, NOOP, 8)

    assert game.over is deadly


def test_moving_shell_kills_turtles_too_and_falls_into_pits() -> None:
    game = make_game(pits=((20, 22),))
    shell = Turtle(100.0, TURTLE_Y)
    shell.to_shell()
    shell.kick(1)
    victim = Turtle(200.0, TURTLE_Y)
    game.entities += [shell, victim]

    hold(game, NOOP, 120)

    assert not victim.alive and not shell.alive
    assert game.entities == []
    assert game.score == SCORE_SHELL_KILL


def test_walker_turns_around_at_a_resting_shell() -> None:
    game = make_game()
    shell = Turtle(200.0, TURTLE_Y)
    shell.to_shell()
    walker = Walker(240.0, WALKER_Y)
    game.entities += [shell, walker]

    hold(game, NOOP, 120)

    assert walker.alive and walker.facing == 1 and walker.x > 240.0
    assert shell.state == "shell" and shell.x == 200.0


# --- spawns ---------------------------------------------------------------------------------


def test_dormant_spawns_activate_half_a_screen_ahead() -> None:
    game = make_game(width=100, put={(10, 12): "g", (50, 12): "k"})
    spawn_x = 50 * TILE

    assert [e.kind for e in game.entities] == ["walker"]  # near spawns are live from frame 0
    game.entities.clear()

    activated_at = None
    for _ in range(600):
        game.step(RUN_RIGHT)
        present = any(e.kind == "turtle" for e in game.entities)
        assert present == (spawn_x < game.camera_x + SPAWN_ACTIVATION_DISTANCE)
        if present and activated_at is None:
            activated_at = game.frame
            turtle = game.entities[0]
            assert turtle.facing == -1 and turtle.vx == -ENEMY_SPEED
            assert turtle.x == spawn_x + 1 and turtle.y == TURTLE_Y
            break

    assert activated_at is not None


def test_spawn_activation_boundary_is_strict() -> None:
    game = make_game(width=100, put={(50, 12): "k"})
    p = game.player
    p.x = 50 * TILE - SPAWN_ACTIVATION_DISTANCE + 112.0

    game.step(NOOP)
    assert game.camera_x + SPAWN_ACTIVATION_DISTANCE == 50 * TILE
    assert game.entities == []

    p.x += 0.5
    game.step(NOOP)
    assert [e.kind for e in game.entities] == ["turtle"]


def test_spawns_activate_only_once() -> None:
    game = make_game(put={(10, 12): "g"})
    game.entities.clear()

    hold(game, RIGHT, 50)

    assert game.entities == []


# --- death and victory ---------------------------------------------------------------------------


def test_falling_into_a_pit_kills() -> None:
    game = make_game(pits=((6, 8),))

    events = step_until(game, RIGHT, lambda e: e.died, limit=300)

    assert events.death_cause == "pit"
    assert game.over and not game.won and game.death_cause == "pit" and game.player.dead
    assert game.player.y > LEVEL_H_PX - 1


def test_big_player_also_dies_in_a_pit() -> None:
    game = make_game(pits=((6, 8),))
    game.player.grow()

    events = step_until(game, RIGHT, lambda e: e.died, limit=300)

    assert events.death_cause == "pit"


def test_timer_counts_down_and_times_out() -> None:
    game = make_game(time=2)
    assert game.time_left == 2

    hold(game, NOOP, 23)
    assert game.time_left == 2
    game.step(NOOP)
    assert game.time_left == 1
    hold(game, NOOP, 23)
    assert not game.over

    events = game.step(NOOP)

    assert game.time_left == 0
    assert events.died and events.death_cause == "timeout"
    assert game.over and game.death_cause == "timeout"


def test_touching_the_flagpole_wins_immediately() -> None:
    game = make_game(width=40)

    events = step_until(game, RUN_RIGHT, lambda e: e.won, limit=600)

    bonus = SCORE_FLAG + SCORE_FLAG_PER_TIME * game.time_left
    assert events.score_delta == bonus and game.score == bonus
    assert game.over and game.won and game.death_cause is None
    assert not events.died and not game.player.dead
    flag_x = game.level.flag_col * TILE
    assert flag_x - 3 < game.player.x + game.player.w <= flag_x + 3


def test_flag_top_counts_as_the_flag() -> None:
    game = make_game(width=40)
    p = game.player
    p.x, p.y = game.level.flag_col * TILE - p.w - 1.0, 2 * TILE + 1.0
    p.on_ground = False
    assert game.level.tiles[2, game.level.flag_col] == Tile.FLAG_TOP

    events = step_until(game, RIGHT, lambda e: e.won or e.died, limit=20)

    assert events.won


def test_step_after_game_over_is_a_no_op() -> None:
    game = make_game(pits=((6, 8),))
    step_until(game, RIGHT, lambda e: e.died, limit=300)
    before = game.snapshot()

    events = hold(game, RUN_RIGHT, 5)

    assert all(e == StepEvents() for e in events)
    assert game.snapshot() == before


def test_camera_never_scrolls_left() -> None:
    game = make_game(width=100)
    cameras = []
    for buttons in (RUN_RIGHT, LEFT, RUN_RIGHT, LEFT):
        for _ in range(90):
            game.step(buttons)
            cameras.append(game.camera_x)

    assert cameras == sorted(cameras)
    assert cameras[-1] > 100


def test_events_account_for_the_score_and_coins() -> None:
    game = make_game(put={(5, 12): "o", (7, 9): "?", (12, 12): "g", (9, 12): "o"})
    total_score = total_coins = 0
    for frame in range(400):
        buttons = Buttons(right=True, jump=(frame // 20) % 2 == 0)
        events = game.step(buttons)
        total_score += events.score_delta
        total_coins += events.coins
        if game.over:
            break

    assert total_score == game.score
    assert total_coins == game.coins
    assert game.coins >= 1


def test_nothing_is_ever_solid_where_the_player_stands_in_these_scenarios() -> None:
    game = make_game(put={(6, 12): "X", (7, 11): "X", (7, 12): "X", (9, 9): "B", (10, 9): "M"})
    for frame in range(500):
        game.step(Buttons(right=True, run=frame % 50 < 25, jump=frame % 30 < 15))
        assert not player_in_solid(game)
        assert not IS_SOLID[Tile.EMPTY]
        if game.over:
            break
