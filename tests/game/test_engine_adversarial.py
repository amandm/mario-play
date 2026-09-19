"""Adversarial tests of the game core: edge cases, a reference collision model and heavy fuzzing.

Written by a tester whose only goal was to break `mario_play.game`: corner and
sub-pixel collisions, the big hitbox, stomp/side-hit boundaries, shells, items,
timers, the flag, the level edges, `reset()`/`clone()` hygiene, determinism, and a
fuzz campaign in which `Invariants` re-derives after every single frame what the
rules of spec 3.1-3.5 allow (collisions, contacts, tiles, score, camera, timer)
from nothing but the public state before and after the step.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from mario_play.game.constants import (
    CAMERA_PLAYER_OFFSET,
    ENEMY_SPEED,
    FRAMES_PER_TIME_UNIT,
    INVULN_FRAMES,
    JUMP_IMPULSE_FAST,
    LEVEL_H_PX,
    LEVEL_H_TILES,
    MAX_FALL,
    MUSHROOM_H,
    MUSHROOM_SPEED,
    MUSHROOM_W,
    PLAYER_BIG_H,
    PLAYER_SMALL_H,
    PLAYER_W,
    RUN_MAX,
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
    TURTLE_W,
    VIEW_W,
    WALKER_H,
    WALKER_W,
)
from mario_play.game.engine import Buttons, Game, StepEvents
from mario_play.game.entities import Entity, Mushroom, Turtle, Walker
from mario_play.game.level import Level, list_levels, load_level
from mario_play.game.physics import HIT_CEILING, HIT_FLOOR, HIT_NONE, box_hits_solid, move_x, move_y
from mario_play.game.tiles import IS_SOLID, Tile

NOOP = Buttons()
RIGHT = Buttons(right=True)
LEFT = Buttons(left=True)
JUMP = Buttons(jump=True)
RIGHT_JUMP = Buttons(right=True, jump=True)
RUN_RIGHT = Buttons(right=True, run=True)
RUN_RIGHT_JUMP = Buttons(right=True, run=True, jump=True)

FLOOR_Y = 13 * TILE  # top of the ground in `make_game` levels
WALKER_Y = float(FLOOR_Y - WALKER_H)
TURTLE_Y = float(FLOOR_Y - TURTLE_H)
SHELL_Y = float(FLOOR_Y - SHELL_H)


# --- helpers -------------------------------------------------------------------------------------


def make_game(
    width: int = 64,
    put: dict[tuple[int, int], str] | None = None,
    pits: tuple[tuple[int, int], ...] = (),
    start: tuple[int, int] = (2, 12),
    time: int | None = None,
    flag_rows: range = range(2, 13),
    seed: int | None = None,
) -> Game:
    """A flat test level: ground rows 13-14, flag near the right end, `put[(col, row)] = char`."""
    grid = [["."] * width for _ in range(LEVEL_H_TILES)]
    for row in (13, 14):
        grid[row] = ["#"] * width
    for first, last in pits:
        for col in range(first, last + 1):
            grid[13][col] = grid[14][col] = "."
    for row in flag_rows:
        grid[row][width - 10] = "F"
    grid[start[1]][start[0]] = "S"
    for (col, row), char in (put or {}).items():
        grid[row][col] = char
    header = f"; time={time}\n" if time else ""
    text = header + "\n".join("".join(row) for row in grid)
    return Game(Level.from_string(text, name="adversarial"), seed=seed)


def hold(game: Game, buttons: Buttons, frames: int) -> list[StepEvents]:
    return [game.step(buttons) for _ in range(frames)]


def step_until(game: Game, buttons: Buttons, condition, limit: int = 600) -> StepEvents:
    """Step until `condition(events)` holds; returns the events of that frame."""
    for _ in range(limit):
        events = game.step(buttons)
        if condition(events):
            return events
    raise AssertionError(f"condition not reached within {limit} frames: {game!r}")


def place(game: Game, x: float, bottom: float = FLOOR_Y, vx: float = 0.0, vy: float = 0.0) -> None:
    """Teleport the player: left edge at `x`, feet at `bottom`, airborne until the next frame."""
    p = game.player
    p.x, p.y, p.vx, p.vy = float(x), float(bottom - p.h), float(vx), float(vy)
    p.on_ground = False


def still(entity: Entity) -> Entity:
    """Freeze an enemy horizontally so that contact geometry is exact."""
    entity.vx = 0.0
    entity.frozen_by_test = True  # tells `Invariants` that the odd speed is on purpose
    return entity


def resting_shell(x: float, y: float = SHELL_Y) -> Turtle:
    shell = Turtle(x, y - (TURTLE_H - SHELL_H))
    shell.to_shell()
    shell.contact_cooldown = 0
    assert shell.y == y and shell.h == SHELL_H
    return shell


def moving_shell(x: float, direction: int, y: float = SHELL_Y) -> Turtle:
    shell = resting_shell(x, y)
    shell.kick(direction)
    shell.contact_cooldown = 0
    return shell


def solid_cells(game: Game, box) -> list[tuple[int, int]]:
    """Solid tiles (or out-of-level columns) overlapped by the half-open hitbox of `box`.

    Deliberately independent of `physics.py`: plain `math.floor`/`math.ceil` on the box.
    """
    level = game.level
    cells = []
    for col in range(math.floor(box.x / TILE), math.ceil((box.x + box.w) / TILE)):
        for row in range(math.floor(box.y / TILE), math.ceil((box.y + box.h) / TILE)):
            if not 0 <= col < level.width_tiles:
                cells.append((col, row))
            elif 0 <= row < LEVEL_H_TILES and IS_SOLID[level.tiles[row, col]]:
                cells.append((col, row))
    return cells


def tiles_under(game: Game, box) -> set[int]:
    """Ids of the tiles overlapped by the half-open hitbox of `box` (inside the level only)."""
    level = game.level
    return {
        int(level.tiles[row, col])
        for col in range(math.floor(box.x / TILE), math.ceil((box.x + box.w) / TILE))
        for row in range(math.floor(box.y / TILE), math.ceil((box.y + box.h) / TILE))
        if 0 <= col < level.width_tiles and 0 <= row < LEVEL_H_TILES
    }


def entity_state(e: Entity) -> tuple:
    return tuple(sorted((k, v) for k, v in vars(e).items()))


def full_state(game: Game, rng: bool = True) -> tuple:
    """Every bit of simulation state, including private counters, entities and the tile grid."""
    scalars = tuple(
        sorted(
            (k, v)
            for k, v in vars(game).items()
            if isinstance(v, (int, float, bool, str, type(None)))
        )
    )
    return (
        scalars,
        entity_state(game.player),
        tuple((type(e).__name__, entity_state(e)) for e in game.entities),
        game.level.tiles.tobytes(),
        game.level.tiles.shape,
        game.rng.getstate() if rng else None,
    )


def _overlap(a, b) -> bool:
    return a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h


def _dangerous(e: Entity, p: Entity) -> bool:
    """Would touching `e` hurt a vulnerable player who is not stomping it?"""
    if isinstance(e, Walker):
        return e.squished_frames == 0
    if not isinstance(e, Turtle) or e.contact_cooldown > 0:
        return False
    if e.state == "shell_moving":  # only its leading side is dangerous
        return (e.vx > 0) != (e.x + e.w / 2 >= p.x + p.w / 2)
    return e.state == "walk"


def _condition(e: Entity) -> object:
    return e.squished_frames if isinstance(e, Walker) else getattr(e, "state", None)


class Invariants:
    """Frame-by-frame global invariants and rule oracles; call `check` after every `Game.step`.

    The checker remembers the previous frame (tiles, entities, player) and verifies that
    what changed is explained by the returned `StepEvents`. After touching the game by
    hand (teleports, `reset()`, added entities, edited counters) call `rebase()`.
    """

    def __init__(self, game: Game) -> None:
        self.game = game
        self.rebase()

    def rebase(self) -> None:
        """Take the current state as the new reference."""
        game = self.game
        p = game.player
        self.camera_x = game.camera_x
        self.score = game.score
        self.coins = game.coins
        self.frame = game.frame
        self.time_offset = game.time_left + game.frame // FRAMES_PER_TIME_UNIT
        self.tiles = game.level.tiles.copy()
        self.entities = list(game.entities)  # holding them also keeps their ids unique
        self.before = {id(e): (e.x, e.y + e.h, _condition(e)) for e in game.entities}
        self.player = (p.x, p.y + p.h, p.big, p.invuln_frames)

    def check(self, events: StepEvents, context: str = "", buttons: Buttons | None = None) -> None:
        self._check_numbers(context)
        self._check_bookkeeping(events, context)
        self._check_tiles_and_score(events, context)
        self._check_player(events, context, buttons)
        self._check_entities(context)
        self._check_contacts(events, context)
        self._check_enemy_pairs(context)
        self._check_ending(events, context)
        self.rebase()

    def _check_numbers(self, context: str) -> None:
        game = self.game
        p = game.player
        for name in ("x", "y", "vx", "vy"):
            value = getattr(p, name)
            assert type(value) is float and math.isfinite(value), (context, name, value)
        assert type(game.camera_x) is float and math.isfinite(game.camera_x), context
        assert abs(p.vx) <= RUN_MAX + 1e-9, (context, p.vx)
        assert JUMP_IMPULSE_FAST - 1e-9 <= p.vy <= MAX_FALL + 1e-9, (context, p.vy)

    def _check_bookkeeping(self, events: StepEvents, context: str) -> None:
        game = self.game
        p = game.player
        level = game.level
        assert game.frame == self.frame + 1, context
        assert game.score - self.score == events.score_delta >= 0, context
        assert game.coins - self.coins == events.coins >= 0, context

        # Spec 3.4: camera_x = clamp(player.x - 112, prev_camera_x, width_px - 256).
        furthest = float(max(level.width_px - VIEW_W, 0))
        assert game.camera_x == max(self.camera_x, min(p.x - CAMERA_PLAYER_OFFSET, furthest)), (
            context,
            game.camera_x,
            self.camera_x,
            p.x,
        )
        assert 0.0 <= game.camera_x <= furthest, context

        # Spec 3.1: one unit of time every 24 frames; whoever dies or wins first stops the clock.
        expected = self.time_offset - game.frame // FRAMES_PER_TIME_UNIT
        if game.over and game.death_cause != "timeout":
            assert game.time_left in (expected, expected + 1), context
        else:
            assert game.time_left == max(expected, 0), (context, game.time_left, expected)

    def _check_tiles_and_score(self, events: StepEvents, context: str) -> None:
        game = self.game
        tiles = game.level.tiles
        changes: dict[tuple[int, int], int] = {}
        for row, col in np.argwhere(self.tiles != tiles):
            key = (int(self.tiles[row, col]), int(tiles[row, col]))
            changes[key] = changes.get(key, 0) + 1
        coins_taken = changes.pop((Tile.COIN, Tile.EMPTY), 0)
        coin_blocks = changes.pop((Tile.QUESTION_COIN, Tile.USED), 0)
        shroom_blocks = changes.pop((Tile.QUESTION_MUSHROOM, Tile.USED), 0)
        bricks = changes.pop((Tile.BRICK, Tile.EMPTY), 0)
        assert not changes, (context, "illegal tile change", changes)
        assert coin_blocks + shroom_blocks + bricks <= 1, (context, "one head bump per frame")
        assert events.coins == coins_taken + coin_blocks, context
        assert events.bricks == bricks and (not bricks or self.player[2]), context

        born = [e for e in game.entities if id(e) not in self.before]
        assert sum(isinstance(e, Mushroom) for e in born) == shroom_blocks, context
        alive_ids = {id(e) for e in game.entities}
        gone = [e for e in self.entities if id(e) not in alive_ids]
        assert all(not e.alive for e in gone), (context, "removed although alive")
        assert events.powerups <= sum(isinstance(e, Mushroom) for e in gone), context

        explained = (
            SCORE_COIN * events.coins
            + SCORE_BRICK * events.bricks
            + SCORE_STOMP * events.stomps
            + SCORE_MUSHROOM * events.powerups
            + (SCORE_FLAG + SCORE_FLAG_PER_TIME * game.time_left if events.won else 0)
        )
        shell_kills, rest = divmod(events.score_delta - explained, SCORE_SHELL_KILL)
        enemies_gone = sum(not isinstance(e, Mushroom) for e in gone)
        assert rest == 0 and 0 <= shell_kills <= enemies_gone, (context, events, shell_kills)

    def _check_player(self, events: StepEvents, context: str, buttons: Buttons | None) -> None:
        game = self.game
        p = game.player
        level = game.level
        x, feet, was_big, was_invuln = self.player
        assert game.camera_x <= p.x <= level.width_px - p.w, (context, p, game.camera_x)
        assert p.w == PLAYER_W and p.h == (PLAYER_BIG_H if p.big else PLAYER_SMALL_H), context
        assert not solid_cells(game, p), (context, "player inside solid", solid_cells(game, p), p)
        if p.on_ground and not game.over:
            ground = Entity(p.x, p.y + p.h, p.w, 1)
            assert solid_cells(game, ground), (context, "on_ground in mid-air", p)

        # No teleporting: one frame moves at most one frame's worth; the feet ignore resizing.
        assert abs(p.x - x) <= RUN_MAX + 1e-9, (context, "x jumped", x, p.x)
        assert JUMP_IMPULSE_FAST - 1e-9 <= p.y + p.h - feet <= MAX_FALL + 1e-9, (context, feet, p)

        if events.hurt:
            assert (was_big or events.powerups) and not p.big and was_invuln == 0, context
            assert p.invuln_frames == INVULN_FRAMES, context
            # Shrinking deep in a pit can put the head below the level in the same frame.
            assert events.death_cause in (None, "pit", "timeout"), (context, events)
        else:
            assert p.invuln_frames == max(was_invuln - 1, 0), (context, was_invuln, p.invuln_frames)
            assert p.big == (was_big or events.powerups > 0), (context, "size changed", events)
        if events.stomps and not game.over:
            assert not p.on_ground, context
            if buttons is None:
                assert p.vy in (STOMP_BOUNCE, STOMP_BOUNCE_HELD), (context, p.vy)
            else:
                assert p.vy == (STOMP_BOUNCE_HELD if buttons.jump else STOMP_BOUNCE), context

    def _check_entities(self, context: str) -> None:
        game = self.game
        level = game.level
        for e in game.entities:
            for name in ("x", "y", "vx", "vy"):
                value = getattr(e, name)
                assert type(value) is float and math.isfinite(value), (context, e, name)
            assert e.alive, (context, "dead entity kept", e)
            assert 0.0 <= e.x <= level.width_px - e.w, (context, e)
            # A turtle stomped at the very bottom shrinks to a shell whose top is 8 px lower.
            lowest = LEVEL_H_PX + (TURTLE_H - SHELL_H if isinstance(e, Turtle) else 0)
            assert e.y <= lowest, (context, "below the level but still there", e)
            assert 0.0 <= e.vy <= MAX_FALL, (context, e)
            assert e.facing in (-1, 1), (context, e)
            assert not solid_cells(game, e), (context, "entity in solid", solid_cells(game, e), e)
            if isinstance(e, Walker):
                assert (e.w, e.h) == (WALKER_W, WALKER_H), context
                speed = 0.0 if e.squished_frames else ENEMY_SPEED
                assert 0 <= e.squished_frames <= SQUISH_FRAMES, context
            elif isinstance(e, Turtle):
                speed = {"walk": ENEMY_SPEED, "shell": 0.0, "shell_moving": SHELL_SPEED}[e.state]
                assert (e.w, e.h) == (TURTLE_W, TURTLE_H if e.state == "walk" else SHELL_H)
            else:
                assert isinstance(e, Mushroom), (context, e)
                speed = MUSHROOM_SPEED
            if not hasattr(e, "frozen_by_test"):
                assert abs(e.vx) == speed, (context, e, vars(e))
            if e.vx:
                assert (e.vx > 0) == (e.facing > 0), (context, "moonwalking", e, e.facing)
            if id(e) in self.before:
                x, feet, _ = self.before[id(e)]
                assert abs(e.x - x) <= SHELL_SPEED + 1e-9, (context, "entity x jumped", e, x)
                fell = e.y + e.h - feet
                assert -1e-9 <= fell <= MAX_FALL + 1e-9, (context, "entity feet jumped", e, feet)

    def _check_contacts(self, events: StepEvents, context: str) -> None:
        """Stomps, hurts and deaths by enemy happen exactly when the geometry says so.

        Contacts are resolved after all movement, so the positions at the end of the
        frame are the ones the engine looked at (only heights may differ, feet never do).
        """
        game = self.game
        p = game.player
        _, feet_before, was_big, was_invuln = self.player
        feet = p.y + p.h
        # Equivalent to "vy > 0 when the contacts were resolved", which the bounce overwrites.
        moving_down = feet > feet_before and not p.on_ground

        stomped = []
        for e in self.entities:
            then, now = self.before[id(e)][2], _condition(e)
            if isinstance(e, Walker) and then == 0 and now == SQUISH_FRAMES:
                stomped.append(e.y + e.h / 2)
            elif isinstance(e, Turtle) and now == "shell" and then != "shell":
                stomped.append(e.y + e.h - (TURTLE_H if then == "walk" else SHELL_H) / 2)
        assert events.stomps == len(stomped), (context, events, stomped)
        if stomped and not game.over:
            # `>=`: at the very apex of a jump vy can be +2e-15, too little to move the feet.
            rising = feet < feet_before or p.on_ground
            assert not rising, (context, "stomp without moving down", p, feet_before)
            assert all(feet < midpoint for midpoint in stomped), (context, "stomp from the side")

        harmed = events.hurt or (events.died and events.death_cause == "enemy")
        if harmed:
            assert was_invuln == 0 and not events.stomps, (context, events)
            # The big hitbox is gone by now; rebuilding it from the feet is off by float rounding.
            slack = 1e-9
            body = Entity(p.x, feet - PLAYER_BIG_H - slack, p.w, PLAYER_BIG_H + slack)
            body = body if events.hurt else p
            culprits = [e for e in self.entities if _overlap(body, e) and _dangerous(e, p)]
            assert culprits, (context, "hurt by nothing", events, p)
            if moving_down:
                assert all(feet >= e.y + e.h / 2 for e in culprits), (context, "missed stomp", p)
        elif not (game.over or events.stomps or events.powerups or was_invuln):
            for e in game.entities:
                if id(e) in self.before and _overlap(p, e) and _dangerous(e, p):
                    raise AssertionError((context, "touched a dangerous enemy for free", e, p))

    def _check_enemy_pairs(self, context: str) -> None:
        """After a frame no live enemy overlaps a moving shell and bumping walkers head apart.

        Enemies removed during this frame still count as neighbours: only a bump between
        exactly two enemies has an outcome that does not depend on who was handled last.
        """
        enemies = [
            e
            for e in self.entities
            if not isinstance(e, Mushroom) and not (isinstance(e, Walker) and e.squished_frames)
        ]
        enemies.sort(key=lambda e: e.x)
        touching: dict[int, list[Entity]] = {id(e): [] for e in enemies}
        for i, a in enumerate(enemies):
            for b in enemies[i + 1 :]:
                if b.x >= a.x + a.w:
                    break
                if _overlap(a, b):
                    touching[id(a)].append(b)
                    touching[id(b)].append(a)
        for a in enemies:
            for b in touching[id(a)]:
                if not (a.alive and b.alive):
                    continue
                shells = [e for e in (a, b) if isinstance(e, Turtle) and e.state == "shell_moving"]
                assert not shells, (context, "survived a moving shell", a, b)
                just_the_two = len(touching[id(a)]) == len(touching[id(b)]) == 1
                if just_the_two and a.x + a.w / 2 < b.x + b.w / 2:
                    assert a.vx <= 0.0 <= b.vx, (context, "bumped but not heading apart", a, b)

    def _check_ending(self, events: StepEvents, context: str) -> None:
        game = self.game
        p = game.player
        under = tiles_under(game, p)
        touches_flag = bool(under & {Tile.FLAGPOLE, Tile.FLAG_TOP})
        if game.won:
            assert touches_flag, (context, "won without touching the flag", p)
        elif not events.powerups:  # growing happens after the tiles were touched
            assert not touches_flag, (context, "touched the flag without winning", p)
            assert Tile.COIN not in under, (context, "walked through a coin", p)
        if game.over:
            assert game.won != (game.death_cause is not None), context
            assert game.death_cause in (None, "pit", "enemy", "timeout"), context
            assert (events.won, events.died) == (game.won, not game.won), (context, events)
            assert events.death_cause == game.death_cause, context
            assert p.dead == (not game.won), context
            if game.death_cause == "timeout":
                assert game.time_left == 0, context
            if game.death_cause == "pit":
                assert p.y > LEVEL_H_PX, context
        else:
            assert not (events.won or events.died) and events.death_cause is None, context
            assert game.death_cause is None and not game.won and not p.dead, context
            assert game.time_left > 0 and p.y <= LEVEL_H_PX, context


def run_checked(game: Game, script, context: str = "") -> list[StepEvents]:
    """Step through `script` checking every invariant after every frame (stops when over)."""
    watch = Invariants(game)
    out = []
    for frame, buttons in enumerate(script):
        events = game.step(buttons)
        watch.check(events, f"{context} frame={frame}", buttons)
        out.append(events)
        if game.over:
            break
    return out


def test_the_invariant_checker_itself_notices_a_player_inside_a_wall() -> None:
    game = make_game(put={(8, 12): "X"})
    watch = Invariants(game)
    place(game, 8 * TILE + 1.0)

    with pytest.raises(AssertionError, match="player inside solid"):
        watch.check(game.step(NOOP))


# --- corners, gaps and walls ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "lands"),
    [
        (10 * TILE - PLAYER_W, False),  # right edge flush with the block's left edge: slips by
        (10 * TILE - PLAYER_W + 0.25, True),  # a quarter pixel over the corner: lands
        (11 * TILE - 0.25, True),
        (11 * TILE, False),
    ],
)
def test_landing_on_the_exact_corner_of_a_block(x: float, lands: bool) -> None:
    game = make_game(put={(10, 10): "X"})
    p = game.player
    place(game, x, bottom=6 * TILE)

    run_checked(game, [NOOP] * 60)

    assert p.on_ground and p.x == x
    assert p.y + p.h == (10 * TILE if lands else FLOOR_Y)


def test_hugging_a_floating_wall_while_falling_slides_down_and_slips_under_it() -> None:
    game = make_game(put={(10, row): "X" for row in range(4, 10)})
    p = game.player
    wall_left, wall_top, wall_bottom = 10 * TILE, 4 * TILE, 10 * TILE
    place(game, wall_left - PLAYER_W - 2.0, bottom=wall_top + 20.0)
    watch = Invariants(game)

    ys, flush_frames = [], 0
    for _ in range(80):
        watch.check(game.step(RIGHT))
        ys.append(p.y)
        if p.y < wall_bottom:  # still beside the wall
            assert p.x <= wall_left - PLAYER_W
            flush_frames += p.x == wall_left - PLAYER_W and p.vx == 0.0
        if p.on_ground:
            break

    assert p.on_ground and p.y + p.h == FLOOR_Y
    assert flush_frames > 10, "the player really slid down the wall"
    assert ys == sorted(ys), "pushing into a wall must not slow the fall"
    hold(game, RIGHT, 30)
    assert p.x > wall_left, "once below the wall the player walks under it"


@pytest.mark.parametrize(
    ("x", "passes"),
    [
        (6 * TILE + 2.0, True),  # centred under the hole
        (6 * TILE, True),  # flush with the left rim
        (6 * TILE + 4.0, True),  # flush with the right rim
        (6 * TILE - 0.25, False),  # a quarter pixel under the left block
        (6 * TILE + 4.25, False),  # a quarter pixel under the right block
    ],
)
def test_jumping_into_a_one_tile_hole_in_a_ceiling(x: float, passes: bool) -> None:
    game = make_game(put={(col, 9): "X" for col in range(3, 12) if col != 6})
    p = game.player
    place(game, x)
    game.step(NOOP)
    watch = Invariants(game)

    top = p.y
    for _ in range(70):
        watch.check(game.step(JUMP))
        top = min(top, p.y)

    assert p.x == x
    if passes:
        assert top < 9 * TILE, "the head should rise through the hole"
    else:
        assert top == 10 * TILE, "the head should stop flush under the ceiling"
    assert p.on_ground and p.y + p.h == FLOOR_Y


def test_one_tile_slot_in_the_floor_can_be_entered_and_left() -> None:
    game = make_game(put={(8, 12): "X", (10, 12): "X"})
    p = game.player
    place(game, 9 * TILE + 2.0, bottom=10 * TILE)

    run_checked(game, [NOOP] * 40)
    assert p.on_ground and p.y + p.h == FLOOR_Y
    run_checked(game, [LEFT] * 20)
    assert p.x == 9 * TILE and p.vx == 0.0
    run_checked(game, [RIGHT] * 20)
    assert p.x == 10 * TILE - PLAYER_W and p.vx == 0.0

    run_checked(game, [RIGHT_JUMP] * 45)
    run_checked(game, [RIGHT] * 30)
    assert p.x > 10 * TILE, "jumping while pushing right must get the player out of the slot"


@pytest.mark.parametrize("offset", [i * 0.37 for i in range(0, 44, 3)])
@pytest.mark.parametrize("run", [False, True])
def test_sub_pixel_sweep_through_a_block_garden(offset: float, run: bool) -> None:
    """Steps, slots, low ceilings and floating blocks approached from every sub-pixel phase."""
    put = {
        (8, 12): "X",
        (10, 12): "X",  # a one-tile slot
        (13, 12): "X",
        (14, 11): "X",
        (14, 12): "X",  # stairs
        (17, 10): "B",
        (18, 10): "?",
        (19, 10): "M",  # a low roof, 2 tiles above the floor
        (22, 9): "X",
        (23, 9): "X",
        (25, 12): "[",
        (26, 12): "]",
        (25, 11): "[",
        (26, 11): "]",
        (29, 8): "B",
        (30, 8): "B",
    }
    game = make_game(width=48, put=put)
    place(game, 40.0 + offset)
    script = [
        Buttons(right=True, run=run, jump=(frame // 9) % 3 != 0, left=frame % 131 > 120)
        for frame in range(700)
    ]

    run_checked(game, script, f"offset={offset} run={run}")

    assert game.player.x > 12 * TILE


@pytest.mark.parametrize("phase", [i * 0.31 for i in range(10)])
@pytest.mark.parametrize("buttons", [RIGHT, RUN_RIGHT], ids=["walk", "run"])
def test_one_tile_pit_never_clips_the_player_into_its_walls(buttons: Buttons, phase: float) -> None:
    """The snag case: a 12 px player dips into a 16 px pit and meets the far wall side-on."""
    game = make_game(pits=((12, 12),))
    place(game, 40.0 + phase)

    events = run_checked(game, [buttons] * 400)

    p = game.player
    if game.over:
        assert events[-1].death_cause == "pit" and 12 * TILE <= p.x <= 13 * TILE - PLAYER_W
    else:
        assert p.x > 13 * TILE and p.on_ground


def test_left_camera_wall_holds_against_every_input() -> None:
    game = make_game(width=100)
    p = game.player
    place(game, 40.3)
    run_checked(game, [RUN_RIGHT] * 200)
    run_checked(game, [LEFT] * 30)  # skid to a halt and turn around
    camera = game.camera_x
    assert camera > 100 and camera != int(camera), "a fractional camera is the nasty case"

    script = [Buttons(left=True, run=True, jump=frame % 40 < 20) for frame in range(200)]
    run_checked(game, script)

    assert game.camera_x == camera
    assert p.x == camera and p.vx == 0.0
    run_checked(game, [RIGHT] * 5)
    assert p.x > camera


def test_enemies_are_not_stopped_by_the_camera_wall_or_the_level_edges() -> None:
    game = make_game(width=100)
    hold(game, RUN_RIGHT, 300)
    camera = game.camera_x
    walker = Walker(camera + 4.0, WALKER_Y, facing=-1)
    game.entities.append(walker)
    place(game, camera + 150.0)

    hold(game, NOOP, 60)
    assert walker.x < camera - 20.0, "the screen edge is a wall for the player only"

    walker.x = 3.0
    hold(game, NOOP, 20)
    assert walker.facing == 1 and walker.vx == ENEMY_SPEED and walker.x >= 0.0

    right_end = game.level.width_px - WALKER_W
    walker.x = right_end - 3.0
    xs = [(game.step(NOOP), walker.x)[1] for _ in range(20)]
    assert max(xs) == right_end and walker.facing == -1


def test_head_bump_at_full_run_speed_keeps_the_momentum() -> None:
    game = make_game(put={(col, 9): "?" for col in range(20, 30)})
    p = game.player
    step_until(game, RUN_RIGHT, lambda e: p.x > 22 * TILE)
    assert p.vx == RUN_MAX
    watch = Invariants(game)

    for _ in range(40):
        centre_before = p.x + p.w / 2 + p.vx
        events = game.step(RUN_RIGHT_JUMP)
        watch.check(events)
        if events.coins:
            break

    assert events.coins == 1 and events.score_delta == SCORE_COIN
    assert p.vx == RUN_MAX and p.vy == 0.0 and p.y == 10 * TILE
    used = [col for col in range(20, 30) if game.level.tiles[9, col] == Tile.USED]
    assert used == [int(centre_before // TILE)], "the block above the player's centre is hit"


# --- a slow reference model of tile collision ----------------------------------------------------


def _random_room(rng: random.Random, width: int = 24) -> Game:
    grid = [["."] * width for _ in range(LEVEL_H_TILES)]
    for row in range(LEVEL_H_TILES):
        for col in range(width):
            if rng.random() < 0.22:
                grid[row][col] = rng.choice("#XB?M")
    grid[12][1], grid[11][1] = "S", "."
    grid[3][width - 2] = "F"
    return Game(Level.from_string("\n".join("".join(row) for row in grid), name="room"))


def _reference_slide(game: Game, box: Entity, dx: float, dy: float) -> tuple[float, float, bool]:
    """Where a box ends up when slid by brute force in 1/8 px increments (exact in binary)."""
    step = 0.125
    x, y = box.x, box.y
    remaining = abs(dx) if dx else abs(dy)
    sign = 1.0 if (dx or dy) > 0 else -1.0
    while remaining > 0:
        delta = min(step, remaining) * sign
        probe = Entity(x + (delta if dx else 0.0), y + (delta if dy else 0.0), box.w, box.h)
        if solid_cells(game, probe):
            return x, y, True
        x, y = probe.x, probe.y
        remaining -= abs(delta)
    return x, y, False


@pytest.mark.parametrize("seed", range(6))
def test_move_helpers_agree_with_a_brute_force_reference(seed: int) -> None:
    rng = random.Random(seed)
    game = _random_room(rng)
    width_px = game.level.width_px
    sizes = [(12, 15), (12, 30), (14, 14), (14, 22)]
    checked = 0
    while checked < 250:
        w, h = rng.choice(sizes)
        # Multiples of 1/8 px keep the reference exact; rows may poke out above and below.
        x = rng.randrange(0, (width_px - w) * 8) / 8
        y = rng.randrange(-40 * 8, (LEVEL_H_PX + 20) * 8) / 8
        box = Entity(x, y, w, h)
        if solid_cells(game, box):
            assert box_hits_solid(game.level, x, y, w, h)
            continue
        assert not box_hits_solid(game.level, x, y, w, h)
        distance = rng.choice(
            [rng.randrange(-60 * 8, 60 * 8) / 8, rng.choice([-16.0, -8.0, 8.0, 16.0])]
        )
        if distance == 0:
            continue
        checked += 1

        horizontal, vertical = Entity(x, y, w, h), Entity(x, y, w, h)
        blocked = move_x(game.level, horizontal, distance)
        want_x, _, want_blocked = _reference_slide(game, box, distance, 0.0)
        assert (horizontal.x, horizontal.y, blocked) == (want_x, y, want_blocked), (box, distance)

        hit = move_y(game.level, vertical, distance)
        _, want_y, want_blocked = _reference_slide(game, box, 0.0, distance)
        want_hit = (HIT_FLOOR if distance > 0 else HIT_CEILING) if want_blocked else HIT_NONE
        assert (vertical.x, vertical.y, hit) == (x, want_y, want_hit), (box, distance)


# --- the big hitbox ------------------------------------------------------------------------------


def _room(ceiling_row: int) -> Game:
    """A closed room between the walls at columns 4 and 14 with a roof at `ceiling_row`."""
    put = {(col, ceiling_row): "X" for col in range(4, 15)}
    put.update({(col, row): "X" for col in (4, 14) for row in range(ceiling_row, 13)})
    return make_game(put=put)


def test_growing_waits_for_head_room_and_happens_once_there_is_some() -> None:
    game = _room(ceiling_row=10)  # 32 px high inside: the big player fits only near the floor
    p = game.player
    place(game, 8 * TILE, bottom=FLOOR_Y - 5.0)
    shroom = Mushroom(p.x - 1.0, FLOOR_Y - MUSHROOM_H)
    game.entities.append(shroom)
    watch = Invariants(game)

    events = game.step(NOOP)
    watch.check(events)
    assert not p.big and shroom.alive and events.powerups == 0, "no head room 4.5 px up"

    for _ in range(10):
        events = game.step(NOOP)
        watch.check(events)
        if events.powerups:
            break

    assert events.powerups == 1 and events.score_delta == SCORE_MUSHROOM
    assert p.big and p.y >= 11 * TILE and not shroom.alive
    for _ in range(5):
        watch.check(game.step(NOOP))
    assert p.on_ground and p.y + p.h == FLOOR_Y


def test_big_player_keeps_playing_in_a_two_tile_corridor() -> None:
    game = _room(ceiling_row=10)
    game.level.tiles[10, 7:11] = Tile.BRICK
    game.level.tiles[11:13, 14] = Tile.EMPTY  # open the right wall
    p = game.player
    p.grow()
    place(game, 5 * TILE + 1.0)
    script = [Buttons(right=True, jump=frame % 12 < 6) for frame in range(400)]

    events = run_checked(game, script)

    assert p.big and p.x > 15 * TILE, "jumping into the low roof must not trap the player"
    broken = sum(e.bricks for e in events)
    assert broken >= 1 and game.score == broken * SCORE_BRICK
    assert int((game.level.tiles[10, 7:11] == Tile.EMPTY).sum()) == broken


def test_block_at_head_height_stops_the_big_player_but_not_the_small_one() -> None:
    for big, expected_x in ((True, 8 * TILE - PLAYER_W), (False, None)):
        game = make_game(put={(8, 11): "X"})
        p = game.player
        if big:
            p.grow()
            place(game, p.x)
        run_checked(game, [RIGHT] * 200)
        if expected_x is None:
            assert p.x > 9 * TILE
        else:
            assert p.x == expected_x and p.vx == 0.0


def test_shrinking_keeps_the_feet_and_never_clips() -> None:
    game = _room(ceiling_row=10)
    p = game.player
    p.grow()
    place(game, 8 * TILE, bottom=FLOOR_Y - 1.0, vy=-1.0)
    game.entities.append(still(Walker(p.x + 4.0, WALKER_Y)))
    watch = Invariants(game)

    events = game.step(NOOP)
    watch.check(events)

    assert events.hurt and not events.died
    assert not p.big and p.h == PLAYER_SMALL_H and p.invuln_frames == INVULN_FRAMES
    assert p.y + p.h == FLOOR_Y - 1.0 - 0.5, "the feet moved only by this frame's velocity"


def test_one_tile_tunnel_lets_the_small_player_in_and_keeps_the_big_one_out() -> None:
    put = {(col, 11): "X" for col in range(6, 12)}
    tunnel_left, tunnel_right, roof_top = 6 * TILE, 12 * TILE, 11 * TILE
    for big in (False, True):
        game = make_game(put=put)
        p = game.player
        if big:
            p.grow()
            place(game, p.x)
        watch = Invariants(game)
        frames_inside = 0
        for frame in range(300):
            watch.check(game.step(Buttons(right=True, jump=frame > 60 and frame % 50 > 44)))
            if tunnel_left - p.w < p.x < tunnel_right and p.y + p.h > roof_top:
                frames_inside += 1
                assert not big or p.x <= tunnel_left - p.w, "the big player never gets under it"

        assert p.x > tunnel_right, "through it when small, over it when big"
        assert big or frames_inside > 50


# --- stomp versus side hit -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("feet_above_midpoint", "stomp"),
    [(4.0, True), (0.25, True), (0.0, False), (-0.25, False), (-4.0, False)],
)
@pytest.mark.parametrize("enemy", ["walker", "turtle", "shell_moving"])
def test_stomp_needs_the_feet_strictly_above_the_enemy_midpoint(
    enemy: str, feet_above_midpoint: float, stomp: bool
) -> None:
    game = make_game()
    p = game.player
    if enemy == "walker":
        target: Entity = still(Walker(100.0, WALKER_Y))
    elif enemy == "turtle":
        target = still(Turtle(100.0, TURTLE_Y))
    else:
        target = moving_shell(100.0 + SHELL_SPEED, -1)  # at x=100, coming at the player, on contact
    game.entities.append(target)
    midpoint = target.y + target.h / 2
    # vy 0.5 becomes 1.0 after gravity, so the feet end this frame exactly where we want them.
    place(game, 99.0, bottom=midpoint - feet_above_midpoint - 1.0, vy=0.5)

    events = game.step(NOOP)

    assert target.x == 100.0 and p.y + p.h == midpoint - feet_above_midpoint
    assert events.stomps == int(stomp) and events.died == (not stomp)
    if stomp:
        assert p.vy == STOMP_BOUNCE and events.score_delta == SCORE_STOMP and not game.over
    else:
        assert game.death_cause == "enemy"


def test_rising_into_an_enemy_is_never_a_stomp() -> None:
    game = make_game()
    walker = still(Walker(100.0, WALKER_Y - 40.0))
    walker.vy = 0.0
    game.entities.append(walker)
    place(game, 101.0, bottom=walker.y + 6.0, vy=-3.0)  # feet above the midpoint, moving up

    events = game.step(NOOP)

    assert events.died and events.stomps == 0 and game.death_cause == "enemy"


def test_two_enemies_stomped_in_the_same_frame() -> None:
    game = make_game()
    p = game.player
    left, right = still(Walker(96.0, WALKER_Y)), still(Walker(112.0, WALKER_Y))
    game.entities += [left, right]
    place(game, 105.0, bottom=WALKER_Y - 0.5, vy=2.0)  # 12 px wide: over both walkers

    events = game.step(NOOP)

    assert events.stomps == 2 and events.score_delta == 2 * SCORE_STOMP
    assert left.squished_frames == right.squished_frames == SQUISH_FRAMES
    assert p.vy == STOMP_BOUNCE and not game.over, "one bounce, however many enemies"


@pytest.mark.parametrize("turtle_first", [False, True])
def test_stomp_and_side_contact_in_one_frame_do_not_depend_on_entity_order(
    turtle_first: bool,
) -> None:
    """Feet between the turtle's midpoint and the walker's: stomps the walker, grazes the turtle."""
    game = make_game()
    p = game.player
    walker, turtle = still(Walker(100.0, WALKER_Y)), still(Turtle(108.0, TURTLE_Y))
    game.entities += [turtle, walker] if turtle_first else [walker, turtle]
    place(game, 99.0, bottom=FLOOR_Y - 9.5 - 3.5, vy=3.0)

    events = game.step(NOOP)

    assert p.y + p.h == FLOOR_Y - 9.5
    assert events.stomps == 1 and walker.squished_frames == SQUISH_FRAMES
    assert not events.died and not game.over, "the stomp of this frame protects the player"
    assert turtle.state == "walk" and p.vy == STOMP_BOUNCE


@pytest.mark.parametrize("mushroom_first", [False, True])
def test_mushroom_and_enemy_touched_in_one_frame_do_not_depend_on_entity_order(
    mushroom_first: bool,
) -> None:
    game = make_game()
    p = game.player
    place(game, 100.0)
    walker, shroom = still(Walker(108.0, WALKER_Y)), Mushroom(90.0, FLOOR_Y - MUSHROOM_H)
    game.entities += [shroom, walker] if mushroom_first else [walker, shroom]

    events = game.step(NOOP)

    assert events.powerups == 1 and events.hurt and not events.died
    assert not game.over and not p.big and p.invuln_frames == INVULN_FRAMES
    assert game.score == SCORE_MUSHROOM


def test_stomping_works_while_invulnerable_and_side_hits_do_not() -> None:
    game = make_game()
    p = game.player
    p.invuln_frames = 50
    walker = still(Walker(100.0, WALKER_Y))
    game.entities.append(walker)
    place(game, 101.0)

    assert not any(e.hurt or e.died for e in hold(game, NOOP, 10))

    place(game, 101.0, bottom=WALKER_Y - 1.0, vy=1.0)
    events = step_until(game, NOOP, lambda e: e.stomps or e.died, limit=5)
    assert events.stomps == 1 and walker.squished_frames == SQUISH_FRAMES


def test_hurt_and_pit_death_can_share_a_frame() -> None:
    """Found by the fuzzer: shrinking deep inside a pit drops the head below the level."""
    game = make_game(pits=((10, 12),))
    p = game.player
    p.grow()
    place(game, 11 * TILE + 2.0, bottom=LEVEL_H_PX + 21.0, vy=0.5)  # head at y=232 after the move
    game.entities.append(still(Walker(11 * TILE + 1.0, LEVEL_H_PX - 4.0)))

    (events,) = run_checked(game, [NOOP])

    assert events.hurt and events.died and events.death_cause == "pit"
    assert not p.big and p.y > LEVEL_H_PX and game.over and not game.won


def test_invulnerability_protects_for_exactly_120_frames_inside_an_enemy() -> None:
    game = make_game()
    p = game.player
    p.grow()
    place(game, 100.0)
    game.entities.append(still(Walker(104.0, WALKER_Y)))
    watch = Invariants(game)

    first = game.step(NOOP)
    watch.check(first)
    assert first.hurt and p.invuln_frames == INVULN_FRAMES

    for frame in range(INVULN_FRAMES):
        events = game.step(NOOP)
        watch.check(events)
        assert not events.hurt and not events.died, f"safe frame {frame + 1} of {INVULN_FRAMES}"
    assert p.invuln_frames == 0

    events = game.step(NOOP)
    assert events.died and game.death_cause == "enemy"


# --- shells --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("player_dx", "direction"), [(-5.0, 1), (7.0, -1)])
def test_landing_on_a_resting_shell_kicks_it_away_without_a_bounce(
    player_dx: float, direction: int
) -> None:
    game = make_game()
    p = game.player
    shell = resting_shell(200.0)
    game.entities.append(shell)
    place(game, shell.x + player_dx, bottom=shell.y - 0.5, vy=1.0)

    events = game.step(NOOP)

    assert shell.state == "shell_moving" and shell.vx == direction * SHELL_SPEED
    assert shell.facing == direction
    assert events.stomps == 0 and events.score_delta == 0 and p.vy > 0
    later = run_checked(game, [NOOP] * 40)
    assert not any(e.hurt or e.died or e.stomps for e in later)


def test_shell_bouncing_between_walls_hurts_then_kills() -> None:
    put = {(col, row): "X" for col in (4, 14) for row in (11, 12)}
    game = make_game(put=put)
    p = game.player
    p.grow()
    place(game, 9 * TILE)
    game.entities.append(resting_shell(9 * TILE + 20.0))

    step_until(game, RIGHT, lambda e: game.entities[0].state == "shell_moving", limit=40)
    script = [NOOP] * 600
    events = run_checked(game, script)

    hurt_at = [i for i, e in enumerate(events) if e.hurt]
    assert len(hurt_at) == 1, "shrunk once by the returning shell"
    assert events[-1].died and game.death_cause == "enemy"
    assert len(events) - 1 - hurt_at[0] > INVULN_FRAMES, "and killed only after the grace period"


def test_stomped_moving_shell_stops_and_the_next_landing_kicks_it_again() -> None:
    game = make_game()
    p = game.player
    shell = moving_shell(150.0, 1)
    game.entities.append(shell)
    place(game, 150.0 + 3 * SHELL_SPEED, bottom=SHELL_Y - 0.5, vy=1.0)

    events = step_until(game, NOOP, lambda e: e.stomps or e.died, limit=5)
    assert events.stomps == 1 and shell.state == "shell" and shell.vx == 0.0
    assert p.vy == STOMP_BOUNCE
    stopped_at = shell.x

    later = step_until(game, NOOP, lambda e: shell.state == "shell_moving" or e.died, limit=60)
    assert shell.state == "shell_moving" and shell.x == stopped_at and not later.died
    assert later.stomps == 0, "landing on a resting shell is a kick, not a stomp"
    assert not any(e.hurt or e.died for e in run_checked(game, [NOOP] * 60))


def test_moving_shell_kills_everything_in_its_way() -> None:
    game = make_game(width=80)
    shell = moving_shell(100.0, 1)
    victims = [
        Walker(160.0, WALKER_Y),
        Walker(200.0, WALKER_Y, facing=1),
        Turtle(260.0, TURTLE_Y),
        resting_shell(330.0),
        Walker(400.0, WALKER_Y),
    ]
    flat = Walker(125.0, WALKER_Y)
    flat.squish()
    shroom = Mushroom(430.0, FLOOR_Y - MUSHROOM_H)
    game.entities += [shell, *victims, flat, shroom]

    events = run_checked(game, [NOOP] * 200)

    assert not any(v.alive for v in victims) and all(v not in game.entities for v in victims)
    assert shell.alive and shell.state == "shell_moving" and shell.vx == SHELL_SPEED
    assert shroom.alive, "items are not enemies"
    assert game.score == len(victims) * SCORE_SHELL_KILL
    assert sum(e.stomps for e in events) == 0


def test_two_moving_shells_destroy_each_other() -> None:
    game = make_game()
    a, b = moving_shell(100.0, 1), moving_shell(300.0, -1)
    game.entities += [a, b]

    run_checked(game, [NOOP] * 60)

    assert not a.alive and not b.alive and game.entities == []
    assert game.score == 2 * SCORE_SHELL_KILL


@pytest.mark.parametrize("walker_first", [False, True])
def test_shells_colliding_on_top_of_a_walker_kill_it_whatever_the_entity_order(
    walker_first: bool,
) -> None:
    game = make_game()
    a, b = moving_shell(200.0 - SHELL_SPEED, 1), moving_shell(206.0 + SHELL_SPEED, -1)
    walker = Walker(203.0 + ENEMY_SPEED, WALKER_Y)  # under both shells once everything has moved
    game.entities += [walker, a, b] if walker_first else [a, b, walker]

    events = game.step(NOOP)

    assert (a.x, b.x, walker.x) == (200.0, 206.0, 203.0)
    assert not (a.alive or b.alive or walker.alive) and game.entities == []
    assert events.score_delta == 3 * SCORE_SHELL_KILL


def test_chasing_a_kicked_shell_at_full_speed_is_safe() -> None:
    game = make_game(width=120)
    game.entities.append(resting_shell(200.0))
    shell = game.entities[0]

    events = run_checked(game, [RUN_RIGHT] * 300)

    assert shell.state == "shell_moving" and shell.vx == SHELL_SPEED
    assert not any(e.hurt or e.died for e in events)


def test_shell_kicked_against_an_adjacent_wall_stays_sane() -> None:
    """The shell is back within its contact cooldown: whatever happens to the player, nothing
    may get stuck, stop or clip (today it passes through; see SHELL_CONTACT_COOLDOWN)."""
    game = make_game(put={(8, 12): "X", (8, 11): "X"})
    shell = resting_shell(8 * TILE - TURTLE_W)  # flush against the wall
    game.entities.append(shell)
    place(game, shell.x - 30.0)

    step_until(game, RIGHT, lambda e: shell.state == "shell_moving", limit=60)
    assert shell.vx == SHELL_SPEED, "kicked towards the wall, away from the player"
    run_checked(game, [NOOP] * 20)

    assert shell.state == "shell_moving" and shell.vx == -SHELL_SPEED
    assert shell.x + shell.w < game.player.x, "it bounced and left to the other side"


def test_kicked_shell_falls_into_a_pit_and_is_gone() -> None:
    game = make_game(pits=((14, 15),))
    game.entities.append(resting_shell(150.0))
    shell = game.entities[0]

    step_until(game, RIGHT, lambda e: shell.state == "shell_moving", limit=120)
    run_checked(game, [NOOP] * 80)

    assert not shell.alive and game.entities == [] and not game.over


# --- enemies from above, ledges ------------------------------------------------------------------


@pytest.mark.parametrize("big", [False, True])
def test_walker_stepping_off_a_ledge_onto_the_player_hurts(big: bool) -> None:
    game = make_game(put={(col, 9): "X" for col in range(8, 12)})
    p = game.player
    if big:
        p.grow()
    place(game, 7 * TILE - 2.0)
    hold(game, NOOP, 2)
    walker = Walker(9.0 * TILE, 9 * TILE - WALKER_H)
    game.entities.append(walker)

    events = step_until(game, NOOP, lambda e: e.hurt or e.died or e.stomps, limit=200)

    assert walker.y + walker.h < p.y + p.h, "it came from above"
    assert events.stomps == 0
    assert (events.hurt, events.died) == (big, not big)


def test_turtle_walks_off_a_ledge_and_keeps_its_direction() -> None:
    game = make_game(put={(col, 9): "X" for col in range(8, 12)})
    turtle = Turtle(9.0 * TILE, 9 * TILE - TURTLE_H)
    game.entities.append(turtle)
    place(game, 30 * TILE)

    run_checked(game, [NOOP] * 200)

    assert turtle.y == TURTLE_Y and turtle.vx == -ENEMY_SPEED and turtle.x < 8 * TILE


# --- mushrooms -----------------------------------------------------------------------------------


def test_mushroom_bounces_off_walls_falls_off_ledges_and_into_pits() -> None:
    put = {(col, 9): "X" for col in range(10, 14)}
    put[(13, 8)] = "X"  # a wall on the platform, right of the mushroom
    game = make_game(put=put, pits=((3, 4),))
    shroom = Mushroom(11.0 * TILE, 9 * TILE - MUSHROOM_H)
    game.entities.append(shroom)
    place(game, 30 * TILE)
    watch = Invariants(game)

    xs, ys = [], []
    for _ in range(300):
        watch.check(game.step(NOOP))
        if shroom.alive:
            xs.append(shroom.x)
            ys.append(shroom.y)

    assert max(xs) == 13 * TILE - MUSHROOM_W, "turned around flush at the wall"
    assert FLOOR_Y - MUSHROOM_H in ys, "fell off the left end of the platform onto the ground"
    assert not shroom.alive and shroom not in game.entities, "and finally into the pit"


@pytest.mark.parametrize("approach", ["from_below", "from_above", "from_the_side"])
def test_mushroom_is_collected_from_every_direction_without_a_bounce(approach: str) -> None:
    game = make_game(put={(col, 9): "X" for col in range(9, 13)})
    p = game.player
    shroom = Mushroom(10.0 * TILE, 9 * TILE - MUSHROOM_H)
    shroom.vx = 0.0
    game.entities.append(shroom)
    if approach == "from_below":
        shroom.y = 5 * TILE
        place(game, shroom.x + 1.0, bottom=5 * TILE + MUSHROOM_H + 30.0, vy=-4.0)
    elif approach == "from_above":
        place(game, shroom.x + 1.0, bottom=shroom.y - 1.0, vy=2.0)
    else:
        place(game, shroom.x - PLAYER_W - 1.0, bottom=9 * TILE, vx=1.5)

    events = step_until(game, RIGHT if approach == "from_the_side" else NOOP, lambda e: e.powerups)

    assert events.stomps == 0 and events.score_delta == SCORE_MUSHROOM
    assert p.big and p.h == PLAYER_BIG_H and p.vy != STOMP_BOUNCE
    assert not solid_cells(game, p) and game.entities == []


def test_mushroom_pops_out_on_top_of_stacked_blocks_and_out_of_the_top_row() -> None:
    game = make_game(put={(2, 9): "M", (2, 8): "B", (2, 7): "X"})
    step_until(game, JUMP, lambda e: game.level.tiles[9, 2] == Tile.USED, limit=30)
    (shroom,) = game.entities
    assert shroom.y == 7 * TILE - MUSHROOM_H and not solid_cells(game, shroom)

    put = {(2, row): "X" for row in range(0, 9)}
    put[(2, 9)] = "M"
    game = make_game(put=put)
    step_until(game, JUMP, lambda e: game.level.tiles[9, 2] == Tile.USED, limit=30)
    (shroom,) = game.entities
    assert shroom.y + shroom.h <= 0.5, "above the level, on top of the full-height column"
    run_checked(game, [NOOP] * 200)  # walks off, falls all the way down, no crash


def test_mushrooms_and_enemies_ignore_each_other() -> None:
    game = make_game()
    shroom, walker = Mushroom(100.0, FLOOR_Y - MUSHROOM_H), Walker(140.0, WALKER_Y)
    game.entities += [shroom, walker]
    place(game, 30 * TILE)

    run_checked(game, [NOOP] * 80)

    assert shroom.vx == MUSHROOM_SPEED and walker.vx == -ENEMY_SPEED
    assert shroom.x > walker.x, "they walked through each other"


# --- timer ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("time", [1, 3])
def test_timer_kills_exactly_when_it_reaches_zero(time: int) -> None:
    game = make_game(time=time)
    last_frame = time * FRAMES_PER_TIME_UNIT
    watch = Invariants(game)

    for frame in range(1, last_frame):
        watch.check(game.step(NOOP))
        assert game.time_left == time - frame // FRAMES_PER_TIME_UNIT > 0 and not game.over

    events = game.step(NOOP)
    watch.check(events)
    assert game.frame == last_frame and game.time_left == 0
    assert events.died and events.death_cause == "timeout" and game.death_cause == "timeout"
    assert game.player.dead and not game.won


def test_bundled_level_times_out_after_its_full_time() -> None:
    game = Game("flat")
    frames = game.level.time * FRAMES_PER_TIME_UNIT

    events = run_checked(game, [NOOP] * (frames + 10))

    assert len(events) == frames == game.frame and game.death_cause == "timeout"
    assert game.step(RIGHT) == StepEvents() and game.frame == frames


def test_reaching_the_flag_on_the_timeout_frame_wins() -> None:
    probe = make_game(width=40, time=1)
    hold(probe, RIGHT, FRAMES_PER_TIME_UNIT - 1)
    travelled = probe.player.x - make_game(width=40, time=1).player.x
    events = probe.step(RIGHT)
    assert events.died, "sanity: without the flag this is the timeout frame"

    game = make_game(width=40, time=1)
    p = game.player
    # Start so that the 24th frame is the first one overlapping the flagpole.
    place(game, game.level.flag_col * TILE - PLAYER_W - travelled - 0.5)
    p.on_ground = True
    before = hold(game, RIGHT, FRAMES_PER_TIME_UNIT - 1)
    assert not game.over and not any(e.won for e in before)

    events = game.step(RIGHT)

    assert events.won and not events.died and game.won and game.death_cause is None
    assert game.time_left == 1 and game.score == SCORE_FLAG + SCORE_FLAG_PER_TIME


# --- the flag ------------------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["left", "right", "above", "below"])
def test_flag_wins_from_every_direction(direction: str) -> None:
    game = make_game(width=40, flag_rows=range(4, 9) if direction == "below" else range(2, 13))
    p = game.player
    flag_x = game.level.flag_col * TILE
    if direction == "left":
        place(game, flag_x - PLAYER_W - 3.0)
        buttons = RIGHT
    elif direction == "right":
        place(game, flag_x + TILE + 3.0)
        buttons = LEFT
    elif direction == "above":
        place(game, flag_x + 2.0, bottom=-20.0)
        buttons = NOOP
    else:
        place(game, flag_x + 2.0)
        p.on_ground = True
        buttons = JUMP
    watch = Invariants(game)
    watch.check(game.step(NOOP))
    assert not game.over

    for _ in range(60):
        events = game.step(buttons)
        watch.check(events)
        if game.over:
            break

    bonus = SCORE_FLAG + SCORE_FLAG_PER_TIME * game.time_left
    assert events.won and game.won and events.score_delta == bonus == game.score
    assert hold(game, buttons, 3) == [StepEvents()] * 3 and game.score == bonus


def test_flag_beats_an_enemy_touched_in_the_same_frame() -> None:
    game = make_game(width=40)
    flag_x = game.level.flag_col * TILE
    place(game, flag_x - PLAYER_W - 0.5)
    game.player.on_ground = True
    game.entities.append(still(Walker(float(flag_x), WALKER_Y)))  # reached in the same frame

    events = step_until(game, RIGHT, lambda e: e.won or e.died, limit=10)

    assert events.won and not events.died and not game.player.dead


# --- the right end of the level ------------------------------------------------------------------


def test_right_level_edge_is_a_solid_wall_for_everything() -> None:
    game = make_game(width=40)
    p = game.player
    end = game.level.width_px
    shell = moving_shell(end - 40.0, 1)
    game.entities.append(shell)
    watch = Invariants(game)

    xs = []
    for _ in range(40):
        watch.check(game.step(NOOP))
        xs.append(shell.x)
    assert max(xs) == end - shell.w and shell.vx == -SHELL_SPEED, "bounced off the level end"
    game.entities.clear()

    place(game, end - 100.0)  # right of the flag
    watch.rebase()
    for frame in range(200):
        watch.check(game.step(Buttons(right=True, run=True, jump=frame % 30 < 15)))
    assert p.x == end - PLAYER_W and p.vx == 0.0 and not game.over
    assert game.camera_x == end - VIEW_W


# --- reset ---------------------------------------------------------------------------------------


MESSY = """\
; time=90
................................................
..........................................F.....
..........................................F.....
...?M?..BBB...............................F.....
..........................................F.....
..................oo......................F.....
S...o..g....k....g....[]....k...g.........F.....
######################[]######..################
######################[]######..################
"""


def _mess_up(game: Game, frames: int = 500) -> None:
    rng = random.Random(7)
    game.player.grow()
    for frame in range(frames):
        if frame % 10 == 0:
            buttons = Buttons(rng.random() < 0.2, rng.random() < 0.8, rng.random() < 0.5, True)
        game.step(buttons)
        if game.over and frame < frames - 60:
            game.reset()
            game.player.grow()
            game.player.invuln_frames = 10_000


def test_reset_restores_every_bit_of_state() -> None:
    level = Level.from_string(MESSY, name="messy")
    game = Game(level, seed=3)
    fresh = full_state(Game(level, seed=3), rng=False)
    assert full_state(game, rng=False) == fresh
    initial_entities = [(type(e), e.x, e.y) for e in game.entities]
    assert len(initial_entities) == 3, "the spawns of the first screen and a half"

    _mess_up(game)
    game.level.tiles[:, 5:9] = Tile.HARD
    game.level.spawns.clear()
    game.entities.append(Mushroom(50.0, 50.0))
    game.score += 5
    assert full_state(game, rng=False) != fresh

    game.reset()

    assert full_state(game, rng=False) == fresh
    assert [(type(e), e.x, e.y) for e in game.entities] == initial_entities
    assert np.array_equal(game.level.tiles, level.tiles) and game.level.spawns == level.spawns
    p = game.player
    assert (p.big, p.h, p.invuln_frames, p.dead, p.alive, p.facing) == (
        False,
        PLAYER_SMALL_H,
        0,
        False,
        True,
        1,
    )
    assert (game.score, game.coins, game.frame, game.camera_x) == (0, 0, 0, 0.0)
    assert (game.over, game.won, game.death_cause, game.time_left) == (False, False, None, 90)


@pytest.mark.parametrize("ending", ["pit", "enemy", "timeout", "won"])
def test_reset_after_every_kind_of_ending_replays_like_a_fresh_game(ending: str) -> None:
    level = Level.from_string(MESSY, name="messy")
    game = Game(level, seed=1)
    if ending == "pit":
        place(game, 30 * TILE + 2.0)
    elif ending == "enemy":
        place(game, 6 * TILE)
    elif ending == "won":
        place(game, 41 * TILE)
    for _ in range(90 * FRAMES_PER_TIME_UNIT):
        game.step(RIGHT if ending != "timeout" else NOOP)
        if ending == "timeout":
            game.entities.clear()
    assert game.over and (game.death_cause or "won") == ending

    game.reset(seed=1)
    fresh = Game(level, seed=1)
    assert full_state(game) == full_state(fresh)
    rng = random.Random(11)
    for _ in range(400):
        buttons = Buttons(False, True, rng.random() < 0.4, rng.random() < 0.5)
        assert game.step(buttons) == fresh.step(buttons)
    assert full_state(game) == full_state(fresh)


def test_the_level_given_to_the_game_is_never_touched() -> None:
    level = Level.from_string(MESSY, name="messy")
    tiles, spawns = level.tiles.copy(), list(level.spawns)
    game = Game(level)

    _mess_up(game, frames=300)
    game.reset()
    _mess_up(game, frames=300)

    assert np.array_equal(level.tiles, tiles) and level.spawns == spawns
    level.tiles[:] = Tile.HARD  # and the game does not look at it any more either
    game.reset()
    assert np.array_equal(game.level.tiles, tiles)


# --- clone ---------------------------------------------------------------------------------------


def test_clone_shares_nothing_mutable_in_either_direction() -> None:
    game = Game(Level.from_string(MESSY, name="messy"), seed=5)
    _mess_up(game, frames=120)
    reference = full_state(game)

    twin = game.clone()
    assert full_state(twin) == reference
    assert twin.rng.random() == game.clone().rng.random(), "clones draw the same numbers"
    assert full_state(game) == reference, "drawing from a clone's rng leaves the original alone"

    # Wreck the clone.
    twin.level.tiles[:] = Tile.HARD
    twin.level.spawns.clear()
    for e in twin.entities:
        e.x, e.alive = -100.0, False
    twin.entities.clear()
    twin.player.x += 99.0
    twin.player.grow()
    twin.rng.seed(123)
    _mess_up(twin, frames=50)
    twin.reset(seed=9)
    assert full_state(game) == reference

    # Wreck the original: an earlier clone must not notice.
    keeper = game.clone()
    game.level.tiles[:] = Tile.EMPTY
    game.entities.clear()
    game.player.die()
    game.rng.random()
    game.reset(seed=77)
    assert full_state(keeper) == reference


def test_clone_in_the_middle_of_everything_continues_identically() -> None:
    game = make_game(put={(8, 12): "k", (12, 12): "g", (5, 9): "M", (20, 12): "X", (20, 11): "X"})
    p = game.player
    p.grow()
    p.invuln_frames = 60
    game.entities.append(moving_shell(150.0, 1))
    hold(game, RIGHT_JUMP, 25)
    assert not p.on_ground and p.invuln_frames > 0

    twins = [game.clone(), game.clone().clone()]
    rng = random.Random(0)
    for _ in range(500):
        buttons = Buttons(rng.random() < 0.3, rng.random() < 0.6, rng.random() < 0.5, False)
        events = game.step(buttons)
        for twin in twins:
            assert twin.step(buttons) == events
    assert all(full_state(twin) == full_state(game) for twin in twins)


# --- determinism ---------------------------------------------------------------------------------


def biased_script(kind: str, seed: int, frames: int) -> list[Buttons]:
    """Button scripts with very different statistics; every one is a pure function of its args."""
    rng = random.Random(f"{kind}-{seed}")
    script: list[Buttons] = []
    while len(script) < frames:
        if kind == "uniform":
            chunk = [Buttons(*(rng.random() < 0.5 for _ in range(4)))]
        elif kind == "right_random_jumps":
            chunk = [Buttons(right=True, jump=rng.random() < 0.35)]
        elif kind == "run_right_long_jumps":
            chunk = [RUN_RIGHT_JUMP] * rng.randint(1, 30) + [RUN_RIGHT] * rng.randint(1, 20)
        elif kind == "held":
            buttons = Buttons(
                rng.random() < 0.15, rng.random() < 0.75, rng.random() < 0.45, rng.random() < 0.5
            )
            chunk = [buttons] * rng.randint(1, 40)
        elif kind == "left_heavy":
            buttons = Buttons(
                rng.random() < 0.55, rng.random() < 0.45, rng.random() < 0.5, rng.random() < 0.7
            )
            chunk = [buttons] * rng.randint(1, 25)
        elif kind == "jump_mash":
            chunk = [Buttons(right=rng.random() < 0.9, jump=len(script) % 2 == 0, run=True)]
        else:
            raise ValueError(kind)
        script += chunk
    return script[:frames]


SCRIPT_KINDS = [
    "uniform",
    "right_random_jumps",
    "run_right_long_jumps",
    "held",
    "left_heavy",
    "jump_mash",
]


@pytest.mark.parametrize("level", ["1-1", "1-2", "1-3", "flat"])
def test_long_random_runs_are_bit_for_bit_reproducible(level: str) -> None:
    script = biased_script("held", seed=5, frames=5_000)

    def run() -> tuple[list[tuple], int]:
        game = Game(level, seed=12)
        states, resets = [], 0
        for frame, buttons in enumerate(script):
            game.step(buttons)
            if frame % 250 == 0 or game.over:
                states.append(full_state(game))
            if game.over:
                game.reset()
                resets += 1
        states.append(full_state(game))
        return states, resets

    first, resets = run()
    assert run() == (first, resets)
    assert resets > 0 or level == "flat"


def test_the_game_never_touches_the_global_random_generators() -> None:
    python_state, numpy_state = random.getstate(), np.random.get_state()

    for level in list_levels():
        game = Game(level, seed=0)
        for buttons in biased_script("held", seed=1, frames=800):
            game.step(buttons)
            if game.over:
                game.reset()
        game.clone().step(RIGHT)

    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


# --- spawn activation ----------------------------------------------------------------------------


@pytest.mark.parametrize("level_name", ["1-1", "1-2", "1-3"])
def test_every_spawn_of_a_bundled_level_wakes_up_once_in_order_and_in_free_space(
    level_name: str,
) -> None:
    level = load_level(level_name)
    game = Game(level_name)
    p = game.player
    seen: list[Entity] = []
    born: dict[int, tuple[float, float, float]] = {}
    x = p.x
    while x < level.width_px - PLAYER_W:
        # Glide along the top of the level: the camera follows, nothing can touch the player.
        p.x, p.y, p.vx, p.vy = x, -200.0, 0.0, 0.0
        camera_before = game.camera_x
        game.step(NOOP)
        for e in game.entities:
            if id(e) not in born:
                seen.append(e)
                born[id(e)] = (e.x, e.y + e.h, camera_before)
                assert not solid_cells(game, e), (level_name, e)
                assert e.facing == -1 and e.vx == -ENEMY_SPEED
        x += 2.0

    expected = sorted(level.spawns, key=lambda s: s.col)
    assert [e.kind for e in seen] == [s.kind for s in expected]
    for e, spawn in zip(seen, expected, strict=True):
        born_x, feet, camera_before = born[id(e)]
        assert feet == (spawn.row + 1) * TILE
        assert spawn.col * TILE <= born_x < (spawn.col + 1) * TILE
        if camera_before > 0:  # not one of the spawns that are awake from frame 0
            assert camera_before + SPAWN_ACTIVATION_DISTANCE <= spawn.col * TILE


def test_a_camera_jump_wakes_up_every_spawn_it_passes_exactly_once() -> None:
    put = {(col, 12): "g" for col in range(30, 90, 7)}
    put.update({(col, 12): "k" for col in range(33, 90, 7)})
    game = make_game(width=120, put=put)
    assert game.entities == []

    place(game, 100 * TILE)
    game.step(NOOP)

    assert len(game.entities) == len(put)
    xs = [e.x for e in game.entities]
    assert xs == sorted(xs), "woken up from left to right"
    hold(game, NOOP, 5)
    hold(game, RIGHT, 50)
    assert len(game.entities) <= len(put), "nothing is ever created twice"


# --- fuzz campaign -------------------------------------------------------------------------------


def junk_level(seed: int, width: int = 80) -> Level:
    """Random clutter: blocks everywhere (also in the top rows), pits, pipes, dense enemies."""
    rng = random.Random(seed)
    density = 0.06 + 0.02 * (seed % 6)
    grid = [["."] * width for _ in range(LEVEL_H_TILES)]
    for row in (13, 14):
        grid[row] = ["#"] * width
    col = 8
    while col < width - 14:
        if rng.random() < 0.12:
            pit = rng.randint(1, 4)
            for k in range(pit):
                grid[13][col + k] = grid[14][col + k] = "."
            col += pit + 2
        else:
            col += 1
    for col in range(5, width - 12):
        for row in range(0, 13):
            if rng.random() < density:
                grid[row][col] = rng.choice("XXBB?M[]")
    for col in range(6, width - 12):
        for row in range(2, 13):
            free = grid[row][col] == "." and grid[row - 1][col] == "."
            if free and grid[row + 1][col] in "XB?M#[]" and rng.random() < 0.15:
                grid[row][col] = rng.choice("ggk")
            elif grid[row][col] == "." and rng.random() < 0.04:
                grid[row][col] = "o"
    grid[12][2], grid[11][2] = "S", "."
    for row in range(1, 13):
        grid[row][width - 10] = "F"
    return Level.from_string("\n".join("".join(row) for row in grid), name=f"junk-{seed}")


def standing_spots(level: Level) -> list[tuple[float, float]]:
    """(x, feet) positions where a big player fits with solid ground right below."""
    spots = []
    for col in range(1, level.width_tiles - 1):
        for row in range(2, LEVEL_H_TILES - 1):
            free = not (level.solid_at(col, row) or level.solid_at(col, row - 1))
            if free and level.solid_at(col, row + 1) and int(level.tiles[row, col]) == Tile.EMPTY:
                spots.append((col * TILE + 2.0, float((row + 1) * TILE)))
    return spots


def random_intruder(game: Game, rng: random.Random) -> Entity | None:
    """A random enemy or item in a random free spot around the player, or None if none is found."""
    kind = rng.choice(["walker", "turtle", "shell", "shell_moving", "mushroom"])
    facing = rng.choice([-1, 1])
    x = min(max(game.player.x + rng.uniform(-150.0, 250.0), 0.0), game.level.width_px - TURTLE_W)
    y = rng.uniform(-20.0, 200.0)
    if kind == "walker":
        intruder: Entity = Walker(x, y, facing)
    elif kind == "mushroom":
        intruder = Mushroom(x, y, facing)
    else:
        intruder = Turtle(x, y, facing)
        if kind != "turtle":
            intruder.to_shell()
            intruder.contact_cooldown = 0
        if kind == "shell_moving":
            intruder.kick(facing)
    return None if solid_cells(game, intruder) else intruder


def fuzz(
    level: Level | str,
    kind: str,
    seed: int,
    frames: int,
    teleport_every: int = 0,
    intruder_every: int = 0,
) -> dict:
    """Play `frames` frames of a biased random script, checking every invariant on every frame.

    A clone made at a random frame is stepped in lockstep and must stay identical.
    With `teleport_every`, the player is regularly dropped on a random standing spot,
    big or small, so that every part of a level is exercised, not just its first screens.
    With `intruder_every`, enemies, shells and mushrooms keep appearing out of thin air.
    """
    game = Game(level, seed=seed)
    rng = random.Random(f"fuzz-{kind}-{seed}")
    spots = standing_spots(game.level) if teleport_every else []
    watch = Invariants(game)
    twin: Game | None = None
    clone_at = rng.randrange(frames)
    stats = {"won": 0, "pit": 0, "enemy": 0, "timeout": 0, "stomps": 0, "powerups": 0, "max_x": 0.0}
    for frame, buttons in enumerate(biased_script(kind, seed, frames)):
        if frame == clone_at:
            twin = game.clone()
        games = (game, twin) if twin else (game,)
        if teleport_every and frame % teleport_every == teleport_every - 1:
            x, feet = rng.choice(spots)
            big = rng.random() < 0.5
            for g in games:
                if big:
                    g.player.grow()
                else:
                    g.player.shrink()
                g.player.x, g.player.y = x, feet - g.player.h
                g.player.vx = g.player.vy = 0.0
                g.camera_x = min(g.camera_x, x)  # a teleport behind the camera drags it back
            watch.rebase()
        if intruder_every and frame % intruder_every == 0 and len(game.entities) < 40:
            intruder = random_intruder(game, rng)
            if intruder is not None:
                for g in games:
                    g.entities.append(intruder.clone())
                watch.rebase()
        context = f"{game.level.name} {kind} seed={seed} frame={frame}"
        events = game.step(buttons)
        watch.check(events, context, buttons)
        if twin is not None:
            assert twin.step(buttons) == events, context
            assert twin.snapshot() == game.snapshot(), context
        stats["stomps"] += events.stomps
        stats["powerups"] += events.powerups
        stats["max_x"] = max(stats["max_x"], game.player.x)
        if game.over:
            stats["won" if game.won else str(game.death_cause)] += 1
            game.reset()
            watch.rebase()
            if twin is not None:
                twin.reset()
    if twin is not None:
        assert full_state(twin) == full_state(game)
    return stats


@pytest.mark.parametrize("kind", SCRIPT_KINDS)
@pytest.mark.parametrize("level", ["1-1", "1-2", "1-3", "flat"])
def test_fuzz_bundled_levels_with_biased_inputs(level: str, kind: str) -> None:
    stats = fuzz(level, kind, seed=0, frames=2_000)
    if kind not in ("uniform", "left_heavy"):  # those two hang around the start
        assert stats["max_x"] > 10 * TILE


@pytest.mark.parametrize("level", ["1-1", "1-2", "1-3", "flat"])
def test_fuzz_bundled_levels_everywhere_by_teleporting(level: str) -> None:
    stats = fuzz(level, "held", seed=1, frames=5_000, teleport_every=100)
    assert stats["max_x"] > load_level(level).flag_col * TILE - 5 * TILE, "the far end was visited"


@pytest.mark.parametrize("seed", range(12))
def test_fuzz_junk_levels(seed: int) -> None:
    kind = SCRIPT_KINDS[seed % len(SCRIPT_KINDS)]
    fuzz(junk_level(seed), kind, seed, frames=1_500, teleport_every=200 if seed % 2 else 0)


@pytest.mark.parametrize("level", ["1-1", "1-3", "flat", "junk"])
def test_fuzz_with_enemies_and_items_appearing_out_of_thin_air(level: str) -> None:
    for seed in range(2):
        target = junk_level(50 + seed) if level == "junk" else level
        stats = fuzz(target, "held", seed, frames=1_500, intruder_every=15)
        assert stats["enemy"] + stats["stomps"] > 0, "the intruders met the player"


@pytest.mark.slow
@pytest.mark.parametrize("kind", SCRIPT_KINDS)
@pytest.mark.parametrize("level", ["1-1", "1-2", "1-3", "flat"])
def test_fuzz_campaign_bundled_levels(level: str, kind: str) -> None:
    outcomes = {"won": 0, "pit": 0, "enemy": 0, "timeout": 0}
    for seed in range(100, 106):
        stats = fuzz(
            level,
            kind,
            seed,
            frames=4_000,
            teleport_every=(0, 120, 400)[seed % 3],
            intruder_every=(0, 0, 0, 30)[seed % 4],
        )
        for key in outcomes:
            outcomes[key] += stats[key]
    if level == "flat":  # no pits; the only enemies are the fuzzer's intruders
        assert outcomes["pit"] == 0
        assert outcomes["won"] > 0 or kind in ("uniform", "left_heavy")
    else:
        assert outcomes["pit"] + outcomes["enemy"] > 0


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(100, 124))
def test_fuzz_campaign_junk_levels(seed: int) -> None:
    for kind in SCRIPT_KINDS:
        fuzz(
            junk_level(seed),
            kind,
            seed,
            frames=2_000,
            teleport_every=250 if seed % 2 else 0,
            intruder_every=(0, 0, 25)[seed % 3],
        )
