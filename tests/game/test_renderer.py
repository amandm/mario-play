"""Renderer contract (spec 3.6): frame format, cached background with tile diffing, camera,
sprite anchoring and animation, HUD, decorations, speed, and the preview script."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

from mario_play.game.constants import LEVEL_H_TILES, TILE, VIEW_H, VIEW_W
from mario_play.game.engine import Buttons, Game
from mario_play.game.entities import Entity, Mushroom, Turtle, Walker
from mario_play.game.level import Level, list_levels
from mario_play.game.renderer import HUD_HEIGHT, Renderer
from mario_play.game.sprites import SKY_COLOR, blit, get_sprite
from mario_play.game.tiles import Tile

REPO_ROOT = Path(__file__).resolve().parents[2]

NOOP = Buttons()
RIGHT = Buttons(right=True)
RUN_RIGHT = Buttons(right=True, run=True)
JUMP = Buttons(jump=True)

SKY = np.array(SKY_COLOR, dtype=np.uint8)
FLOOR_ROW = 13  # first ground row of the test levels, and of every bundled level
FLOOR_Y = FLOOR_ROW * TILE
FAR_AWAY = 10_000.0


# --------------------------------------------------------------------------------- helpers


def make_level(
    width: int = 40,
    put: dict[tuple[int, int], str] | None = None,
    pits: tuple[tuple[int, int], ...] = (),
    start: tuple[int, int] = (2, 12),
    name: str = "test",
) -> Level:
    """A flat test level: ground rows 13-14, flag near the right end, `put[(col, row)] = char`."""
    grid = [["."] * width for _ in range(LEVEL_H_TILES)]
    for row in (13, 14):
        grid[row] = ["#"] * width
    for first, last in pits:
        for col in range(first, last + 1):
            grid[13][col] = grid[14][col] = "."
    for row in range(2, 13):
        grid[row][width - 3] = "F"
    grid[start[1]][start[0]] = "S"
    for (col, row), char in (put or {}).items():
        grid[row][col] = char
    return Level.from_string("\n".join("".join(row) for row in grid), name=name)


def make_game(**kwargs) -> Game:
    return Game(make_level(**kwargs))


def plain() -> Renderer:
    """A renderer that draws nothing but the level itself: no HUD, no scenery."""
    return Renderer(hud=False, decorations=False)


def bare(game: Game) -> Game:
    """A copy of `game` without anything that moves on screen."""
    twin = game.clone()
    twin.entities.clear()
    twin.player.y = FAR_AWAY
    return twin


def sprite_on_sky(name: str) -> np.ndarray:
    """A 16x16 (or sprite-sized) RGB patch: the sprite composited over plain sky."""
    sprite = get_sprite(name)
    patch = np.empty((*sprite.shape[:2], 3), dtype=np.uint8)
    patch[:] = SKY
    blit(patch, sprite, 0, 0)
    return patch


def cell(frame: np.ndarray, col: int, row: int, cam: int = 0) -> np.ndarray:
    x = col * TILE - cam
    return frame[row * TILE : (row + 1) * TILE, x : x + TILE]


def anchor(entity: Entity, sprite_name: str, cam: int = 0) -> tuple[int, int]:
    """Top-left screen pixel of a sprite anchored bottom-centre on the entity's hitbox."""
    height, width = get_sprite(sprite_name).shape[:2]
    left = math.floor(entity.x + entity.w / 2 - width / 2) - cam
    top = math.floor(entity.y + entity.h) - height
    return left, top


def expected_frame(game: Game, draws: list[tuple[str, Entity, bool]]) -> np.ndarray:
    """The plain scene with the given (sprite name, entity, flip) drawn in order."""
    frame = plain().render(bare(game))
    cam = int(game.camera_x)
    for name, entity, flip in draws:
        blit(frame, get_sprite(name), *anchor(entity, name, cam), flip)
    return frame


def which(frame: np.ndarray, candidates: dict[str, np.ndarray]) -> str:
    """Name of the candidate frame equal to `frame` (fails when there is none)."""
    for name, candidate in candidates.items():
        if np.array_equal(frame, candidate):
            return name
    raise AssertionError(f"frame matches none of {sorted(candidates)}")


def run_lengths(values: list) -> list[int]:
    runs = [1]
    for previous, current in zip(values, values[1:], strict=False):
        if current == previous:
            runs[-1] += 1
        else:
            runs.append(1)
    return runs


def panorama(renderer: Renderer, game: Game) -> np.ndarray:
    """The static scene of the whole level as one image, made by scrolling a bare copy."""
    twin = bare(game)
    width = max(twin.level.width_px, VIEW_W)
    image = np.empty((VIEW_H, width, 3), dtype=np.uint8)
    for cam in [*range(0, width - VIEW_W, VIEW_W), width - VIEW_W]:
        twin.camera_x = float(cam)
        image[:, cam : cam + VIEW_W] = renderer.render(twin)
    return image


def play(game: Game, buttons: Buttons, frames: int) -> None:
    for _ in range(frames):
        game.step(buttons)


@pytest.fixture
def full_repaints(monkeypatch: pytest.MonkeyPatch) -> list[Renderer]:
    """White box: the renderers that painted a whole level from scratch, one entry per time."""
    calls: list[Renderer] = []
    original = Renderer._repaint_level

    def spy(self: Renderer, level: Level) -> None:
        calls.append(self)
        original(self, level)

    monkeypatch.setattr(Renderer, "_repaint_level", spy)
    return calls


def times(calls: list[Renderer], renderer: Renderer) -> int:
    return sum(1 for entry in calls if entry is renderer)


# ----------------------------------------------------------------------------- frame format


@pytest.mark.parametrize("level", list_levels())
@pytest.mark.parametrize("hud", [True, False])
def test_frame_is_a_240x256_rgb_uint8_array(level: str, hud: bool) -> None:
    frame = Renderer(hud=hud).render(Game(level))
    assert frame.shape == (VIEW_H, VIEW_W, 3) == (240, 256, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous and frame.flags.writeable


def test_every_call_returns_a_new_independent_array() -> None:
    game = Game("flat")
    renderer = Renderer()
    first = renderer.render(game)
    second = renderer.render(game)
    assert first is not second
    assert not np.shares_memory(first, second)
    assert np.array_equal(first, second)
    first[:] = 0  # scribbling over a returned frame must not reach the cached background
    assert np.array_equal(renderer.render(game), second)


def test_render_does_not_touch_the_game() -> None:
    game = Game("1-1")
    play(game, RUN_RIGHT, 30)
    before, tiles_before = game.snapshot(), game.level.tiles.copy()
    entities_before = [(e.kind, e.x, e.y, e.vx, e.vy) for e in game.entities]
    Renderer().render(game)
    assert game.snapshot() == before
    assert np.array_equal(game.level.tiles, tiles_before)
    assert [(e.kind, e.x, e.y, e.vx, e.vy) for e in game.entities] == entities_before


def test_hud_is_on_by_default() -> None:
    game = Game("flat")
    assert np.array_equal(Renderer().render(game), Renderer(hud=True).render(game))
    assert not np.array_equal(Renderer().render(game), Renderer(hud=False).render(game))


# -------------------------------------------------------------------------------- background


def test_empty_space_is_sky() -> None:
    game = make_game()
    frame = plain().render(bare(game))
    assert (frame[:FLOOR_Y] == SKY).all()  # the flag stands outside the first screen
    assert not (frame[FLOOR_Y:] == SKY).all(axis=2).any()


def test_flat_has_ground_along_the_bottom() -> None:
    frame = Renderer().render(Game("flat"))
    capped, soil = sprite_on_sky("ground_top"), sprite_on_sky("ground")
    for col in range(VIEW_W // TILE):
        assert np.array_equal(cell(frame, col, 13), capped), col
        assert np.array_equal(cell(frame, col, 14), soil), col


@pytest.mark.parametrize(
    ("char", "sprite"),
    [
        ("X", "hard"),
        ("B", "brick"),
        ("?", "question"),
        ("M", "question"),  # what a ?-block holds is a secret
        ("o", "coin"),
    ],
)
def test_tiles_are_drawn_with_their_sprites(char: str, sprite: str) -> None:
    game = make_game(put={(6, 9): char})
    assert np.array_equal(cell(plain().render(bare(game)), 6, 9), sprite_on_sky(sprite))


def test_used_block_pipe_and_flagpole_sprites() -> None:
    put = {(5, 11): "[", (6, 11): "]", (5, 12): "[", (6, 12): "]", (9, 9): "?"}
    game = make_game(width=16, put=put)  # one screen wide: the flag (column 13) is in view
    game.level.tiles[9, 9] = Tile.USED
    frame = plain().render(bare(game))
    expected = {
        (5, 11): "pipe_tl",
        (6, 11): "pipe_tr",
        (5, 12): "pipe_l",
        (6, 12): "pipe_r",
        (9, 9): "used",
        (13, 2): "flag_top",
        (13, 12): "flagpole",
    }
    for (col, row), name in expected.items():
        assert np.array_equal(cell(frame, col, row), sprite_on_sky(name)), name


def test_only_the_top_ground_tile_is_grass_capped() -> None:
    game = make_game(put={(5, 12): "#", (8, 11): "[", (9, 11): "]", (8, 12): "[", (9, 12): "]"})
    frame = plain().render(bare(game))
    capped, soil = sprite_on_sky("ground_top"), sprite_on_sky("ground")
    assert np.array_equal(cell(frame, 5, 12), capped)  # a one-tile bump in the ground
    assert np.array_equal(cell(frame, 5, 13), soil)
    assert np.array_equal(cell(frame, 4, 13), capped)
    assert np.array_equal(cell(frame, 8, 13), capped)  # a pipe stands on the grass
    assert np.array_equal(cell(frame, 4, 14), soil)


def test_a_pennant_hangs_on_the_flagpole() -> None:
    game = make_game(width=16)
    frame = plain().render(bare(game))
    beside_pole = cell(frame, 14, 3)  # right of the pole, just below the FLAG_TOP tile
    assert not (beside_pole == SKY).all()
    assert (cell(frame, 14, 5) == SKY).all()
    assert (cell(frame, 12, 3) == SKY).all()


# ------------------------------------------------------------------------------ tile diffing


@pytest.mark.parametrize(
    ("char", "new_tile", "sprite"),
    [
        ("B", Tile.EMPTY, None),  # a broken brick
        ("?", Tile.USED, "used"),
        ("M", Tile.USED, "used"),
        ("o", Tile.EMPTY, None),  # a collected coin
    ],
)
def test_changed_tile_is_repainted_and_nothing_else(
    char: str, new_tile: Tile, sprite: str | None
) -> None:
    game = bare(make_game(put={(6, 9): char}))
    renderer = plain()
    before = renderer.render(game)
    game.level.tiles[9, 6] = new_tile
    after = renderer.render(game)

    expected = sprite_on_sky(sprite) if sprite else np.broadcast_to(SKY, (TILE, TILE, 3))
    assert np.array_equal(cell(after, 6, 9), expected)
    changed = (before != after).any(axis=2)
    assert changed.any()
    changed[9 * TILE : 10 * TILE, 6 * TILE : 7 * TILE] = False
    assert not changed.any()


def test_bumped_question_block_turns_used_on_screen() -> None:
    game = make_game(put={(2, 9): "?"})  # right above the player's head
    renderer = plain()
    assert np.array_equal(cell(renderer.render(game), 2, 9), sprite_on_sky("question"))
    for _ in range(60):
        game.step(JUMP)
        if game.coins:
            break
    assert game.level.tiles[9, 2] == Tile.USED
    # Without the player: its 16 px sprite pokes one pixel above the 15 px hitbox into the block.
    assert np.array_equal(cell(renderer.render(bare(game)), 2, 9), sprite_on_sky("used"))


def test_collected_coin_disappears_from_the_frame() -> None:
    game = Game("flat")  # coins lie on the way at columns 9-11, row 12
    renderer = plain()
    assert np.array_equal(cell(renderer.render(game), 9, 12), sprite_on_sky("coin"))
    while game.coins < 3:
        game.step(RIGHT)
    cam = int(game.camera_x)
    frame = renderer.render(bare(game))
    for col in (9, 10, 11):
        assert (cell(frame, col, 12, cam) == SKY).all()


def test_big_player_breaking_a_brick_clears_the_cell() -> None:
    game = make_game(put={(2, 8): "B"})
    game.player.grow()
    renderer = plain()
    assert np.array_equal(cell(renderer.render(game), 2, 8), sprite_on_sky("brick"))
    for _ in range(60):
        game.step(JUMP)
        if game.level.tiles[8, 2] == Tile.EMPTY:
            break
    assert game.level.tiles[8, 2] == Tile.EMPTY
    assert (cell(renderer.render(bare(game)), 2, 8) == SKY).all()


def test_repaint_restores_the_scenery_behind_the_cell(full_repaints: list[Renderer]) -> None:
    # Coins in front of wherever bushes and hills (rows 11-12) and clouds (rows 2-5) may be.
    rows = (2, 3, 4, 5, 11, 12)
    put = {(col, row): "o" for col in range(3, 75) for row in rows}
    game = make_game(width=80, put=put)
    renderer = Renderer(hud=False)
    panorama(renderer, game)  # caches the background with every coin in place
    for row in rows:  # one row of coins at a time keeps this on the cell-by-cell path
        line = game.level.tiles[row]
        line[line == Tile.COIN] = Tile.EMPTY
        assert np.array_equal(panorama(renderer, game), panorama(Renderer(hud=False), game))
    assert times(full_repaints, renderer) == 1

    scene = panorama(renderer, game)
    behind = scene[2 * TILE : 6 * TILE], scene[11 * TILE : 13 * TILE]
    assert all((band != SKY).any() for band in behind), "no scenery behind the coins: vacuous"


def test_play_reset_and_clones_never_repaint_the_whole_level(
    full_repaints: list[Renderer],
) -> None:
    game = Game("flat")
    renderer = Renderer()
    renderer.render(game)
    twin = game.clone()
    while game.coins < 3:
        game.step(RIGHT)
        renderer.render(game)
    renderer.render(twin)
    game.reset()
    renderer.render(game)
    assert times(full_repaints, renderer) == 1
    renderer.render(Game("1-1"))
    assert times(full_repaints, renderer) == 2


@pytest.mark.parametrize("level", ["flat", "1-1"])
def test_incremental_rendering_equals_a_fresh_renderer_during_play(level: str) -> None:
    game = Game(level)
    renderer = Renderer()
    for frame_no in range(360):
        game.step(Buttons(right=True, run=True, jump=frame_no % 45 < 20))
        if frame_no % 12 == 0 or game.over:
            assert np.array_equal(renderer.render(game), Renderer().render(game)), frame_no
        if game.over:
            break
    assert game.camera_x > 0


def test_many_changed_cells_at_once() -> None:
    put = {(col, row): "o" for col in range(3, 35) for row in range(3, 12)}
    game = make_game(put=put)
    renderer = plain()
    renderer.render(game)
    game.level.tiles[game.level.tiles == Tile.COIN] = Tile.EMPTY
    assert np.array_equal(renderer.render(game), plain().render(game))


# ------------------------------------------------------------------- one renderer, many games


def test_same_renderer_after_reset() -> None:
    game = Game("flat")
    renderer = Renderer()
    start = renderer.render(game)
    while game.coins < 3:
        game.step(RIGHT)
    renderer.render(game)
    game.reset()
    again = renderer.render(game)
    assert np.array_equal(again, start)
    assert np.array_equal(cell(again, 9, 12), sprite_on_sky("coin"))


def test_same_renderer_alternating_between_a_game_and_its_clone() -> None:
    game = Game("flat")
    renderer = Renderer()
    renderer.render(game)
    twin = game.clone()
    while twin.coins < 3:
        twin.step(RIGHT)
    play(game, NOOP, 5)
    for _ in range(2):
        assert np.array_equal(renderer.render(twin), Renderer().render(twin))
        assert np.array_equal(renderer.render(game), Renderer().render(game))
    assert np.array_equal(cell(renderer.render(game), 9, 12)[8], sprite_on_sky("coin")[8])


def test_same_renderer_on_different_levels() -> None:
    renderer = Renderer()
    for name in ["flat", "1-1", "1-2", "flat", "1-3"]:
        game = Game(name)
        assert np.array_equal(renderer.render(game), Renderer().render(game)), name


def test_same_name_and_shape_but_a_different_layout() -> None:
    first = Game(make_level(put={(6, 12): "X", (9, 9): "B"}))
    second = Game(make_level(put={(7, 11): "[", (8, 11): "]", (7, 12): "[", (8, 12): "]"}))
    third = Game(make_level(pits=((5, 7),)))
    fourth = Game(make_level(pits=((5, 7),), start=(20, 12)))
    renderer = Renderer()
    for game in (first, second, third, fourth, first):
        assert np.array_equal(panorama(renderer, game), panorama(Renderer(), game))


def test_ground_that_disappears_takes_its_scenery_along() -> None:
    whole = Game(make_level(width=60))
    scenery = (panorama(Renderer(hud=False), whole) != panorama(plain(), whole)).any(axis=2)
    grounded = np.unique(np.nonzero(scenery[11 * TILE : 13 * TILE])[1] // TILE)
    assert grounded.size, "no bush or hill on this level: vacuous"
    col = int(grounded[0])
    holed = Game(make_level(width=60, pits=((col, col + 1),)))  # a pit right under it

    renderer = Renderer(hud=False)
    panorama(renderer, whole)
    assert np.array_equal(panorama(renderer, holed), panorama(Renderer(hud=False), holed))
    assert np.array_equal(panorama(renderer, whole), panorama(Renderer(hud=False), whole))


# ------------------------------------------------------------------------------------ camera


def test_camera_scroll_shifts_the_background() -> None:
    game = bare(make_game(put={(10, 9): "B", (14, 12): "X"}))
    renderer = plain()
    at_rest = renderer.render(game)
    game.camera_x = 40.0
    scrolled = renderer.render(game)
    assert np.array_equal(scrolled[:, : VIEW_W - 40], at_rest[:, 40:])
    assert not np.array_equal(scrolled, at_rest)


def test_camera_position_is_truncated_to_whole_pixels() -> None:
    game = make_game(put={(10, 9): "B"})
    game.player.x = 200.0
    renderer = plain()
    game.camera_x = 40.0
    whole = renderer.render(game)
    game.camera_x = 40.9
    assert np.array_equal(renderer.render(game), whole)


def test_camera_follows_the_player_during_play() -> None:
    game = Game("flat")
    play(game, RUN_RIGHT, 150)
    assert game.camera_x > 0 and game.player.on_ground
    poses = {
        name: expected_frame(game, [(name, game.player, False)])
        for name in ("player_small_walk1", "player_small_walk2")
    }
    assert which(plain().render(game), poses)
    left, top = anchor(game.player, "player_small_walk1", int(game.camera_x))
    assert left == 110 and top + 16 == FLOOR_Y  # 112 px from the left edge, feet on the ground


@pytest.mark.parametrize("camera_x", [-50.0, 1e9])
def test_camera_outside_the_level_is_clamped(camera_x: float) -> None:
    game = bare(Game("flat"))
    renderer = plain()
    game.camera_x = 0.0 if camera_x < 0 else float(game.level.width_px - VIEW_W)
    inside = renderer.render(game)
    game.camera_x = camera_x
    frame = renderer.render(game)
    assert frame.shape == (VIEW_H, VIEW_W, 3)
    assert np.array_equal(frame, inside)


def test_level_narrower_than_the_view() -> None:
    tiles = np.zeros((LEVEL_H_TILES, 10), dtype=np.int8)
    tiles[13:] = Tile.GROUND
    tiles[10:13, 8] = Tile.FLAGPOLE
    tiles[9, 8] = Tile.FLAG_TOP
    game = Game(Level("narrow", tiles, (1, 12), [], 100, 8))
    frame = Renderer(hud=False).render(bare(game))
    assert frame.shape == (VIEW_H, VIEW_W, 3)
    assert (frame[:, 10 * TILE :] == SKY).all()
    assert np.array_equal(cell(frame, 9, 13), sprite_on_sky("ground_top"))


# ----------------------------------------------------------------------- player and entities


def test_small_player_stands_with_its_feet_on_the_ground() -> None:
    game = make_game()
    p = game.player
    assert np.array_equal(
        plain().render(game), expected_frame(game, [("player_small_stand", p, False)])
    )
    left, top = anchor(p, "player_small_stand")
    assert top + 16 == FLOOR_Y
    assert left + 8 == p.x + p.w / 2  # centred on the hitbox


def test_big_player_uses_the_tall_sprites() -> None:
    game = make_game()
    game.player.grow()
    expected = expected_frame(game, [("player_big_stand", game.player, False)])
    assert np.array_equal(plain().render(game), expected)
    assert anchor(game.player, "player_big_stand")[1] + 32 == FLOOR_Y


@pytest.mark.parametrize("big", [False, True])
def test_jump_sprite_while_airborne(big: bool) -> None:
    game = make_game()
    if big:
        game.player.grow()
    play(game, JUMP, 10)
    assert not game.player.on_ground
    name = "player_big_jump" if big else "player_small_jump"
    assert np.array_equal(plain().render(game), expected_frame(game, [(name, game.player, False)]))


@pytest.mark.parametrize("size", ["small", "big"])
def test_walk_cycle_follows_the_game_frame(size: str) -> None:
    game = make_game()
    if size == "big":
        game.player.grow()
    game.player.vx = 1.0
    renderer = plain()
    poses = {
        pose: expected_frame(game, [(f"player_{size}_{pose}", game.player, False)])
        for pose in ("walk1", "walk2")
    }
    sequence = []
    for frame_no in range(96):
        game.frame = frame_no
        sequence.append(which(renderer.render(game), poses))
    assert set(sequence) == {"walk1", "walk2"}
    runs = run_lengths(sequence)
    assert min(runs[1:-1]) == max(runs[1:-1]) >= 4  # a steady beat, no flicker

    game.player.vx = 0.0
    standing = expected_frame(game, [(f"player_{size}_stand", game.player, False)])
    assert np.array_equal(renderer.render(game), standing)


def test_player_facing_left_is_mirrored() -> None:
    game = make_game()
    game.player.facing = -1
    mirrored = expected_frame(game, [("player_small_stand", game.player, True)])
    assert np.array_equal(plain().render(game), mirrored)
    assert not np.array_equal(
        mirrored, expected_frame(game, [("player_small_stand", game.player, False)])
    )


def test_player_blinks_in_four_frame_windows_while_invulnerable() -> None:
    game = make_game()
    renderer = plain()
    shown = expected_frame(game, [("player_small_stand", game.player, False)])
    hidden = expected_frame(game, [])
    visible = []
    for invuln in range(1, 121):
        game.player.invuln_frames = invuln
        visible.append(which(renderer.render(game), {"shown": shown, "hidden": hidden}))
    runs = run_lengths(visible)
    assert set(runs[1:-1]) == {4} and max(runs) <= 4
    assert visible.count("hidden") in range(56, 65)
    game.player.invuln_frames = 0
    assert np.array_equal(renderer.render(game), shown)


def test_walker_sprites_anchor_animate_and_flatten() -> None:
    game = make_game()
    walker = Walker(100.0, float(FLOOR_Y - 14))
    game.entities.append(walker)
    renderer = plain()
    p = game.player
    steps = {
        name: expected_frame(game, [(name, walker, True), ("player_small_stand", p, False)])
        for name in ("walker_1", "walker_2")
    }
    sequence = []
    for frame_no in range(96):
        game.frame = frame_no
        sequence.append(which(renderer.render(game), steps))
    assert set(sequence) == {"walker_1", "walker_2"}
    runs = run_lengths(sequence)
    assert min(runs[1:-1]) == max(runs[1:-1]) >= 4
    assert anchor(walker, "walker_1") == (99, FLOOR_Y - 16)

    game.frame = 0
    walker.facing = 1  # sprites are drawn facing right, so no mirroring now
    unflipped = expected_frame(
        game, [(sequence[0], walker, False), ("player_small_stand", p, False)]
    )
    assert np.array_equal(renderer.render(game), unflipped)

    walker.squish()
    flat = expected_frame(game, [("walker_flat", walker, False), ("player_small_stand", p, False)])
    assert np.array_equal(renderer.render(game), flat)


def test_turtle_walks_tall_and_hides_in_a_shell() -> None:
    game = make_game()
    turtle = Turtle(100.0, float(FLOOR_Y - 22))
    game.entities.append(turtle)
    renderer = plain()
    p = ("player_small_stand", game.player, False)
    walking = {
        name: expected_frame(game, [(name, turtle, True), p]) for name in ("turtle_1", "turtle_2")
    }
    seen = set()
    for frame_no in range(64):
        game.frame = frame_no
        seen.add(which(renderer.render(game), walking))
    assert seen == {"turtle_1", "turtle_2"}
    assert anchor(turtle, "turtle_1") == (99, FLOOR_Y - 24)

    turtle.to_shell()
    assert np.array_equal(
        renderer.render(game), expected_frame(game, [("shell", turtle, False), p])
    )
    assert anchor(turtle, "shell")[1] + 16 == FLOOR_Y
    turtle.kick(1)
    assert np.array_equal(
        renderer.render(game), expected_frame(game, [("shell", turtle, False), p])
    )


def test_mushroom_sprite() -> None:
    game = make_game()
    shroom = Mushroom(120.0, float(FLOOR_Y - 14))
    game.entities.append(shroom)
    expected = expected_frame(
        game, [("mushroom", shroom, False), ("player_small_stand", game.player, False)]
    )
    assert np.array_equal(plain().render(game), expected)


def test_fractional_positions_are_floored_to_whole_pixels() -> None:
    game = make_game(width=60)
    game.camera_x = 30.75
    game.player.x, game.player.y = 150.6, 100.7
    game.player.on_ground = False
    shroom = Mushroom(90.5, 60.9)
    game.entities.append(shroom)
    assert anchor(game.player, "player_small_jump", 30) == (118, 99)
    assert anchor(shroom, "mushroom", 30) == (59, 58)
    expected = expected_frame(
        game, [("mushroom", shroom, False), ("player_small_jump", game.player, False)]
    )
    assert np.array_equal(plain().render(game), expected)


def test_player_is_drawn_in_front_of_entities() -> None:
    game = make_game()
    walker = Walker(game.player.x + 4, float(FLOOR_Y - 14))
    game.entities.append(walker)
    expected = expected_frame(
        game, [("walker_1", walker, True), ("player_small_stand", game.player, False)]
    )
    other = expected_frame(
        game, [("walker_2", walker, True), ("player_small_stand", game.player, False)]
    )
    frame = plain().render(game)
    assert np.array_equal(frame, expected) or np.array_equal(frame, other)


def test_sprites_are_clipped_at_every_screen_edge() -> None:
    game = make_game(width=60)
    game.camera_x = 100.0
    game.player.x, game.player.y = 300.0, -8.0  # head above the top edge
    game.player.on_ground = False
    left_edge = Walker(93.0, float(FLOOR_Y - 14))  # straddles the left screen edge
    right_edge = Turtle(349.0, float(FLOOR_Y - 22))  # straddles the right one
    sinking = Mushroom(200.0, float(VIEW_H - 6))  # falling out of the bottom
    game.entities.extend([left_edge, right_edge, sinking])
    frame = plain().render(game)
    assert frame.shape == (VIEW_H, VIEW_W, 3)
    assert not np.array_equal(frame[:, :8], plain().render(bare(game))[:, :8])
    assert not np.array_equal(frame[:, -8:], plain().render(bare(game))[:, -8:])


def test_off_screen_and_unknown_entities_draw_nothing() -> None:
    game = make_game(width=60)
    game.player.y = FAR_AWAY
    game.entities.append(Walker(500.0, float(FLOOR_Y - 14)))  # beyond the right edge
    game.entities.append(Walker(40.0, -200.0))  # far above
    game.entities.append(Entity(64.0, 64.0, 16, 16))  # a kind the renderer has no art for
    assert np.array_equal(plain().render(game), plain().render(bare(game)))


def test_enemies_stand_on_the_ground_during_real_play() -> None:
    game = Game("1-1")
    play(game, NOOP, 120)  # the first walker comes walking towards the start
    walkers = [e for e in game.entities if e.kind == "walker" and e.on_ground]
    assert walkers
    for walker in walkers:
        assert anchor(walker, "walker_1")[1] + 16 == FLOOR_Y


def test_dead_and_finished_games_still_render() -> None:
    dead = make_game(pits=((3, 6),))
    while not dead.over:
        dead.step(RIGHT)
    assert dead.death_cause == "pit"
    won = Game("flat")
    while not won.over:
        won.step(RUN_RIGHT)
    assert won.won
    for game in (dead, won):
        assert Renderer().render(game).shape == (VIEW_H, VIEW_W, 3)


# --------------------------------------------------------------------------------------- HUD


@pytest.mark.parametrize("level", ["flat", "1-1"])
def test_without_hud_the_top_rows_are_sky(level: str) -> None:
    frame = Renderer(hud=False).render(Game(level))
    assert (frame[:HUD_HEIGHT] == SKY).all()


def test_hud_draws_only_inside_its_band() -> None:
    game = Game("1-1")
    with_hud = Renderer(hud=True).render(game)
    without = Renderer(hud=False).render(game)
    assert 16 <= HUD_HEIGHT <= 32
    assert np.array_equal(with_hud[HUD_HEIGHT:], without[HUD_HEIGHT:])
    text = (with_hud[:HUD_HEIGHT] != SKY).any(axis=2)
    assert text.sum() > 200  # four labels and four values worth of glyph pixels
    assert not text[:, :8].any() and not text[:, -8:].any()  # a margin at both sides


@pytest.mark.parametrize("field", ["score", "coins", "time_left"])
def test_hud_shows_score_coins_and_time(field: str) -> None:
    game = Game("flat")
    renderer = Renderer()
    before = renderer.render(game)
    original = getattr(game, field)
    setattr(game, field, original + 7)
    changed = renderer.render(game)
    assert not np.array_equal(changed[:HUD_HEIGHT], before[:HUD_HEIGHT])
    assert np.array_equal(changed[HUD_HEIGHT:], before[HUD_HEIGHT:])
    setattr(game, field, original)
    assert np.array_equal(renderer.render(game), before)


def test_hud_shows_the_level_name() -> None:
    hud = [
        Renderer(decorations=False).render(Game(make_level(name=name)))[:HUD_HEIGHT]
        for name in ("alpha", "omega")
    ]
    assert not np.array_equal(hud[0], hud[1])


def test_hud_survives_huge_numbers_and_long_names() -> None:
    game = Game(make_level(name="a-very-long-level-name_with.odd/chars"))
    game.score, game.coins, game.time_left = 12_345_678_901, 98_765, 99_999
    frame = Renderer().render(game)
    assert np.array_equal(frame[HUD_HEIGHT:], Renderer(hud=False).render(game)[HUD_HEIGHT:])


# ------------------------------------------------------------------------------- decorations


@pytest.mark.parametrize("level", list_levels())
def test_decorations_only_ever_replace_sky(level: str) -> None:
    game = Game(level)
    decorated = panorama(Renderer(hud=False), game)
    undecorated = panorama(plain(), game)
    foreground = (undecorated != SKY).any(axis=2)
    assert np.array_equal(decorated[foreground], undecorated[foreground])

    scenery = (decorated != undecorated).any(axis=2)
    assert scenery.any(), "the level has no decorations at all"
    assert scenery.mean() < 0.10, "decorations are meant to be sparse"
    assert not scenery[:HUD_HEIGHT].any(), "clouds must keep out of the HUD band"


@pytest.mark.parametrize("level", list_levels())
def test_bushes_and_hills_rest_on_something(level: str) -> None:
    game = Game(level)
    decorated = panorama(Renderer(hud=False), game)
    undecorated = panorama(plain(), game)
    scenery = (decorated != undecorated).any(axis=2)
    solid = (undecorated != SKY).any(axis=2)
    low = scenery.copy()
    low[: 7 * TILE] = False  # clouds float; everything lower must stand on the ground
    assert low.any()
    rows, cols = np.nonzero(low)
    for col in np.unique(cols):
        bottom = rows[cols == col].max()
        assert bottom + 1 < VIEW_H and solid[bottom + 1, col], (col, bottom)


@pytest.mark.parametrize("level", list_levels())
def test_player_starts_in_front_of_plain_sky(level: str) -> None:
    game = Game(level)
    frame = Renderer(hud=False).render(bare(game))
    p = game.player
    cam = int(game.camera_x)
    left, right = int(p.x) - TILE - cam, int(p.x + p.w) + TILE - cam
    assert (frame[int(p.y) - TILE : int(p.y + p.h), max(left, 0) : right] == SKY).all()


def test_decorations_can_be_switched_on_a_live_renderer() -> None:
    game = Game("flat")
    renderer = Renderer()
    decorated = renderer.render(game)
    renderer.decorations = False
    assert np.array_equal(renderer.render(game), Renderer(decorations=False).render(game))
    renderer.decorations = True
    assert np.array_equal(renderer.render(game), decorated)
    renderer.hud = False
    assert np.array_equal(renderer.render(game), Renderer(hud=False).render(game))


def test_decorations_do_not_cover_entities_or_the_player() -> None:
    game = Game("1-1")
    play(game, NOOP, 60)
    assert game.entities
    decorated = Renderer().render(game)
    undecorated = Renderer(decorations=False).render(game)
    foreground = (undecorated != SKY).any(axis=2)
    assert np.array_equal(decorated[foreground], undecorated[foreground])


def test_decorations_are_the_same_in_every_process() -> None:
    code = (
        "import zlib\n"
        "from mario_play.game.engine import Game\n"
        "from mario_play.game.renderer import Renderer\n"
        "print(zlib.crc32(Renderer().render(Game('1-1')).tobytes()))\n"
    )
    here = zlib.crc32(Renderer().render(Game("1-1")).tobytes())
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        assert int(result.stdout) == here


# -------------------------------------------------------------------------------- robustness


def test_unknown_tile_id_is_reported() -> None:
    game = make_game()
    game.level.tiles[3, 3] = 99
    with pytest.raises(ValueError, match="tile"):
        Renderer().render(game)
    healthy = make_game()
    renderer = Renderer()
    renderer.render(healthy)
    healthy.level.tiles[3, 3] = 99
    with pytest.raises(ValueError, match="tile"):
        renderer.render(healthy)


def test_wrong_level_height_is_rejected() -> None:
    game = make_game()
    game.level.tiles = np.zeros((10, 40), dtype=np.int8)
    with pytest.raises(ValueError, match="15"):
        Renderer().render(game)


def test_game_modules_stay_free_of_heavy_imports() -> None:
    code = (
        "import sys\n"
        "import mario_play.game.renderer\n"
        "heavy = {'pygame', 'gymnasium', 'torch', 'PIL'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------------------------------- speed


def test_render_is_fast_on_1_1() -> None:
    game = Game("1-1")
    play(game, RUN_RIGHT, 60)
    cam = game.camera_x
    for offset in (20, 60, 100, 140):  # a busy screen: four walkers, a turtle and a mushroom
        game.entities.append(Walker(cam + offset, float(FLOOR_Y - 14)))
    game.entities.append(Turtle(cam + 180, float(FLOOR_Y - 22)))
    game.entities.append(Mushroom(cam + 220, float(FLOOR_Y - 14)))
    renderer = Renderer()
    renderer.render(game)  # builds the background

    samples = []
    for _ in range(300):
        started = time.perf_counter()
        renderer.render(game)
        samples.append(time.perf_counter() - started)
    median_ms = sorted(samples)[len(samples) // 2] * 1000
    print(f"median render time on 1-1: {median_ms:.3f} ms")
    # The target is < 1 ms; the hard limit is generous so that a loaded CI box does not flake.
    assert median_ms <= 3.0


# ---------------------------------------------------------------------------- preview script


def test_preview_script_writes_a_contact_sheet(tmp_path: Path) -> None:
    from PIL import Image

    path = REPO_ROOT / "scripts" / "render_preview.py"
    spec = importlib.util.spec_from_file_location("render_preview", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "nested" / "sheet.png"
    assert module.main(["--level", "flat", "--out", str(out), "--frames", "4"]) == 0
    with Image.open(out) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.width >= 2 * VIEW_W and image.height >= 2 * VIEW_H
