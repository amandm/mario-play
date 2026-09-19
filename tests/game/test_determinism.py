"""Determinism, clone equivalence and a random-input fuzz over every bundled level."""

from __future__ import annotations

import math
import random

import pytest

from mario_play.game.constants import LEVEL_H_TILES, TILE
from mario_play.game.engine import Buttons, Game
from mario_play.game.level import Level, list_levels
from mario_play.game.tiles import IS_SOLID

FUZZ_SEEDS = 20
FUZZ_FRAMES = 3_000


def random_script(seed: int, frames: int) -> list[Buttons]:
    """Random buttons held for random durations, biased to the right so levels get explored."""
    rng = random.Random(seed)
    script: list[Buttons] = []
    while len(script) < frames:
        buttons = Buttons(
            left=rng.random() < 0.15,
            right=rng.random() < 0.75,
            jump=rng.random() < 0.45,
            run=rng.random() < 0.5,
        )
        script += [buttons] * rng.randint(1, 30)
    return script[:frames]


def full_state(game: Game) -> tuple:
    """Everything that defines the simulation state, not just the `snapshot()` summary."""
    entities = tuple(
        (e.kind, e.x, e.y, e.vx, e.vy, e.facing, e.alive, getattr(e, "state", None))
        for e in game.entities
    )
    p = game.player
    return (
        tuple(sorted(game.snapshot().items(), key=lambda kv: kv[0])),
        (p.facing, p.invuln_frames, p.dead, p.h),
        game.camera_x,
        entities,
        game.level.tiles.tobytes(),
        game.rng.getstate(),
    )


@pytest.mark.parametrize("level", ["1-1", "1-3"])
def test_same_seed_and_buttons_give_identical_snapshots_every_frame(level: str) -> None:
    script = random_script(seed=1, frames=1_500)
    a, b = Game(level, seed=42), Game(level, seed=42)

    for buttons in script:
        a.step(buttons)
        b.step(buttons)
        assert a.snapshot() == b.snapshot()
        if a.over:
            a.reset()
            b.reset()
    assert full_state(a) == full_state(b)


def test_reset_replays_identically() -> None:
    script = random_script(seed=2, frames=600)
    game = Game("1-2", seed=0)

    def run() -> list[dict]:
        game.reset(seed=0)
        snapshots = []
        for buttons in script:
            game.step(buttons)
            snapshots.append(game.snapshot())
        return snapshots

    assert run() == run()


def test_clone_continues_identically_to_the_original() -> None:
    script = random_script(seed=3, frames=1_200)
    game = Game("1-2", seed=9)
    for buttons in script[:400]:
        game.step(buttons)
        if game.over:
            game.reset()

    twin = game.clone()
    assert full_state(twin) == full_state(game)
    for buttons in script[400:]:
        events_a, events_b = game.step(buttons), twin.step(buttons)
        assert events_a == events_b
        assert twin.snapshot() == game.snapshot()
        if game.over:
            game.reset()
            twin.reset()
    assert full_state(twin) == full_state(game)


def test_clone_with_diverging_inputs_does_not_affect_the_original() -> None:
    script = random_script(seed=4, frames=900)
    reference, game = Game("1-1", seed=1), Game("1-1", seed=1)
    chaos = random_script(seed=99, frames=900)

    for frame, buttons in enumerate(script):
        if frame % 50 == 0:
            twin = game.clone()
            for other in chaos[frame : frame + 120]:
                twin.step(other)
            twin.level.tiles[:] = 0
            twin.entities.clear()
        reference.step(buttons)
        game.step(buttons)
        assert game.snapshot() == reference.snapshot()
        if game.over:
            game.reset()
            reference.reset()
    assert full_state(game) == full_state(reference)


def test_clone_of_a_clone_and_mid_air_clone() -> None:
    game = Game("1-1")
    jump = Buttons(right=True, run=True, jump=True)
    for _ in range(40):
        game.step(jump)
    assert not game.player.on_ground

    twin = game.clone().clone()
    for _ in range(200):
        game.step(jump)
        twin.step(jump)

    assert full_state(twin) == full_state(game)


def _assert_sane(game: Game, context: str) -> None:
    p = game.player
    for value in (p.x, p.y, p.vx, p.vy, game.camera_x):
        assert math.isfinite(value), context
    assert game.camera_x <= p.x <= game.level.width_px - p.w, context
    assert 0.0 <= game.camera_x <= game.level.width_px - 256, context

    tiles = game.level.tiles
    rows = range(
        max(int(p.y // TILE), 0), min(int((p.y + p.h - 1e-9) // TILE), LEVEL_H_TILES - 1) + 1
    )
    for col in range(int(p.x // TILE), int((p.x + p.w - 1e-9) // TILE) + 1):
        for row in rows:
            assert not IS_SOLID[tiles[row, col]], f"{context}: player inside solid ({col}, {row})"

    for e in game.entities:
        assert math.isfinite(e.x) and math.isfinite(e.y), context
        assert -1.0 <= e.x <= game.level.width_px, context


@pytest.mark.parametrize("level", list_levels() or ["<no levels>"])
def test_fuzz_random_buttons(level: str) -> None:
    game = Game(level)
    furthest = 0.0
    deaths: set[str] = set()
    for seed in range(FUZZ_SEEDS):
        game.reset(seed=seed)
        score = coins = 0
        for frame, buttons in enumerate(random_script(seed=1000 + seed, frames=FUZZ_FRAMES)):
            events = game.step(buttons)
            score += events.score_delta
            coins += events.coins
            _assert_sane(game, f"{level} seed={seed} frame={frame}")
            furthest = max(furthest, game.player.x)
            if game.over:
                assert events.died != events.won
                assert (score, coins) == (game.score, game.coins)
                if events.died:
                    deaths.add(str(events.death_cause))
                game.reset()
                score = coins = 0
    assert furthest > 30 * TILE, "the fuzz should get somewhere"
    assert deaths <= {"pit", "enemy", "timeout"}


def rough_level(seed: int, width: int = 90) -> Level:
    """Random junk terrain: scattered blocks, one-tile gaps, pits, enemies on top of things."""
    rng = random.Random(seed)
    grid = [["."] * width for _ in range(LEVEL_H_TILES)]
    for row in (13, 14):
        grid[row] = ["#"] * width
    col = 8
    while col < width - 14:
        if rng.random() < 0.12:
            pit = rng.randint(1, 3)
            for k in range(pit):
                grid[13][col + k] = grid[14][col + k] = "."
            col += pit + 2
        else:
            col += 1
    for col in range(5, width - 12):
        for row in range(3, 13):
            if rng.random() < 0.13:
                grid[row][col] = rng.choice("XXBB?M")
    for col in range(6, width - 12):
        for row in range(2, 13):
            free = grid[row][col] == "." and grid[row - 1][col] == "."
            if free and grid[row + 1][col] in "XB?M#" and rng.random() < 0.08:
                grid[row][col] = rng.choice("gk")
            elif grid[row][col] == "." and rng.random() < 0.03:
                grid[row][col] = "o"
    grid[12][2] = "S"
    for row in range(1, 13):
        grid[row][width - 10] = "F"
    return Level.from_string("\n".join("".join(row) for row in grid), name=f"rough-{seed}")


def _solid_under(game: Game, e) -> tuple[int, int] | None:
    tiles = game.level.tiles
    last_col = game.level.width_tiles - 1
    rows = range(
        max(int(e.y // TILE), 0), min(int((e.y + e.h - 1e-9) // TILE), LEVEL_H_TILES - 1) + 1
    )
    for col in range(max(int(e.x // TILE), 0), min(int((e.x + e.w - 1e-9) // TILE), last_col) + 1):
        for row in rows:
            if IS_SOLID[tiles[row, col]]:
                return col, row
    return None


@pytest.mark.parametrize("seed", range(12))
def test_fuzz_rough_terrain_nothing_ever_ends_up_inside_a_solid_tile(seed: int) -> None:
    game = Game(rough_level(seed))
    was_big = False
    for frame, buttons in enumerate(random_script(seed=500 + seed, frames=2_500)):
        game.step(buttons)
        was_big = was_big or game.player.big
        if not game.over:
            assert _solid_under(game, game.player) is None, (seed, frame, game.player)
        for e in game.entities:
            assert _solid_under(game, e) is None, (seed, frame, e)
        if game.over:
            game.reset()
