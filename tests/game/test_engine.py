"""`Game` API: construction, reset, snapshot, clone, speed, and bundled-level completability."""

from __future__ import annotations

import copy
import dataclasses
import heapq
import pickle
import random
import subprocess
import sys
import time

import numpy as np
import pytest

from mario_play.game.constants import TILE
from mario_play.game.engine import Buttons, Game, StepEvents
from mario_play.game.entities import Entity, Mushroom, Player, Turtle, Walker
from mario_play.game.level import Level, Spawn, list_levels, load_level
from mario_play.game.tiles import Tile

NOOP = Buttons()
RIGHT = Buttons(right=True)
LEFT = Buttons(left=True)
JUMP = Buttons(jump=True)
RIGHT_JUMP = Buttons(right=True, jump=True)
RUN_RIGHT = Buttons(right=True, run=True)
RUN_RIGHT_JUMP = Buttons(right=True, run=True, jump=True)

SNAPSHOT_KEYS = {
    "x",
    "y",
    "vx",
    "vy",
    "on_ground",
    "big",
    "coins",
    "score",
    "time_left",
    "over",
    "won",
    "death_cause",
    "frame",
}

SMALL = """\
; time=50
................................
..?M..B.........................
................................
.........................F......
S...o....g...k...........F......
################################
################################
"""


def small_game() -> Game:
    return Game(Level.from_string(SMALL, name="small"))


def scripted(frame: int) -> Buttons:
    """A fixed, lively input script: run right, jump often, sometimes turn back."""
    phase = frame % 97
    return Buttons(left=phase > 90, right=phase <= 90, jump=frame % 41 < 22, run=frame % 300 < 250)


# --- value types ---------------------------------------------------------------------------------


def test_buttons_are_frozen_and_default_to_released() -> None:
    assert Buttons() == Buttons(left=False, right=False, jump=False, run=False)
    assert Buttons(right=True) == RIGHT and hash(Buttons(right=True)) == hash(RIGHT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        RIGHT.left = True  # type: ignore[misc]


def test_step_events_default_to_nothing_happened() -> None:
    events = StepEvents()

    assert (events.coins, events.stomps, events.bricks, events.powerups) == (0, 0, 0, 0)
    assert (events.hurt, events.died, events.won) == (False, False, False)
    assert events.death_cause is None and events.score_delta == 0


# --- construction and reset ----------------------------------------------------------------------


def test_default_level_is_1_1_and_names_load_bundled_levels() -> None:
    assert Game().level.name == "1-1"
    assert Game("flat").level.name == "flat"
    with pytest.raises(ValueError):
        Game("no-such-level")


def test_game_accepts_level_file_paths(tmp_path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text(SMALL)

    assert Game(path).level.name == "tiny"
    assert Game(str(path)).level.time == 50


def test_unknown_spawn_kind_is_rejected() -> None:
    level = Level.from_string(SMALL)
    level.spawns.append(Spawn("dragon", 5, 12))

    with pytest.raises(ValueError, match="dragon"):
        Game(level)


def test_initial_state() -> None:
    game = small_game()
    p = game.player

    assert isinstance(p, Player) and p.kind == "player"
    assert (p.x, p.y + p.h) == (2.0, 13 * TILE)
    assert not p.big and not p.dead and p.alive and p.invuln_frames == 0 and p.facing == 1
    assert (game.frame, game.score, game.coins, game.camera_x) == (0, 0, 0, 0.0)
    assert game.time_left == 50
    assert (game.over, game.won, game.death_cause) == (False, False, None)
    assert isinstance(game.rng, random.Random)
    assert [type(e) for e in game.entities] == [Walker, Turtle]
    assert all(isinstance(e, Entity) and e is not p for e in game.entities)


def test_entities_expose_the_contract_attributes() -> None:
    game = small_game()
    walker, turtle = game.entities
    shroom = Mushroom(0.0, 0.0)

    for entity in (game.player, walker, turtle, shroom):
        for attr in ("x", "y", "w", "h", "vx", "vy", "alive", "kind", "facing"):
            assert hasattr(entity, attr), (entity, attr)
        assert isinstance(entity.x, float) and isinstance(entity.y, float)
    assert (walker.kind, turtle.kind, shroom.kind) == ("walker", "turtle", "mushroom")
    assert (walker.w, walker.h) == (14, 14) and walker.squished_frames == 0
    assert (turtle.w, turtle.h) == (14, 22) and turtle.state == "walk"
    assert (shroom.w, shroom.h) == (14, 14)


def test_game_plays_on_a_private_copy_of_the_level() -> None:
    level = Level.from_string(SMALL, name="small")
    game = Game(level)

    game.level.tiles[:] = Tile.EMPTY

    assert game.level is not level
    assert level.tiles.any()


def test_step_returns_events_and_advances_the_frame_counter() -> None:
    game = small_game()

    events = game.step(RIGHT)

    assert isinstance(events, StepEvents)
    assert game.frame == 1
    for _ in range(9):
        game.step(RIGHT)
    assert game.frame == 10 and game.player.x > 2.0


def test_player_can_jump_on_the_very_first_frame() -> None:
    game = small_game()
    assert game.player.on_ground

    game.step(JUMP)

    assert game.player.vy < 0 and not game.player.on_ground


def test_entities_list_keeps_its_identity_when_entities_are_removed() -> None:
    game = small_game()
    entities = game.entities
    walker = entities[0]
    walker.squish()

    for _ in range(40):
        game.step(NOOP)

    assert game.entities is entities
    assert walker not in entities and len(entities) == 1


def test_snapshot_has_exactly_the_contract_keys_and_plain_values() -> None:
    game = small_game()
    game.step(RIGHT_JUMP)

    snap = game.snapshot()

    assert set(snap) == SNAPSHOT_KEYS
    assert snap["frame"] == 1 and snap["on_ground"] is False and snap["big"] is False
    assert snap["x"] == game.player.x and snap["vy"] == game.player.vy
    assert all(type(v) in (int, float, bool, str, type(None)) for v in snap.values())


def test_reset_restores_the_pristine_level_and_state() -> None:
    game = small_game()
    fresh = game.snapshot()
    tiles = game.level.tiles.copy()
    game.player.grow()
    for frame in range(400):
        game.step(scripted(frame))
    game.level.tiles[9, 6] = Tile.EMPTY
    assert game.snapshot() != fresh

    game.reset()

    assert game.snapshot() == fresh
    assert np.array_equal(game.level.tiles, tiles)
    assert [type(e) for e in game.entities] == [Walker, Turtle]
    assert game.camera_x == 0.0 and not game.player.big


def test_reset_with_seed_reseeds_the_rng() -> None:
    a, b = Game("flat", seed=7), Game("flat", seed=8)
    a.rng.random()

    a.reset(seed=3)
    b.reset(seed=3)

    assert a.rng.random() == b.rng.random()
    assert Game("flat", seed=11).rng.random() == Game("flat", seed=11).rng.random()


def test_reset_without_seed_keeps_the_rng_stream() -> None:
    game = Game("flat", seed=5)
    expected = random.Random(5)
    expected.random()

    game.rng.random()
    game.reset()

    assert game.rng.random() == expected.random()


# --- clone ---------------------------------------------------------------------------------------


def test_clone_is_a_deep_independent_copy() -> None:
    game = small_game()
    for frame in range(30):
        game.step(scripted(frame))

    twin = game.clone()

    assert isinstance(twin, Game)
    assert twin.snapshot() == game.snapshot()
    assert twin.level is not game.level and twin.level.tiles is not game.level.tiles
    assert np.array_equal(twin.level.tiles, game.level.tiles)
    assert twin.player is not game.player
    assert len(twin.entities) == len(game.entities)
    assert all(
        a is not b and type(a) is type(b) for a, b in zip(twin.entities, game.entities, strict=True)
    )
    assert twin.rng is not game.rng and twin.rng.getstate() == game.rng.getstate()

    before = game.snapshot()
    tiles_before = game.level.tiles.copy()
    entity_xs = [e.x for e in game.entities]
    twin.level.tiles[:] = Tile.HARD
    twin.player.x += 50.0
    twin.entities.clear()
    twin.rng.random()
    twin.score += 999

    assert game.snapshot() == before
    assert np.array_equal(game.level.tiles, tiles_before)
    assert [e.x for e in game.entities] == entity_xs
    assert twin.rng.getstate() != game.rng.getstate()


def test_clone_keeps_the_subclass_and_repr_is_informative() -> None:
    class MyGame(Game):
        pass

    game = MyGame("flat")

    assert type(game.clone()) is MyGame
    assert "flat" in repr(game) and "running" in repr(game)


@pytest.mark.parametrize("duplicate", [copy.deepcopy, lambda g: pickle.loads(pickle.dumps(g))])
def test_generic_copies_also_continue_identically(duplicate) -> None:
    game = small_game()
    for frame in range(60):
        game.step(scripted(frame))

    twin = duplicate(game)
    for frame in range(60, 300):
        game.step(scripted(frame))
        twin.step(scripted(frame))

    assert twin.snapshot() == game.snapshot()
    assert np.array_equal(twin.level.tiles, game.level.tiles)


def test_clone_of_a_finished_game_stays_finished() -> None:
    game = Game("flat")
    while not game.over:
        game.step(RUN_RIGHT)

    twin = game.clone()

    assert twin.over and twin.won
    assert twin.step(RIGHT) == StepEvents()


def test_clone_can_be_reset_to_the_pristine_level() -> None:
    game = small_game()
    fresh = game.snapshot()
    for frame in range(100):
        game.step(scripted(frame))

    twin = game.clone()
    twin.reset()

    assert twin.snapshot() == fresh
    assert twin.level.tiles[9, 2] == Tile.QUESTION_COIN


# --- hygiene and speed ---------------------------------------------------------------------------


def test_game_package_does_not_import_heavy_dependencies() -> None:
    code = (
        "import sys\n"
        "import mario_play.game.constants, mario_play.game.tiles, mario_play.game.level\n"
        "import mario_play.game.entities, mario_play.game.physics, mario_play.game.engine\n"
        "heavy = {'pygame', 'torch', 'gymnasium'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_step_throughput_on_1_1() -> None:
    game = Game("1-1")
    frames = 20_000
    start = time.perf_counter()
    for frame in range(frames):
        game.step(scripted(frame))
        if game.over:
            game.reset()
    elapsed = time.perf_counter() - start

    assert frames / elapsed >= 5_000, f"{frames / elapsed:.0f} frames/s"


def test_clone_is_cheap() -> None:
    game = Game("1-1")
    for frame in range(300):
        game.step(scripted(frame))
        if game.over:
            game.reset()
    clones = 2_000
    start = time.perf_counter()
    for _ in range(clones):
        game.clone()
    per_clone = (time.perf_counter() - start) / clones

    assert per_clone < 1e-3, f"{per_clone * 1e6:.0f} us per clone"


# --- completability ------------------------------------------------------------------------------

RIGHT_ONLY = [RUN_RIGHT, RUN_RIGHT_JUMP, RIGHT, RIGHT_JUMP, NOOP]  # the env's smallest action set


def solve(
    level: str, actions: list[Buttons], hold_frames: int = 8, max_nodes: int = 40_000
) -> list[Buttons] | None:
    """Best-first search over `Game.clone()` for a button script that reaches the flag.

    Nodes are expanded furthest-right first; every expansion holds one action for
    `hold_frames` frames (like an env with that frame skip). States falling into a
    pit are pruned early and near-identical states are visited once. Returns one
    `Buttons` per `hold_frames` frames.
    """
    floor_y = 13 * TILE
    root = Game(level)
    tie = 0
    frontier: list[tuple[float, int, Game, tuple[int, ...]]] = [(0.0, tie, root, ())]
    seen: set[tuple] = set()
    for _ in range(max_nodes):
        if not frontier:
            return None
        _, _, game, path = heapq.heappop(frontier)
        for index, action in enumerate(actions):
            child = game.clone()
            for _ in range(hold_frames):
                child.step(action)
            if child.won:
                return [actions[i] for i in (*path, index)]
            p = child.player
            if child.over or p.y + p.h > floor_y + 4:
                continue
            key = (
                int(p.x) >> 1,
                int(p.y) >> 1,
                round(p.vx * 4),
                round(p.vy * 2),
                p.big,
                child.frame // 32,
            )
            if key in seen:
                continue
            seen.add(key)
            tie += 1
            heapq.heappush(frontier, (-p.x, tie, child, (*path, index)))
    return None


def replay(level: str, script: list[Buttons], hold_frames: int = 8) -> Game:
    game = Game(level)
    for action in script:
        for _ in range(hold_frames):
            game.step(action)
    return game


def test_flat_is_finished_by_just_holding_right() -> None:
    game = Game("flat")
    for _ in range(2_000):
        game.step(RIGHT)

    assert game.won


def test_1_1_is_finishable_by_running_right_and_jumping() -> None:
    script = solve("1-1", [RUN_RIGHT, RUN_RIGHT_JUMP])

    assert script is not None, "no run-right/jump script reaches the flag of 1-1"
    assert replay("1-1", script).won  # and the script replays deterministically


@pytest.mark.parametrize("hold_frames", [4, 8])
@pytest.mark.parametrize("name", ["1-1", "1-2", "1-3"])
def test_levels_are_completable_with_the_smallest_action_set(name: str, hold_frames: int) -> None:
    script = solve(name, RIGHT_ONLY, hold_frames)

    assert script is not None, f"the search found no way to the flag of {name}"
    final = replay(name, script, hold_frames)
    assert final.won and final.time_left > 200  # with plenty of time to spare


@pytest.mark.parametrize("name", ["1-1", "1-2", "1-3"])
def test_levels_are_completable_without_the_run_button(name: str) -> None:
    script = solve(name, [RIGHT, RIGHT_JUMP, NOOP, LEFT])

    assert script is not None, f"walking and jumping is not enough for {name}"
    assert replay(name, script).won


def test_every_bundled_level_is_covered_by_a_completability_test() -> None:
    assert list_levels() == ["1-1", "1-2", "1-3", "flat"]
    assert all(load_level(name).flag_col > 0 for name in list_levels())
