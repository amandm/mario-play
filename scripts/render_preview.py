"""Render a contact sheet of a level: several frames along a played run, upscaled, as one PNG.

    .venv/bin/python scripts/render_preview.py --level 1-1 --out preview.png
    .venv/bin/python scripts/render_preview.py --level flat --gallery --out preview-flat.png

The level is played by a small driver: simple reflexes (run right, jump at walls, pits and
enemies) that are checked by look-ahead on `Game.clone()`; when they would get the player
killed, a handful of other openings are tried and the one that gets furthest alive is taken.
So the snapshots show what a real run looks like: a scrolled camera, enemies that have woken
up, collected coins. It is a preview aid, not an agent: on a level it cannot finish the sheet
simply ends where the run did. `--gallery` appends staged scenes that show every tile and
every sprite state at once, the quickest way to eyeball art and alignment after touching
the renderer.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from mario_play.game.constants import TILE, VIEW_H, VIEW_W
from mario_play.game.engine import Buttons, Game
from mario_play.game.entities import Mushroom, Turtle, Walker
from mario_play.game.font import draw_text
from mario_play.game.level import Level
from mario_play.game.renderer import Renderer
from mario_play.game.tiles import Tile

CAPTION_H = 12
GAP = 4
PAPER = (24, 20, 36)
INK = (236, 236, 244)

RUN = Buttons(right=True, run=True)
RUN_JUMP = Buttons(right=True, run=True, jump=True)
# Openings the driver tries when plain reflexes would get the player killed: (buttons, frames
# to hold them). After its opening every plan falls back to the reflexes.
OPENINGS: tuple[tuple[Buttons, int], ...] = (
    (RUN_JUMP, 30),
    (RUN_JUMP, 12),
    (RUN, 8),
    (Buttons(right=True, jump=True), 30),
    (Buttons(), 12),
    (Buttons(left=True), 12),
)
DECIDE_EVERY = 4
HORIZON = 60
LANDING_GRACE = 50
STALL_FRAMES = 600
DEAD = -1e6


def _reflex(game: Game) -> Buttons:
    """Run right; jump at walls, pits and enemies. Good enough to be a rollout policy."""
    p = game.player
    if not p.on_ground:
        return RUN_JUMP if p.vy < 0 else RUN  # full jumps; let go on the way down
    if not p.jump_armed:
        return RUN  # jumps are edge-triggered: the button has to come up before the next one
    level = game.level
    col = int((p.x + p.w + 6 + 6 * abs(p.vx)) // TILE)
    feet = int((p.y + p.h - 1) // TILE)
    wall = level.solid_at(col, feet) or level.solid_at(col, feet - 1)
    pit = not any(level.solid_at(col, row) for row in range(feet + 1, 15))
    enemy = any(
        e.kind != "mushroom" and 0 < e.x - p.x < 44 and abs(e.y + e.h - p.y - p.h) < 24
        for e in game.entities
    )
    return RUN_JUMP if wall or pit or enemy else RUN


def _rollout(game: Game, opening: Buttons | None, hold: int) -> float:
    """How far a copy of the game gets with `opening` held for `hold` frames, then reflexes."""
    sim = game.clone()
    horizon = hold + HORIZON  # every plan gets the same time after its opening: a fair race
    for frame in range(horizon + LANDING_GRACE):
        if sim.over or (frame >= horizon and sim.player.on_ground):
            break
        sim.step(opening if opening is not None and frame < hold else _reflex(sim))
    if sim.won:
        return 1e9 - sim.frame
    # Dead ends only compete with each other: then the furthest one wins, which keeps the run
    # heading for the obstacle until some opening clears it.
    return sim.player.x + (DEAD if sim.over else 0.0)


def autoplay(game: Game) -> Iterator[None]:
    """Play `game` with the look-ahead driver, yielding after every frame.

    Stops when the game is over, or when the player has not got any further for a while.
    """
    opening: Buttons | None = None
    best_x, best_frame = game.player.x, game.frame
    while not game.over and game.frame - best_frame < STALL_FRAMES:
        if game.frame % DECIDE_EVERY == 0:
            opening = None
            if _rollout(game, None, 0) < DEAD / 2:  # reflexes alone end badly: look around
                opening = max(OPENINGS, key=lambda plan: _rollout(game, *plan))[0]
        game.step(opening if opening is not None else _reflex(game))
        if game.player.x > best_x:
            best_x, best_frame = game.player.x, game.frame
        yield


def snapshots(level: str, count: int) -> list[tuple[str, np.ndarray]]:
    """`count` captioned frames, evenly spaced between the start of the level and its flag."""
    game = Game(level)
    renderer = Renderer()
    start = game.player.x
    goal = game.level.flag_col * TILE
    marks = [start + (goal - start) * i / max(count - 1, 1) for i in range(count)]

    def shot(finished: bool = False) -> tuple[str, np.ndarray]:
        state = "clear" if game.won else game.death_cause if game.over else "playing"
        if finished and not game.over:
            state = "stalled"
        caption = f"{game.level.name[:12]}  frame {game.frame}  x {int(game.player.x)}  {state}"
        return caption, renderer.render(game)

    shots = [shot()]
    for _ in autoplay(game):
        if len(shots) < count - 1 and game.player.x >= marks[len(shots)]:
            shots.append(shot())
    shots.append(shot(finished=True))  # how the run ended
    return shots


_GALLERY_LEVEL = """
................
................
..............F.
..............F.
..............F.
..?MB..oo.....F.
..............F.
..............F.
..............F.
..B?B.........F.
.......[].....F.
....X..[].....F.
.S.XX..[].....F.
#####..#########
#####..#########
"""
# (caption, big player, pose, facing, blocks used up and coins collected)
_GALLERY_SCENES = (
    ("small standing", False, "stand", 1, False),
    ("big walking", True, "walk", 1, False),
    ("small jumping left", False, "jump", -1, False),
    ("big jumping, used blocks", True, "jump", 1, True),
)


def gallery() -> list[tuple[str, np.ndarray]]:
    """Staged scenes that put every tile and every sprite state on screen."""
    floor = 13 * TILE
    shots = []
    for caption, big, pose, facing, used in _GALLERY_SCENES:
        game = Game(Level.from_string(_GALLERY_LEVEL, name="gallery"))
        tiles = game.level.tiles
        player = game.player
        player.facing = facing
        if big:
            player.grow()
        if pose == "walk":
            player.vx = 1.5
        elif pose == "jump":  # in mid-air, the head just below the row of blocks
            player.on_ground = False
            player.x, player.y = 3.0 * TILE + 2, 10.0 * TILE + 2
        if used:
            tiles[(tiles == Tile.QUESTION_COIN) | (tiles == Tile.QUESTION_MUSHROOM)] = Tile.USED
            tiles[tiles == Tile.COIN] = Tile.EMPTY
            game.score, game.coins = 1400, 3

        flat_walker = Walker(10 * TILE + 1.0, floor - 14.0)
        flat_walker.squish()
        shell = Turtle(12 * TILE + 1.0, floor - 22.0)
        shell.to_shell()
        game.entities += [
            Walker(9 * TILE + 1.0, floor - 14.0),
            flat_walker,
            Turtle(11 * TILE + 1.0, floor - 22.0, facing=1),
            shell,
            Mushroom(3 * TILE + 1.0, 5 * TILE - 14.0),
        ]
        shots.append((f"gallery: {caption}", Renderer().render(game)))
    return shots


def contact_sheet(
    shots: Sequence[tuple[str, np.ndarray]], columns: int = 3, scale: int = 2
) -> np.ndarray:
    """Lay captioned frames out on a grid and upscale the sheet with nearest neighbour."""
    columns = max(1, min(columns, len(shots)))
    rows = -(-len(shots) // columns)
    cell_w, cell_h = VIEW_W + GAP, VIEW_H + CAPTION_H + GAP
    sheet = np.empty((rows * cell_h + GAP, columns * cell_w + GAP, 3), dtype=np.uint8)
    sheet[:] = PAPER
    for index, (caption, frame) in enumerate(shots):
        x = GAP + (index % columns) * cell_w
        y = GAP + (index // columns) * cell_h
        sheet[y : y + VIEW_H, x : x + VIEW_W] = frame
        draw_text(sheet, caption, x + 2, y + VIEW_H + 3, INK)
    return sheet.repeat(scale, axis=0).repeat(scale, axis=1)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--level", default="1-1", help="bundled level name or level file path")
    parser.add_argument("--out", default="preview.png", help="PNG file to write")
    parser.add_argument("--frames", type=int, default=9, help="snapshots along the level")
    parser.add_argument("--columns", type=int, default=3, help="snapshots per sheet row")
    parser.add_argument("--scale", type=int, default=2, help="integer upscaling factor")
    parser.add_argument("--gallery", action="store_true", help="append staged sprite scenes")
    args = parser.parse_args(argv)
    if args.frames < 2 or args.scale < 1 or args.columns < 1:
        parser.error("--frames must be >= 2, --scale and --columns >= 1")

    from PIL import Image

    shots = snapshots(args.level, args.frames)
    if args.gallery:
        shots += gallery()
    sheet = contact_sheet(shots, args.columns, args.scale)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out, format="PNG")
    print(f"wrote {out} ({sheet.shape[1]}x{sheet.shape[0]}, {len(shots)} frames): {shots[-1][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
