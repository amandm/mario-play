"""Observation builders: the compact grid observation and the observation spaces.

The grid observation is a `(14, 15, 16)` float32 stack of planes over a window of
tiles: all 15 rows, 16 columns, horizontally egocentric. The column holding the
left edge of the player's hitbox is always window column `PLAYER_COLUMN`, and the
`x_offset` plane says how far into that column the edge sits, so together they
give the exact x position relative to the tiles. Columns outside the level read
as solid. Entities and the player mark every cell their hitbox overlaps.

This runs on every env step of CPU training: the tile planes come from one table
lookup on a slice of `level.tiles`, everything else is a handful of scalar writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from mario_play.game.constants import LEVEL_H_TILES, RUN_MAX, TILE, VIEW_H, VIEW_TILES_W, VIEW_W
from mario_play.game.tiles import Tile

if TYPE_CHECKING:
    from mario_play.game.engine import Game

GRID_PLANES: tuple[str, ...] = (
    "solid",  # 0: tiles that block movement (and everything outside the level)
    "breakable",  # 1: bricks and unused ?-blocks (they are solid as well)
    "coin",  # 2: coin tiles
    "enemy",  # 3: walkers, turtles and shells (resting or moving); not squished walkers
    "moving_shell",  # 4: kicked shells
    "mushroom",  # 5
    "goal",  # 6: the flagpole
    "player",  # 7: the player's hitbox
    # Scalars, broadcast over the whole plane:
    "vx",  # 8: vx / 2.5, clipped to [-1, 1]
    "vy",  # 9: vy / 5, clipped to [-1, 1]
    "on_ground",  # 10
    "big",  # 11
    "x_offset",  # 12: (x mod 16) / 16
    "time",  # 13: time_left / level time
)
"""Names of the grid observation's planes, in channel order."""

PLAYER_COLUMN = 4
"""Window column that holds the left edge of the player's hitbox."""

GRID_SHAPE: tuple[int, int, int] = (len(GRID_PLANES), LEVEL_H_TILES, VIEW_TILES_W)
PIXEL_SHAPE: tuple[int, int, int] = (VIEW_H, VIEW_W, 3)

VX_SCALE = RUN_MAX
VY_SCALE = 5.0

_N_PLANES, _ROWS, _COLS = GRID_SHAPE
_SOLID, _BREAKABLE, _COIN, _ENEMY, _MOVING_SHELL, _MUSHROOM, _GOAL, _PLAYER = range(8)
_FIRST_SCALAR = 8

# Tile id -> its column of the 8 spatial planes (the entity planes stay zero), so that
# `_TILE_PLANES[:, window]` turns a (15, n) tile slice into (8, 15, n) planes in one go.
_TILE_PLANES = np.zeros((_FIRST_SCALAR, len(Tile)), dtype=np.float32)
_TILE_PLANES[
    _SOLID,
    [
        Tile.GROUND,
        Tile.HARD,
        Tile.BRICK,
        Tile.QUESTION_COIN,
        Tile.QUESTION_MUSHROOM,
        Tile.USED,
        Tile.PIPE_TL,
        Tile.PIPE_TR,
        Tile.PIPE_L,
        Tile.PIPE_R,
    ],
] = 1.0
_TILE_PLANES[_BREAKABLE, [Tile.BRICK, Tile.QUESTION_COIN, Tile.QUESTION_MUSHROOM]] = 1.0
_TILE_PLANES[_COIN, Tile.COIN] = 1.0
_TILE_PLANES[_GOAL, [Tile.FLAGPOLE, Tile.FLAG_TOP]] = 1.0


def grid_observation_space() -> gym.spaces.Box:
    """`Box(-1, 1, (14, 15, 16), float32)`."""
    return gym.spaces.Box(-1.0, 1.0, shape=GRID_SHAPE, dtype=np.float32)


def pixel_observation_space() -> gym.spaces.Box:
    """`Box(0, 255, (240, 256, 3), uint8)`: the renderer's RGB frame."""
    return gym.spaces.Box(0, 255, shape=PIXEL_SHAPE, dtype=np.uint8)


def _mark(plane: np.ndarray, x: float, y: float, w: float, h: float, first_col: int) -> None:
    """Set the cells of `plane` the box overlaps; window column 0 is level column `first_col`."""
    # Same half-open pixel ranges as the collision code: [x, x + w) x [y, y + h).
    c0 = int(x // TILE) - first_col
    c1 = -int(-(x + w) // TILE) - 1 - first_col
    r0 = int(y // TILE)
    r1 = -int(-(y + h) // TILE) - 1
    if c1 < 0 or c0 >= _COLS or r1 < 0 or r0 >= _ROWS:
        return
    if c0 == c1 and r0 == r1:
        plane[r0, c0] = 1.0  # fully inside, or the early return above would have fired
        return
    plane[max(r0, 0) : r1 + 1, max(c0, 0) : c1 + 1] = 1.0


def grid_observation(game: Game) -> np.ndarray:
    """The grid observation of `game`: a new `(14, 15, 16)` float32 array within [-1, 1].

    See `GRID_PLANES` for the channels. The game is not modified.
    """
    obs = np.zeros(GRID_SHAPE, dtype=np.float32)
    level = game.level
    player = game.player
    first_col = int(player.x // TILE) - PLAYER_COLUMN  # level column of window column 0

    # Tiles: the part of the window that lies inside the level...
    lo = max(first_col, 0)
    hi = min(first_col + _COLS, level.width_tiles)
    if lo < hi:
        obs[:_FIRST_SCALAR, :, lo - first_col : hi - first_col] = _TILE_PLANES[
            :, level.tiles[:, lo:hi]
        ]
    # ... and what sticks out of it is a wall, as it is for the physics.
    if first_col < 0:
        obs[_SOLID, :, : min(-first_col, _COLS)] = 1.0
    if hi - first_col < _COLS:
        obs[_SOLID, :, max(hi - first_col, 0) :] = 1.0

    for entity in game.entities:
        kind = entity.kind
        if kind == "walker":
            if entity.squished_frames > 0:
                continue  # a harmless corpse
            plane = _ENEMY
        elif kind == "turtle":
            plane = _ENEMY
            if entity.state == "shell_moving":
                _mark(obs[_MOVING_SHELL], entity.x, entity.y, entity.w, entity.h, first_col)
        elif kind == "mushroom":
            plane = _MUSHROOM
        else:
            continue
        _mark(obs[plane], entity.x, entity.y, entity.w, entity.h, first_col)
    _mark(obs[_PLAYER], player.x, player.y, player.w, player.h, first_col)

    vx = player.vx / VX_SCALE
    vy = player.vy / VY_SCALE
    obs[_FIRST_SCALAR:] = np.array(
        (
            max(-1.0, min(1.0, vx)),
            max(-1.0, min(1.0, vy)),
            1.0 if player.on_ground else 0.0,
            1.0 if player.big else 0.0,
            (player.x % TILE) / TILE,
            max(0.0, min(1.0, game.time_left / level.time)) if level.time > 0 else 0.0,
        ),
        dtype=np.float32,
    ).reshape(-1, 1, 1)
    return obs
