"""Tile ids of the level grid and their collision class."""

from __future__ import annotations

from enum import IntEnum


class Tile(IntEnum):
    """Value stored in `Level.tiles` for one 16x16 cell."""

    EMPTY = 0
    GROUND = 1
    HARD = 2
    BRICK = 3
    QUESTION_COIN = 4
    QUESTION_MUSHROOM = 5
    USED = 6
    PIPE_TL = 7
    PIPE_TR = 8
    PIPE_L = 9
    PIPE_R = 10
    COIN = 11
    FLAGPOLE = 12
    FLAG_TOP = 13


SOLID: frozenset[int] = frozenset(
    {
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
    }
)

# Lookup tables indexed by tile id: the hot collision loops use these instead of
# set membership, which is slow for the numpy scalars that `tiles[row, col]` returns.
IS_SOLID: tuple[bool, ...] = tuple(t in SOLID for t in Tile)
IS_GOAL: tuple[bool, ...] = tuple(t in (Tile.FLAGPOLE, Tile.FLAG_TOP) for t in Tile)
