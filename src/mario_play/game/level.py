"""ASCII level format, the `Level` container and the bundled level loader.

A level is text with one character per tile (see `LEGEND`). Lines starting with
`;` are comments; a comment of the exact form `; key=value` is metadata (only
`time` is known). Blank lines before the first and after the last row are
ignored - write `.` to make an explicitly empty row. Levels with fewer than 15
rows are padded with empty rows on top, and rows are right-padded to the longest.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mario_play.game.constants import DEFAULT_LEVEL_TIME, LEVEL_H_TILES, TILE, VIEW_TILES_W
from mario_play.game.tiles import IS_SOLID, Tile

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

LEGEND: dict[str, Tile] = {
    " ": Tile.EMPTY,
    ".": Tile.EMPTY,
    "#": Tile.GROUND,
    "X": Tile.HARD,
    "B": Tile.BRICK,
    "?": Tile.QUESTION_COIN,
    "M": Tile.QUESTION_MUSHROOM,
    "o": Tile.COIN,
    "[": Tile.PIPE_L,
    "]": Tile.PIPE_R,
    "F": Tile.FLAGPOLE,
    # Markers: they leave an empty tile behind.
    "S": Tile.EMPTY,
    "g": Tile.EMPTY,
    "k": Tile.EMPTY,
}
SPAWN_KINDS: dict[str, str] = {"g": "walker", "k": "turtle"}

_METADATA = re.compile(r"^;\s*(\w+)\s*=\s*(\S+)\s*$")
_PIPE_LIPS: dict[int, int] = {
    int(Tile.PIPE_L): int(Tile.PIPE_TL),
    int(Tile.PIPE_R): int(Tile.PIPE_TR),
}


class LevelNotFoundError(FileNotFoundError, ValueError):
    """`load_level` got a name that is neither a bundled level nor an existing file."""


@dataclass(frozen=True)
class Spawn:
    """A dormant enemy: `kind` is "walker" or "turtle"; (col, row) is its feet tile."""

    kind: str
    col: int
    row: int


class Level:
    """Tile grid plus everything the engine needs to start a game on it.

    `tiles[row, col]` holds `Tile` values as int8, row 0 on top. The grid is the
    mutable part of a level (blocks get used, bricks break, coins disappear), so
    each `Game` plays on its own `copy()`.
    """

    def __init__(
        self,
        name: str,
        tiles: np.ndarray,
        player_start: tuple[int, int],
        spawns: list[Spawn],
        time: int,
        flag_col: int,
    ) -> None:
        self.name = name
        self.tiles = tiles
        self.width_tiles = int(tiles.shape[1])
        self.width_px = self.width_tiles * TILE
        self.player_start = player_start
        self.spawns = spawns
        self.time = time
        self.flag_col = flag_col

    @classmethod
    def from_string(cls, text: str, name: str = "custom") -> Level:
        """Parse level text; raises `ValueError` (with line and column) on bad input."""
        time = DEFAULT_LEVEL_TIME
        rows: list[tuple[int, str]] = []  # (1-based line number, text)
        for line_no, raw in enumerate(text.splitlines(), start=1):
            if raw.lstrip().startswith(";"):
                time = _parse_metadata(raw.strip(), line_no, time)
            else:
                rows.append((line_no, raw.rstrip()))
        while rows and not rows[0][1]:
            rows.pop(0)
        while rows and not rows[-1][1]:
            rows.pop()

        if not rows:
            raise ValueError(f"level {name!r} has no tile rows")
        if len(rows) > LEVEL_H_TILES:
            raise ValueError(
                f"level {name!r} has {len(rows)} rows (line {rows[0][0]} to line {rows[-1][0]}); "
                f"the maximum is {LEVEL_H_TILES}"
            )
        width = max(len(row) for _, row in rows)
        if width < VIEW_TILES_W:
            raise ValueError(
                f"level {name!r} is {width} tiles wide; the minimum is {VIEW_TILES_W} (one screen)"
            )

        tiles = np.zeros((LEVEL_H_TILES, width), dtype=np.int8)
        top = LEVEL_H_TILES - len(rows)
        start: tuple[int, int] | None = None
        spawns: list[Spawn] = []
        flag_cols: dict[int, str] = {}
        for offset, (line_no, row_text) in enumerate(rows):
            row = top + offset
            for col, char in enumerate(row_text):
                where = f"line {line_no}, column {col + 1}"
                tile = LEGEND.get(char)
                if tile is None:
                    raise ValueError(f"unknown level character {char!r} at {where}")
                tiles[row, col] = tile
                if char == "S":
                    if start is not None:
                        raise ValueError(f"second player start 'S' at {where}")
                    start = (col, row)
                elif char in SPAWN_KINDS:
                    spawns.append(Spawn(SPAWN_KINDS[char], col, row))
                elif char == "F":
                    flag_cols.setdefault(col, where)

        if start is None:
            raise ValueError(f"level {name!r} has no player start 'S'")
        if not flag_cols:
            raise ValueError(f"level {name!r} has no flagpole 'F'")
        if len(flag_cols) > 1:
            first, second = sorted(flag_cols)[:2]
            raise ValueError(
                f"flagpole 'F' tiles must share one column; found column {first + 1} and "
                f"column {second + 1} ({flag_cols[second]})"
            )
        flag_col = next(iter(flag_cols))

        _detect_top_tiles(tiles, flag_col)
        return cls(name, tiles, start, spawns, time, flag_col)

    def copy(self) -> Level:
        """An independent copy (own tile grid, own spawn list)."""
        return Level(
            self.name,
            self.tiles.copy(),
            self.player_start,
            list(self.spawns),
            self.time,
            self.flag_col,
        )

    def solid_at(self, col: int, row: int) -> bool:
        """Is tile (col, row) solid? Left/right of the level is solid, above/below is empty."""
        if col < 0 or col >= self.width_tiles:
            return True
        if row < 0 or row >= LEVEL_H_TILES:
            return False
        return IS_SOLID[self.tiles[row, col]]

    def __repr__(self) -> str:
        return f"Level(name={self.name!r}, width_tiles={self.width_tiles}, time={self.time})"


def _parse_metadata(comment: str, line_no: int, time: int) -> int:
    """Return the level time after reading one `;` line (free-text comments change nothing)."""
    match = _METADATA.match(comment)
    if match is None:
        return time
    key, value = match.groups()
    if key != "time":
        raise ValueError(f"unknown level metadata key {key!r} on line {line_no} (known: time)")
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"level time must be a positive integer, got {value!r} on line {line_no}")
    return int(value)


def _detect_top_tiles(tiles: np.ndarray, flag_col: int) -> None:
    """Turn the topmost tile of every pipe column into a lip and of the flagpole into its top."""
    for row in range(LEVEL_H_TILES):
        for col in range(tiles.shape[1]):
            body = int(tiles[row, col])
            lip = _PIPE_LIPS.get(body)
            if lip is not None and (row == 0 or int(tiles[row - 1, col]) not in (body, lip)):
                tiles[row, col] = lip
    flag_rows = np.flatnonzero(tiles[:, flag_col] == Tile.FLAGPOLE)
    tiles[flag_rows[0], flag_col] = Tile.FLAG_TOP


def _levels_dir() -> Traversable:
    """The bundled level files: package data, so this also works from a wheel or a zip."""
    return files("mario_play") / "levels"


@cache
def _bundled_names() -> tuple[str, ...]:
    entries = (entry.name for entry in _levels_dir().iterdir())
    return tuple(sorted(name[: -len(".txt")] for name in entries if name.endswith(".txt")))


def list_levels() -> list[str]:
    """Names of the bundled levels, sorted: `["1-1", "1-2", "1-3", "flat"]`."""
    return list(_bundled_names())


@cache
def _load_bundled(name: str) -> Level:
    text = (_levels_dir() / f"{name}.txt").read_text(encoding="utf-8")
    return Level.from_string(text, name=name)


def load_level(name_or_path: str | os.PathLike[str]) -> Level:
    """Load a bundled level by name, or a level file by filesystem path.

    Always returns a fresh copy, so callers may mutate it. An unknown name raises
    `LevelNotFoundError`, which is both a `ValueError` and a `FileNotFoundError`.
    """
    name = os.fspath(name_or_path)
    if name in _bundled_names():
        return _load_bundled(name).copy()
    path = Path(name)
    if path.is_file():
        return Level.from_string(path.read_text(encoding="utf-8"), name=path.stem)
    raise LevelNotFoundError(
        f"unknown level {name!r}: not a file, and the bundled levels are {', '.join(list_levels())}"
    )
