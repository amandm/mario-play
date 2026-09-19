"""Level parsing, bundled level loading and the design rules every bundled level must obey."""

from __future__ import annotations

import numpy as np
import pytest

from mario_play.game.constants import DEFAULT_LEVEL_TIME, LEVEL_H_TILES, TILE
from mario_play.game.level import Level, Spawn, list_levels, load_level
from mario_play.game.tiles import IS_SOLID, SOLID, Tile

LEGEND = """\
; every legend character appears once
; time=300
 . o
..?MBX..........
.........F......
.........F......
..[].....F......
S.[]g.k..F......
################
"""

BUNDLED = ["1-1", "1-2", "1-3", "flat"]
GROUND_ROWS = (13, 14)


def minimal(rows: list[str] | None = None, header: str = "") -> str:
    """A tiny valid level (16 wide) whose rows can be overridden by the caller."""
    body = rows if rows is not None else ["S..........F....", "################"]
    return header + "\n".join(body) + "\n"


# --- parsing ---------------------------------------------------------------------------


def test_parses_every_legend_char_and_pads_to_15_rows() -> None:
    level = Level.from_string(LEGEND, name="legend")

    assert level.name == "legend"
    assert level.tiles.shape == (LEVEL_H_TILES, 16)
    assert level.tiles.dtype == np.int8
    assert level.width_tiles == 16
    assert level.width_px == 16 * TILE
    # 7 text rows -> 8 empty rows of padding on top.
    assert not level.tiles[:8].any()

    t = level.tiles
    assert t[8, 0] == Tile.EMPTY and t[8, 1] == Tile.EMPTY  # ' ' and '.'
    assert t[8, 3] == Tile.COIN
    assert t[9, 2] == Tile.QUESTION_COIN
    assert t[9, 3] == Tile.QUESTION_MUSHROOM
    assert t[9, 4] == Tile.BRICK
    assert t[9, 5] == Tile.HARD
    assert (t[14] == Tile.GROUND).all()


def test_player_start_and_spawns_leave_empty_tiles() -> None:
    level = Level.from_string(LEGEND)

    assert level.player_start == (0, 13)
    assert level.spawns == [Spawn("walker", 4, 13), Spawn("turtle", 6, 13)]
    for col in (0, 4, 6):
        assert level.tiles[13, col] == Tile.EMPTY


def test_pipe_top_lip_is_detected() -> None:
    t = Level.from_string(LEGEND).tiles

    assert (t[12, 2], t[12, 3]) == (Tile.PIPE_TL, Tile.PIPE_TR)
    assert (t[13, 2], t[13, 3]) == (Tile.PIPE_L, Tile.PIPE_R)


def test_topmost_flag_tile_becomes_flag_top() -> None:
    level = Level.from_string(LEGEND)

    assert level.flag_col == 9
    assert level.tiles[10, 9] == Tile.FLAG_TOP
    assert [level.tiles[r, 9] for r in (11, 12, 13)] == [Tile.FLAGPOLE] * 3


def test_time_metadata_and_default() -> None:
    assert Level.from_string(LEGEND).time == 300
    assert Level.from_string(minimal()).time == DEFAULT_LEVEL_TIME


def test_rows_are_right_padded_to_the_longest_row() -> None:
    level = Level.from_string(minimal(["S", "...........F", "################"]))

    assert level.tiles.shape == (LEVEL_H_TILES, 16)
    assert level.player_start == (0, 12)
    assert level.tiles[13, 11] == Tile.FLAG_TOP


def test_blank_lines_around_the_rows_and_crlf_are_ignored() -> None:
    text = "\r\n\r\n" + minimal().replace("\n", "\r\n") + "\r\n\r\n"

    level = Level.from_string(text)

    assert level.player_start == (0, 13)
    assert (level.tiles[14] == Tile.GROUND).all()


def test_exactly_15_rows_is_accepted() -> None:
    rows = ["." * 16] * 13 + ["S..........F....", "################"]

    level = Level.from_string(minimal(rows))

    assert level.player_start == (0, 13)


def test_an_all_space_bottom_row_is_a_row_of_empty_tiles() -> None:
    # Regression: the space row used to be dropped as "blank", which made this a 14-row level
    # that was padded on top, so everything silently ended up one row lower than written.
    rows = ["." * 16] * 12 + ["S..........F....", "################", " " * 16]

    level = Level.from_string(minimal(rows))

    assert level.player_start == (0, 12)
    assert (level.tiles[13] == Tile.GROUND).all()
    assert (level.tiles[14] == Tile.EMPTY).all()


@pytest.mark.parametrize("spaces", [" ", " " * 16])
@pytest.mark.parametrize("where", ["top", "bottom"])
def test_a_row_of_spaces_parses_like_a_row_of_dots(where: str, spaces: str) -> None:
    body = ["S..........F....", "################"]
    with_spaces = [spaces, *body] if where == "top" else [*body, spaces]
    with_dots = ["." * 16, *body] if where == "top" else [*body, "." * 16]

    level = Level.from_string(minimal(with_spaces))
    expected = Level.from_string(minimal(with_dots))

    assert level.player_start == expected.player_start
    assert np.array_equal(level.tiles, expected.tiles)


# --- parsing errors --------------------------------------------------------------------


def test_missing_start_raises() -> None:
    with pytest.raises(ValueError, match="'S'"):
        Level.from_string(minimal(["...........F....", "################"]))


def test_two_starts_raise_with_position() -> None:
    with pytest.raises(ValueError, match=r"line 1, column 3"):
        Level.from_string(minimal(["S.S........F....", "################"]))


def test_missing_flag_raises() -> None:
    with pytest.raises(ValueError, match="'F'"):
        Level.from_string(minimal(["S...............", "################"]))


def test_flag_in_two_columns_raises() -> None:
    with pytest.raises(ValueError, match="column"):
        Level.from_string(minimal(["S.......F..F....", "################"]))


def test_unknown_char_reports_line_and_column() -> None:
    text = minimal(["S...............", "....Q......F....", "################"], header="; note\n")

    with pytest.raises(ValueError, match=r"'Q'.*line 3, column 5"):
        Level.from_string(text)


def test_more_than_15_rows_raises() -> None:
    rows = ["." * 16] * 14 + ["S..........F....", "################"]

    with pytest.raises(ValueError, match="16 rows"):
        Level.from_string(minimal(rows))


@pytest.mark.parametrize("where", ["top", "bottom"])
def test_a_row_of_spaces_counts_towards_the_15_row_limit(where: str) -> None:
    rows = ["." * 16] * 13 + ["S..........F....", "################"]
    rows = [" " * 16, *rows] if where == "top" else [*rows, " " * 16]

    with pytest.raises(ValueError, match="16 rows"):
        Level.from_string(minimal(rows))


def test_width_below_16_raises() -> None:
    with pytest.raises(ValueError, match="16"):
        Level.from_string(minimal(["S.........F", "###########"]))


def test_empty_text_raises() -> None:
    with pytest.raises(ValueError):
        Level.from_string("; only a comment\n")


@pytest.mark.parametrize("value", ["abc", "0", "-5", "1.5"])
def test_bad_time_metadata_raises(value: str) -> None:
    with pytest.raises(ValueError, match="time"):
        Level.from_string(minimal(header=f"; time={value}\n"))


def test_unknown_metadata_key_raises_but_free_text_comments_do_not() -> None:
    assert Level.from_string(minimal(header="; pits are <= 3 wide, a=b style prose\n")).time == 400
    with pytest.raises(ValueError, match="tmie"):
        Level.from_string(minimal(header="; tmie=300\n"))


# --- Level API -------------------------------------------------------------------------


def test_copy_is_independent() -> None:
    level = Level.from_string(LEGEND, name="legend")

    dup = level.copy()
    dup.tiles[9, 2] = Tile.USED
    dup.spawns.clear()

    assert dup.name == "legend" and dup.time == 300 and dup.flag_col == 9
    assert dup.player_start == level.player_start
    assert level.tiles[9, 2] == Tile.QUESTION_COIN
    assert len(level.spawns) == 2


def test_solid_at_in_and_out_of_bounds() -> None:
    level = Level.from_string(LEGEND)

    assert level.solid_at(0, 14) is True
    assert level.solid_at(0, 13) is False
    assert level.solid_at(3, 8) is False  # a coin is not solid
    assert level.solid_at(9, 12) is False  # neither is the flagpole
    assert level.solid_at(2, 12) is True  # pipes are
    # Left and right of the level are walls; above and below are open.
    assert level.solid_at(-1, 5) is True
    assert level.solid_at(16, 5) is True
    assert level.solid_at(-1, -1) is True
    assert level.solid_at(5, -1) is False
    assert level.solid_at(5, 15) is False


def test_solid_set_matches_lookup_table() -> None:
    assert {int(t) for t in Tile if IS_SOLID[t]} == set(SOLID)
    assert Tile.COIN not in SOLID and Tile.FLAGPOLE not in SOLID and Tile.EMPTY not in SOLID
    assert len(SOLID) == 10


# --- bundled levels --------------------------------------------------------------------


def test_list_levels_returns_the_four_bundled_names() -> None:
    assert list_levels() == BUNDLED


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_level_loads(name: str) -> None:
    level = load_level(name)

    assert level.name == name
    assert level.tiles.shape == (LEVEL_H_TILES, level.width_tiles)
    assert level.width_tiles >= 16
    assert level.time > 0


def test_load_level_returns_independent_copies() -> None:
    first = load_level("flat")
    first.tiles[:] = Tile.EMPTY

    assert load_level("flat").tiles.any()


def test_load_level_from_a_filesystem_path(tmp_path) -> None:
    path = tmp_path / "my-level.txt"
    path.write_text(LEGEND)

    level = load_level(str(path))

    assert level.name == "my-level"
    assert level.time == 300


def test_load_level_unknown_name_raises_a_helpful_error() -> None:
    with pytest.raises(ValueError, match="1-1"):
        load_level("9-9")
    with pytest.raises(FileNotFoundError):
        load_level("9-9")


# --- design rules of the bundled levels ------------------------------------------------


def _pit_runs(level: Level) -> list[int]:
    """Widths of the maximal runs of columns that have no floor in the ground rows."""
    runs, current = [], 0
    for col in range(level.width_tiles):
        if not any(IS_SOLID[level.tiles[r, col]] for r in GROUND_ROWS):
            current += 1
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


@pytest.mark.parametrize("name", BUNDLED)
def test_level_file_starts_with_a_comment_header(name: str) -> None:
    from importlib.resources import files

    text = (files("mario_play") / "levels" / f"{name}.txt").read_text()

    assert text.startswith(";")
    assert len(text.splitlines()) > LEVEL_H_TILES  # header lines + 15 rows


@pytest.mark.parametrize(
    ("name", "min_width", "max_width", "max_pit"),
    [("flat", 50, 70, 0), ("1-1", 180, 220, 3), ("1-2", 180, 260, 4), ("1-3", 180, 260, 4)],
)
def test_level_size_and_pit_widths(name: str, min_width: int, max_width: int, max_pit: int) -> None:
    level = load_level(name)

    assert min_width <= level.width_tiles <= max_width
    runs = _pit_runs(level)
    assert max(runs, default=0) <= max_pit
    if name == "flat":
        assert not level.spawns
    else:
        assert runs, "the numbered levels are supposed to have pits"
        assert level.spawns


@pytest.mark.parametrize("name", BUNDLED)
def test_ground_rows_are_ground_or_pit_and_pits_go_all_the_way_down(name: str) -> None:
    level = load_level(name)

    for col in range(level.width_tiles):
        cells = [int(level.tiles[r, col]) for r in GROUND_ROWS]
        assert cells in ([Tile.GROUND, Tile.GROUND], [Tile.EMPTY, Tile.EMPTY]), (name, col)


@pytest.mark.parametrize("name", BUNDLED)
def test_start_is_on_the_ground_near_the_left(name: str) -> None:
    level = load_level(name)
    col, row = level.player_start

    assert col <= 5
    assert row == 12
    assert level.tiles[13, col] == Tile.GROUND


@pytest.mark.parametrize("name", BUNDLED)
def test_flag_stands_on_the_ground_with_room_after_it(name: str) -> None:
    level = load_level(name)
    col = level.flag_col

    assert level.tiles[12, col] == Tile.FLAGPOLE
    assert level.tiles[13, col] == Tile.GROUND
    assert level.width_tiles - 1 - col >= 8
    assert (level.tiles[13, col:] == Tile.GROUND).all()
    # Tall enough that it cannot be jumped over from the staircase in front of it.
    pole_rows = [r for r in range(LEVEL_H_TILES) if level.tiles[r, col] != Tile.EMPTY]
    assert level.tiles[pole_rows[0], col] == Tile.FLAG_TOP
    assert pole_rows[0] <= 3


@pytest.mark.parametrize("name", BUNDLED)
def test_the_rows_under_the_hud_are_empty(name: str) -> None:
    # The HUD is written over the top 28 px = tile rows 0 and 1. Regression: the flag top of
    # 1-3 stood in row 1, where its ball ended up behind the digits of the HUD values.
    level = load_level(name)

    assert (level.tiles[:2] == Tile.EMPTY).all()


@pytest.mark.parametrize("name", ["1-1", "1-2", "1-3"])
def test_hard_block_staircase_before_the_flag(name: str) -> None:
    level = load_level(name)
    flag = level.flag_col

    heights = []
    for col in range(flag - 16, flag):
        column = level.tiles[:13, col]
        heights.append(int((column == Tile.HARD).sum()))
    assert max(heights) >= 4
    peak = heights.index(max(heights))
    climb = heights[: peak + 1]
    rising = [h for h in climb if h > 0]
    assert rising == sorted(rising), "the staircase climbs towards the flag"
    steps = [b - a for a, b in zip([0, *rising[:-1]], rising, strict=True)]
    assert max(steps) <= 1, "steps are one tile high"


@pytest.mark.parametrize("name", BUNDLED)
def test_enemies_spawn_on_flat_solid_ground(name: str) -> None:
    level = load_level(name)

    for spawn in level.spawns:
        assert spawn.kind in ("walker", "turtle")
        assert level.tiles[spawn.row, spawn.col] == Tile.EMPTY
        for col in (spawn.col - 1, spawn.col, spawn.col + 1):
            assert level.solid_at(col, spawn.row + 1), (name, spawn)
            assert not level.solid_at(col, spawn.row), (name, spawn)


def test_only_the_harder_levels_have_turtles() -> None:
    assert {s.kind for s in load_level("1-1").spawns} == {"walker"}
    for name in ("1-2", "1-3"):
        assert {s.kind for s in load_level(name).spawns} == {"walker", "turtle"}
    assert len(load_level("1-2").spawns) > len(load_level("1-1").spawns)


@pytest.mark.parametrize("name", BUNDLED)
def test_pipes_are_two_wide_two_to_four_tall_and_stand_on_the_ground(name: str) -> None:
    level = load_level(name)
    t = level.tiles

    for row in range(LEVEL_H_TILES):
        for col in range(level.width_tiles):
            if t[row, col] == Tile.PIPE_TL:
                assert t[row, col + 1] == Tile.PIPE_TR
                height, r = 1, row + 1
                while t[r, col] == Tile.PIPE_L:
                    assert t[r, col + 1] == Tile.PIPE_R
                    height, r = height + 1, r + 1
                assert 2 <= height <= 4, (name, col, height)
                assert r == 13 and t[13, col] == Tile.GROUND and t[13, col + 1] == Tile.GROUND
            if t[row, col] in (Tile.PIPE_L, Tile.PIPE_R, Tile.PIPE_TR):
                # Every pipe tile belongs to a pipe that has a lip on top.
                top = row
                while t[top, col] in (Tile.PIPE_L, Tile.PIPE_R):
                    top -= 1
                assert t[top, col] in (Tile.PIPE_TL, Tile.PIPE_TR)


@pytest.mark.parametrize("name", BUNDLED)
def test_no_one_tile_high_corridors(name: str) -> None:
    level = load_level(name)

    for col in range(level.width_tiles):
        for row in range(1, LEVEL_H_TILES - 1):
            if not level.solid_at(col, row):
                squeezed = level.solid_at(col, row - 1) and level.solid_at(col, row + 1)
                assert not squeezed, (name, col, row)


@pytest.mark.parametrize("name", BUNDLED)
def test_bumpable_blocks_hang_exactly_three_tiles_above_their_floor(name: str) -> None:
    level = load_level(name)
    bumpable = (Tile.BRICK, Tile.QUESTION_COIN, Tile.QUESTION_MUSHROOM)
    found = 0

    for col in range(level.width_tiles):
        for row in range(LEVEL_H_TILES):
            if level.tiles[row, col] in bumpable:
                found += 1
                below = [level.solid_at(col, row + k) for k in (1, 2, 3, 4)]
                assert below == [False, False, False, True], (name, col, row)
    assert found > 0
    assert (level.tiles == Tile.QUESTION_MUSHROOM).sum() >= 1
