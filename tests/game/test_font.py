"""5x7 bitmap font: glyph coverage, metrics, colour, and clipping safety."""

import string

import numpy as np
import pytest

from mario_play.game.font import GLYPH_H, GLYPH_W, draw_text, supported_chars, text_width

REQUIRED = string.ascii_uppercase + string.digits + "-x: "
BG = 9  # canvas fill value; no test colour uses it


def _canvas(h=16, w=64):
    return np.full((h, w, 3), BG, dtype=np.uint8)


def _ink(canvas):
    """Boolean mask of pixels that draw_text touched."""
    return (canvas != BG).any(axis=2)


def _render(text, h=16, w=64, x=0, y=0, color=(255, 255, 255), **kwargs):
    canvas = _canvas(h, w)
    draw_text(canvas, text, x, y, color, **kwargs)
    return canvas


def _glyph(char):
    return _ink(_render(char, h=GLYPH_H, w=GLYPH_W))


# ---------------------------------------------------------------- metrics


def test_glyph_cell_is_5_by_7():
    assert (GLYPH_W, GLYPH_H) == (5, 7)


@pytest.mark.parametrize(
    ("text", "width"),
    [("", 0), ("A", 5), ("AB", 11), ("SCORE 000100", 71), (" ", 5), ("a~", 11)],
)
def test_text_width_is_five_per_glyph_plus_one_pixel_gaps(text, width):
    assert text_width(text) == width


def test_text_width_matches_the_pixels_actually_drawn():
    text = "MW-10"
    ink = _ink(_render(text))
    columns = np.flatnonzero(ink.any(axis=0))
    assert columns[0] == 0 and columns[-1] == text_width(text) - 1
    rows = np.flatnonzero(ink.any(axis=1))
    assert rows[0] == 0 and rows[-1] == GLYPH_H - 1


# ---------------------------------------------------------------- glyphs


def test_required_characters_are_supported():
    assert set(REQUIRED.upper()) <= set(supported_chars())
    assert "×" in supported_chars()  # the spec's multiplication sign for "x 03" coin counts
    for char in REQUIRED.replace(" ", ""):
        assert _glyph(char).any(), char


@pytest.mark.parametrize(
    ("char", "pixels"),
    [("-", 3), (":", 4), ("T", 11), ("L", 11), ("I", 11), ("1", 10), (" ", 0)],
)
def test_known_glyphs_write_the_expected_pixel_count(char, pixels):
    assert _glyph(char).sum() == pixels


def test_known_glyph_shape():
    expected = np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )
    assert np.array_equal(_glyph("T"), expected)


def test_every_visible_glyph_is_distinct_and_fits_its_cell():
    glyphs = {}
    for char in supported_chars():
        canvas = _render(char, h=GLYPH_H + 4, w=GLYPH_W + 4, x=2, y=2)
        ink = _ink(canvas)
        assert not ink[:2].any() and not ink[-2:].any(), char
        assert not ink[:, :2].any() and not ink[:, -2:].any(), char
        if char != " ":
            assert ink.any(), char
            glyphs[char] = ink.tobytes()
    assert len(set(glyphs.values())) == len(glyphs), "two characters share a bitmap (O vs 0?)"


def test_letters_and_digits_use_the_full_cell_height():
    for char in string.ascii_uppercase + string.digits:
        rows = np.flatnonzero(_glyph(char).any(axis=1))
        assert rows[0] == 0 and rows[-1] == GLYPH_H - 1, char


def test_lowercase_renders_as_uppercase():
    assert np.array_equal(_render("game over x3"), _render("GAME OVER X3"))


def test_unknown_characters_render_as_a_space():
    assert np.array_equal(_render("A~B@C"), _render("A B C"))
    assert not _ink(_render("~@#")).any()


def test_multiplication_sign_is_smaller_than_the_letter_x():
    times, letter = _glyph("×"), _glyph("X")
    assert 0 < times.sum() < letter.sum()


# ---------------------------------------------------------------- colour


def test_text_uses_the_requested_colour_and_touches_nothing_else():
    canvas = _render("HI 5", color=(12, 200, 99))
    ink = _ink(canvas)
    assert ink.any()
    assert (canvas[ink] == np.array([12, 200, 99], dtype=np.uint8)).all()
    assert (canvas[~ink] == BG).all()


def test_default_colour_is_white_and_nothing_is_returned():
    canvas = _canvas()
    assert draw_text(canvas, "OK", 1, 1) is None
    assert (canvas[_ink(canvas)] == 255).all()


def test_position_offsets_the_text():
    at_origin = _ink(_render("A9", h=20, w=40))
    moved = _ink(_render("A9", h=20, w=40, x=7, y=4))
    assert np.array_equal(moved[4 : 4 + GLYPH_H, 7 : 7 + 11], at_origin[:GLYPH_H, :11])
    assert moved.sum() == at_origin.sum()


def test_scale_enlarges_glyphs_and_metrics():
    assert text_width("AB", scale=2) == 22
    small = _ink(_render("AB", h=14, w=22))
    big = _ink(_render("AB", h=14, w=22, scale=2))
    assert np.array_equal(big, np.kron(small[:7, :11], np.ones((2, 2), dtype=bool)))
    with pytest.raises(ValueError):
        draw_text(_canvas(), "A", 0, 0, scale=0)


def test_shadow_draws_a_second_colour_one_pixel_down_right():
    plain = _ink(_render("T", h=10, w=10, x=1, y=1))
    canvas = _render("T", h=10, w=10, x=1, y=1, color=(255, 255, 255), shadow=(0, 0, 0))
    white = (canvas == 255).all(axis=2)
    black = (canvas == 0).all(axis=2)
    assert np.array_equal(white, plain), "the shadow must never cover the text itself"
    assert black.any()
    assert np.array_equal(black, np.roll(plain, (1, 1), axis=(0, 1)) & ~plain)


# ---------------------------------------------------------------- clipping


@pytest.mark.parametrize("scale", [1, 2])
def test_clipped_text_equals_a_crop_of_unclipped_text(scale):
    text = "TIME 399"
    pad = 80
    for x, y in [(-3, 2), (-17, -4), (30, -6), (44, 5), (20, 9), (-200, 3), (3, -200), (70, 3)]:
        big = _canvas(12 + 2 * pad, 48 + 2 * pad + 200)
        draw_text(big, text, x + pad + 200, y + pad, (250, 1, 2), scale=scale, shadow=(1, 2, 3))
        small = _canvas(12, 48)
        draw_text(small, text, x, y, (250, 1, 2), scale=scale, shadow=(1, 2, 3))
        assert np.array_equal(small, big[pad : pad + 12, pad + 200 : pad + 200 + 48]), (x, y)


@pytest.mark.parametrize(("x", "y"), [(-500, 0), (500, 0), (0, -50), (0, 50), (64, 16), (-71, -7)])
def test_fully_off_screen_text_is_a_no_op(x, y):
    canvas = _canvas()
    draw_text(canvas, "SCORE 000100", x, y, shadow=(0, 0, 0))
    assert not _ink(canvas).any()


def test_text_never_writes_outside_a_dst_view():
    guard = np.full((40, 60, 3), BG, dtype=np.uint8)
    window = guard[10:22, 20:40]
    for y in range(-9, 14, 2):
        for x in range(-40, 24, 5):
            draw_text(window, "WORLD 1-1", x, y, (200, 100, 50), shadow=(0, 0, 0))
    outside = np.ones(guard.shape[:2], dtype=bool)
    outside[10:22, 20:40] = False
    assert (guard[outside] == BG).all()
    assert _ink(window).any()


def test_empty_text_and_numpy_coordinates_are_fine():
    canvas = _canvas()
    draw_text(canvas, "", 3, 3)
    assert not _ink(canvas).any()
    draw_text(canvas, "7", np.int64(2), np.float64(1.0))
    assert _ink(canvas)[1:8, 2:7].any()


def test_draws_on_a_full_frame_in_place():
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    draw_text(frame, "SCORE 001200  COINS ×07  TIME 388  1-1", 8, 8, (255, 255, 255))
    assert frame.shape == (240, 256, 3) and frame.dtype == np.uint8
    assert frame[8:15].any() and not frame[:8].any() and not frame[16:].any()


@pytest.mark.parametrize(
    "dst",
    [np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.float32)],
)
def test_malformed_destination_is_rejected(dst):
    with pytest.raises(ValueError):
        draw_text(dst, "A", 0, 0)
