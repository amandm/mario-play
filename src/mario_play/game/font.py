"""A tiny original 5x7 bitmap font drawn straight into RGB numpy frames.

Covers ``A-Z``, ``0-9``, space and a little punctuation (``- : x`` for the HUD, plus the
multiplication sign and a few extras for overlays). Lowercase letters render as uppercase and
anything unknown renders as a space, so callers never have to sanitise their strings.

Numpy only: this module must stay importable without the engine, pygame or Pillow.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

GLYPH_W = 5
GLYPH_H = 7
SPACING = 1  # blank columns between neighbouring glyphs

Color = tuple[int, int, int]

# fmt: off
_GLYPH_ROWS: dict[str, tuple[str, ...]] = {
    " ": (".....", ".....", ".....", ".....", ".....", ".....", "....."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "J": ("..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("####.", "....#", "....#", ".###.", "....#", "....#", "####."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": (".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", "..#..", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."),
    "-": (".....", ".....", ".....", ".###.", ".....", ".....", "....."),
    ":": (".....", "..#..", "..#..", ".....", "..#..", "..#..", "....."),
    "×": (".....", ".....", ".#.#.", "..#..", ".#.#.", ".....", "....."),
    ".": (".....", ".....", ".....", ".....", ".....", "..#..", "..#.."),
    ",": (".....", ".....", ".....", ".....", "..#..", "..#..", ".#..."),
    "!": ("..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."),
    "?": (".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."),
    "/": ("....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    "%": ("##..#", "##..#", "...#.", "..#..", ".#...", "#..##", "#..##"),
    "=": (".....", ".....", "#####", ".....", "#####", ".....", "....."),
    "'": ("..#..", "..#..", ".....", ".....", ".....", ".....", "....."),
    "(": ("...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."),
    ")": (".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."),
    "<": ("...#.", "..#..", ".#...", "#....", ".#...", "..#..", "...#."),
    ">": (".#...", "..#..", "...#.", "....#", "...#.", "..#..", ".#..."),
}
# fmt: on


def _build_glyphs() -> dict[str, np.ndarray]:
    """Turn the string rows into read-only boolean masks, failing loudly on typos."""
    glyphs: dict[str, np.ndarray] = {}
    for char, rows in _GLYPH_ROWS.items():
        if len(rows) != GLYPH_H or any(len(row) != GLYPH_W for row in rows):
            raise ValueError(f"font glyph {char!r} is not {GLYPH_W}x{GLYPH_H}: {rows!r}")
        bad = set("".join(rows)) - {"#", "."}
        if bad:
            raise ValueError(f"font glyph {char!r} uses unknown symbols {sorted(bad)!r}")
        mask = np.array([[cell == "#" for cell in row] for row in rows], dtype=bool)
        mask.flags.writeable = False
        glyphs[char] = mask
    return glyphs


_GLYPHS = _build_glyphs()
_BLANK = _GLYPHS[" "]
_GAP = np.zeros((GLYPH_H, SPACING), dtype=bool)


def supported_chars() -> list[str]:
    """Characters with their own glyph (lowercase letters are accepted too and drawn uppercase)."""
    return sorted(_GLYPHS)


def _glyph(char: str) -> np.ndarray:
    glyph = _GLYPHS.get(char)
    if glyph is None:
        # Per-character upper() keeps one glyph per input character (unlike "ß".upper() == "SS").
        glyph = _GLYPHS.get(char.upper(), _BLANK)
    return glyph


def _check_scale(scale: int) -> int:
    if not isinstance(scale, (int, np.integer)) or isinstance(scale, bool) or scale < 1:
        raise ValueError(f"scale must be a positive integer, got {scale!r}")
    return int(scale)


def text_width(text: str, scale: int = 1) -> int:
    """Width in pixels of ``text`` as drawn by :func:`draw_text` (0 for the empty string)."""
    scale = _check_scale(scale)
    if not text:
        return 0
    return (len(text) * (GLYPH_W + SPACING) - SPACING) * scale


@lru_cache(maxsize=2048)
def _text_mask(text: str, scale: int) -> np.ndarray:
    """Boolean ``(7 * scale, text_width, 1)`` coverage mask of a whole string, cached.

    HUD strings repeat frame after frame, so drawing a cached string is a single masked copy.
    """
    parts: list[np.ndarray] = []
    for char in text:
        if parts:
            parts.append(_GAP)
        parts.append(_glyph(char))
    mask = np.concatenate(parts, axis=1)
    if scale > 1:
        mask = mask.repeat(scale, axis=0).repeat(scale, axis=1)
    mask = np.ascontiguousarray(mask[..., None])
    mask.flags.writeable = False
    return mask


def _paint(dst: np.ndarray, mask: np.ndarray, x: int, y: int, color: np.ndarray) -> None:
    """Set ``dst`` to ``color`` wherever ``mask`` is true, clipped to the bounds of ``dst``."""
    height, width = mask.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, dst.shape[1]), min(y + height, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    np.copyto(dst[y0:y1, x0:x1], color, where=mask[y0 - y : y1 - y, x0 - x : x1 - x])


def draw_text(
    dst: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Color = (255, 255, 255),
    *,
    scale: int = 1,
    shadow: Color | None = None,
) -> None:
    """Draw ``text`` into the RGB frame ``dst`` in place, top-left corner at ``(x, y)``.

    Args:
        dst: ``(H, W, 3)`` uint8 array. Text is clipped at every edge and may lie fully outside.
        text: Single line. Lowercase maps to uppercase; unknown characters render as a space.
        x, y: Pixel position of the first glyph's top-left corner (may be negative; floats floor).
        color: RGB colour of the glyphs.
        scale: Integer magnification (``2`` draws 10x14 glyphs) for overlays and titles.
        shadow: Optional RGB colour of a drop shadow offset by ``scale`` pixels down-right,
            which keeps white HUD text legible over clouds and sky.
    """
    if dst.ndim != 3 or dst.shape[2] != 3 or dst.dtype != np.uint8:
        raise ValueError(f"dst must be an (H, W, 3) uint8 array, got {dst.shape} {dst.dtype}")
    scale = _check_scale(scale)
    if not text:
        return
    left, top = math.floor(x), math.floor(y)
    mask = _text_mask(text, scale)
    if shadow is not None:
        _paint(dst, mask, left + scale, top + scale, np.asarray(shadow, dtype=np.uint8))
    _paint(dst, mask, left, top, np.asarray(color, dtype=np.uint8))
