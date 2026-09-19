"""Original 8-bit style pixel art and the alpha-aware blitter that draws it.

Every sprite is authored below as rows of palette characters (``.`` is transparent) with a
small per-sprite palette, and is compiled once at import time into a cached, read-only
``(h, w, 4)`` uint8 RGBA array. Row lengths, sizes and palette characters are validated during
that compile step, so a typo in the art fails loudly on import instead of drawing garbage.

All characters face right; pass ``flip=True`` to :func:`blit` to face them left. Characters
stand on the bottom row of their sprite, so anchor them bottom-centre on the hitbox.

The cast (all original designs): the player is a little explorer bundled up in a teal hood
with an orange bobble and a trailing orange scarf, the walker is a grumpy purple blob with a
sprig on its head and orange feet, and the turtle is a tortoise under a tall ringed green dome.

Two notes for whoever draws with these. ``flag`` is a pennant whose hoist is its left edge
and the pole occupies columns 7-8 of a ``flagpole`` tile, so blit the flag at ``pole_x + 9``
(or flipped at ``pole_x - 9``). ``ground_top`` is an optional grass-capped variant of
``ground`` meant for ground tiles with nothing solid above them; ``ground`` alone also works.

Numpy only: this module must stay importable without the engine, pygame or Pillow.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

RGBA = tuple[int, int, int, int]

SKY_COLOR: tuple[int, int, int] = (104, 184, 240)

_TRANSPARENT = "."


def _rgb(red: int, green: int, blue: int) -> RGBA:
    return (red, green, blue, 255)


# ------------------------------------------------------------------------------ colours

_INK = _rgb(30, 24, 48)
_WHITE = _rgb(252, 250, 240)
_CREAM = _rgb(252, 232, 176)

_TEAL = _rgb(24, 164, 152)
_TEAL_DARK = _rgb(12, 96, 108)
_SKIN = _rgb(255, 212, 164)
_ORANGE = _rgb(248, 136, 32)

_PURPLE = _rgb(152, 88, 214)
_PURPLE_DARK = _rgb(88, 44, 148)
_FOOT = _rgb(248, 168, 56)

_SHELL = _rgb(64, 180, 84)
_SHELL_DARK = _rgb(20, 104, 64)

_SOIL = _rgb(188, 112, 60)
_SOIL_DARK = _rgb(112, 60, 40)
_SOIL_LIGHT = _rgb(232, 168, 100)
_GRASS = _rgb(96, 200, 80)
_GRASS_DARK = _rgb(40, 136, 64)

_STONE = _rgb(168, 176, 196)
_STONE_DARK = _rgb(84, 92, 120)
_STONE_LIGHT = _rgb(228, 232, 244)

_BRICK = _rgb(204, 92, 56)
_BRICK_DARK = _rgb(104, 40, 36)
_BRICK_LIGHT = _rgb(244, 156, 104)

_GOLD = _rgb(252, 196, 48)
_GOLD_DARK = _rgb(196, 116, 20)
_GOLD_LIGHT = _rgb(255, 240, 168)

_DULL = _rgb(156, 124, 108)
_DULL_DARK = _rgb(88, 64, 60)
_DULL_LIGHT = _rgb(200, 172, 148)

_PIPE = _rgb(72, 184, 80)
_PIPE_DARK = _rgb(20, 108, 64)
_PIPE_LIGHT = _rgb(176, 240, 128)

_POLE = _rgb(224, 228, 236)
_POLE_DARK = _rgb(124, 132, 156)

_CLOUD_SHADE = _rgb(196, 224, 248)
_BUSH = _rgb(56, 164, 76)
_BUSH_DARK = _rgb(24, 104, 60)
_BUSH_LIGHT = _rgb(140, 220, 100)
_HILL = _rgb(112, 204, 164)
_HILL_DARK = _rgb(76, 172, 140)

# ------------------------------------------------------------------------------ registry


class _Prepared(NamedTuple):
    """Blit-ready views of a registered sprite, indexed by ``flip`` (0 or 1)."""

    sprite: np.ndarray
    rgb: tuple[np.ndarray, np.ndarray]
    mask: tuple[np.ndarray | None, np.ndarray | None]  # None: fully opaque, plain copy


_SPRITES: dict[str, np.ndarray] = {}
# Keyed by id(): safe because each entry keeps its (immortal, read-only) sprite alive and
# blit() double-checks identity before trusting the entry.
_PREPARED: dict[int, _Prepared] = {}


def _compile(
    name: str, size: tuple[int, int] | None, palette: dict[str, RGBA], rows: list[str]
) -> np.ndarray:
    """Validate one piece of art and turn it into a read-only RGBA array."""
    if not rows or not rows[0]:
        raise ValueError(f"sprite {name!r} has no pixels")
    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"sprite {name!r} row {index} is {len(row)} px wide, expected {width}: {row!r}"
            )
    if size is not None and size != (width, len(rows)):
        raise ValueError(f"sprite {name!r} is {width}x{len(rows)}, expected {size[0]}x{size[1]}")
    for char, colour in palette.items():
        if len(char) != 1 or not char.isascii() or char == _TRANSPARENT:
            raise ValueError(f"sprite {name!r} has an invalid palette key {char!r}")
        if len(colour) != 4 or colour[3] != 255 or not all(0 <= c <= 255 for c in colour):
            raise ValueError(f"sprite {name!r} palette entry {char!r} is not opaque RGBA: {colour}")
    used = set("".join(rows)) - {_TRANSPARENT}
    if used - set(palette):
        raise ValueError(f"sprite {name!r} uses undefined symbols {sorted(used - set(palette))}")
    if set(palette) - used:
        raise ValueError(f"sprite {name!r} never uses symbols {sorted(set(palette) - used)}")

    lut = np.zeros((128, 4), dtype=np.uint8)
    # Transparent texels carry the sky colour so even a naive RGB copy looks sane.
    lut[ord(_TRANSPARENT)] = (*SKY_COLOR, 0)
    for char, colour in palette.items():
        lut[ord(char)] = colour
    indices = np.frombuffer("".join(rows).encode("ascii"), dtype=np.uint8)
    sprite = lut[indices.reshape(len(rows), width)]
    sprite.flags.writeable = False
    return sprite


def _prepare(sprite: np.ndarray) -> _Prepared:
    rgb, mask = [], []
    for source in (sprite, sprite[:, ::-1]):
        pixels = np.ascontiguousarray(source[..., :3])
        opaque = np.ascontiguousarray(source[..., 3:] != 0)
        pixels.flags.writeable = False
        opaque.flags.writeable = False
        rgb.append(pixels)
        mask.append(None if opaque.all() else opaque)
    return _Prepared(sprite, (rgb[0], rgb[1]), (mask[0], mask[1]))


def _sprite(
    name: str, size: tuple[int, int] | None, palette: dict[str, RGBA], rows: list[str]
) -> None:
    """Register one sprite; ``size`` is ``(width, height)`` or ``None`` for free-size art."""
    if name in _SPRITES:
        raise ValueError(f"sprite {name!r} is defined twice")
    sprite = _compile(name, size, palette, rows)
    _SPRITES[name] = sprite
    _PREPARED[id(sprite)] = _prepare(sprite)


def get_sprite(name: str) -> np.ndarray:
    """Return the cached, read-only ``(h, w, 4)`` uint8 RGBA array of a sprite.

    Raises:
        KeyError: if ``name`` is not one of :func:`sprite_names`.
    """
    try:
        return _SPRITES[name]
    except KeyError:
        raise KeyError(f"unknown sprite {name!r}; known: {', '.join(sorted(_SPRITES))}") from None


def sprite_names() -> list[str]:
    """Sorted names of every available sprite."""
    return sorted(_SPRITES)


def blit(dst: np.ndarray, sprite: np.ndarray, x: int, y: int, flip: bool = False) -> None:
    """Draw ``sprite`` into ``dst`` in place with its top-left corner at ``(x, y)``.

    Args:
        dst: ``(H, W, 3)`` uint8 RGB frame.
        sprite: ``(h, w, 4)`` uint8 RGBA array. Alpha is binary: 0 leaves ``dst`` untouched,
            anything else overwrites it.
        x, y: Destination of the sprite's top-left pixel. May be negative or beyond the frame;
            the sprite is clipped at all four edges. Floats are floored.
        flip: Mirror the sprite horizontally (characters face left).
    """
    if dst.ndim != 3 or dst.shape[2] != 3 or dst.dtype != np.uint8:
        raise ValueError(f"dst must be an (H, W, 3) uint8 array, got {dst.shape} {dst.dtype}")
    prepared = _PREPARED.get(id(sprite))
    if prepared is not None and prepared.sprite is sprite:
        rgb, mask = prepared.rgb[1 if flip else 0], prepared.mask[1 if flip else 0]
    else:
        if sprite.ndim != 3 or sprite.shape[2] != 4 or sprite.dtype != np.uint8:
            raise ValueError(
                f"sprite must be an (h, w, 4) uint8 array, got {sprite.shape} {sprite.dtype}"
            )
        source = sprite[:, ::-1] if flip else sprite
        rgb, mask = source[..., :3], source[..., 3:] != 0

    left, top = math.floor(x), math.floor(y)
    height, width = rgb.shape[:2]
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + width, dst.shape[1]), min(top + height, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    rows, cols = slice(y0 - top, y1 - top), slice(x0 - left, x1 - left)
    if mask is None:
        dst[y0:y1, x0:x1] = rgb[rows, cols]
    else:
        np.copyto(dst[y0:y1, x0:x1], rgb[rows, cols], where=mask[rows, cols])


# ------------------------------------------------------------------------------ tiles
# fmt: off

_sprite("ground", (16, 16), {"L": _SOIL_LIGHT, "D": _SOIL, "d": _SOIL_DARK}, [
    "LLLLLLLLLLLLLLLd",
    "LDDDDDDDDDDDDDDd",
    "LDDDDDDDdDDDDDDd",
    "LDDLDDDDDDDDLDDd",
    "LDDDDDDDDDDDDDDd",
    "LDDDDDdDDDDDDDDd",
    "LDDDDDDDDDDLDDDd",
    "LDdDDDDDDDDDDDDd",
    "LDDDDDDDLDDDDdDd",
    "LDDDDDDDDDDDDDDd",
    "LDDDLDDDDDDDDDDd",
    "LDDDDDDDDdDDDDDd",
    "LDDDDDDDDDDDLDDd",
    "LDDdDDDDDDDDDDDd",
    "LDDDDDDDDDDDDDDd",
    "dddddddddddddddd",
])

_sprite("ground_top", (16, 16), {"G": _GRASS, "g": _GRASS_DARK, "D": _SOIL, "d": _SOIL_DARK}, [
    "GGGGGGGGGGGGGGGG",
    "GGGGGGGGGGGGGGGG",
    "GGGGGGGGGGGGGGGG",
    "gGGgggGGgGGggGGg",
    "dggdddggdggddggd",
    "DddDDDddDddDDddD",
    "DDDDDDDDDDDDdDDD",
    "DDdDDDDDDDDDDDDD",
    "DDDDDDDDdDDDDDDD",
    "DDDDDDDDDDDDDDdD",
    "DDDDdDDDDDDDDDDD",
    "DDDDDDDDDDdDDDDD",
    "DdDDDDDDDDDDDDDD",
    "DDDDDDDdDDDDDdDD",
    "DDDDDDDDDDDDDDDD",
    "DDDdDDDDDDDdDDDD",
])

_sprite("hard", (16, 16), {"L": _STONE_LIGHT, "S": _STONE, "d": _STONE_DARK}, [
    "LLLLLLLLLLLLLLLd",
    "LLLLLLLLLLLLLLdd",
    "LLSSSSSSSSSSSSdd",
    "LLSdSSSSSSSSdSdd",
    "LLSSSSSSSSSSSSdd",
    "LLSSSSSddSSSSSdd",
    "LLSSSSdSSLSSSSdd",
    "LLSSSdSSSSLSSSdd",
    "LLSSSdSSSSLSSSdd",
    "LLSSSSdSSLSSSSdd",
    "LLSSSSSLLSSSSSdd",
    "LLSSSSSSSSSSSSdd",
    "LLSdSSSSSSSSdSdd",
    "LLSSSSSSSSSSSSdd",
    "Lddddddddddddddd",
    "dddddddddddddddd",
])

_sprite("brick", (16, 16), {"L": _BRICK_LIGHT, "R": _BRICK, "d": _BRICK_DARK}, [
    "LLLLLLLdLLLLLLLd",
    "RRRRRRRdRRRRRRRd",
    "RRRRRRRdRRRRRRRd",
    "dddddddddddddddd",
    "LLLdLLLLLLLdLLLL",
    "RRRdRRRRRRRdRRRR",
    "RRRdRRRRRRRdRRRR",
    "dddddddddddddddd",
    "LLLLLLLdLLLLLLLd",
    "RRRRRRRdRRRRRRRd",
    "RRRRRRRdRRRRRRRd",
    "dddddddddddddddd",
    "LLLdLLLLLLLdLLLL",
    "RRRdRRRRRRRdRRRR",
    "RRRdRRRRRRRdRRRR",
    "dddddddddddddddd",
])

_sprite("question", (16, 16), {"L": _GOLD_LIGHT, "Y": _GOLD, "y": _GOLD_DARK, "K": _INK}, [
    "yLLLLLLLLLLLLLLy",
    "LYYYYYYYYYYYYYYy",
    "LYKYYYYYYYYYYKYy",
    "LYYYYKKKKKKYYYYy",
    "LYYYKKYYYYKKYYYy",
    "LYYYKKYYYYKKYYYy",
    "LYYYYYYYYKKKYYYy",
    "LYYYYYYYKKKYYYYy",
    "LYYYYYYKKKYYYYYy",
    "LYYYYYYKKYYYYYYy",
    "LYYYYYYYYYYYYYYy",
    "LYYYYYYKKYYYYYYy",
    "LYYYYYYKKYYYYYYy",
    "LYKYYYYYYYYYYKYy",
    "LYYYYYYYYYYYYYYy",
    "yyyyyyyyyyyyyyyy",
])

_sprite("used", (16, 16), {"L": _DULL_LIGHT, "U": _DULL, "d": _DULL_DARK}, [
    "dLLLLLLLLLLLLLLd",
    "LUUUUUUUUUUUUUUd",
    "LUdUUUUUUUUUUdUd",
    "LUUUUUUUUUUUUUUd",
    "LUUUddddddddUUUd",
    "LUUUdUUUUUUULUUd",
    "LUUUdUUUUUUULUUd",
    "LUUUdUUUUUUULUUd",
    "LUUUdUUUUUUULUUd",
    "LUUUdUUUUUUULUUd",
    "LUUUdUUUUUUULUUd",
    "LUUUdLLLLLLLLUUd",
    "LUUUUUUUUUUUUUUd",
    "LUdUUUUUUUUUUdUd",
    "LUUUUUUUUUUUUUUd",
    "dddddddddddddddd",
])

_PIPE_PALETTE = {"K": _INK, "L": _PIPE_LIGHT, "G": _PIPE, "g": _PIPE_DARK}

_sprite("pipe_tl", (16, 16), _PIPE_PALETTE, [
    "KKKKKKKKKKKKKKKK",
    "KLLLLLLLLLLLLLLL",
    "KLLGGLGGGGGGGGGG",
    "KLLGGLGGGGGGGGGG",
    "KLLGGLGGGGGGGGGG",
    "KLLGGLGGGGGGGGGG",
    "KLLGGLGGGGGGGGGG",
    "KLLGGLGGGGGGGGGG",
    "KLLGGLGGGGGGGGGG",
    "KLLGGLGGGGGGGGGG",
    "KLLGGLGGGGGGGGGG",
    "KLLGGLGGGGGGGGGG",
    "KggggggggggggggG",
    "KKKKKKKKKKKKKKKK",
    "..KggggggggggggG",
    "..KLLGGLGGGGGGGG",
])

_sprite("pipe_tr", (16, 16), _PIPE_PALETTE, [
    "KKKKKKKKKKKKKKKK",
    "LLLLLLLLLLLLLLLK",
    "GGGGGGGGGgGgGggK",
    "GGGGGGGGGgGgGggK",
    "GGGGGGGGGgGgGggK",
    "GGGGGGGGGgGgGggK",
    "GGGGGGGGGgGgGggK",
    "GGGGGGGGGgGgGggK",
    "GGGGGGGGGgGgGggK",
    "GGGGGGGGGgGgGggK",
    "GGGGGGGGGgGgGggK",
    "GGGGGGGGGgGgGggK",
    "GggggggggggggggK",
    "KKKKKKKKKKKKKKKK",
    "GggggggggggggK..",
    "GGGGGGGgGgGggK..",
])

_sprite("pipe_l", (16, 16), {"K": _INK, "L": _PIPE_LIGHT, "G": _PIPE}, ["..KLLGGLGGGGGGGG"] * 16)

_sprite("pipe_r", (16, 16), {"K": _INK, "G": _PIPE, "g": _PIPE_DARK}, ["GGGGGGGgGgGggK.."] * 16)

_sprite("coin", (16, 16), {"L": _GOLD_LIGHT, "Y": _GOLD, "y": _GOLD_DARK}, [
    "................",
    "......yyyy......",
    "....yyYYYYyy....",
    "...yYYLLYYYYy...",
    "...yYLYYYyYYy...",
    "..yYLYYyyYyYYy..",
    "..yYLYYyYYyYYy..",
    "..yYLYYyYYyYYy..",
    "..yYLYYyYYyYYy..",
    "..yYLYYyYYyYYy..",
    "..yYYYYyyYyYYy..",
    "...yYYYYYyYYy...",
    "...yYYYYYYYYy...",
    "....yyYYYYyy....",
    "......yyyy......",
    "................",
])

_sprite("flagpole", (16, 16), {"P": _POLE, "p": _POLE_DARK}, [".......Pp......."] * 16)

_sprite("flag_top", (16, 16), {"L": _GOLD_LIGHT, "Y": _GOLD, "y": _GOLD_DARK, "P": _POLE, "p": _POLE_DARK}, [
    "................",
    "................",
    "......yyyy......",
    ".....yYLLYy.....",
    "....yYLYYYYy....",
    "....yYLYYYYy....",
    "....yYYYYYYy....",
    "....yYYYYYYy....",
    ".....yYYYYy.....",
    "......yyyy......",
    ".......Pp.......",
    ".......Pp.......",
    ".......Pp.......",
    ".......Pp.......",
    ".......Pp.......",
    ".......Pp.......",
])

_sprite("flag", (16, 16), {"O": _ORANGE, "C": _CREAM, "T": _TEAL_DARK}, [
    "................",
    "TTTTTTTTTTTTTTT.",
    "OOOOOOOOOOOOOO..",
    "OOOCCOOOOOOOO...",
    "OOCCCCOOOOOO....",
    "OCCCCCCOOOO.....",
    "OCCCCCCOOOOO....",
    "OOCCCCOOOOOOO...",
    "OOOCCOOOOOOOOO..",
    "OOOOOOOOOOOOOOO.",
    "TTTTTTTTTTTTTTT.",
    "................",
    "................",
    "................",
    "................",
    "................",
])

# ------------------------------------------------------------------------------ player

_HERO_PALETTE = {"K": _INK, "T": _TEAL, "t": _TEAL_DARK, "S": _SKIN, "O": _ORANGE}

_SMALL_HEAD = [
    "......OO........",
    ".....tOOtt......",
    "...ttTTTTTtt....",
    "..tTTTTTTTTTt...",
    "..tTTTTTTTTTTt..",
    "..tTTTtSSSSSSt..",
    "..tTTtSSKSSKS...",
    "..tTTtSSKSSKS...",
    "...tTtSSSSSSS...",
]

_sprite("player_small_stand", (16, 16), _HERO_PALETTE, [
    *_SMALL_HEAD,
    "..OOOOOOOOOOO...",
    "OOO.tOOOOOOt....",
    "O...tTTTTTTSS...",
    "....tTTTTTTt....",
    "....ttTTTTtt....",
    "....KKK..KKK....",
    "....KKKK.KKKK...",
])

_sprite("player_small_walk1", (16, 16), _HERO_PALETTE, [
    *_SMALL_HEAD,
    "OOOOOOOOOOOOO...",
    "O...tOOOOOOt....",
    "....tTTTTTTtSS..",
    "....tTTTTTTt....",
    "....ttTTTTtt....",
    "...KKK....KKK...",
    "..KKK......KKKK.",
])

_sprite("player_small_walk2", (16, 16), _HERO_PALETTE, [
    *_SMALL_HEAD,
    "..OOOOOOOOOOO...",
    ".OO.tOOOOOOt....",
    "OO..tTTTTTSS....",
    "....tTTTTTTt....",
    "....ttTTTTtt....",
    ".....KKKKK......",
    ".....KKKKKK.....",
])

_sprite("player_small_jump", (16, 16), _HERO_PALETTE, [
    *_SMALL_HEAD[:6],
    "O.tTTtSSKSSKS...",
    "OOtTTtSSKSSKS.SS",
    ".OOtTtSSSSSSS.SS",
    "..OOOOOOOOOOOtt.",
    "....tOOOOOOtt...",
    "SS.tTTTTTTTt....",
    "SSttTTTTTTTt....",
    "....ttTTTTtKKK..",
    "..KKKtt..ttKKK..",
    ".KKKK...........",
])

_BIG_HEAD = [
    "......OOO.......",
    ".....OOOOO......",
    "....ttOOOttt....",
    "...tTTTTTTTTtt..",
    "..tTTTTTTTTTTTt.",
    "..tTTTTTTTTTTTt.",
    "..tTTTTtttttttt.",
    "..tTTTtSSSSSSSt.",
    "..tTTtSSSKSSKSS.",
    "..tTTtSSSKSSKSS.",
    "..tTTtSSSSSSSSS.",
    "..tTTTtSSSSSSS..",
    "...tTTTtSSSSS...",
]

_sprite("player_big_stand", (16, 32), _HERO_PALETTE, [
    *_BIG_HEAD,
    "..OOOOOOOOOOOO..",
    "OOOOOOOOOOOOOO..",
    "OO.OtTTTTTTt....",
    "O...tTTTTTTTt...",
    "...tTTTTTTTTt...",
    "...tTTtTTTTtSS..",
    "...tTTtTTTTtSS..",
    "...tTTTTTTTTt...",
    "...KKKKOOKKKK...",
    "...tTTTTTTTTt...",
    "...tTTTTTTTTt...",
    "....tttttttt....",
    "....ttt..ttt....",
    "....ttt..ttt....",
    "....ttt..ttt....",
    "....KKK..KKK....",
    "....KKK..KKK....",
    "....KKKK.KKKK...",
    "....KKKK.KKKK...",
])

_sprite("player_big_walk1", (16, 32), _HERO_PALETTE, [
    *_BIG_HEAD,
    "OOOOOOOOOOOOOO..",
    "OOOOOOOOOOOOOO..",
    "O..OtTTTTTTt....",
    "....tTTTTTTTt...",
    "...tTTTTTTTTt...",
    "...tTTtTTTTttSS.",
    "...tTTtTTTTt.SS.",
    "...tTTTTTTTTt...",
    "...KKKKOOKKKK...",
    "...tTTTTTTTTt...",
    "...tTTTTTTTTt...",
    "....tttttttt....",
    "...tttt..tttt...",
    "...ttt....ttt...",
    "..ttt......ttt..",
    "..KKK......KKK..",
    ".KKK.......KKK..",
    ".KKKK......KKKK.",
    ".KKKK......KKKK.",
])

_sprite("player_big_walk2", (16, 32), _HERO_PALETTE, [
    *_BIG_HEAD,
    "..OOOOOOOOOOOO..",
    ".OOOOOOOOOOOOO..",
    "OO.OtTTTTTTt....",
    "OO..tTTTTTTTt...",
    "...tTTTTTTTTt...",
    "...tTTtTTTSSt...",
    "...tTTtTTTSSt...",
    "...tTTTTTTTTt...",
    "...KKKKOOKKKK...",
    "...tTTTTTTTTt...",
    "...tTTTTTTTTt...",
    "....tttttttt....",
    ".....tttttt.....",
    ".....tttttt.....",
    ".....ttttt......",
    ".....KKKKK......",
    ".....KKKKK......",
    ".....KKKKKK.....",
    ".....KKKKKK.....",
])

_sprite("player_big_jump", (16, 32), _HERO_PALETTE, [
    *_BIG_HEAD[:7],
    "O.tTTTtSSSSSSSt.",
    "OOtTTtSSSKSSKSS.",
    ".OtTTtSSSKSSKSS.",
    ".OOTTtSSSSSSSSS.",
    "..OTTTtSSSSSSS..",
    "..OtTTTtSSSSS.SS",
    "..OOOOOOOOOOO.SS",
    "...OOOOOOOOOOtt.",
    "....tTTTTTTttt..",
    "....tTTTTTTTt...",
    "...tTTTTTTTTt...",
    "SS.tTTTTTTTTt...",
    "SSttTTTTTTTTt...",
    "...tTTTTTTTTt...",
    "...KKKKOOKKKK...",
    "...tTTTTTTTTt...",
    "...tTTTTTTTTt...",
    "....tttttttt....",
    "...tttt.tttttKK.",
    "...ttt...ttttKK.",
    "..ttt.....tttKK.",
    "..ttt......KK...",
    "..KKK...........",
    ".KKKK...........",
    ".KKKK...........",
])

# ------------------------------------------------------------------------------ enemies

_WALKER_PALETTE = {"K": _INK, "P": _PURPLE, "p": _PURPLE_DARK, "W": _WHITE, "F": _FOOT}

_WALKER_BODY = [
    ".....ppp........",
    ".......p........",
    ".....pppppp.....",
    "...ppPPPPPPpp...",
    "..pPPPPPPPPPPp..",
    ".pPKKPPPPPPKKPp.",
    ".pPPPKKPPKKPPPp.",
    ".pPPWWKPPKWWPPp.",
    ".pPPWWKPPKWWPPp.",
    ".pPPWWWPPWWWPPp.",
    ".pPPPPPPPPPPPPp.",
    ".pPPPPKKKKPPPPp.",
    "..pPPKPPPPKPPp..",
    "...pppppppppp...",
]

_sprite("walker_1", (16, 16), _WALKER_PALETTE, [
    *_WALKER_BODY,
    "..FFFF....FFFF..",
    ".FFFFF....FFFFF.",
])

_sprite("walker_2", (16, 16), _WALKER_PALETTE, [
    *_WALKER_BODY,
    "...FFFF..FFFF...",
    "...FFFF..FFFFF..",
])

_sprite("walker_flat", (16, 16), _WALKER_PALETTE, [
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "....pppppppp....",
    ".pppPPPPPPPPppp.",
    "pPKKKPPPPPPKKKPp",
    "pPPWWKPPPPKWWPPp",
    ".ppppppppppppp..",
    ".FFFFF....FFFFF.",
])

_TURTLE_PALETTE = {"K": _INK, "G": _SHELL, "g": _SHELL_DARK, "C": _CREAM}

_TURTLE_BODY = [
    "................",
    "................",
    "....ggggg.......",
    "..ggGGGGGgg.....",
    ".gGGGGGGGGGg....",
    ".gGGGgggGGGg....",
    "gGGggGGGggGGg...",
    "gGgGGGGGGGgGg...",
    "gGgGGgggGGgGg...",
    "gGgGgGGGgGgGg...",
    "gGgGgGGGgGgGg...",
    "gGgGGgggGGgGg...",
    "gGgGGGGGGGKKKKK.",
    "gGGggGGGggKCCCCK",
    "gGGGGgggGGKCCKCK",
    "gGGGGGGGGGKCCKCK",
    ".gGGGGGGGGKCCCCK",
    "KCCCCCCCCCKCCKK.",
    "KCCCCCCCCCCKK...",
    ".KKKKKKKKKKK....",
]

_sprite("turtle_1", (16, 24), _TURTLE_PALETTE, [
    *_TURTLE_BODY,
    "..KCCK..KCCK....",
    "..KCCK..KCCK....",
    "..KCCCK.KCCCK...",
    "..KKKKK.KKKKK...",
])

_sprite("turtle_2", (16, 24), _TURTLE_PALETTE, [
    *_TURTLE_BODY,
    "...KCCKKCCK.....",
    "...KCCKKCCK.....",
    "...KCCCKCCCK....",
    "...KKKKKKKKK....",
])

_sprite("shell", (16, 16), _TURTLE_PALETTE, [
    "................",
    "................",
    ".....gggggg.....",
    "...ggGGGGGGgg...",
    "..gGGGGggGGGGg..",
    ".gGGGggGGggGGGg.",
    ".gGGgGGGGGGgGGg.",
    "gGGgGGGggGGGgGGg",
    "gGGgGGgGGgGGgGGg",
    "gGGgGGGGGGGGgGGg",
    "gGGGggGGGGggGGGg",
    "gGGGGGggggGGGGGg",
    ".gGGGGGGGGGGGGg.",
    "KCCCCCCCCCCCCCCK",
    "KCCKKCCCCCCKKCCK",
    ".KKKKKKKKKKKKKK.",
])

_sprite("mushroom", (16, 16), {"K": _INK, "O": _ORANGE, "C": _CREAM, "c": _DULL_LIGHT}, [
    "................",
    ".....KKKKKK.....",
    "...KKOOOOOOKK...",
    "..KOOOOOOOOOOK..",
    ".KOOOCOOOOCOOOK.",
    ".KOOCCCOOCCCOOK.",
    "KOOCCOCCCCOCCOOK",
    "KOCCOOOCCOOOCCOK",
    "KOOOOOOOOOOOOOOK",
    "KKOOOOOOOOOOOOKK",
    ".KKKKKKKKKKKKKK.",
    "...KCCCCCCCCK...",
    "...KCCCCCCCcK...",
    "...KCCCCCCCcK...",
    "...KCCCCCCccK...",
    "....KKKKKKKK....",
])

# ------------------------------------------------------------------------------ decorations

_sprite("cloud", None, {"W": _WHITE, "b": _CLOUD_SHADE}, [
    "..............WWWWW.............",
    "...........WWWWWWWWWW...........",
    ".........WWWWWWWWWWWWW..........",
    "....WWWW.WWWWWWWWWWWWWW.........",
    "..WWWWWWWWWWWWWWWWWWWWWW.WWWW...",
    ".WWWWWWWWWWWWWWWWWWWWWWWWWWWWW..",
    ".WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW.",
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    "WWbWWWWWWWWWWWWWWWWWWWWWWWWWWbWW",
    ".WbbWWWWWWWbWWWWWWWWWbWWWWWWbbW.",
    ".WWbbbWWWbbbbWWWWWWbbbbWWWbbbW..",
    "..WWbbbbbbbbbbbWWbbbbbbbbbbbW...",
    "....bbbbbbb.bbbbbbbbbb.bbbbb....",
    "......bbb.....bbbbbb.....b......",
    "................................",
])

_sprite("bush", None, {"L": _BUSH_LIGHT, "B": _BUSH, "b": _BUSH_DARK}, [
    "................................",
    "................................",
    "..............bbbb..............",
    "............bbLLLLbb............",
    "...........bLLLBBBBBb...........",
    "..........bLLBBBBBBBBb..........",
    ".....bbbb.bLBBBBBBBBBb..bbbb....",
    "...bbLLLLbbBBBBBBBBBBBbbLLLLbb..",
    "..bLLLBBBBBbBBBBBBBBBbLLLBBBBBb.",
    ".bLLBBBBBBBBbBBBBBBBbLLBBBBBBBBb",
    ".bLBBBBBBBBBBBBBBBBBBLBBBBBBBBBb",
    "bLBBBBBBBBBBBBBBBBBBBBBBBBBBBBBb",
    "bLBBBBBBBBBBBBBBBBBBBBBBBBBBBBBb",
    "bBBBBBBBbBBBBBBBBBBBBBBbBBBBBBBb",
    "bBBBBBBBBBBBBBBbBBBBBBBBBBBBBBBb",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
])

_sprite("hill", None, {"H": _HILL, "h": _HILL_DARK}, [
    "............................hhhhhhhh............................",
    ".........................hhhHHHHHHHHhhh.........................",
    ".......................hhHHHHHHHHHHHHHHhh.......................",
    ".....................hhHHHHHHHHHHHHHHHHHHhh.....................",
    "....................hHHHHHHHHHHHHHHHHHHHHHHh....................",
    "...................hHHHHHHHHHHHHHHHHHHHHHHHHh...................",
    "..................hHHHHHHHHHHHhhHHHHHHHHHHHHHh..................",
    ".................hHHHHHHHHHHHhHHhHHHHHHHHHHHHHh.................",
    "................hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh................",
    "...............hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh...............",
    "..............hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh..............",
    ".............hHHHHHHHhhHHHHHHHHHHHHHHHHHHHHHHHHHHHh.............",
    "............hHHHHHHHhHHhHHHHHHHHHHHHHHHHhhHHHHHHHHHh............",
    "...........hHHHHHHHHHHHHHHHHHHHHHHHHHHHhHHhHHHHHHHHHh...........",
    "..........hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh..........",
    "..........hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh..........",
    ".........hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh.........",
    "........hHHHHHhhHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh........",
    "........hHHHHhHHhHHHHHHHHHHHHHHhhHHHHHHHHHHHHHHHHHHHHHHh........",
    ".......hHHHHHHHHHHHHHHHHHHHHHHhHHhHHHHHHHHHHHHHhhHHHHHHHh.......",
    "......hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHhHHhHHHHHHHh......",
    "......hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh......",
    ".....hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh.....",
    ".....hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh.....",
    "....hHHHhhHHHHHHHHHHHHHHHHHHHHHHHHHHHHHhhHHHHHHHHHHHHHHHHHHh....",
    "...hHHHhHHhHHHHHHHHHHHHhhHHHHHHHHHHHHHhHHhHHHHHHHHHHHHHHHHHHh...",
    "...hHHHHHHHHHHHHHHHHHHhHHhHHHHHHHHHHHHHHHHHHHHHHHHHHHHhhHHHHh...",
    "..hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHhHHhHHHHh..",
    "..hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh..",
    ".hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh.",
    ".hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh.",
    "hHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh",
])

# fmt: on
