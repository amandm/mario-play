"""Draws a `Game` into a ``(240, 256, 3)`` uint8 RGB frame: the one view humans and agents share.

Almost everything on screen is static, so the whole level is painted once into a wide
background image and a frame is just a crop of it at the camera position, plus the few sprites
that move and the HUD. Two wide images are kept per level:

* the *scenery*: sky, decorations (hills, bushes, clouds) and the flag's pennant;
* the *background*: the scenery with every tile drawn on top.

Tiles change during play (coins vanish, bricks break, ?-blocks get used), so every `render`
compares `game.level.tiles` with the copy the background was painted from and repaints only
the cells that differ: scenery first, then the new tile. That is why the scenery is kept.
The comparison is by content, never by object identity, so one renderer can be pointed at any
game at any time: clones, a game after `reset`, another level.

Decorations are a pure function of the level (its name, the player's start and the tiles
that never change during play): no random state, the same picture in every process. They are
only ever drawn behind tiles and sprites, they keep out of the HUD band and away from the
spot where the player starts.

Numpy only: this module must stay importable without pygame, gymnasium or Pillow.
"""

from __future__ import annotations

import math
import zlib
from typing import TYPE_CHECKING

import numpy as np

from mario_play.game.constants import LEVEL_H_TILES, TILE, VIEW_H, VIEW_W
from mario_play.game.font import draw_text
from mario_play.game.sprites import SKY_COLOR, blit, get_sprite
from mario_play.game.tiles import Tile

if TYPE_CHECKING:
    from mario_play.game.engine import Game
    from mario_play.game.entities import Entity, Player
    from mario_play.game.level import Level

HUD_HEIGHT = 28
"""Rows at the top of the frame that belong to the HUD; decorations never enter them."""

_SKY = np.array(SKY_COLOR, dtype=np.uint8)

# ------------------------------------------------------------------------------------- tiles

_N_TILES = len(Tile)
_GROUND_TOP = _N_TILES  # extra sprite slot: a ground tile without ground above it

_TILE_SPRITE_NAMES: dict[int, str] = {
    Tile.GROUND: "ground",
    Tile.HARD: "hard",
    Tile.BRICK: "brick",
    Tile.QUESTION_COIN: "question",
    Tile.QUESTION_MUSHROOM: "question",  # what a ?-block holds is not visible
    Tile.USED: "used",
    Tile.PIPE_TL: "pipe_tl",
    Tile.PIPE_TR: "pipe_tr",
    Tile.PIPE_L: "pipe_l",
    Tile.PIPE_R: "pipe_r",
    Tile.COIN: "coin",
    Tile.FLAGPOLE: "flagpole",
    Tile.FLAG_TOP: "flag_top",
    _GROUND_TOP: "ground_top",
}
# Sprite per slot; None: nothing to draw.
_TILE_SPRITES: list[np.ndarray | None] = [
    get_sprite(_TILE_SPRITE_NAMES[slot]) if slot in _TILE_SPRITE_NAMES else None
    for slot in range(_N_TILES + 1)
]

# Tiles the engine rewrites during play. Any other difference between two tile grids means
# that they belong to different levels, and then the scenery has to be redone as well.
_IS_DYNAMIC = np.zeros(_N_TILES, dtype=bool)
_IS_DYNAMIC[
    [Tile.EMPTY, Tile.BRICK, Tile.QUESTION_COIN, Tile.QUESTION_MUSHROOM, Tile.USED, Tile.COIN]
] = True
# Tiles that never change: the only ones decorations are allowed to depend on.
_IS_STATIC = ~_IS_DYNAMIC

# Beyond this many changed cells a whole-level repaint is about as cheap as going cell by cell.
_MAX_REPAINT_CELLS = 256

# ----------------------------------------------------------------------------------- sprites

_WALK_PERIOD = 6  # frames per player walk pose
_ENEMY_PERIOD = 8  # frames per enemy walk pose
_BLINK_PERIOD = 4  # frames per visible/hidden window while invulnerable

_PLAYER_STAND = (get_sprite("player_small_stand"), get_sprite("player_big_stand"))
_PLAYER_JUMP = (get_sprite("player_small_jump"), get_sprite("player_big_jump"))
_PLAYER_WALK = (
    (get_sprite("player_small_walk1"), get_sprite("player_small_walk2")),
    (get_sprite("player_big_walk1"), get_sprite("player_big_walk2")),
)
_WALKER_WALK = (get_sprite("walker_1"), get_sprite("walker_2"))
_WALKER_FLAT = get_sprite("walker_flat")
_TURTLE_WALK = (get_sprite("turtle_1"), get_sprite("turtle_2"))
_SHELL = get_sprite("shell")
_MUSHROOM = get_sprite("mushroom")

_FLAG = get_sprite("flag")
_FLAG_OFFSET_X = 9  # the pennant's hoist sits just right of the pole (see sprites.py)

# --------------------------------------------------------------------------------------- HUD

_HUD_TEXT = (252, 252, 252)
_HUD_SHADOW = (30, 24, 48)
_HUD_KEY = (1, 2, 3)  # a colour no text uses: marks the see-through part of the HUD strip
_HUD_LABEL_Y = 7
_HUD_VALUE_Y = 16
_HUD_LABELS = ("SCORE", "COINS", "LEVEL", "TIME")
_HUD_COLUMNS = (16, 80, 144, 208)  # x of each label and its value; 64 px (10 glyphs) apart
_HUD_NAME_CHARS = 9

# ------------------------------------------------------------------------------- decorations

_CLOUD = get_sprite("cloud")
_BUSH = get_sprite("bush")
_HILL = get_sprite("hill")

# One candidate per slot of this many tile columns, kept with this probability (in 1/256).
_HILL_SLOT, _HILL_CHANCE = 22, 176
_BUSH_SLOT, _BUSH_CHANCE = 9, 150
_CLOUD_SLOT, _CLOUD_CHANCE = 9, 176
_CLOUD_Y_MIN, _CLOUD_Y_MAX = 32, 84  # top edge of a cloud: below the HUD, above the action
_START_CLEARANCE = 2  # tile columns on either side of the player's start without bush or hill


def _mix(seed: int, index: int, channel: int) -> int:
    """A stateless 32-bit integer hash (lowbias32): the decorations' only source of variety."""
    value = (seed ^ (index * 0x9E3779B1) ^ (channel * 0x85EBCA6B)) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def _surface_rows(tiles: np.ndarray) -> np.ndarray:
    """Per column, the top row of the ground that reaches the bottom of the level (-1: a pit)."""
    ground = tiles == Tile.GROUND
    depth = np.cumprod(ground[::-1], axis=0, dtype=np.intp).sum(axis=0)
    return np.where(depth > 0, tiles.shape[0] - depth, -1)


def _paint_grounded(
    scenery: np.ndarray,
    sprite: np.ndarray,
    surface: np.ndarray,
    occupied: np.ndarray,
    start_col: int,
    seed: int,
    channel: int,
    slot_cols: int,
    chance: int,
) -> None:
    """Stand `sprite` on level ground at most once per slot of `slot_cols` columns.

    A spot qualifies when the ground below is one level surface as wide as the sprite and no
    static tile is in the way, so nothing ever hangs over a pit or pokes out of a pipe. The
    columns around `start_col` stay free: the first frame shows the player against plain sky.
    """
    n_cols = surface.shape[0]
    span = -(-sprite.shape[1] // TILE)
    rise = -(-sprite.shape[0] // TILE)
    for slot in range(-(-n_cols // slot_cols)):
        if _mix(seed, slot, channel) % 256 >= chance:
            continue
        first = slot * slot_cols
        last = min(first + slot_cols, n_cols) - span
        start = first + _mix(seed, slot, channel + 1) % slot_cols
        for col in range(start, last + 1):
            row = int(surface[col])
            if row < rise or (surface[col : col + span] != row).any():
                continue
            if occupied[row - rise : row, col : col + span].any():
                continue
            if col - _START_CLEARANCE <= start_col < col + span + _START_CLEARANCE:
                continue
            blit(scenery, sprite, col * TILE, row * TILE - sprite.shape[0])
            break


def _paint_clouds(scenery: np.ndarray, occupied: np.ndarray, seed: int) -> None:
    """Scatter clouds over the upper sky, at most one per slot, never across a static tile."""
    n_cols = occupied.shape[1]
    height, width = _CLOUD.shape[:2]
    for slot in range(-(-n_cols // _CLOUD_SLOT)):
        if _mix(seed, slot, 20) % 256 >= _CLOUD_CHANCE:
            continue
        x = slot * _CLOUD_SLOT * TILE + _mix(seed, slot, 21) % (_CLOUD_SLOT * TILE - width)
        y = _CLOUD_Y_MIN + _mix(seed, slot, 22) % (_CLOUD_Y_MAX - _CLOUD_Y_MIN)
        if x + width > n_cols * TILE:
            continue
        cells = occupied[
            y // TILE : (y + height - 1) // TILE + 1, x // TILE : (x + width - 1) // TILE + 1
        ]
        if not cells.any():
            blit(scenery, _CLOUD, x, y)


def _paint_decorations(scenery: np.ndarray, level: Level) -> None:
    """Hills, bushes and clouds: a pure function of the level's name, start and static tiles."""
    tiles = level.tiles
    seed = zlib.crc32(str(level.name).encode("utf-8"))  # not hash(): it varies by process
    occupied = _IS_STATIC[tiles]
    surface = _surface_rows(tiles)
    start = level.player_start[0]
    _paint_grounded(scenery, _HILL, surface, occupied, start, seed, 0, _HILL_SLOT, _HILL_CHANCE)
    _paint_grounded(scenery, _BUSH, surface, occupied, start, seed, 10, _BUSH_SLOT, _BUSH_CHANCE)
    _paint_clouds(scenery, occupied, seed)


def _paint_pennants(scenery: np.ndarray, tiles: np.ndarray) -> None:
    """Hang the flag's pennant on the pole, right below its top."""
    rows, cols = np.nonzero(tiles == Tile.FLAG_TOP)
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if row + 1 < tiles.shape[0] and tiles[row + 1, col] == Tile.FLAGPOLE:
            blit(scenery, _FLAG, col * TILE + _FLAG_OFFSET_X, (row + 1) * TILE)


def _sky(width: int) -> np.ndarray:
    """A ``(240, width, 3)`` image of nothing but sky."""
    image = np.empty((VIEW_H, width, 3), dtype=np.uint8)
    # One flat row broadcast down the image: ~50x faster than broadcasting the single colour.
    image.reshape(VIEW_H, width * 3)[:] = np.tile(_SKY, width)
    return image


def _sprite_slots(tiles: np.ndarray) -> np.ndarray:
    """Tile grid -> index into `_TILE_SPRITES` (grass-capped ground has a slot of its own)."""
    if tiles.size and (tiles.min() < 0 or tiles.max() >= _N_TILES):
        bad = sorted({int(t) for t in np.unique(tiles) if not 0 <= t < _N_TILES})
        raise ValueError(f"level contains unknown tile ids {bad}; known: 0..{_N_TILES - 1}")
    slots = tiles.astype(np.intp)
    ground = tiles == Tile.GROUND
    capped = ground.copy()
    capped[1:] &= ~ground[:-1]
    slots[capped] = _GROUND_TOP
    return slots


class Renderer:
    """Turns games into frames; keeps the painted level of the last game it has seen.

    Args:
        hud: Draw score, coins, level name and time across the top of the frame.
        decorations: Paint hills, bushes and clouds behind the level. They never hide a tile
            or a sprite; switch them off for the plainest possible picture.
    """

    def __init__(self, hud: bool = True, decorations: bool = True) -> None:
        self.hud = hud
        self.decorations = decorations
        self._level_key: tuple | None = None  # what the scenery depends on besides the tiles
        self._tiles: np.ndarray | None = None  # the grid the background was painted from
        self._scenery = np.empty((VIEW_H, 0, 3), dtype=np.uint8)
        self._background = np.empty((VIEW_H, 0, 3), dtype=np.uint8)
        self._hud_state: tuple | None = None
        self._hud_rgb = np.empty((HUD_HEIGHT, VIEW_W, 3), dtype=np.uint8)
        self._hud_mask = np.zeros((HUD_HEIGHT, VIEW_W, 1), dtype=bool)

    def render(self, game: Game) -> np.ndarray:
        """Draw the current view of `game`; returns a new ``(240, 256, 3)`` uint8 RGB array.

        Works with any game at any time (clones, after `reset`, another level): the cached
        background is brought up to date first. The game is never modified.
        """
        self._sync_background(game.level)
        background = self._background
        cam = min(max(int(game.camera_x), 0), background.shape[1] - VIEW_W)
        frame = background[:, cam : cam + VIEW_W].copy()

        tick = game.frame
        for entity in game.entities:
            sprite = _entity_sprite(entity, tick)
            if sprite is not None:
                # Only things that walk have a front; shells and mushrooms look the same both ways.
                walks = sprite is not _SHELL and sprite is not _MUSHROOM
                _draw(frame, sprite, entity, cam, walks and entity.facing < 0)
        player = game.player
        if (player.invuln_frames // _BLINK_PERIOD) % 2 == 0:
            _draw(frame, _player_sprite(player, tick), player, cam, player.facing < 0)

        if self.hud:
            self._draw_hud(frame, game)
        return frame

    # --- background ----------------------------------------------------------------------------

    def _sync_background(self, level: Level) -> None:
        """Make the cached background show `level.tiles`, repainting as little as possible."""
        tiles = level.tiles
        cached = self._tiles
        if cached is None or tiles.shape != cached.shape or self._key_of(level) != self._level_key:
            self._repaint_level(level)
            return
        changed = tiles != cached
        if not changed.any():
            return
        rows, cols = np.nonzero(changed)
        new = tiles[rows, cols]
        playable = (
            rows.size <= _MAX_REPAINT_CELLS
            and new.min() >= 0
            and new.max() < _N_TILES
            and _IS_DYNAMIC[new].all()
            and _IS_DYNAMIC[cached[rows, cols]].all()
        )
        if not playable:  # another level under the same name (or a broken one)
            self._repaint_level(level)
            return
        background, scenery = self._background, self._scenery
        for row, col, tile in zip(rows.tolist(), cols.tolist(), new.tolist(), strict=True):
            x, y = col * TILE, row * TILE
            background[y : y + TILE, x : x + TILE] = scenery[y : y + TILE, x : x + TILE]
            sprite = _TILE_SPRITES[tile]
            if sprite is not None:
                blit(background, sprite, x, y)
        cached[rows, cols] = new

    def _key_of(self, level: Level) -> tuple:
        """Everything the scenery depends on apart from the tile grid itself."""
        return (level.name, level.player_start, self.decorations)

    def _repaint_level(self, level: Level) -> None:
        """Paint scenery and background of a whole level from scratch."""
        tiles = level.tiles
        if tiles.ndim != 2 or tiles.shape[0] != LEVEL_H_TILES:
            raise ValueError(
                f"level {level.name!r} has a tile grid of shape {tiles.shape}; "
                f"expected ({LEVEL_H_TILES}, width)"
            )
        self._tiles = None  # nothing is cached should painting fail half-way
        slots = _sprite_slots(tiles)
        # At least one screen wide, so that the camera crop always has the full frame size.
        scenery = _sky(max(tiles.shape[1] * TILE, VIEW_W))
        if self.decorations:
            _paint_decorations(scenery, level)
        _paint_pennants(scenery, tiles)

        background = scenery.copy()
        rows, cols = np.nonzero(slots)
        for row, col, slot in zip(
            rows.tolist(), cols.tolist(), slots[rows, cols].tolist(), strict=True
        ):
            blit(background, _TILE_SPRITES[slot], col * TILE, row * TILE)

        self._scenery = scenery
        self._background = background
        self._level_key = self._key_of(level)
        self._tiles = tiles.copy()

    # --- HUD -----------------------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, game: Game) -> None:
        state = (game.score, game.coins, game.time_left, game.level.name)
        if state != self._hud_state:
            # The HUD changes a few times per second at most: compose it once, then stamp it.
            self._hud_rgb, self._hud_mask = _compose_hud(*state)
            self._hud_state = state
        np.copyto(frame[:HUD_HEIGHT], self._hud_rgb, where=self._hud_mask)


def _compose_hud(
    score: int, coins: int, time_left: int, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """The HUD as an RGB strip plus the mask of the pixels that carry text."""
    strip = np.empty((HUD_HEIGHT, VIEW_W, 3), dtype=np.uint8)
    strip[:] = _HUD_KEY
    values = (
        f"{max(score, 0):06d}"[-10:],
        f"\u00d7{max(coins, 0):02d}"[:10],
        str(name)[:_HUD_NAME_CHARS],
        f"{max(time_left, 0):03d}"[-6:],
    )
    for x, label, value in zip(_HUD_COLUMNS, _HUD_LABELS, values, strict=True):
        draw_text(strip, label, x, _HUD_LABEL_Y, _HUD_TEXT, shadow=_HUD_SHADOW)
        draw_text(strip, value, x, _HUD_VALUE_Y, _HUD_TEXT, shadow=_HUD_SHADOW)
    mask = (strip != np.array(_HUD_KEY, dtype=np.uint8)).any(axis=2, keepdims=True)
    return strip, mask


# --- sprites -----------------------------------------------------------------------------------


def _player_sprite(player: Player, tick: int) -> np.ndarray:
    big = 1 if player.big else 0
    if not player.on_ground:
        return _PLAYER_JUMP[big]
    if player.vx != 0.0:
        return _PLAYER_WALK[big][(tick // _WALK_PERIOD) % 2]
    return _PLAYER_STAND[big]


def _entity_sprite(entity: Entity, tick: int) -> np.ndarray | None:
    kind = entity.kind
    if kind == "walker":
        if entity.squished_frames > 0:
            return _WALKER_FLAT
        return _WALKER_WALK[(tick // _ENEMY_PERIOD) % 2]
    if kind == "turtle":
        if entity.state == "walk":
            return _TURTLE_WALK[(tick // _ENEMY_PERIOD) % 2]
        return _SHELL
    if kind == "mushroom":
        return _MUSHROOM
    return None  # no art for this kind


def _draw(frame: np.ndarray, sprite: np.ndarray, entity: Entity, cam: int, flip: bool) -> None:
    """Blit `sprite` with its bottom centre on the bottom centre of the entity's hitbox."""
    height, width = sprite.shape[:2]
    x = math.floor(entity.x + (entity.w - width) / 2) - cam
    y = math.floor(entity.y + entity.h) - height
    if -width < x < VIEW_W and -height < y < VIEW_H:
        blit(frame, sprite, x, y, flip)
