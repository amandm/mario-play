"""Tile collision and velocity integration.

Movement is axis-separated: callers move an entity along X, then along Y, and
each move is resolved against the tile grid on its own. A hitbox covers the
half-open pixel ranges [x, x + w) x [y, y + h), so a hitbox that ends exactly on a
tile boundary does not touch the next tile; flush positions are exact floats and
need no epsilon. Moves longer than `MAX_SUBSTEP` are split up, so nothing can
tunnel through a tile at any speed.

The functions here are the hot path of the simulation: they use plain Python
floats/ints and index `level.tiles` directly.
"""

from __future__ import annotations

from mario_play.game.constants import (
    AIR_ACCEL,
    GRAVITY,
    GRAVITY_JUMP_HELD,
    GROUND_ACCEL_RUN,
    GROUND_ACCEL_WALK,
    JUMP_FAST_THRESHOLD,
    JUMP_IMPULSE,
    JUMP_IMPULSE_FAST,
    LEVEL_H_TILES,
    MAX_FALL,
    MAX_SUBSTEP,
    RELEASE_DECEL,
    RUN_MAX,
    SKID_DECEL,
    TILE,
    WALK_MAX,
)
from mario_play.game.entities import Entity, Player
from mario_play.game.level import Level
from mario_play.game.tiles import IS_SOLID

HIT_NONE = 0
HIT_FLOOR = 1
HIT_CEILING = 2

_LAST_ROW = LEVEL_H_TILES - 1


def overlaps(a: Entity, b: Entity) -> bool:
    """Do the two hitboxes overlap? Touching edges do not count."""
    return a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h


def box_hits_solid(level: Level, x: float, y: float, w: float, h: float) -> bool:
    """Does the box overlap any solid tile (level edges count as walls)?"""
    c0 = int(x // TILE)
    c1 = -int(-(x + w) // TILE) - 1
    if c0 < 0 or c1 >= level.width_tiles:
        return True
    r0 = max(int(y // TILE), 0)
    r1 = min(-int(-(y + h) // TILE) - 1, _LAST_ROW)
    tiles = level.tiles
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            if IS_SOLID[tiles[row, col]]:
                return True
    return False


def move_x(level: Level, e: Entity, dx: float) -> bool:
    """Move `e` horizontally by `dx`, stopping flush at solid tiles and the level edges.

    Returns True when the move was blocked. The velocity is left alone: the caller
    decides whether a blocked entity stops (player) or turns around (enemies).
    """
    while dx > MAX_SUBSTEP or dx < -MAX_SUBSTEP:
        part = MAX_SUBSTEP if dx > 0 else -MAX_SUBSTEP
        if _move_x_once(level, e, part):
            return True
        dx -= part
    return _move_x_once(level, e, dx)


def _move_x_once(level: Level, e: Entity, dx: float) -> bool:
    x = e.x + dx
    if dx > 0:
        col = -int(-(x + e.w) // TILE) - 1  # column the right edge ends up in
        if col >= level.width_tiles:
            e.x = float(level.width_tiles * TILE - e.w)
            return True
    elif dx < 0:
        col = int(x // TILE)
        if col < 0:
            e.x = 0.0
            return True
    else:
        return False

    y = e.y
    r0 = int(y // TILE)
    r1 = -int(-(y + e.h) // TILE) - 1
    if r0 < 0:
        r0 = 0
    if r1 > _LAST_ROW:
        r1 = _LAST_ROW
    tiles = level.tiles
    while r0 <= r1:
        if IS_SOLID[tiles[r0, col]]:
            e.x = float(col * TILE - e.w) if dx > 0 else float((col + 1) * TILE)
            return True
        r0 += 1
    e.x = x
    return False


def move_y(level: Level, e: Entity, dy: float) -> int:
    """Move `e` vertically by `dy`; returns `HIT_FLOOR`, `HIT_CEILING` or `HIT_NONE`.

    Above the top and below the bottom of the level there is nothing to collide with.
    """
    while dy > MAX_SUBSTEP or dy < -MAX_SUBSTEP:
        part = MAX_SUBSTEP if dy > 0 else -MAX_SUBSTEP
        hit = _move_y_once(level, e, part)
        if hit:
            return hit
        dy -= part
    return _move_y_once(level, e, dy)


def _move_y_once(level: Level, e: Entity, dy: float) -> int:
    y = e.y + dy
    if dy > 0:
        row = -int(-(y + e.h) // TILE) - 1  # row the bottom edge ends up in
    elif dy < 0:
        row = int(y // TILE)
    else:
        return HIT_NONE
    if row < 0 or row > _LAST_ROW:
        e.y = y
        return HIT_NONE

    x = e.x
    c0 = int(x // TILE)
    c1 = -int(-(x + e.w) // TILE) - 1
    if c0 < 0:
        c0 = 0
    if c1 >= level.width_tiles:
        c1 = level.width_tiles - 1
    tiles = level.tiles
    while c0 <= c1:
        if IS_SOLID[tiles[row, c0]]:
            if dy > 0:
                e.y = float(row * TILE - e.h)
                return HIT_FLOOR
            e.y = float((row + 1) * TILE)
            return HIT_CEILING
        c0 += 1
    e.y = y
    return HIT_NONE


def bump_column(level: Level, e: Entity, row: int) -> int:
    """Column of the solid tile in `row` above `e` that is nearest to its horizontal centre.

    Returns -1 when no tile of that row above the hitbox is solid.
    """
    if row < 0 or row > _LAST_ROW:
        return -1
    center = e.x + e.w / 2
    c0 = max(int(e.x // TILE), 0)
    c1 = min(-int(-(e.x + e.w) // TILE) - 1, level.width_tiles - 1)
    tiles = level.tiles
    best, best_distance = -1, 0.0
    for col in range(c0, c1 + 1):
        if IS_SOLID[tiles[row, col]]:
            distance = abs(col * TILE + TILE / 2 - center)
            if best < 0 or distance < best_distance:
                best, best_distance = col, distance
    return best


def apply_gravity(e: Entity) -> None:
    """Accelerate `e` downwards by normal gravity, up to terminal velocity."""
    vy = e.vy + GRAVITY
    e.vy = MAX_FALL if vy > MAX_FALL else vy


def control_player(p: Player, left: bool, right: bool, jump: bool, run: bool) -> None:
    """Turn one frame of button state into the player's new velocity (no movement yet)."""
    direction = (1 if right else 0) - (1 if left else 0)
    limit = RUN_MAX if run else WALK_MAX
    vx = p.vx
    if p.on_ground:
        if direction == 0:
            if vx > 0:
                vx = vx - RELEASE_DECEL if vx > RELEASE_DECEL else 0.0
            elif vx < 0:
                vx = vx + RELEASE_DECEL if vx < -RELEASE_DECEL else 0.0
        else:
            p.facing = direction
            speed = vx * direction
            if speed < 0:
                speed += SKID_DECEL
            elif speed > limit:
                speed = max(limit, speed - RELEASE_DECEL)
            else:
                speed = min(limit, speed + (GROUND_ACCEL_RUN if run else GROUND_ACCEL_WALK))
            vx = speed * direction
    elif direction != 0:
        p.facing = direction
        speed = vx * direction
        if speed < limit:
            vx = min(limit, speed + AIR_ACCEL) * direction
    p.vx = vx

    if not jump:
        p.jump_armed = True
    elif p.jump_armed and p.on_ground:
        p.vy = JUMP_IMPULSE_FAST if abs(vx) > JUMP_FAST_THRESHOLD else JUMP_IMPULSE
        p.jump_armed = False
        p.on_ground = False

    vy = p.vy
    vy += GRAVITY_JUMP_HELD if (jump and vy < 0) else GRAVITY
    p.vy = MAX_FALL if vy > MAX_FALL else vy
