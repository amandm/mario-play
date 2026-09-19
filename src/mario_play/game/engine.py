"""The game loop: one `Game.step(buttons)` advances the world by one frame (1/60 s).

Frame order: player control and movement (X, then Y, with head bumps) -> tile
pickups and the flag -> enemy and item movement -> player/entity contacts ->
entity/entity contacts -> pits and cleanup -> camera -> spawn activation -> timer.
The simulation is deterministic: the same level, seed and button sequence always
produce the same states, also across `clone()`. The current rules draw no random
numbers; `Game.rng` is the one source any future stochastic rule must use.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

from mario_play.game.constants import (
    CAMERA_PLAYER_OFFSET,
    ENEMY_SPEED,
    FRAMES_PER_TIME_UNIT,
    LEVEL_H_PX,
    MUSHROOM_H,
    MUSHROOM_W,
    PLAYER_BIG_H,
    SCORE_BRICK,
    SCORE_COIN,
    SCORE_FLAG,
    SCORE_FLAG_PER_TIME,
    SCORE_MUSHROOM,
    SCORE_SHELL_KILL,
    SCORE_STOMP,
    SPAWN_ACTIVATION_DISTANCE,
    STOMP_BOUNCE,
    STOMP_BOUNCE_HELD,
    TILE,
    TURTLE_H,
    TURTLE_W,
    VIEW_W,
    WALKER_H,
    WALKER_W,
)
from mario_play.game.entities import Entity, Mushroom, Player, Turtle, Walker
from mario_play.game.level import Level, load_level
from mario_play.game.physics import (
    HIT_CEILING,
    HIT_FLOOR,
    apply_gravity,
    box_hits_solid,
    bump_column,
    control_player,
    move_x,
    move_y,
)
from mario_play.game.tiles import IS_GOAL, IS_SOLID, Tile

_EMPTY = int(Tile.EMPTY)
_COIN = int(Tile.COIN)
_BRICK = int(Tile.BRICK)
_QUESTION_COIN = int(Tile.QUESTION_COIN)
_QUESTION_MUSHROOM = int(Tile.QUESTION_MUSHROOM)
_USED = int(Tile.USED)
_LAST_ROW = LEVEL_H_PX // TILE - 1


@dataclass(frozen=True)
class Buttons:
    """The controller state for one frame."""

    left: bool = False
    right: bool = False
    jump: bool = False
    run: bool = False


@dataclass
class StepEvents:
    """What happened during one `Game.step` (all zero/False once the game is over)."""

    coins: int = 0
    stomps: int = 0
    bricks: int = 0
    powerups: int = 0
    hurt: bool = False
    died: bool = False
    won: bool = False
    death_cause: str | None = None  # "pit" | "enemy" | "timeout"
    score_delta: int = 0


class Game:
    """A headless, deterministic platformer simulation on one level.

    Attributes: `level` (a private, mutable copy), `player`, `entities` (active
    non-player entities), `frame`, `time_left`, `score`, `coins`, `camera_x`,
    `over`, `won`, `death_cause` and `rng`.
    """

    def __init__(
        self, level: Level | str | os.PathLike[str] = "1-1", seed: int | None = None
    ) -> None:
        """`level`: a `Level` (copied, never mutated), a bundled level name or a level file path."""
        source = level.copy() if isinstance(level, Level) else load_level(level)
        unknown = {s.kind for s in source.spawns} - {"walker", "turtle"}
        if unknown:
            raise ValueError(f"level {source.name!r} has unknown spawn kinds: {sorted(unknown)}")
        self._pristine: Level = source
        # Spawns wake up in order of their x position because the camera only moves right.
        self._spawn_order = sorted(source.spawns, key=lambda s: s.col)
        self.rng = random.Random(seed)
        self.reset()

    # --- lifecycle -----------------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Restart the level from its pristine state; a given `seed` reseeds `rng`."""
        if seed is not None:
            self.rng.seed(seed)
        self.level: Level = self._pristine.copy()
        col, row = self.level.player_start
        player = Player(0.0, 0.0)
        player.x = float(col * TILE + (TILE - player.w) // 2)
        player.y = float((row + 1) * TILE - player.h)
        # Standing from frame 0, so that a jump on the very first frame works.
        player.on_ground = box_hits_solid(self.level, player.x, player.y + player.h, player.w, 1)
        self.player: Player = player
        self.entities: list[Entity] = []
        self.frame = 0
        self.time_left = self.level.time
        self.score = 0
        self.coins = 0
        self.over = False
        self.won = False
        self.death_cause: str | None = None
        self._max_camera_x = float(max(self.level.width_px - VIEW_W, 0))
        self.camera_x = min(max(player.x - CAMERA_PLAYER_OFFSET, 0.0), self._max_camera_x)
        self._next_spawn = 0
        self._activate_spawns()

    def clone(self) -> Game:
        """An independent deep copy that continues exactly like the original."""
        twin = object.__new__(type(self))
        twin.__dict__.update(self.__dict__)  # immutable fields and the shared pristine level
        twin.level = self.level.copy()
        twin.player = self.player.clone()
        twin.entities = [e.clone() for e in self.entities]
        rng = random.Random.__new__(random.Random)
        rng.setstate(self.rng.getstate())
        twin.rng = rng
        return twin

    def snapshot(self) -> dict:
        """A plain-value summary of the state, handy for determinism checks and logging."""
        p = self.player
        return {
            "x": p.x,
            "y": p.y,
            "vx": p.vx,
            "vy": p.vy,
            "on_ground": p.on_ground,
            "big": p.big,
            "coins": self.coins,
            "score": self.score,
            "time_left": self.time_left,
            "over": self.over,
            "won": self.won,
            "death_cause": self.death_cause,
            "frame": self.frame,
        }

    def __repr__(self) -> str:
        state = "won" if self.won else self.death_cause if self.over else "running"
        return (
            f"Game(level={self.level.name!r}, frame={self.frame}, x={self.player.x:.1f}, "
            f"entities={len(self.entities)}, {state})"
        )

    # --- the frame -----------------------------------------------------------------------------

    def step(self, buttons: Buttons) -> StepEvents:
        """Advance one frame. Once the game is over this is a no-op returning empty events."""
        events = StepEvents()
        if self.over:
            return events
        self.frame += 1
        score_before = self.score
        p = self.player
        # Decided before counting down, so a hurt player is safe for exactly INVULN_FRAMES frames.
        invulnerable = p.invuln_frames > 0
        if invulnerable:
            p.invuln_frames -= 1

        self._move_player(buttons, events)
        self._touch_tiles(events)
        if not self.over:
            entities = self.entities
            if entities:
                self._update_entities()
                self._player_contacts(buttons.jump, invulnerable, events)
                if len(entities) > 1:
                    self._entity_contacts()
                if not all(e.alive for e in entities):
                    # In place, so that references to the list held elsewhere stay valid.
                    entities[:] = [e for e in entities if e.alive]
            if p.y > LEVEL_H_PX and not self.over:
                self._kill_player("pit", events)

        camera = p.x - CAMERA_PLAYER_OFFSET
        if camera > self._max_camera_x:
            camera = self._max_camera_x
        if camera > self.camera_x:
            self.camera_x = camera
        self._activate_spawns()

        if not self.over and self.frame % FRAMES_PER_TIME_UNIT == 0:
            self.time_left -= 1
            if self.time_left <= 0:
                self.time_left = 0
                self._kill_player("timeout", events)

        events.score_delta = self.score - score_before
        return events

    # --- player --------------------------------------------------------------------------------

    def _move_player(self, buttons: Buttons, events: StepEvents) -> None:
        p = self.player
        level = self.level
        control_player(p, buttons.left, buttons.right, buttons.jump, buttons.run)

        vx = p.vx
        if vx != 0.0:
            if move_x(level, p, vx):
                p.vx = 0.0
            if p.x < self.camera_x:  # the left screen edge is a wall
                p.x = self.camera_x
                if vx < 0:
                    p.vx = 0.0

        hit = move_y(level, p, p.vy)
        if hit == HIT_FLOOR:
            p.vy = 0.0
            p.on_ground = True
        else:
            p.on_ground = False
            if hit == HIT_CEILING:
                p.vy = 0.0
                row = int(p.y // TILE) - 1
                col = bump_column(level, p, row)
                if col >= 0:
                    self._bump_tile(col, row, events)

    def _bump_tile(self, col: int, row: int, events: StepEvents) -> None:
        tiles = self.level.tiles
        tile = tiles[row, col]
        if tile == _QUESTION_COIN:
            tiles[row, col] = _USED
            self.coins += 1
            self.score += SCORE_COIN
            events.coins += 1
        elif tile == _QUESTION_MUSHROOM:
            tiles[row, col] = _USED
            self._spawn_mushroom(col, row)
        elif tile == _BRICK and self.player.big:
            tiles[row, col] = _EMPTY
            self.score += SCORE_BRICK
            events.bricks += 1

    def _spawn_mushroom(self, col: int, row: int) -> None:
        tiles = self.level.tiles
        above = row - 1
        while above >= 0 and IS_SOLID[tiles[above, col]]:  # stacked blocks: pop out on top
            above -= 1
        x = col * TILE + (TILE - MUSHROOM_W) / 2
        self.entities.append(Mushroom(x, float((above + 1) * TILE - MUSHROOM_H)))

    def _touch_tiles(self, events: StepEvents) -> None:
        """Collect coin tiles under the player's hitbox and detect the flagpole."""
        p = self.player
        level = self.level
        c0 = int(p.x // TILE)
        c1 = -int(-(p.x + p.w) // TILE) - 1
        r0 = int(p.y // TILE)
        r1 = -int(-(p.y + p.h) // TILE) - 1
        if r0 < 0:
            r0 = 0
        if r1 > _LAST_ROW:
            r1 = _LAST_ROW
        if c1 >= level.width_tiles:
            c1 = level.width_tiles - 1
        tiles = level.tiles
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                tile = tiles[row, col]
                if tile == _EMPTY:
                    continue
                if tile == _COIN:
                    tiles[row, col] = _EMPTY
                    self.coins += 1
                    self.score += SCORE_COIN
                    events.coins += 1
                elif IS_GOAL[tile]:
                    self.over = True
                    self.won = True
                    self.score += SCORE_FLAG + SCORE_FLAG_PER_TIME * self.time_left
                    events.won = True
                    return

    def _kill_player(self, cause: str, events: StepEvents) -> None:
        self.over = True
        self.won = False
        self.death_cause = cause
        self.player.die()
        events.died = True
        events.death_cause = cause

    # --- enemies and items ---------------------------------------------------------------------

    def _activate_spawns(self) -> None:
        order = self._spawn_order
        limit = self.camera_x + SPAWN_ACTIVATION_DISTANCE
        while self._next_spawn < len(order) and order[self._next_spawn].col * TILE < limit:
            spawn = order[self._next_spawn]
            self._next_spawn += 1
            feet = float((spawn.row + 1) * TILE)
            if spawn.kind == "turtle":
                x = spawn.col * TILE + (TILE - TURTLE_W) / 2
                self.entities.append(Turtle(x, feet - TURTLE_H))
            else:
                x = spawn.col * TILE + (TILE - WALKER_W) / 2
                self.entities.append(Walker(x, feet - WALKER_H))

    def _update_entities(self) -> None:
        level = self.level
        for e in self.entities:
            if e.kind == "walker":
                if e.squished_frames > 0:  # a corpse: no walking, but it still falls
                    e.squished_frames -= 1
                    if e.squished_frames == 0:
                        e.alive = False
                        continue
            elif e.kind == "turtle" and e.contact_cooldown > 0:
                e.contact_cooldown -= 1
            apply_gravity(e)
            if e.vx != 0.0 and move_x(level, e, e.vx):
                e.vx = -e.vx
                e.facing = -e.facing
            e.on_ground = move_y(level, e, e.vy) == HIT_FLOOR
            if e.on_ground:
                e.vy = 0.0
            elif e.y > LEVEL_H_PX:
                e.alive = False

    def _player_contacts(self, jump_held: bool, invulnerable: bool, events: StepEvents) -> None:
        """Resolve everything the player touches this frame, whatever the order of `entities`.

        All contacts are classified against the hitbox the player had after moving;
        growing and getting hurt change it, so both happen after the loop: first the
        mushrooms, then harm - once, and never in a frame with a stomp.
        """
        p = self.player
        left, right, top, bottom = p.x, p.x + p.w, p.y, p.y + p.h
        bounced = harmed = False
        mushrooms: tuple[Entity, ...] = ()
        for e in self.entities:
            if not (e.alive and left < e.x + e.w and e.x < right):
                continue
            if not (top < e.y + e.h and e.y < bottom):
                continue
            kind = e.kind
            if kind == "mushroom":
                mushrooms += (e,)
                continue
            if kind == "walker":
                if e.squished_frames > 0:
                    continue
            elif e.contact_cooldown > 0:
                continue
            elif e.state == "shell":
                e.kick(1 if e.x + e.w / 2 >= p.x + p.w / 2 else -1)
                continue

            # A harmful enemy: walker, walking turtle or moving shell.
            if p.vy > 0 and bottom < e.y + e.h / 2:
                if kind == "walker":
                    e.squish()
                elif e.state == "walk":
                    e.to_shell()
                else:
                    e.stop()
                self.score += SCORE_STOMP
                events.stomps += 1
                bounced = True
                continue
            if kind == "turtle" and e.state == "shell_moving":
                moving_away = (e.vx > 0) == (e.x + e.w / 2 >= p.x + p.w / 2)
                if moving_away:
                    continue
            harmed = True

        for shroom in mushrooms:
            self._collect_mushroom(shroom, events)
        if bounced:
            p.vy = STOMP_BOUNCE_HELD if jump_held else STOMP_BOUNCE
            p.on_ground = False
        elif harmed and not invulnerable:
            if p.big:
                p.hurt()
                events.hurt = True
            else:
                self._kill_player("enemy", events)

    def _collect_mushroom(self, shroom: Entity, events: StepEvents) -> None:
        p = self.player
        if not p.big:
            if box_hits_solid(self.level, p.x, p.y + p.h - PLAYER_BIG_H, p.w, PLAYER_BIG_H):
                return  # no head room to grow here; the mushroom stays in the world
            p.grow()
        shroom.alive = False
        self.score += SCORE_MUSHROOM
        events.powerups += 1

    def _entity_contacts(self) -> None:
        """Moving shells kill what they touch; other enemies turn away from each other."""
        enemies = [
            e
            for e in self.entities
            if e.alive and e.kind != "mushroom" and not (e.kind == "walker" and e.squished_frames)
        ]
        count = len(enemies)
        # Kills take effect after the pass: a shell destroyed by another shell this frame still
        # kills everything else it touches, whatever the order of `entities`.
        doomed: list[Entity] = []
        for i in range(count - 1):
            a = enemies[i]
            ax1 = a.x + a.w
            for j in range(i + 1, count):
                b = enemies[j]
                if not (a.x < b.x + b.w and b.x < ax1 and a.y < b.y + b.h and b.y < a.y + a.h):
                    continue
                a_shell = a.kind == "turtle" and a.state == "shell_moving"
                b_shell = b.kind == "turtle" and b.state == "shell_moving"
                if a_shell or b_shell:
                    if a_shell:
                        doomed.append(b)
                    if b_shell:
                        doomed.append(a)
                    continue
                left, right = (a, b) if a.x + a.w / 2 <= b.x + b.w / 2 else (b, a)
                _turn(left, -1)
                _turn(right, 1)
        for e in doomed:
            if e.alive:  # touched by two shells at once: still one kill
                e.alive = False
                self.score += SCORE_SHELL_KILL


def _turn(e: Entity, direction: int) -> None:
    """Make a walking enemy head towards `direction`; resting shells stay put."""
    if e.vx != 0.0:
        e.facing = direction
        e.vx = direction * ENEMY_SPEED
