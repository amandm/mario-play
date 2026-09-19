"""Everything that moves: the player, enemies and items.

Entities are plain mutable objects with float positions. `x, y` is the top-left
of the hitbox, `w, h` its size in pixels. They hold state and the small state
transitions that keep the hitbox consistent (feet stay put when the height
changes); the rules that decide *when* those happen live in `engine.py`.
"""

from __future__ import annotations

from mario_play.game.constants import (
    ENEMY_SPEED,
    INVULN_FRAMES,
    MUSHROOM_H,
    MUSHROOM_SPEED,
    MUSHROOM_W,
    PLAYER_BIG_H,
    PLAYER_SMALL_H,
    PLAYER_W,
    SHELL_CONTACT_COOLDOWN,
    SHELL_H,
    SHELL_SPEED,
    SQUISH_FRAMES,
    TURTLE_H,
    TURTLE_W,
    WALKER_H,
    WALKER_W,
)


class Entity:
    """Base class: an axis-aligned hitbox with a velocity.

    `alive` turns False when the entity leaves the world; the engine then drops it
    from `Game.entities` at the end of the frame. `facing` is +1 (right) or -1 (left).
    """

    kind: str = "entity"

    def __init__(
        self, x: float, y: float, w: int, h: int, vx: float = 0.0, vy: float = 0.0, facing: int = -1
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.w = w
        self.h = h
        self.vx = float(vx)
        self.vy = float(vy)
        self.facing = facing
        self.alive = True
        self.on_ground = False

    @property
    def right(self) -> float:
        """x of the right edge of the hitbox."""
        return self.x + self.w

    @property
    def bottom(self) -> float:
        """y of the bottom edge (the feet) of the hitbox."""
        return self.y + self.h

    @property
    def center_x(self) -> float:
        """x of the horizontal centre of the hitbox."""
        return self.x + self.w / 2

    def set_height(self, h: int) -> None:
        """Change the hitbox height while keeping the feet where they are."""
        self.y += self.h - h
        self.h = h

    def clone(self) -> Entity:
        """A shallow copy; enough for a deep one because every attribute is a primitive."""
        twin = object.__new__(type(self))
        twin.__dict__.update(self.__dict__)
        return twin

    def __repr__(self) -> str:
        return f"{type(self).__name__}(x={self.x:.2f}, y={self.y:.2f}, vx={self.vx}, vy={self.vy})"


class Player(Entity):
    """The player. `dead` is set when the game ends by death; `big` after a mushroom."""

    kind = "player"

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y, PLAYER_W, PLAYER_SMALL_H, facing=1)
        self.big = False
        self.invuln_frames = 0
        self.dead = False
        # False from a jump until the jump button is released: makes jumps edge-triggered.
        self.jump_armed = True

    def grow(self) -> None:
        """Become big; the feet stay put. The caller checks that there is head room."""
        if not self.big:
            self.big = True
            self.set_height(PLAYER_BIG_H)

    def shrink(self) -> None:
        """Become small (after being hurt); the feet stay put."""
        if self.big:
            self.big = False
            self.set_height(PLAYER_SMALL_H)

    def hurt(self) -> None:
        """Take a hit while big: shrink and become invulnerable for a while."""
        self.shrink()
        self.invuln_frames = INVULN_FRAMES

    def die(self) -> None:
        """Mark the player dead and frozen in place."""
        self.dead = True
        self.alive = False
        self.vx = 0.0
        self.vy = 0.0


class Walker(Entity):
    """Basic ground enemy: walks, turns at walls, falls off ledges, dies when stomped.

    A stomped walker stays in `Game.entities` (still `alive`) as a flat corpse while
    `squished_frames > 0`: check that counter, not `alive`, to know whether it is dangerous.
    """

    kind = "walker"

    def __init__(self, x: float, y: float, facing: int = -1) -> None:
        super().__init__(x, y, WALKER_W, WALKER_H, vx=facing * ENEMY_SPEED, facing=facing)
        self.squished_frames = 0  # > 0: a flat, harmless corpse that disappears at 0

    def squish(self) -> None:
        """Flatten the walker; it stays visible (and harmless) for `SQUISH_FRAMES` frames."""
        self.squished_frames = SQUISH_FRAMES
        self.vx = 0.0


class Turtle(Entity):
    """Shelled enemy. `state` is "walk", "shell" (resting) or "shell_moving" (kicked)."""

    kind = "turtle"

    def __init__(self, x: float, y: float, facing: int = -1) -> None:
        super().__init__(x, y, TURTLE_W, TURTLE_H, vx=facing * ENEMY_SPEED, facing=facing)
        self.state = "walk"
        self.contact_cooldown = 0  # frames during which the player cannot touch the shell

    def to_shell(self) -> None:
        """Retreat into a resting shell (shorter hitbox, feet stay put)."""
        self.state = "shell"
        self.vx = 0.0
        self.set_height(SHELL_H)
        self.contact_cooldown = SHELL_CONTACT_COOLDOWN

    def kick(self, direction: int) -> None:
        """Send the shell sliding towards `direction` (+1 right, -1 left)."""
        self.state = "shell_moving"
        self.facing = direction
        self.vx = direction * SHELL_SPEED
        self.contact_cooldown = SHELL_CONTACT_COOLDOWN

    def stop(self) -> None:
        """Bring a sliding shell to rest."""
        self.state = "shell"
        self.vx = 0.0
        self.contact_cooldown = SHELL_CONTACT_COOLDOWN


class Mushroom(Entity):
    """Power-up that slides along the ground; makes the player big."""

    kind = "mushroom"

    def __init__(self, x: float, y: float, facing: int = 1) -> None:
        super().__init__(x, y, MUSHROOM_W, MUSHROOM_H, vx=facing * MUSHROOM_SPEED, facing=facing)
