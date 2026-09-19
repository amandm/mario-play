"""Every tunable of the simulation lives here: units, physics, hitboxes, timers, scores.

World coordinates are pixels (floats), origin top-left, +y down. One `Game.step` is
one frame (1/60 s); velocities are px/frame and accelerations px/frame^2.
"""

from __future__ import annotations

# --- units ---------------------------------------------------------------------
TILE = 16
VIEW_W = 256
VIEW_H = 240
VIEW_TILES_W = 16
LEVEL_H_TILES = 15
LEVEL_H_PX = LEVEL_H_TILES * TILE
FPS = 60

# --- player horizontal movement ------------------------------------------------
WALK_MAX = 1.5
RUN_MAX = 2.5
GROUND_ACCEL_WALK = 0.07
GROUND_ACCEL_RUN = 0.10
RELEASE_DECEL = 0.08  # no direction held, or faster than the current speed cap
SKID_DECEL = 0.18  # direction held against the current velocity
AIR_ACCEL = 0.06

# --- player vertical movement --------------------------------------------------
JUMP_IMPULSE = -5.0
JUMP_IMPULSE_FAST = -5.4
JUMP_FAST_THRESHOLD = 2.0  # |vx| above this takes off with JUMP_IMPULSE_FAST
GRAVITY_JUMP_HELD = 0.20  # while rising with jump held (variable jump height)
GRAVITY = 0.50
MAX_FALL = 4.5
STOMP_BOUNCE = -3.5
STOMP_BOUNCE_HELD = -5.0

# Longest distance moved against the tile grid in one collision pass. Anything
# faster is split into several passes, so nothing tunnels at any speed.
MAX_SUBSTEP = 8.0

# --- hitboxes (w, h); x, y are always the hitbox top-left ------------------------
PLAYER_W = 12
PLAYER_SMALL_H = 15
PLAYER_BIG_H = 30
WALKER_W = 14
WALKER_H = 14
TURTLE_W = 14
TURTLE_H = 22
SHELL_W = 14
SHELL_H = 14
MUSHROOM_W = 14
MUSHROOM_H = 14

# --- enemies and items -----------------------------------------------------------
ENEMY_SPEED = 0.5
SHELL_SPEED = 3.5
MUSHROOM_SPEED = 1.0
SPAWN_ACTIVATION_DISTANCE = 384  # a spawn wakes up once spawn_x < camera_x + this
SQUISH_FRAMES = 30  # how long a flat walker corpse stays around
INVULN_FRAMES = 120  # after a big player is hurt
SHELL_CONTACT_COOLDOWN = 10  # frames a shell ignores the player after a stomp or kick

# --- camera ------------------------------------------------------------------------
CAMERA_PLAYER_OFFSET = 112  # camera_x follows player.x - this, never scrolling left

# --- timer -------------------------------------------------------------------------
DEFAULT_LEVEL_TIME = 400
FRAMES_PER_TIME_UNIT = 24

# --- score ---------------------------------------------------------------------------
SCORE_COIN = 200
SCORE_STOMP = 100
SCORE_SHELL_KILL = 100
SCORE_BRICK = 50
SCORE_MUSHROOM = 1000
SCORE_FLAG = 1000
SCORE_FLAG_PER_TIME = 10
