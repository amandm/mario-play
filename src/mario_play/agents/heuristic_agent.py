"""`HeuristicAgent`: a handful of reactive rules on the game state - no lookahead, no learning.

Run right; when standing in front of a wall, a pit or an approaching enemy, jump
and keep the button down while rising. It finishes `flat` and gets some way into
the real levels; `SearchAgent` is the baseline that completes them.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from mario_play.envs.actions import button_name
from mario_play.game.constants import LEVEL_H_TILES, TILE
from mario_play.game.engine import Game

WALL_LOOKAHEAD_PX = 28
PIT_LOOKAHEAD_PX = 16
ENEMY_LOOKAHEAD_PX = 44
ENEMY_BAND_PX = 24  # enemies whose feet are this close to the player's feet height count


class HeuristicAgent:
    """Reactive rule-based agent reading `env.unwrapped.game`.

    Works with every bundled action set (it needs "right+run" and "right+run+jump",
    falling back to "right" / "right+jump").
    """

    def __init__(self, env: gym.Env) -> None:
        base = env.unwrapped
        if not hasattr(base, "game") or not hasattr(base, "action_buttons"):
            raise TypeError(f"HeuristicAgent needs a MarioEnv, got {base!r}")
        self._env = base
        index = {button_name(b): i for i, b in reversed(list(enumerate(base.action_buttons)))}
        try:
            self._run = index["right+run"] if "right+run" in index else index["right"]
            self._jump = (
                index["right+run+jump"] if "right+run+jump" in index else index["right+jump"]
            )
        except KeyError:
            raise ValueError("the env's action set has no way to move right and jump") from None

    def reset(self) -> None:
        """Nothing to forget: every decision is made from the current game state."""

    def act(self, obs: Any = None) -> int:
        """The action for the env's live game; `obs` is ignored (the state is read)."""
        game: Game = self._env.game
        player = game.player
        if not player.on_ground:
            # Rising: keep the button down for a full jump. Falling: let go, which also
            # re-arms the (edge-triggered) jump for the next take-off.
            return self._jump if player.vy < 0 else self._run
        if not player.jump_armed:
            return self._run
        if self.wall_ahead(game) or self.pit_ahead(game) or self.enemy_ahead(game):
            return self._jump
        return self._run

    # --- the three senses ------------------------------------------------------------------------

    @staticmethod
    def wall_ahead(game: Game) -> bool:
        """Is a solid tile at body height within `WALL_LOOKAHEAD_PX` of the player's front?"""
        player = game.player
        level = game.level
        front = player.x + player.w
        rows = range(int(player.y // TILE), int((player.y + player.h - 1) // TILE) + 1)
        cols = range(int(front // TILE), int((front + WALL_LOOKAHEAD_PX) // TILE) + 1)
        return any(level.solid_at(col, row) for col in cols for row in rows if row >= 0)

    @staticmethod
    def pit_ahead(game: Game) -> bool:
        """Is there a column without any floor below the feet just ahead of the player?"""
        player = game.player
        level = game.level
        front = player.x + player.w
        feet_row = int((player.y + player.h) // TILE)
        for col in range(int(front // TILE), int((front + PIT_LOOKAHEAD_PX) // TILE) + 1):
            if not any(level.solid_at(col, row) for row in range(feet_row, LEVEL_H_TILES)):
                return True
        return False

    @staticmethod
    def enemy_ahead(game: Game) -> bool:
        """Is a dangerous enemy close in front of the player, at about the same height?"""
        player = game.player
        front = player.x + player.w
        feet = player.y + player.h
        for e in game.entities:
            if not e.alive or e.kind == "mushroom":
                continue
            if e.kind == "walker" and e.squished_frames > 0:
                continue
            if e.kind == "turtle" and e.state == "shell":
                continue  # a resting shell is harmless: running into it kicks it away
            gap = e.x - front
            if -player.w <= gap <= ENEMY_LOOKAHEAD_PX and abs(e.y + e.h - feet) <= ENEMY_BAND_PX:
                return True
        return False
