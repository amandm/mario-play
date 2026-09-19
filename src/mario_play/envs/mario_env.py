"""`MarioEnv`: the Gymnasium environment around `mario_play.game.engine.Game`.

Registered as `MarioPlay-v0` by `import mario_play.envs`. Observation wrappers and
step limits are added by `mario_play.envs.factory.make_env`; this class is the
bare env: one discrete action -> `frame_skip` game frames -> observation, shaped
reward, termination (death or flag) and optional stall truncation.
"""

from __future__ import annotations

import operator
import os
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from mario_play.envs.actions import get_action_set
from mario_play.envs.observations import (
    grid_observation,
    grid_observation_space,
    pixel_observation_space,
)
from mario_play.envs.rewards import RewardConfig, add_events, compute_reward, make_reward_config
from mario_play.game.constants import FPS, TILE
from mario_play.game.engine import Buttons, Game, StepEvents
from mario_play.game.level import Level, load_level
from mario_play.game.renderer import Renderer

OBS_MODES: tuple[str, ...] = ("pixels", "grid")
HUMAN_WINDOW_SCALE = 3
_MAX_GAME_SEED = 2**31 - 1


class MarioEnv(gym.Env):
    """One level (or a list of levels, one drawn per episode) as a Gymnasium env.

    Args:
        level: Bundled level name or level file path, or a sequence of them; with
            more than one, every `reset` draws a level with `self.np_random`.
        obs_mode: `"pixels"` - the `(240, 256, 3)` uint8 RGB frame a human sees;
            `"grid"` - the `(14, 15, 16)` float32 planes of `grid_observation`.
        action_set: Name of one of `mario_play.envs.actions.ACTION_SETS`.
        frame_skip: Game frames per env step; the action is held for all of them.
        reward: `RewardConfig`, a dict of field overrides, or `None` for the defaults.
        stall_steps: Truncate the episode after this many consecutive env steps
            without a new maximum x; `None` disables it.
        hud: Draw the HUD into frames (pixel observations and `render`).
        render_mode: `None`, `"rgb_array"` (`render()` returns the frame) or
            `"human"` (every reset/step is shown in a pygame window in real time;
            P pauses, and Esc or closing the window raises `KeyboardInterrupt`,
            like Ctrl-C would).

    Public attributes: `game` (the live `Game`; replaced on `reset` when several
    levels are configured), `action_buttons` (`Buttons` per action index),
    `frame_skip`, `obs_mode`, `reward_config`, `stall_steps`, `level_names`.

    `info` (from `reset` and `step`): `x_pos`, `max_x` (furthest x at the end of
    any step of this episode), `progress` (`max_x` as a 0-1 fraction of the way from
    the start to the flag; exactly 1.0 with the flag), `coins`, `score`,
    `time_left`, `flag_get`, `death_cause` (`"pit"`, `"enemy"`, `"timeout"` or
    `None`) and `level`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        level: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] = "1-1",
        obs_mode: str = "pixels",
        action_set: str = "simple",
        frame_skip: int = 4,
        reward: RewardConfig | dict | None = None,
        stall_steps: int | None = None,
        hud: bool = True,
        render_mode: str | None = None,
    ) -> None:
        if obs_mode not in OBS_MODES:
            raise ValueError(f"unknown obs_mode {obs_mode!r}; choose one of {', '.join(OBS_MODES)}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"unknown render_mode {render_mode!r}; choose None or one of "
                f"{', '.join(self.metadata['render_modes'])}"
            )
        self.frame_skip = _positive_int("frame_skip", frame_skip)
        self.stall_steps = (
            None if stall_steps is None else _positive_int("stall_steps", stall_steps)
        )
        self.obs_mode = obs_mode
        self.hud = bool(hud)
        self.render_mode = render_mode
        self.action_buttons: list[Buttons] = get_action_set(action_set)
        self.reward_config: RewardConfig = make_reward_config(reward)

        # Parsed once; every episode plays on a copy (`Game` never touches its source level).
        self._levels: dict[str, Level] = {}
        self._level_keys: list[str] = _level_keys(level)
        for key in self._level_keys:
            self._load(key)
        self.level_names: list[str] = [self._levels[key].name for key in self._level_keys]

        self.action_space = gym.spaces.Discrete(len(self.action_buttons))
        self.observation_space = (
            pixel_observation_space() if obs_mode == "pixels" else grid_observation_space()
        )
        if self.frame_skip != 4:  # keep `render_fps` honest: real time is 60 game frames a second
            fps = max(1, round(FPS / self.frame_skip))
            self.metadata = {**self.metadata, "render_fps": fps}

        self._game_key = self._level_keys[0]
        self.game: Game = Game(self._levels[self._game_key])
        self._renderer: Renderer | None = None
        self._window: Any = None
        self._needs_reset = True
        self._warned_after_end = False
        self._start_x = self.game.player.x
        self._max_x = self._start_x
        self._stalled = 0

    # --- Gymnasium API -------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode. `options={"level": name_or_path}` forces this episode's level."""
        super().reset(seed=seed)
        forced = options.get("level") if isinstance(options, Mapping) else None
        if forced is not None:
            (key,) = _level_keys(forced)
            self._load(key)
        elif len(self._level_keys) > 1:
            key = self._level_keys[int(self.np_random.integers(len(self._level_keys)))]
        else:
            key = self._level_keys[0]
        # The engine's own RNG hangs off the env's, so one seed pins down everything.
        game_seed = int(self.np_random.integers(_MAX_GAME_SEED))
        if key == self._game_key:
            self.game.reset(seed=game_seed)
        else:
            self.game = Game(self._levels[key], seed=game_seed)
            self._game_key = key

        self._start_x = self._max_x = self.game.player.x
        self._stalled = 0
        self._needs_reset = False
        self._warned_after_end = False
        if self.render_mode == "human":
            self._show()
        return self.observe(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Hold `action` for `frame_skip` frames (fewer when the game ends) and sum the reward."""
        if self._needs_reset:
            raise gym.error.ResetNeeded("call reset() before the first step()")
        buttons = self.action_buttons[self._action_index(action)]
        game = self.game
        if game.over:
            if not self._warned_after_end:
                self._warned_after_end = True
                gym.logger.warn(
                    "step() was called although the episode has ended; call reset() first. "
                    "Further steps change nothing and give no reward."
                )
            return self.observe(), 0.0, True, False, self._info()

        x_before = game.player.x
        events = StepEvents()
        for _ in range(self.frame_skip):
            add_events(events, game.step(buttons))
            if game.over:
                break
        x = game.player.x
        reward = compute_reward(self.reward_config, x - x_before, events)

        if x > self._max_x:
            self._max_x = x
            self._stalled = 0
        else:
            self._stalled += 1
        terminated = game.over
        truncated = (
            not terminated and self.stall_steps is not None and self._stalled >= self.stall_steps
        )
        if self.render_mode == "human":
            self._show()
        return self.observe(), reward, terminated, truncated, self._info()

    def render(self) -> np.ndarray | None:
        """`rgb_array`: a new `(240, 256, 3)` uint8 frame. `human`: updates the window, `None`."""
        if self.render_mode == "rgb_array":
            return self.render_frame()
        if self.render_mode == "human":
            self._show()
            return None
        gym.logger.warn(
            "render() does nothing without a render_mode; "
            'create the env with render_mode="rgb_array" or "human".'
        )
        return None

    def close(self) -> None:
        """Close the human window, if one is open. Safe to call more than once."""
        window, self._window = self._window, None
        if window is not None:
            window.close()

    # --- extras for agents, tools and tests ------------------------------------------------------

    def observe(self) -> np.ndarray:
        """The observation of the current game state (a new array on every call)."""
        if self.obs_mode == "grid":
            return grid_observation(self.game)
        return self.render_frame()

    def render_frame(self) -> np.ndarray:
        """The current `(240, 256, 3)` uint8 RGB frame, whatever the render mode."""
        if self._renderer is None:  # lazily: grid-mode training never pays for a renderer
            self._renderer = Renderer(hud=self.hud)
        return self._renderer.render(self.game)

    # --- internals -------------------------------------------------------------------------------

    def _load(self, key: str) -> None:
        if key not in self._levels:
            self._levels[key] = load_level(key)

    def _action_index(self, action: Any) -> int:
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.reshape(-1)[0]
        try:
            index = operator.index(action)
        except TypeError:
            raise ValueError(f"action must be an integer, got {action!r}") from None
        if not 0 <= index < len(self.action_buttons):
            raise ValueError(f"action {index} is out of range 0..{len(self.action_buttons) - 1}")
        return index

    def _info(self) -> dict[str, Any]:
        game = self.game
        player = game.player
        if game.won:
            progress = 1.0
        else:
            # The flag is touched as soon as the hitbox reaches into its column.
            flag_x = game.level.flag_col * TILE - player.w
            span = flag_x - self._start_x
            progress = (self._max_x - self._start_x) / span if span > 0 else 0.0
            progress = max(0.0, min(1.0, progress))
        return {
            "x_pos": float(player.x),
            "max_x": float(self._max_x),
            "progress": float(progress),
            "coins": int(game.coins),
            "score": int(game.score),
            "time_left": int(game.time_left),
            "flag_get": bool(game.won),
            "death_cause": game.death_cause,
            "level": game.level.name,
        }

    def _show(self) -> None:
        """Put the current frame into the human window (opened on first use) and keep real time."""
        if self._window is None:
            from mario_play.game.human import Window  # lazy: only this mode needs pygame

            self._window = Window(scale=HUMAN_WINDOW_SCALE, title=f"mario-play - {self._title()}")
        window = self._window
        fps = int(self.metadata["render_fps"])
        window.show(self.render_frame())
        inputs = window.poll()
        if inputs.get("pause") and not inputs.get("quit"):
            # P freezes the show (the env simply does not return) until P or quit.
            while True:
                window.tick(fps)
                inputs = window.poll()
                if inputs.get("pause") or inputs.get("quit"):
                    break
        if inputs.get("quit"):
            self.close()
            raise KeyboardInterrupt("the mario-play window was closed")
        window.tick(fps)

    def _title(self) -> str:
        return ", ".join(self.level_names)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _level_keys(level: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]) -> list[str]:
    """The `level` argument as a non-empty list of level names / file paths."""
    if isinstance(level, (str, os.PathLike)):
        entries: list[Any] = [level]
    elif isinstance(level, Sequence) and not isinstance(level, bytes):
        entries = list(level)
    else:
        raise ValueError(f"level must be a name, a path or a sequence of them, got {level!r}")
    if not entries:
        raise ValueError("level must name at least one level")
    keys: list[str] = []
    for entry in entries:
        if not isinstance(entry, (str, os.PathLike)) or not os.fspath(entry):
            raise ValueError(f"level names must be non-empty strings or paths, got {entry!r}")
        keys.append(os.fspath(entry))
    return keys
