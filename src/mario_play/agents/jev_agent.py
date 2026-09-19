"""A bounded, hosted Jev policy over the same grid observations used by PPO.

This controller performs inference only. It neither trains Jev nor substitutes
actions into a PPO rollout. The environment must wait for ``act`` before stepping.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from mario_play.envs.actions import action_names
from mario_play.envs.observations import GRID_PLANES, GRID_SHAPE, PLAYER_COLUMN, VX_SCALE, VY_SCALE
from mario_play.game.constants import TILE

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
ACTION_NAMES = tuple(action_names("simple"))
ACTION_CRITERIA = {
    "noop": "Release all buttons; coast or wait, and release jump.",
    "right": "Move right at walking speed; release jump.",
    "right+jump": "Move right at walking speed and press or hold jump.",
    "right+run": "Move right at running speed; release jump.",
    "right+run+jump": "Move right at running speed and press or hold jump.",
    "jump": "Press or hold jump without a horizontal direction.",
    "left": "Move left at walking speed; release jump.",
}
DECISION_INSTRUCTIONS = (
    "Choose the next controller action in this side-scrolling platform game. "
    "Reach the flag to the right while avoiding pits and side contact with enemies. "
    "Use semantic_local_facts to assess current nearby hazards before choosing movement. "
    "Side contact with an enemy is dangerous; passing above it avoids side contact. "
    "Tile bounds can touch or overlap without exact hitbox contact; treat their stated "
    "distance uncertainty explicitly. No future collision or action outcome is supplied. "
    "The action is held for action_frames simulation frames. The game pauses during inference. "
    "All geometry comes from the current observed tile window, not a future simulation. "
    "Rectangles are inclusive occupied tile bounds [left,top,right,bottom]; x increases "
    "rightward, y downward. Player left-edge tile is x=0. One tile is 16 pixels. "
    "Occupied tiles bound hitboxes approximately; exact vertical sub-tile position is unknown. "
    "solid blocks motion; breakable is also solid; empty columns without support are pits. "
    "Negative vertical velocity means rising. Press jump while grounded to take off; "
    "holding jump while rising extends the jump. Jump must be released between takeoffs. "
    "Horizontal momentum persists. Running is faster than walking. Use geometry, motion, "
    "and recent button history to select the action that advances safely. "
    "Coins and mushrooms are optional; reaching the flag without dying is the objective."
)


class JevError(RuntimeError):
    """A sanitized failure: no API response body, request headers, or key is exposed."""


class JevBudgetExceeded(JevError):
    """The configured lifetime request cap has been reached."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a credential to another endpoint, including HTTP redirects.
        return None


def _rectangles(plane: np.ndarray) -> list[list[int]]:
    """Encode occupied cells exactly as inclusive rectangles, with x relative to player."""
    finished: list[list[int]] = []
    active: dict[tuple[int, int], list[int]] = {}
    for row, values in enumerate(plane):
        cols = np.flatnonzero(values > 0.5).tolist()
        spans: list[tuple[int, int]] = []
        for col in cols:
            if spans and col == spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], col)
            else:
                spans.append((col, col))
        current: dict[tuple[int, int], list[int]] = {}
        for span in spans:
            rectangle = active.pop(
                span, [span[0] - PLAYER_COLUMN, row, span[1] - PLAYER_COLUMN, row]
            )
            rectangle[3] = row
            current[span] = rectangle
        finished.extend(active.values())
        active = current
    finished.extend(active.values())
    return sorted(finished, key=lambda box: (box[1], box[0], box[3], box[2]))


def semantic_state(obs: np.ndarray, *, frame_skip: int = 4) -> dict[str, Any]:
    """Losslessly encode grid tile occupancy and its scalar planes as semantic JSON.

    No environment, reward, absolute world position, or simulator is consulted.
    Velocity retains the observation's clipping; it is not a hidden true velocity.
    """
    grid = np.asarray(obs)
    if grid.shape != GRID_SHAPE or not np.issubdtype(grid.dtype, np.number):
        raise ValueError(f"JevAgent requires a numeric grid observation of shape {GRID_SHAPE}")
    if not np.isfinite(grid).all() or np.any(grid < -1) or np.any(grid > 1):
        raise ValueError("grid observation must contain finite values in [-1, 1]")
    if not np.isin(grid[:8], [0, 1]).all():
        raise ValueError("grid occupancy planes must be binary")
    scalars = grid[8:, 0, 0]
    if not np.all(grid[8:] == scalars[:, None, None]):
        raise ValueError("grid scalar planes must be constant")
    if not np.any(grid[7]):
        raise ValueError("grid observation has no visible player")
    state = {
        "action_frames": _positive_int(frame_skip, "frame_skip"),
        "window": {
            "left": -PLAYER_COLUMN,
            "right": GRID_SHAPE[2] - PLAYER_COLUMN - 1,
            "top": 0,
            "bottom": GRID_SHAPE[1] - 1,
        },
        "occupied_tile_rectangles": {
            name: _rectangles(grid[index]) for index, name in enumerate(GRID_PLANES[:8])
        },
        "player": {
            "x_offset_pixels": float(scalars[4]) * TILE,
            "vx_pixels_per_frame": float(scalars[0]) * VX_SCALE,
            "vy_pixels_per_frame_clipped": float(scalars[1]) * VY_SCALE,
            "on_ground": bool(scalars[2] > 0.5),
            "big": bool(scalars[3] > 0.5),
        },
        "time_fraction_remaining": float(scalars[5]),
    }
    state["semantic_local_facts"] = _local_hazard_facts(grid, state)
    return state


def _local_hazard_facts(grid: np.ndarray, state: dict[str, Any]) -> dict[str, Any]:
    """Describe observed spatial relations, without trajectories or action recommendations."""
    player_rows, player_cols = np.nonzero(grid[7])
    top, bottom = int(player_rows.min()), int(player_rows.max())
    left, right = int(player_cols.min()), int(player_cols.max())
    relative_right = right - PLAYER_COLUMN
    enemies = [
        box for box in state["occupied_tile_rectangles"]["enemy"] if box[2] >= left - PLAYER_COLUMN
    ]
    enemies.sort(key=lambda box: (max(0, box[0] - relative_right), abs(box[1] - top)))
    nearest = None
    if enemies:
        box = enemies[0]
        gap = max(0, box[0] - relative_right - 1)
        overlaps_rows = box[1] <= bottom and box[3] >= top
        columns_relation = (
            "overlapping occupied columns"
            if box[0] <= relative_right
            else "adjacent occupied columns"
            if gap == 0
            else f"{gap} empty tile columns between occupied bounds"
        )
        vertical = (
            "at the player's body height"
            if overlaps_rows
            else "above the player's body"
            if box[3] < top
            else "below the player's body"
        )
        nearest = {
            "description": f"An enemy is ahead or horizontally overlapping, {vertical}, "
            f"with {columns_relation}.",
            "occupied_tile_bounds": box,
            "body_rows_overlap": overlaps_rows,
            "empty_tile_columns_between": gap,
            "occupied_columns_relation": columns_relation,
        }
    body_solids = np.flatnonzero(np.any(grid[0, top : bottom + 1], axis=0))
    body_solids_ahead = [int(col) for col in body_solids if col > right]
    ahead = list(range(right + 1, grid.shape[2]))
    unsupported = [col for col in ahead if not np.any(grid[0, bottom + 1 :, col])]
    player = state["player"]
    vx, vy = player["vx_pixels_per_frame"], player["vy_pixels_per_frame_clipped"]
    horizontal = "moving right" if vx > 0 else "moving left" if vx < 0 else "horizontally still"
    vertical = "rising" if vy < 0 else "falling" if vy > 0 else "no vertical motion"
    grounded = "grounded" if player["on_ground"] else "airborne"
    return {
        "player_motion": f"The player is {grounded}, {horizontal}, with {vertical}.",
        "nearest_forward_enemy": nearest,
        "empty_columns_to_solid_at_body_height": (
            body_solids_ahead[0] - right - 1 if body_solids_ahead else None
        ),
        "next_column_ahead_has_solid_below_body": (
            bool(np.any(grid[0, bottom + 1 :, ahead[0]])) if ahead else None
        ),
        "first_visible_column_ahead_without_solid_below_body": (
            unsupported[0] - PLAYER_COLUMN if unsupported else None
        ),
        "distance_precision": "Observed tile bounds only. Exact enemy hitbox gap, "
        "enemy velocity, landing position and collision time are unknown. "
        "Solid below body means visible support, not a predicted landing.",
    }


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError("Jev returned an invalid probability or confidence")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise JevError("Jev returned an invalid probability or confidence")
    return float(value)


class JevAgent:
    """Choose one of the seven ``simple`` actions using one Jev Choice per step.

    ``max_calls`` counts attempted HTTP requests, including failures, across resets.
    There are no automatic retries or replacement actions. Callers must independently
    verify account allowance before invoking this client; a call cap is not billing control.
    """

    def __init__(
        self,
        api_key: str,
        *,
        max_calls: int = 400,
        timeout: float = 20.0,
        frame_skip: int = 4,
        model: str = MODEL,
    ) -> None:
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
            or "\n" in api_key
            or "\r" in api_key
        ):
            raise ValueError("a non-empty TypeSafe API key is required")
        self.max_calls = _positive_int(max_calls, "max_calls")
        self.frame_skip = _positive_int(frame_skip, "frame_skip")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if model != MODEL:
            raise ValueError(f"this controller requires the pinned model {MODEL}")
        self.timeout = float(timeout)
        self.model = model
        self.calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.usage_missing_calls = 0
        self.last_decision: dict[str, Any] | None = None
        self._history: deque[str] = deque(maxlen=4)
        self._api_key = api_key.strip()
        self._opener = build_opener(_NoRedirects())
        self._closed = False

    def reset(self) -> None:
        """Forget episode history, preserving lifetime request and token accounting."""
        self._history.clear()
        self.last_decision = None

    def close(self) -> None:
        """Discard the credential and prevent further requests; safe to repeat."""
        self._api_key = ""
        self._closed = True

    def build_request(self, obs: np.ndarray) -> dict[str, Any]:
        """Build a credential-free payload, useful for inspecting the controller prompt."""
        state = semantic_state(obs, frame_skip=self.frame_skip)
        state["recent_actions_oldest_first"] = list(self._history)
        state["semantic_local_facts"]["previous_jump_button"] = (
            "unknown at episode start"
            if not self._history
            else "held during the previous action"
            if "jump" in self._history[-1]
            else "released during the previous action"
        )
        return {
            "model": self.model,
            "state": state,
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": DECISION_INSTRUCTIONS,
                    "criteria": dict(ACTION_CRITERIA),
                }
            },
        }

    def act(self, obs: np.ndarray) -> int:
        """Request and validate a decision; fail closed on budgets or invalid responses."""
        self.last_decision = None
        if self._closed:
            raise JevError("JevAgent is closed")
        if self.calls >= self.max_calls:
            raise JevBudgetExceeded("Jev request cap reached; no additional request was sent")
        payload = self.build_request(obs)
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.calls += 1
        started = time.perf_counter()
        try:
            response = self._post(body)
        except JevError:
            self.usage_missing_calls += 1
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = self._account_usage(response)
        try:
            if response["model"] != self.model:
                raise JevError("Jev response did not confirm the requested model version")
            answer = response["answers"]["action"]
            if answer["type"] != "choice":
                raise JevError("Jev returned a non-Choice answer")
            choice = answer["choice"]
            probabilities = answer["probabilities"]
            if not isinstance(choice, str) or choice not in ACTION_NAMES:
                raise JevError("Jev returned an action outside the seven-action space")
            if not isinstance(probabilities, dict) or set(probabilities) != set(ACTION_NAMES):
                raise JevError("Jev returned an incomplete or unknown action distribution")
            probabilities = {name: _probability(probabilities[name]) for name in ACTION_NAMES}
            # The API can round each of seven probabilities to two decimal places.
            if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.04):
                raise JevError("Jev action probabilities do not sum to one")
            if probabilities[choice] + 1e-6 < max(probabilities.values()):
                raise JevError("Jev choice disagrees with its highest-probability action")
            confidence = _probability(answer["confidence"])
        except (KeyError, TypeError, AttributeError):
            raise JevError("Jev returned a malformed Choice response") from None
        action = ACTION_NAMES.index(choice)
        self.last_decision = {
            "model": self.model,
            "state": payload["state"],
            "choice": choice,
            "action": action,
            "probabilities": probabilities,
            "probabilities_sum": sum(probabilities.values()),
            "confidence": confidence,
            "latency_ms": elapsed_ms,
            "usage": usage,
            "call_number": self.calls,
        }
        self._history.append(choice)
        return action

    def _account_usage(self, response: dict[str, Any]) -> dict[str, int]:
        usage = response.get("usage")
        if not isinstance(usage, dict) or any(
            isinstance(usage.get(key), bool)
            or not isinstance(usage.get(key), int)
            or usage[key] < 0
            for key in self.usage
        ):
            self.usage_missing_calls += 1
            raise JevError("Jev response omitted valid token usage; stopped accounting")
        observed = {key: usage[key] for key in self.usage}
        for key, count in observed.items():
            self.usage[key] += count
        return observed

    def _post(self, body: bytes) -> dict[str, Any]:
        request = Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(1_048_577)
        except HTTPError as error:
            status = error.code
            error.close()
            raise JevError(f"TypeSafe HTTP {status}; request stopped without retry") from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise JevError("TypeSafe connection failed; request stopped without retry") from None
        if len(raw) > 1_048_576:
            raise JevError("TypeSafe response exceeded the size limit")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError):
            raise JevError("TypeSafe returned invalid JSON") from None
        if not isinstance(result, dict):
            raise JevError("TypeSafe returned a non-object response")
        return result
