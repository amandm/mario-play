"""Reward shaping: a weighted sum of what happened during one env step.

The reward of an env step depends only on how far the player moved (pixels of
x) and on the `StepEvents` of its frames, so the return of an episode can be
accounted for exactly: `progress_weight * (x_end - x_start) + time_penalty *
steps + death_penalty * died + flag_bonus * won + coin_bonus * coins +
score_weight * score` (before clipping).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mario_play.game.engine import StepEvents


@dataclass
class RewardConfig:
    """Weights of the reward components; every value is coerced to `float`."""

    progress_weight: float = 1 / 16  # per pixel moved right: +1 per tile (negative to the left)
    time_penalty: float = -0.01  # per env step, whatever the frame skip
    death_penalty: float = -15.0  # once, on the step the player dies (any cause)
    flag_bonus: float = 50.0  # once, on the step the flag is reached
    coin_bonus: float = 0.0  # per coin collected
    score_weight: float = 0.0  # per point of game score gained
    clip: float | None = None  # symmetric clip of the env-step reward to [-clip, clip]

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == "clip" and value is None:
                continue
            setattr(self, field.name, _as_finite_float(field.name, value))
        if self.clip is not None and self.clip <= 0.0:
            raise ValueError(f"reward clip must be positive or None, got {self.clip!r}")


def _as_finite_float(name: str, value: Any) -> float:
    # bool is an int, but `coin_bonus: true` in a config is a mistake, not a 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"reward {name} must be a number, got {value!r}")
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"reward {name} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"reward {name} must be finite, got {value!r}")
    return number


def make_reward_config(reward: RewardConfig | Mapping[str, Any] | None = None) -> RewardConfig:
    """Normalise the `reward` argument of `MarioEnv` into a private `RewardConfig`.

    `None` gives the defaults, a mapping overrides fields by name (unknown names
    and non-numeric values raise `ValueError`), a `RewardConfig` is copied.
    """
    if reward is None:
        return RewardConfig()
    if isinstance(reward, RewardConfig):
        return dataclasses.replace(reward)
    if isinstance(reward, Mapping):
        names = [field.name for field in dataclasses.fields(RewardConfig)]
        unknown = sorted(str(key) for key in reward if key not in names)
        if unknown:
            raise ValueError(f"unknown reward key(s) {unknown}; valid keys: {names}")
        return RewardConfig(**reward)
    raise TypeError(
        f"reward must be a RewardConfig, a mapping or None, got {type(reward).__name__}"
    )


def add_events(total: StepEvents, events: StepEvents) -> None:
    """Accumulate the `events` of one frame into `total` (the events of a whole env step)."""
    total.coins += events.coins
    total.stomps += events.stomps
    total.bricks += events.bricks
    total.powerups += events.powerups
    total.score_delta += events.score_delta
    if events.hurt:
        total.hurt = True
    if events.died:
        total.died = True
        total.death_cause = events.death_cause
    if events.won:
        total.won = True


def compute_reward(cfg: RewardConfig, dx: float, events: StepEvents) -> float:
    """Reward of one env step.

    `dx` is the change of the player's x in pixels over the step and `events` the
    accumulated `StepEvents` of its frames (see `add_events`). The time penalty is
    added once per call; `cfg.clip` bounds the total.
    """
    reward = cfg.progress_weight * dx + cfg.time_penalty
    if events.died:
        reward += cfg.death_penalty
    if events.won:
        reward += cfg.flag_bonus
    if events.coins:
        reward += cfg.coin_bonus * events.coins
    if events.score_delta:
        reward += cfg.score_weight * events.score_delta
    clip = cfg.clip
    if clip is not None:
        reward = max(-clip, min(clip, reward))
    return float(reward)
