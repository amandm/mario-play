"""Append frozen Jev assessments to PPO observations without network or reward access."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from mario_play.envs.jev_features import FEATURE_COUNT, load_table
from mario_play.envs.observations import GRID_SHAPE


class JevFeatureWrapper(gym.ObservationWrapper):
    """Add four broadcast probability planes, or matching zero planes for a control.

    Table advice refreshes on reset and after every ``interval`` actions. Between
    refreshes the previous assessments are held; only the raw grid changes. Age
    and refresh counts are diagnostic info, not additional policy inputs. Terminal
    observations follow the same schedule, before any vector-env auto-reset.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: str,
        path: str | None = None,
        expected_sha256: str | None = None,
        interval: int = 1,
    ) -> None:
        super().__init__(env)
        if mode not in ("zeros", "table"):
            raise ValueError("JevFeatureWrapper mode must be zeros or table")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("Jev feature interval must be a positive integer")
        space = env.observation_space
        if (
            not isinstance(space, gym.spaces.Box)
            or space.shape != GRID_SHAPE
            or space.dtype != np.dtype(np.float32)
        ):
            raise ValueError("JevFeatureWrapper requires the unstacked float32 grid space")
        if mode == "table" and (not path or not expected_sha256):
            raise ValueError("table mode requires a local path and expected SHA-256")
        if mode == "zeros" and (path is not None or expected_sha256 is not None):
            raise ValueError("zero features do not load a feature table")
        self.mode = mode
        self.interval = interval
        self.table = load_table(path, expected_sha256.lower()) if mode == "table" else None
        extra_shape = (FEATURE_COUNT, *GRID_SHAPE[1:])
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((space.low, np.zeros(extra_shape, dtype=np.float32))),
            high=np.concatenate((space.high, np.ones(extra_shape, dtype=np.float32))),
            dtype=np.float32,
        )
        self._features: np.ndarray | None = None
        self._steps = 0
        self._age = 0
        self._refreshes = 0
        self._episode_refreshes = 0

    @property
    def jev_features_refreshes(self) -> int:
        """Lifetime table lookups, including advice refreshed by same-step auto-resets."""
        return self._refreshes

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Append scheduled advice without modifying the supplied raw grid."""
        raw = np.asarray(observation)
        if raw.shape != GRID_SHAPE or raw.dtype != np.float32:
            raise ValueError("JevFeatureWrapper received an incompatible raw grid")
        if not np.isfinite(raw).all() or np.any(raw < -1) or np.any(raw > 1):
            raise ValueError("JevFeatureWrapper received non-finite or out-of-range grid values")
        if self._features is None:
            features = (
                np.zeros(FEATURE_COUNT, dtype=np.float32)
                if self.table is None
                else np.asarray(self.table.features(raw), dtype=np.float32)
            )
            if (
                features.shape != (FEATURE_COUNT,)
                or not np.isfinite(features).all()
                or np.any(features < 0)
                or np.any(features > 1)
            ):
                raise ValueError("Jev table returned invalid probability features")
            self._features = features.copy()
            self._features.flags.writeable = False
            if self.table is not None:
                self._refreshes += 1
                self._episode_refreshes += 1
        advice = np.broadcast_to(self._features[:, None, None], (FEATURE_COUNT, *raw.shape[1:]))
        return np.concatenate((raw, advice)).astype(np.float32, copy=False)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Refresh advice for the new episode; lifetime accounting is preserved."""
        raw, info = self.env.reset(**kwargs)
        self._features = None
        self._steps = self._age = self._episode_refreshes = 0
        return self.observation(raw), self._with_info(info)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance once and refresh on schedule, including the episode's final observation."""
        raw, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        if self.table is not None:
            self._age += 1
            if self._steps % self.interval == 0:
                self._features = None
                self._age = 0
        return self.observation(raw), reward, terminated, truncated, self._with_info(info)

    def _with_info(self, info: dict[str, Any]) -> dict[str, Any]:
        return {
            **info,
            "jev_features_mode": self.mode,
            "jev_features_interval": self.interval,
            "jev_features_age": self._age,
            "jev_features_refreshes": self._refreshes,
            "jev_features_episode_refreshes": self._episode_refreshes,
        }
