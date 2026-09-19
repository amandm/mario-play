"""Observation wrappers for the pixel pipeline: `GrayscaleResize` and `FrameStack`.

Both return new arrays on every call, so observations can be stored as they are
(replay buffers, vector-env buffers) without being overwritten later.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

# Pillow widens the filter with the scale factor, so shrinking with BILINEAR averages over
# the source pixels of each target pixel instead of sampling them: no aliasing of small sprites.
_RESAMPLE = Image.Resampling.BILINEAR


class GrayscaleResize(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Turn `(H, W, 3)` uint8 RGB frames into small network inputs.

    Args:
        env: An env whose observation space is a `(H, W, 3)` uint8 `Box`.
        size: Target `(height, width)`; `None` keeps the frame size.
        grayscale: `True` -> `(H, W)` luma (ITU-R 601) frames; `False` -> the RGB
            frame channel-first, `(3, H, W)`.
    """

    def __init__(
        self,
        env: gym.Env,
        size: Sequence[int] | None = (84, 84),
        grayscale: bool = True,
    ) -> None:
        gym.utils.RecordConstructorArgs.__init__(self, size=size, grayscale=grayscale)
        gym.ObservationWrapper.__init__(self, env)
        space = env.observation_space
        if (
            not isinstance(space, gym.spaces.Box)
            or space.dtype != np.uint8
            or len(space.shape) != 3
            or space.shape[2] != 3
        ):
            raise ValueError(
                f"GrayscaleResize needs (H, W, 3) uint8 image observations, got {space}"
            )
        self.size: tuple[int, int] | None = _checked_size(size)
        self.grayscale = bool(grayscale)
        height, width = self.size if self.size is not None else space.shape[:2]
        # `resize` is skipped when it would not change anything.
        self._resize_to: tuple[int, int] | None = (
            None if (height, width) == tuple(space.shape[:2]) else (width, height)  # PIL: (W, H)
        )
        shape = (height, width) if self.grayscale else (3, height, width)
        self.observation_space = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Convert one frame; the result is a new, writable, C-contiguous uint8 array."""
        image = Image.fromarray(observation)
        if self.grayscale:
            image = image.convert("L")  # before resizing: one channel is a third of the work
        if self._resize_to is not None:
            image = image.resize(self._resize_to, _RESAMPLE)
        frame = np.array(image, dtype=np.uint8)
        if self.grayscale:
            return frame
        return np.ascontiguousarray(frame.transpose(2, 0, 1))


class FrameStack(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Stack the last `k` observations along axis 0, oldest first.

    2-D observations `(H, W)` are stacked on a new leading axis -> `(k, H, W)`;
    channel-first 3-D observations `(C, H, W)` are concatenated -> `(k * C, H, W)`.
    After `reset` the stack holds the first observation `k` times. Dtype and bounds
    of the wrapped space are kept.
    """

    def __init__(self, env: gym.Env, k: int) -> None:
        gym.utils.RecordConstructorArgs.__init__(self, k=k)
        gym.Wrapper.__init__(self, env)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"FrameStack needs a positive integer k, got {k!r}")
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) not in (2, 3):
            raise ValueError(f"FrameStack needs (H, W) or (C, H, W) Box observations, got {space}")
        self.k = int(k)
        low, high = space.low, space.high
        if len(space.shape) == 2:
            low, high = low[None], high[None]
        self._channels = low.shape[0]  # channels per frame in the stack
        reps = (self.k, 1, 1)
        self.observation_space = gym.spaces.Box(
            np.tile(low, reps), np.tile(high, reps), dtype=space.dtype
        )
        self._stack = np.zeros(self.observation_space.shape, dtype=space.dtype)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the env and fill the whole stack with its first observation."""
        observation, info = self.env.reset(seed=seed, options=options)
        frame = np.asarray(observation).reshape(self._channels, *self._stack.shape[1:])
        self._stack[:] = np.tile(frame, (self.k, 1, 1))
        return self._stack.copy(), info

    def step(self, action: Any) -> tuple[np.ndarray, Any, bool, bool, dict[str, Any]]:
        """Step the env, push its observation onto the stack and return a copy of the stack."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        channels = self._channels
        stack = self._stack
        if self.k > 1:
            stack[:-channels] = stack[channels:]
        stack[-channels:] = np.asarray(observation).reshape(channels, *stack.shape[1:])
        return stack.copy(), reward, terminated, truncated, info


def _checked_size(size: Sequence[int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    ok = (
        isinstance(size, Sequence)
        and not isinstance(size, (str, bytes))
        and len(size) == 2
        and all(isinstance(n, (int, np.integer)) and not isinstance(n, bool) for n in size)
        and all(n >= 1 for n in size)
    )
    if not ok:
        raise ValueError(
            f"size must be (height, width) with positive integers or None, got {size!r}"
        )
    return int(size[0]), int(size[1])
