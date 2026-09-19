"""Build a fully wrapped environment from an `EnvConfig`.

`make_env` is a module-level function on purpose: `functools.partial(make_env, cfg)`
is picklable, which `SubprocVecEnv` needs under the "spawn" start method.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

import mario_play.envs  # noqa: F401 - registers MarioPlay-v0
from mario_play.rl.config import MARIO_ENV_ID, EnvConfig


def make_env(cfg: EnvConfig, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """Create the env described by `cfg`.

    `MarioPlay-v0` gets the observation wrappers selected by the config (pixel
    observations always end up channel-first `(C, H, W)` uint8); any other id goes
    through `gym.make(cfg.id, **cfg.kwargs)`. The result is always wrapped in
    `RecordEpisodeStatistics`, so finished episodes carry `info["episode"]`.
    `seed` seeds the action space only; observation seeding happens in `reset(seed=...)`.
    """
    if cfg.id == MARIO_ENV_ID:
        from mario_play.envs.mario_env import MarioEnv
        from mario_play.envs.wrappers import FrameStack, GrayscaleResize

        env: gym.Env = MarioEnv(
            level=cfg.level,
            obs_mode=cfg.obs_mode,
            action_set=cfg.action_set,
            frame_skip=cfg.frame_skip,
            reward=cfg.reward or None,
            stall_steps=cfg.stall_steps,
            hud=cfg.hud,
            render_mode=render_mode,
        )
        if cfg.max_episode_steps:
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        if cfg.obs_mode == "pixels":
            size = tuple(cfg.resize) if cfg.resize else None
            env = GrayscaleResize(env, size=size, grayscale=cfg.grayscale)
            env = FrameStack(env, max(1, cfg.frame_stack))
        elif cfg.frame_stack > 1:
            env = FrameStack(env, cfg.frame_stack)
    else:
        kwargs = dict(cfg.kwargs)
        if cfg.max_episode_steps:
            kwargs["max_episode_steps"] = cfg.max_episode_steps
        env = gym.make(cfg.id, render_mode=render_mode, **kwargs)

    env = RecordEpisodeStatistics(env)
    if seed is not None:
        env.action_space.seed(seed)
    return env
