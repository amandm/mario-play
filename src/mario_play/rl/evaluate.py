"""Policy evaluation and checkpoint loading.

`evaluate` always builds its *own* environment from the `EnvConfig` - never the
training envs, whose episodes are in full swing - and plays the episodes one
after another with `Algorithm.predict`. Episode `k` is reset with `seed + k`, so
two evaluations with the same seed face exactly the same initial conditions and
their scores are comparable over the course of a training run.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mario_play.envs.factory import make_env
from mario_play.rl.algos import get_algorithm
from mario_play.rl.algos.base import Algorithm
from mario_play.rl.checkpoint import load_checkpoint
from mario_play.rl.config import EnvConfig, TrainConfig, config_from_dict
from mario_play.rl.utils import get_rng_state, resolve_device, set_rng_state

FrameCallback = Callable[[np.ndarray], None]


def as_float(value: Any) -> float:
    """Python float from a Python/numpy scalar or a one-element array (as found in env infos)."""
    return float(np.asarray(value).reshape(-1)[0])


def evaluate(
    algo: Algorithm,
    env_cfg: EnvConfig,
    episodes: int,
    seed: int,
    deterministic: bool = True,
    render_mode: str | None = None,
    frame_callback: FrameCallback | None = None,
    max_steps: int | None = None,
) -> dict[str, float | int]:
    """Play `episodes` full episodes with `algo.predict` and summarise them.

    Returns `mean_return`, `std_return` (population standard deviation),
    `mean_length` and `episodes`. When the env reports them in its info dict - the
    Mario env does - `flag_rate` (share of episodes that reached the flag) and
    `mean_progress` (mean final progress, 0-1) are included as well.

    `render_mode="human"` lets the env show its own window (Gymnasium envs render
    on every step in that mode). `frame_callback` receives one RGB frame after every
    reset and every step; it needs `render_mode="rgb_array"`, which is also what
    `render_mode=None` is turned into when a callback is given. `max_steps` cuts
    off episodes of envs that might never end; `None` trusts the env.

    Only `algo.predict` is used, so evaluation never touches training state. That
    includes the global random number generators: a sampled (`deterministic=False`)
    evaluation runs on its own stream seeded with `seed` and puts the global state
    back afterwards, so its result depends on `seed` alone and a periodic evaluation
    does not change the training run around it.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 or None, got {max_steps}")
    if frame_callback is not None:
        if render_mode is None:
            render_mode = "rgb_array"
        elif render_mode != "rgb_array":
            raise ValueError(
                f"frame_callback needs render_mode='rgb_array' (or None), got {render_mode!r}"
            )

    returns: list[float] = []
    lengths: list[int] = []
    flags: list[float] = []
    progress: list[float] = []

    # Greedy prediction draws no random numbers; only the sampled path needs its own stream.
    rng_state = None if deterministic else get_rng_state()
    if rng_state is not None:
        # Not `set_seed`: that would also reset the run's torch determinism flags.
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)  # every device
    env = make_env(env_cfg, seed=seed, render_mode=render_mode)
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            if frame_callback is not None:
                frame_callback(env.render())
            episode_return, episode_length = 0.0, 0
            while True:
                batch = np.asarray(obs)[None]
                action = int(np.asarray(algo.predict(batch, deterministic=deterministic))[0])
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                if frame_callback is not None:
                    frame_callback(env.render())
                if terminated or truncated:
                    break
                if max_steps is not None and episode_length >= max_steps:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
            if "flag_get" in info:
                flags.append(float(bool(as_float(info["flag_get"]))))
            if "progress" in info:
                progress.append(as_float(info["progress"]))
    finally:
        env.close()
        if rng_state is not None:
            set_rng_state(rng_state)

    result: dict[str, float | int] = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "episodes": int(episodes),
    }
    if flags:
        result["flag_rate"] = float(np.mean(flags))
    if progress:
        result["mean_progress"] = float(np.mean(progress))
    return result


def load_algorithm(
    checkpoint_path: str | Path, device: str | torch.device = "auto"
) -> tuple[Algorithm, TrainConfig]:
    """Rebuild the trained algorithm from a checkpoint alone; returns `(algo, cfg)`.

    The config stored in the checkpoint says which env the policy was trained on;
    a throwaway env built from it provides the observation and action spaces. The
    result is meant for inference (`predict`, `evaluate`): to continue training use
    `Trainer(cfg, resume=...)`. For that reason a DQN is built with a minimal replay
    buffer instead of the (possibly multi-gigabyte) one of the training config;
    the returned `cfg` is the unmodified training config.
    """
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = config_from_dict(payload["config"])
    algo_cls = get_algorithm(payload["algo_name"])

    env = make_env(cfg.env)
    try:
        obs_space, action_space = env.observation_space, env.action_space
    finally:
        env.close()

    build_cfg = copy.deepcopy(cfg)
    build_cfg.dqn.buffer_size = max(1, build_cfg.dqn.batch_size)
    algo = algo_cls(obs_space, action_space, build_cfg, resolve_device(device), cfg.n_envs)
    algo.load_state_dict(payload["algo_state"])
    return algo, cfg
