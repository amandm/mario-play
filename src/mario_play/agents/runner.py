"""Run one episode of an agent in an env and summarise it."""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym


class Agent(Protocol):
    """What every baseline agent offers."""

    def act(self, obs: Any) -> int:
        """The action index for the env's current state."""
        ...

    def reset(self) -> None:
        """Forget everything about the previous episode."""
        ...


def run_episode(
    env: gym.Env, agent: Agent, max_steps: int | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Play one episode of `agent` in `env` (reset with `seed`) and return its summary.

    The episode ends when the env terminates or truncates, or after `max_steps`
    env steps. Keys: `return` (sum of rewards), `length` (env steps), `flag_get`,
    `progress` and `death_cause` (the last three from the env's final `info`;
    `False`, `0.0` and `None` for envs that do not report them).
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError(f"max_steps must be >= 0 or None, got {max_steps!r}")
    obs, info = env.reset(seed=seed)
    agent.reset()
    total = 0.0
    length = 0
    while max_steps is None or length < max_steps:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
        total += float(reward)
        length += 1
        if terminated or truncated:
            break
    return {
        "return": total,
        "length": length,
        "flag_get": bool(info.get("flag_get", False)),
        "progress": float(info.get("progress", 0.0)),
        "death_cause": info.get("death_cause"),
    }
