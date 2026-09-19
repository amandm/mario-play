"""Algorithm registry. Implementations are imported lazily so that importing the
package never pulls in a learner that is not needed."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mario_play.rl.algos.base import Algorithm

ALGORITHMS = ("ppo", "dqn")


def get_algorithm(name: str) -> type[Algorithm]:
    """Return the algorithm class registered under `name` ("ppo" | "dqn")."""
    key = name.lower()
    if key == "ppo":
        from mario_play.rl.algos.ppo import PPO

        return PPO
    if key == "dqn":
        from mario_play.rl.algos.dqn import DQN

        return DQN
    raise ValueError(f"unknown algorithm {name!r}; available: {list(ALGORITHMS)}")
