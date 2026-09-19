"""Gymnasium environment for the mario-play game.

Importing this package registers `MarioPlay-v0`. The registration uses a string
entry point, so nothing heavy is imported until the env is actually created.
Import concrete names from their modules (`mario_play.envs.mario_env.MarioEnv`,
`mario_play.envs.factory.make_env`, ...).
"""

from gymnasium.envs.registration import register, registry

ENV_ID = "MarioPlay-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="mario_play.envs.mario_env:MarioEnv")
