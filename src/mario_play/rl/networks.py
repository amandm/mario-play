"""Neural networks: observation encoders plus the actor-critic and Q-value heads.

Conventions follow CleanRL / Nature-DQN:

* Encoders take a *batch* of raw observations in the dtype of the observation
  space. uint8 tensors are cast to float32 and scaled by 1/255 inside the
  encoder; every other dtype is cast to float32 and passed through unchanged
  (so a float image is expected to be normalised already).
* Nothing about the observation is hard-coded: channel counts come from
  `obs_space.shape` (frame stacking multiplies them) and flatten sizes from a
  dummy forward pass.
* Weights use orthogonal initialisation: gain sqrt(2) for hidden layers, 0.01
  for the policy head (near-uniform initial policy), 1.0 for value/Q heads.
* Everything is float32 (MPS has no float64).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from mario_play.rl.config import NetworkConfig

LayerT = TypeVar("LayerT", bound=nn.Module)

ENCODERS = ("auto", "cnn", "grid", "mlp")

# Smallest H and W for which the three Nature-DQN convolutions still produce a >= 1x1 map.
NATURE_CNN_MIN_SIZE = 36


def layer_init(layer: LayerT, std: float = math.sqrt(2), bias: float = 0.0) -> LayerT:
    """Orthogonal-initialise `layer.weight` with gain `std`, fill its bias, and return the layer."""
    nn.init.orthogonal_(layer.weight, gain=std)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias)
    return layer


def _as_shape(obs_shape: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in obs_shape)
    if len(shape) == 0 or any(dim <= 0 for dim in shape):
        raise ValueError(f"observation shape must have positive dimensions, got {shape}")
    return shape


def _as_image_shape(obs_shape: Sequence[int], who: str) -> tuple[int, int, int]:
    shape = _as_shape(obs_shape)
    if len(shape) != 3:
        raise ValueError(f"{who} needs channel-first (C, H, W) observations, got shape {shape}")
    channels, height, width = shape
    return channels, height, width


def _check_positive(name: str, value: int) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _prepare_obs(obs: Tensor, obs_shape: tuple[int, ...]) -> Tensor:
    """Validate that `obs` is a batch `(B, *obs_shape)` and return it as float32.

    The batch check matters because `conv2d` and `linear` both accept unbatched
    input: a single observation would otherwise yield features of the wrong rank
    and broadcast silently further down.
    """
    if not isinstance(obs, Tensor):
        raise TypeError(f"observations must be a torch.Tensor, got {type(obs).__name__}")
    got = tuple(obs.shape)
    if got[1:] != obs_shape:  # obs_shape is never empty, so this also rejects a missing batch dim
        expected = f"(B, {', '.join(str(dim) for dim in obs_shape)})"
        hint = " - add a batch dimension with obs.unsqueeze(0)" if got == obs_shape else ""
        raise ValueError(f"expected a batch of observations with shape {expected}, got {got}{hint}")
    if obs.dtype == torch.uint8:
        return obs.to(torch.float32) / 255.0
    return obs.to(torch.float32)


def _flat_dim(module: nn.Module, obs_shape: tuple[int, ...]) -> int:
    """Number of features `module` produces per observation, measured with a dummy forward."""
    with torch.no_grad():
        return int(module(torch.zeros((1, *obs_shape), dtype=torch.float32)).shape[1])


class NatureCNN(nn.Module):
    """Nature-DQN encoder for channel-first images `(C, H, W)` with `H, W >= 36`.

    conv(32, 8x8, stride 4) -> conv(64, 4x4, stride 2) -> conv(64, 3x3, stride 1)
    -> flatten -> linear(hidden_size), ReLU after every layer.
    """

    def __init__(self, obs_shape: Sequence[int], hidden_size: int = 512) -> None:
        super().__init__()
        channels, height, width = _as_image_shape(obs_shape, "NatureCNN")
        if min(height, width) < NATURE_CNN_MIN_SIZE:
            raise ValueError(
                f"NatureCNN needs H and W >= {NATURE_CNN_MIN_SIZE}, got (C, H, W) = "
                f"{(channels, height, width)}; use encoder='grid' for small channel-first "
                "observations"
            )
        self.obs_shape: tuple[int, ...] = (channels, height, width)
        self.feature_dim = _check_positive("hidden_size", hidden_size)
        self.convs = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flatten_dim = _flat_dim(self.convs, self.obs_shape)
        self.head = nn.Sequential(
            layer_init(nn.Linear(self.flatten_dim, self.feature_dim)), nn.ReLU()
        )

    def forward(self, obs: Tensor) -> Tensor:
        """`(B, C, H, W)` uint8 (scaled by 1/255) or float -> `(B, feature_dim)` float32."""
        return self.head(self.convs(_prepare_obs(obs, self.obs_shape)))


class GridEncoder(nn.Module):
    """Small padded conv stack for low-resolution grids such as the `(C, 15, 16)` tile view.

    conv(32, 3x3) -> conv(64, 3x3, stride 2) -> conv(64, 3x3) -> flatten ->
    linear(hidden_size), padding 1 and ReLU everywhere. The first convolution runs
    at full resolution so tile-exact positions survive; the stride-2 layer then
    quarters the flatten size (4096 for a 15x16 grid). Works for any `H, W >= 1`.
    """

    def __init__(self, obs_shape: Sequence[int], hidden_size: int = 512) -> None:
        super().__init__()
        self.obs_shape: tuple[int, ...] = _as_image_shape(obs_shape, "GridEncoder")
        self.feature_dim = _check_positive("hidden_size", hidden_size)
        self.convs = nn.Sequential(
            layer_init(nn.Conv2d(self.obs_shape[0], 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flatten_dim = _flat_dim(self.convs, self.obs_shape)
        self.head = nn.Sequential(
            layer_init(nn.Linear(self.flatten_dim, self.feature_dim)), nn.ReLU()
        )

    def forward(self, obs: Tensor) -> Tensor:
        """`(B, C, H, W)` -> `(B, feature_dim)` float32."""
        return self.head(self.convs(_prepare_obs(obs, self.obs_shape)))


class MLPEncoder(nn.Module):
    """Flatten -> [linear -> activation] per entry of `hidden` (Tanh by default).

    Meant for 1-D vector observations but accepts any shape (it flattens), which
    is what an explicit `encoder="mlp"` on a grid does. With an empty `hidden`
    it is just the flatten.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        hidden: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.obs_shape: tuple[int, ...] = _as_shape(obs_shape)
        widths = [_check_positive("mlp_hidden entries", width) for width in hidden]
        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = math.prod(self.obs_shape)
        for width in widths:
            layers += [layer_init(nn.Linear(in_dim, width)), activation()]
            in_dim = width
        self.net = nn.Sequential(*layers)
        self.feature_dim = in_dim

    def forward(self, obs: Tensor) -> Tensor:
        """`(B, *obs_shape)` -> `(B, feature_dim)` float32."""
        return self.net(_prepare_obs(obs, self.obs_shape))


def _auto_encoder_name(obs_space: gym.spaces.Box) -> str:
    shape, dtype = obs_space.shape, obs_space.dtype
    if len(shape) == 1:
        return "mlp"
    if len(shape) == 3 and dtype == np.uint8:
        return "cnn"
    if len(shape) == 3 and np.issubdtype(dtype, np.floating):
        return "grid"
    raise ValueError(
        f"encoder='auto' supports 1-D vectors, uint8 (C, H, W) images and float (C, H, W) grids; "
        f"got shape {shape} with dtype {dtype}. Wrap the env (e.g. FrameStack gives images a "
        "channel axis) or set network.encoder explicitly"
    )


def build_encoder(obs_space: gym.spaces.Box, cfg: NetworkConfig) -> tuple[nn.Module, int]:
    """Build the observation encoder for `obs_space` and return `(encoder, feature_dim)`.

    `cfg.encoder="auto"` picks `NatureCNN` for uint8 `(C, H, W)` images,
    `GridEncoder` for float `(C, H, W)` grids and `MLPEncoder` for 1-D vectors;
    `"cnn" | "grid" | "mlp"` force a choice. Raises `ValueError` for shapes the
    chosen encoder cannot handle and for unknown encoder names.
    """
    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"observation space must be Box, got {type(obs_space).__name__}")
    shape = _as_shape(obs_space.shape)
    name = str(cfg.encoder).lower()
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {cfg.encoder!r}; available: {list(ENCODERS)}")
    if name == "auto":
        name = _auto_encoder_name(obs_space)

    encoder: nn.Module
    if name == "cnn":
        encoder = NatureCNN(shape, cfg.hidden_size)
    elif name == "grid":
        encoder = GridEncoder(shape, cfg.hidden_size)
    else:
        encoder = MLPEncoder(shape, cfg.mlp_hidden)
    return encoder, encoder.feature_dim


class ActorCritic(nn.Module):
    """Categorical policy and state-value function for PPO.

    Conv encoders (`NatureCNN`, `GridEncoder`) are shared by the policy and value
    heads, as in CleanRL's Atari PPO. With an `MLPEncoder` the critic gets its own
    trunk (`critic_encoder`), as in CleanRL's classic-control PPO, because value
    gradients through a small shared MLP noticeably destabilise the policy.
    """

    def __init__(self, obs_space: gym.spaces.Box, n_actions: int, cfg: NetworkConfig) -> None:
        super().__init__()
        self.n_actions = _check_positive("n_actions", n_actions)
        self.encoder, feature_dim = build_encoder(obs_space, cfg)
        self.critic_encoder: nn.Module | None = None
        if isinstance(self.encoder, MLPEncoder):
            self.critic_encoder, _ = build_encoder(obs_space, cfg)
        self.policy_head = layer_init(nn.Linear(feature_dim, self.n_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(feature_dim, 1), std=1.0)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Policy logits `(B, n_actions)` and state values `(B,)` for a batch of observations."""
        features = self.encoder(obs)
        critic_features = features if self.critic_encoder is None else self.critic_encoder(obs)
        return self.policy_head(features), self.value_head(critic_features).squeeze(-1)

    def get_value(self, obs: Tensor) -> Tensor:
        """State values `(B,)`; skips the policy path."""
        encoder = self.encoder if self.critic_encoder is None else self.critic_encoder
        return self.value_head(encoder(obs)).squeeze(-1)

    def get_action_and_value(
        self, obs: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return `(action, log_prob, entropy, value)`, each of shape `(B,)`.

        With `action=None` an action is sampled from the policy (or its argmax is
        taken when `deterministic`). A given `action` `(B,)` is returned unchanged
        and evaluated under the current policy, which is what the PPO update needs.
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        elif tuple(action.shape) != (logits.shape[0],):
            raise ValueError(
                f"action must have shape ({logits.shape[0]},) to match the observation batch, "
                f"got {tuple(action.shape)}"
            )
        return action, dist.log_prob(action), dist.entropy(), value


class QNetwork(nn.Module):
    """Action-value network for DQN: encoder -> linear Q head, or a dueling head.

    Dueling (Wang et al., 2016): `Q = V + A - mean_a(A)` with separate linear value
    and advantage streams on the shared features.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Box,
        n_actions: int,
        cfg: NetworkConfig,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.n_actions = _check_positive("n_actions", n_actions)
        self.dueling = bool(dueling)
        self.encoder, feature_dim = build_encoder(obs_space, cfg)
        if self.dueling:
            self.value_head = layer_init(nn.Linear(feature_dim, 1), std=1.0)
            self.advantage_head = layer_init(nn.Linear(feature_dim, self.n_actions), std=1.0)
        else:
            self.q_head = layer_init(nn.Linear(feature_dim, self.n_actions), std=1.0)

    def forward(self, obs: Tensor) -> Tensor:
        """Q-values `(B, n_actions)` for a batch of observations."""
        features = self.encoder(obs)
        if not self.dueling:
            return self.q_head(features)
        advantage = self.advantage_head(features)
        return self.value_head(features) + advantage - advantage.mean(dim=1, keepdim=True)
