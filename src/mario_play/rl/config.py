"""Training configuration: dataclasses, YAML loading and dotted CLI overrides.

The dataclass tree below is the single source of truth for every tunable. YAML
files mirror it one-to-one, unknown keys are rejected, and values are coerced to
the annotated field type so that `lr: 1e-4` (a string in YAML 1.1) or
`total_timesteps: 1e7` behave as expected. `key=value` overrides of string fields
keep their text (`run_name=2026-09-19`, `run_name=007`); in a YAML file such a
value needs quotes, as anywhere in YAML.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml

MARIO_ENV_ID = "MarioPlay-v0"


@dataclass
class EnvConfig:
    """Which environment to build and how to wrap it (see `mario_play.envs.factory`)."""

    id: str = MARIO_ENV_ID
    # --- MarioPlay-v0 only -------------------------------------------------
    level: str | list[str] = "1-1"  # a list samples one level per episode
    obs_mode: str = "grid"  # "grid" | "pixels"
    action_set: str = "simple"  # "right_only" | "simple" | "complex"
    frame_skip: int = 4
    frame_stack: int = 1  # 4 is the usual choice for pixels
    grayscale: bool = True  # pixels only
    resize: list[int] | None = field(default_factory=lambda: [84, 84])  # (H, W), pixels only
    stall_steps: int | None = None  # truncate after N env steps without progress
    hud: bool = True
    reward: dict[str, Any] = field(default_factory=dict)  # RewardConfig field overrides
    # --- any env ------------------------------------------------------------
    max_episode_steps: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)  # extra gym.make kwargs (non-Mario ids)


@dataclass
class NetworkConfig:
    """Encoder selection and sizes (see `mario_play.rl.networks`)."""

    encoder: str = "auto"  # "auto" | "cnn" | "grid" | "mlp"
    hidden_size: int = 512  # width of the final feature layer of cnn / grid encoders
    mlp_hidden: list[int] = field(default_factory=lambda: [64, 64])


@dataclass
class PPOConfig:
    lr: float = 2.5e-4
    anneal_lr: bool = True
    n_steps: int = 128  # rollout length per env; batch = n_steps * n_envs
    n_epochs: int = 4
    n_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    target_kl: float | None = None  # stop the update's epochs early above this KL


@dataclass
class DQNConfig:
    lr: float = 1e-4
    buffer_size: int = 100_000  # transitions
    learning_starts: int = 10_000  # env transitions (global_step) before the first update
    batch_size: int = 32
    gamma: float = 0.99
    train_freq: int = 4  # vector steps between updates
    gradient_steps: int = 1  # gradient steps per update
    target_update_interval: int = 10_000  # env transitions between hard target syncs
    tau: float = 1.0  # < 1.0 switches to Polyak averaging on every update
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 250_000  # env transitions over which epsilon decays linearly
    double_q: bool = True
    dueling: bool = False
    max_grad_norm: float = 10.0


@dataclass
class EvalConfig:
    interval: int = 50_000  # env transitions between evaluations; 0 disables periodic eval
    episodes: int = 5
    deterministic: bool = True
    seed: int = 10_000


@dataclass
class TrainConfig:
    algo: str = "ppo"  # "ppo" | "dqn"
    total_timesteps: int = 1_000_000  # env transitions summed over all envs
    n_envs: int = 8
    vec_env: str = "sync"  # "sync" | "subproc"
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    torch_deterministic: bool = False
    run_dir: str = "runs"
    run_name: str | None = None  # default: <algo>_<env>_<timestamp>
    log_interval: int = 10_000  # env transitions
    checkpoint_interval: int = 100_000  # env transitions
    keep_checkpoints: int = 3
    tensorboard: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# --------------------------------------------------------------------------- #
# dict <-> dataclass
# --------------------------------------------------------------------------- #


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """Coerce `value` to the annotated type `tp`, raising ValueError with `path` on mismatch."""
    origin = typing.get_origin(tp)

    if tp is Any:
        return value

    if origin in (Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ValueError(f"{path}: None is not allowed")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, path)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(f"{path}: {value!r} does not match {tp}: {'; '.join(errors)}")

    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a mapping, got {type(value).__name__}")
        return _from_dict(tp, value, path)

    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}: expected a list, got {type(value).__name__}")
        (item_tp,) = typing.get_args(tp) or (Any,)
        return [_coerce(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a mapping, got {type(value).__name__}")
        return dict(value)

    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError(f"{path}: expected a bool, got {value!r}")

    if tp is int:
        if isinstance(value, bool):
            raise ValueError(f"{path}: expected an int, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, (float, str)):
            try:
                as_float = float(value)
            except ValueError:
                raise ValueError(f"{path}: expected an int, got {value!r}") from None
            if as_float.is_integer():
                return int(as_float)
        raise ValueError(f"{path}: expected an int, got {value!r}")

    if tp is float:
        if isinstance(value, bool):
            raise ValueError(f"{path}: expected a float, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"{path}: expected a float, got {value!r}") from None
        raise ValueError(f"{path}: expected a float, got {value!r}")

    if tp is str:
        if isinstance(value, str):
            return value
        raise ValueError(
            f"{path}: expected a string, got {value!r} (quote the value in YAML to make it one)"
        )

    return value


def _from_dict(cls: type, data: dict[str, Any], path: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        where = path or cls.__name__
        raise ValueError(f"{where}: unknown config key(s) {unknown}; valid keys: {sorted(names)}")
    kwargs = {
        key: _coerce(value, hints[key], f"{path}.{key}" if path else key)
        for key, value in data.items()
    }
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """Build a `TrainConfig` from a nested dict (strict keys, type-coerced values)."""
    return _from_dict(TrainConfig, data or {})


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Nested plain-Python dict (safe for YAML and `torch.save(..., weights_only)` checkpoints)."""
    return dataclasses.asdict(cfg)


# --------------------------------------------------------------------------- #
# YAML + overrides
# --------------------------------------------------------------------------- #


def _accepts_str(tp: Any) -> bool:
    """Whether a field annotated `tp` takes a plain string (`str`, `str | None`, `str | list`)."""
    if tp is str:
        return True
    if typing.get_origin(tp) in (Union, types.UnionType):
        return any(arg is str for arg in typing.get_args(tp))
    return False


def _field_type(keys: list[str]) -> Any:
    """Annotation of the config field at the dotted path `keys`; None if it is not a field."""
    tp: Any = TrainConfig
    for key in keys:
        if not dataclasses.is_dataclass(tp):
            return None  # inside an untyped dict (`env.reward`, `env.kwargs`)
        hints = typing.get_type_hints(tp)
        if key not in hints:
            return None  # unknown key: left to the strict check in `_from_dict`
        tp = hints[key]
    return tp


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides to a nested dict and return it.

    Values are parsed as YAML (`env.level=[1-1,1-2]`, `ppo.target_kl=null`);
    numeric strings such as `1e-4` are later coerced by the dataclass field type.
    A scalar for a string field keeps its text instead, so that `run_name=007`,
    `run_name=2026-09-19` or `run_name=on` name a run rather than fail as an int,
    a date or a bool (`null` / `~` still mean None). The result stays plain Python.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.subkey=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = data
        for key in keys[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"override {item!r}: {key!r} is not a section")
            node = child
        value = yaml.safe_load(raw)
        if (
            value is not None
            and not isinstance(value, (str, list, dict))
            and _accepts_str(_field_type(keys))
        ):
            value = raw.strip()
        node[keys[-1]] = value
    return data


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    """Load a YAML config (or defaults when `path` is None) and apply dotted overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: top level of a config must be a mapping")
        data = loaded
    return config_from_dict(apply_overrides(data, overrides))


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    """Write `cfg` as YAML."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
