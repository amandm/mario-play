"""Small training utilities: seeding, device selection, RNG state capture, schedules."""

from __future__ import annotations

import random
import re
from typing import Any

import numpy as np
import torch

_NUMPY_BIT_GENERATOR = "MT19937"  # what numpy's global (legacy) RandomState uses


def set_seed(seed: int, torch_deterministic: bool = False) -> None:
    """Seed python, numpy and torch (all devices) and configure torch determinism.

    With `torch_deterministic` torch is asked for deterministic kernels; ops that
    have none only warn instead of raising, so training never dies over it.
    Environments are seeded separately through `VecEnv.reset(seed)`.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)  # numpy's legacy seeding only accepts 32 bits
    torch.manual_seed(seed)  # covers CPU, CUDA and MPS generators
    torch.backends.cudnn.deterministic = torch_deterministic
    if torch_deterministic:
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(torch_deterministic, warn_only=True)


def resolve_device(name: str | torch.device = "auto") -> torch.device:
    """Turn a config string into a usable `torch.device`.

    `"auto"` picks CUDA, else MPS, else CPU. An explicit `"cuda"`, `"cuda:N"` or
    `"mps"` raises `RuntimeError` when that device is not usable on this machine,
    rather than silently training on something slower; anything else raises
    `ValueError`.
    """
    text = str(name).strip().lower()
    if text == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if text == "cpu":
        return torch.device("cpu")
    if text == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "device 'mps' was requested but MPS is not available in this torch build / "
                "on this machine; use device=cpu or device=auto"
            )
        return torch.device("mps")
    match = re.fullmatch(r"cuda(?::(\d+))?", text)
    if match:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {text!r} was requested but CUDA is not available; "
                "use device=cpu or device=auto"
            )
        if match.group(1) is None:
            return torch.device("cuda")
        index = int(match.group(1))
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"device {text!r} was requested but only {torch.cuda.device_count()} "
                "CUDA device(s) are visible"
            )
        return torch.device("cuda", index)
    raise ValueError(f"unknown device {name!r}; expected auto, cpu, cuda, cuda:<index> or mps")


def get_rng_state() -> dict[str, Any]:
    """Snapshot the global python, numpy and torch RNGs for a checkpoint.

    The result holds only tensors and plain Python values, so it survives
    `torch.load(weights_only=True)`: python's state becomes nested lists, numpy's
    MT19937 key an int64 tensor, torch states stay ByteTensors. Accelerator states
    are included only when they exist *and* matter: CUDA when it is available and
    already initialised (capturing would otherwise create a CUDA context in a
    CPU-only run), MPS when it is available.
    """
    version, internal, gauss_next = random.getstate()
    np_state = np.random.get_state(legacy=False)
    if np_state["bit_generator"] != _NUMPY_BIT_GENERATOR:  # pragma: no cover - numpy default
        raise RuntimeError(f"unexpected numpy bit generator {np_state['bit_generator']!r}")
    state: dict[str, Any] = {
        "python": [int(version), [int(v) for v in internal], gauss_next],
        "numpy": {
            "bit_generator": _NUMPY_BIT_GENERATOR,
            "key": torch.from_numpy(np_state["state"]["key"].astype(np.int64)),
            "pos": int(np_state["state"]["pos"]),
            "has_gauss": int(np_state["has_gauss"]),
            "gauss": float(np_state["gauss"]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    if torch.backends.mps.is_available():
        try:
            state["mps"] = torch.mps.get_rng_state()
        except Exception:  # reported as available but unusable (e.g. headless CI macs)
            pass
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore a `get_rng_state()` snapshot (also after a checkpoint round trip).

    Accelerator states that cannot be applied here - the checkpoint comes from a
    machine with CUDA/MPS and this one has none - are skipped, so checkpoints stay
    portable; the CPU generators are always restored.
    """
    version, internal, gauss_next = state["python"]
    random.setstate((int(version), tuple(int(v) for v in internal), gauss_next))

    np_state = state["numpy"]
    if np_state["bit_generator"] != _NUMPY_BIT_GENERATOR:
        raise ValueError(
            f"cannot restore numpy RNG state of bit generator {np_state['bit_generator']!r}; "
            f"expected {_NUMPY_BIT_GENERATOR}"
        )
    key = torch.as_tensor(np_state["key"]).cpu().numpy().astype(np.uint32)
    np.random.set_state(
        {
            "bit_generator": _NUMPY_BIT_GENERATOR,
            "state": {"key": key, "pos": int(np_state["pos"])},
            "has_gauss": int(np_state["has_gauss"]),
            "gauss": float(np_state["gauss"]),
        }
    )

    # Checkpoints may have been loaded with map_location=<accelerator>; RNG states must be on CPU.
    torch.set_rng_state(_as_cpu_bytes(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        for index, cuda_state in enumerate(state["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(_as_cpu_bytes(cuda_state), index)
    if "mps" in state and torch.backends.mps.is_available():
        try:
            torch.mps.set_rng_state(_as_cpu_bytes(state["mps"]))
        except Exception:  # see get_rng_state
            pass


def _as_cpu_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.uint8)


def linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    """Linear interpolation from `start` (at `t <= 0`) to `end` (at `t >= duration`), clamped.

    A non-positive `duration` means the schedule is already over and yields `end`.
    """
    if duration <= 0:
        return float(end)
    fraction = min(max(float(t) / float(duration), 0.0), 1.0)
    return float(start + fraction * (end - start))
