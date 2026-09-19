"""Training checkpoints: atomic writes, `weights_only` loading and rotation.

A checkpoint is a single `torch.save` file holding nothing but tensors and plain
Python containers/primitives, so it can be read back with
`torch.load(weights_only=True)` - loading a checkpoint never executes pickled code.
Replay buffers are deliberately not part of it (see `mario_play.rl.algos.dqn`).
"""

from __future__ import annotations

import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1
_REQUIRED_KEYS = ("algo_name", "algo_state", "config", "global_step", "best_eval", "rng_state")
_NUMBERED = re.compile(r"ckpt_(\d+)\.pt")
_PRIMITIVES = (str, bool, int, float)
_TMP_SUFFIX = ".tmp"


def save_checkpoint(
    path: str | Path,
    *,
    algo_name: str,
    algo_state: dict[str, Any],
    config_dict: dict[str, Any],
    global_step: int,
    best_eval: float | None,
    rng_state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write a checkpoint to `path` (parent directories are created).

    The payload is validated *before* anything touches the disk: numpy scalars
    (`best_eval=np.mean(...)`) are converted to Python scalars, and any value that
    `torch.load(weights_only=True)` would refuse raises `TypeError` naming its
    location - far better than discovering an unreadable checkpoint at resume time.
    The file is written next to `path` and renamed into place, so a crash mid-write
    never corrupts an existing checkpoint.
    """
    payload = {
        "format_version": FORMAT_VERSION,
        "algo_name": algo_name,
        "algo_state": algo_state,
        "config": config_dict,
        "global_step": global_step,
        "best_eval": best_eval,
        "rng_state": rng_state,
        "extra": extra if extra is not None else {},
    }
    payload = {key: _sanitize(value, key) for key, value in payload.items()}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint written by `save_checkpoint` using `torch.load(weights_only=True)`.

    Returns a dict with `format_version`, `algo_name`, `algo_state`, `config`,
    `global_step`, `best_eval`, `rng_state` and `extra`. Raises `FileNotFoundError`
    for a missing file and `ValueError` for a file that is not a (compatible)
    checkpoint. That covers everything the reader itself trips over - a corrupt,
    truncated or foreign file, or a pickle that needs arbitrary objects, which
    `weights_only` refuses to build: the message names the file and the reader's
    exception is chained as `__cause__`. A file that cannot be opened keeps its
    `PermissionError`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (PermissionError, FileNotFoundError, MemoryError):
        raise  # the file's content is not the problem
    except Exception as exc:
        # torch's readers fail in many ways on such files: RuntimeError (miniz), OSError,
        # EOFError, KeyError, IndexError, UnpicklingError, ... depending on the bytes.
        reason = str(exc).strip().split("\n", 1)[0][:200]
        raise ValueError(
            f"{path} is not a readable mario-play checkpoint (corrupt, truncated or a different "
            f"kind of file): {type(exc).__name__}: {reason}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a mario-play checkpoint (got {type(payload).__name__})")
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path} is not a mario-play checkpoint: missing keys {missing}")
    version = payload.get("format_version", 1)
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{path} has checkpoint format version {version}, but this code only reads up to "
            f"version {FORMAT_VERSION}; update mario-play"
        )
    payload.setdefault("format_version", version)
    payload.setdefault("extra", {})
    return payload


def atomic_copy(src: str | Path, dst: str | Path) -> None:
    """Copy a finished checkpoint (e.g. `ckpt_<step>.pt` -> `best.pt`) with the same atomicity."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(dst)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_directory(dst.parent)


def list_checkpoints(ckpt_dir: str | Path) -> list[Path]:
    """Numbered checkpoints `ckpt_<step>.pt` in `ckpt_dir`, oldest (lowest step) first."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    found = []
    for entry in ckpt_dir.iterdir():
        match = _NUMBERED.fullmatch(entry.name)
        if match and entry.is_file():
            found.append((int(match.group(1)), entry))
    return [entry for _, entry in sorted(found)]


def rotate_checkpoints(ckpt_dir: str | Path, keep: int) -> None:
    """Delete all but the `keep` highest-step `ckpt_<step>.pt` files.

    `latest.pt`, `best.pt` and any other file are never touched. `keep=0` removes
    every numbered checkpoint; a negative `keep` disables rotation. Temporary files
    orphaned by a save that was killed mid-write are swept as well.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return
    for entry in ckpt_dir.iterdir():
        if entry.name.endswith(_TMP_SUFFIX) and entry.is_file():
            entry.unlink(missing_ok=True)
    if keep < 0:
        return
    numbered = list_checkpoints(ckpt_dir)
    for stale in numbered[: max(0, len(numbered) - keep)]:
        stale.unlink(missing_ok=True)


def _tmp_path(path: Path) -> Path:
    # Same directory as the target: os.replace is only atomic within one filesystem.
    return path.with_name(f".{path.name}{_TMP_SUFFIX}")


def _fsync_directory(directory: Path) -> None:
    """Best effort: make the rename itself durable (not supported on every platform)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _sanitize(value: Any, where: str) -> Any:
    """Return `value` with numpy scalars unwrapped; raise TypeError for unloadable types.

    Containers are rebuilt only when something inside them changed, so untouched
    objects (notably model state_dicts with their `_metadata`) pass through as-is.
    Types are matched exactly: the weights_only unpickler refuses subclasses such as
    `defaultdict`, `IntEnum` members or namedtuples just like any other custom class.
    """
    if isinstance(value, np.generic):  # first: np.float64 is also a `float`
        return value.item()
    if value is None or isinstance(value, torch.Tensor) or type(value) in _PRIMITIVES:
        return value
    if type(value) in (dict, OrderedDict):
        changed = False
        items = []
        for key, item in value.items():
            if not (key is None or type(key) in _PRIMITIVES):
                raise TypeError(
                    f"checkpoint[{where!r}]: dict key {key!r} of type {type(key).__name__} "
                    "cannot be loaded with weights_only=True (use str/int keys)"
                )
            clean = _sanitize(item, f"{where}.{key}")
            changed |= clean is not item
            items.append((key, clean))
        return dict(items) if changed else value
    if type(value) in (list, tuple, torch.Size):
        items = [_sanitize(item, f"{where}[{i}]") for i, item in enumerate(value)]
        if all(clean is item for clean, item in zip(items, value, strict=True)):
            return value
        return items if type(value) is list else tuple(items)
    raise TypeError(
        f"checkpoint[{where!r}] has type {type(value).__name__}, which cannot be loaded with "
        "torch.load(weights_only=True); store tensors or plain Python "
        "dict/list/tuple/str/int/float/bool/None instead"
    )
