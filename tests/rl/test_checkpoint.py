"""Checkpoint files: weights_only round trip, atomic writes, rotation."""

from __future__ import annotations

import collections
import enum
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from mario_play.rl import checkpoint as checkpoint_module
from mario_play.rl.checkpoint import (
    atomic_copy,
    list_checkpoints,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from mario_play.rl.config import TrainConfig, config_from_dict, config_to_dict
from mario_play.rl.utils import get_rng_state, set_rng_state, set_seed


def _algo_state() -> dict:
    """A realistic algorithm state: model + Adam optimizer (with stepped state) + counters."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    model(torch.randn(5, 4)).sum().backward()
    optimizer.step()
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "n_updates": 17}


def _save(path: Path, **overrides) -> dict:
    kwargs = {
        "algo_name": "ppo",
        "config_dict": config_to_dict(TrainConfig()),
        "global_step": 4096,
        "best_eval": 12.5,
        **overrides,
    }
    if "rng_state" not in kwargs:
        kwargs["rng_state"] = get_rng_state()
    if "algo_state" not in kwargs:
        kwargs["algo_state"] = _algo_state()  # note: reseeds torch
    save_checkpoint(path, **kwargs)
    return kwargs


def _draws() -> tuple:
    return (
        random.random(),
        random.gauss(0.0, 1.0),
        float(np.random.rand()),
        float(np.random.randn()),
        torch.rand(2).tolist(),
    )


def _assert_same(a, b) -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_same(x, y)
    else:
        assert a == b and type(a) is type(b)


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #


def test_round_trip_through_weights_only_load(tmp_path) -> None:
    path = tmp_path / "ckpt_4096.pt"
    saved = _save(path, extra={"episodes": 31, "wall_time": 12.5})

    raw = torch.load(path, map_location="cpu", weights_only=True)  # the contract itself
    assert isinstance(raw, dict)

    ckpt = load_checkpoint(path)
    assert ckpt["algo_name"] == "ppo"
    assert ckpt["global_step"] == 4096 and type(ckpt["global_step"]) is int
    assert ckpt["best_eval"] == 12.5
    assert ckpt["extra"] == {"episodes": 31, "wall_time": 12.5}
    assert isinstance(ckpt["format_version"], int)
    _assert_same(saved["algo_state"], ckpt["algo_state"])
    _assert_same(saved["rng_state"], ckpt["rng_state"])
    assert config_from_dict(ckpt["config"]) == TrainConfig()


def test_loaded_state_restores_model_and_optimizer(tmp_path) -> None:
    path = tmp_path / "latest.pt"
    saved = _save(path)
    state = load_checkpoint(path)["algo_state"]
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    for key, value in saved["algo_state"]["model"].items():
        assert torch.equal(model.state_dict()[key], value)


def test_rng_state_round_trip_through_a_checkpoint(tmp_path) -> None:
    algo_state = _algo_state()
    set_seed(21)
    random.gauss(0.0, 1.0)  # leave a cached gaussian behind in python's RNG ...
    np.random.randn()  # ... and in numpy's
    _save(tmp_path / "latest.pt", algo_state=algo_state, rng_state=get_rng_state())
    expected = [_draws() for _ in range(3)]
    set_seed(1234)
    set_rng_state(load_checkpoint(tmp_path / "latest.pt")["rng_state"])
    assert [_draws() for _ in range(3)] == expected


def test_defaults_none_best_eval_and_empty_extra(tmp_path) -> None:
    _save(tmp_path / "c.pt", best_eval=None)
    ckpt = load_checkpoint(tmp_path / "c.pt")
    assert ckpt["best_eval"] is None
    assert ckpt["extra"] == {}


def test_accepts_str_paths_and_creates_parent_directories(tmp_path) -> None:
    path = tmp_path / "runs" / "x" / "checkpoints" / "latest.pt"
    _save(str(path))
    assert load_checkpoint(str(path))["global_step"] == 4096


def test_numpy_scalars_are_converted_to_python_scalars(tmp_path) -> None:
    """`best_eval=np.mean(...)` is the classic way to write a checkpoint that cannot be loaded."""
    path = tmp_path / "c.pt"
    _save(
        path,
        best_eval=np.float64(3.25),
        global_step=np.int64(640),
        extra={"flag_rate": np.float32(0.5), "done": np.bool_(True), "nested": [np.int32(3)]},
    )
    ckpt = load_checkpoint(path)
    assert ckpt["best_eval"] == 3.25 and type(ckpt["best_eval"]) is float
    assert ckpt["global_step"] == 640 and type(ckpt["global_step"]) is int
    assert ckpt["extra"] == {"flag_rate": 0.5, "done": True, "nested": [3]}
    assert type(ckpt["extra"]["done"]) is bool


def test_unsupported_type_is_rejected_at_save_time_with_its_location(tmp_path) -> None:
    path = tmp_path / "c.pt"
    with pytest.raises(TypeError) as excinfo:
        _save(path, algo_state={"model": {}, "replay": {"obs": np.zeros(3)}})
    message = str(excinfo.value)
    assert "algo_state" in message and "replay" in message and "obs" in message
    assert "ndarray" in message
    assert list(tmp_path.iterdir()) == []

    with pytest.raises(TypeError, match="config"):
        _save(path, config_dict={"run_dir": Path("runs")})
    with pytest.raises(TypeError, match="key"):
        _save(path, extra={("a", 1): 2})


class _Color(enum.IntEnum):
    RED = 1


_Point = collections.namedtuple("_Point", "x y")


@pytest.mark.parametrize(
    "value",
    [collections.defaultdict(int, a=1), _Color.RED, _Point(1, 2), {1, 2}, torch.float32],
    ids=["defaultdict", "IntEnum", "namedtuple", "set", "dtype"],
)
def test_subclasses_and_exotic_types_are_rejected_like_torch_would(tmp_path, value) -> None:
    """The weights_only unpickler matches types exactly, so the save-time check must too."""
    with pytest.raises(TypeError, match="extra.thing"):
        _save(tmp_path / "c.pt", extra={"thing": value})
    assert list(tmp_path.iterdir()) == []


def test_containers_torch_itself_produces_are_accepted(tmp_path) -> None:
    extra = {
        "ordered": collections.OrderedDict(a=1, b=(1, 2.5, "x", None, True)),
        "size": torch.Size([2, 3]),
        "int_keys": {0: "a", 1: "b"},
        "param": torch.nn.Parameter(torch.ones(2)),
    }
    _save(tmp_path / "c.pt", extra=extra)
    loaded = load_checkpoint(tmp_path / "c.pt")["extra"]
    assert loaded["ordered"] == extra["ordered"] and loaded["int_keys"] == extra["int_keys"]
    assert tuple(loaded["size"]) == (2, 3)
    assert torch.equal(loaded["param"], torch.ones(2))


def test_sanitizing_does_not_touch_the_callers_objects(tmp_path) -> None:
    extra = {"score": np.float64(1.0)}
    state = _algo_state()
    _save(tmp_path / "c.pt", algo_state=state, extra=extra)
    assert type(extra["score"]) is np.float64
    assert hasattr(state["model"], "_metadata")  # state_dict metadata is left alone
    loaded = load_checkpoint(tmp_path / "c.pt")["algo_state"]["model"]
    assert list(loaded) == list(state["model"])


# --------------------------------------------------------------------------- #
# atomic write
# --------------------------------------------------------------------------- #


def test_save_leaves_only_the_final_file(tmp_path) -> None:
    _save(tmp_path / "ckpt_1.pt")
    _save(tmp_path / "ckpt_1.pt", global_step=2)  # overwriting is fine too
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt_1.pt"]
    assert load_checkpoint(tmp_path / "ckpt_1.pt")["global_step"] == 2


def test_failed_write_keeps_the_previous_checkpoint_and_no_tmp_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "latest.pt"
    _save(path, global_step=100)

    def exploding_save(obj, f, *args, **kwargs) -> None:
        if hasattr(f, "write"):
            f.write(b"partial garbage")
        else:
            Path(f).write_bytes(b"partial garbage")
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint_module.torch, "save", exploding_save)
    with pytest.raises(OSError, match="disk full"):
        _save(path, global_step=200)
    monkeypatch.undo()

    assert [p.name for p in tmp_path.iterdir()] == ["latest.pt"]
    assert load_checkpoint(path)["global_step"] == 100


def test_atomic_copy(tmp_path) -> None:
    src = tmp_path / "ckpt_5.pt"
    _save(src, global_step=5)
    atomic_copy(src, tmp_path / "sub" / "best.pt")
    assert load_checkpoint(tmp_path / "sub" / "best.pt")["global_step"] == 5
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt_5.pt", "sub"]
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["best.pt"]


# --------------------------------------------------------------------------- #
# loading errors
# --------------------------------------------------------------------------- #


def test_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.pt"):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_rejects_files_that_are_not_checkpoints(tmp_path) -> None:
    torch.save({"model": {}}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="global_step"):
        load_checkpoint(tmp_path / "other.pt")
    torch.save([1, 2, 3], tmp_path / "list.pt")
    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(tmp_path / "list.pt")


def test_load_rejects_newer_format_versions(tmp_path) -> None:
    _save(tmp_path / "c.pt")
    payload = torch.load(tmp_path / "c.pt", weights_only=True)
    payload["format_version"] = 999
    torch.save(payload, tmp_path / "future.pt")
    with pytest.raises(ValueError, match="format version 999"):
        load_checkpoint(tmp_path / "future.pt")


def test_load_never_unpickles_arbitrary_objects(tmp_path) -> None:
    """The refusal of `weights_only` surfaces as the documented ValueError, cause attached."""
    torch.save({"global_step": 1, "payload": Path("x")}, tmp_path / "evil.pt")
    with pytest.raises(ValueError, match="evil.pt") as excinfo:
        load_checkpoint(tmp_path / "evil.pt")
    assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)


class _RunsCodeWhenUnpickled:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (str(self.marker),))


def test_load_never_executes_pickled_code(tmp_path) -> None:
    marker = tmp_path / "executed"
    path = tmp_path / "bomb.pt"
    torch.save({"global_step": 1, "payload": _RunsCodeWhenUnpickled(marker)}, path)
    with pytest.raises(ValueError, match="bomb.pt") as excinfo:
        load_checkpoint(path)
    assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)
    assert not marker.exists()
    # The file really is a bomb: an unrestricted load runs it.
    torch.load(path, weights_only=False)
    assert marker.is_dir()


def _checkpoint_bytes(tmp_path: Path, size: str) -> bytes:
    """A few hundred bytes of counters, or ~260 KB with a weight matrix (other zip layout)."""
    state = {"n_updates": 1} if size == "small" else {"weights": torch.randn(256, 256)}
    _save(tmp_path / "source.pt", algo_state=state)
    return (tmp_path / "source.pt").read_bytes()


def _not_a_checkpoint(kind: str, tmp_path: Path) -> bytes:
    if kind == "empty":
        return b""
    if kind == "five-bytes":
        return b"hello"
    if kind == "yaml":
        return b"algo: ppo\ntotal_timesteps: 1000\nenv:\n  id: MarioPlay-v0\n"
    if kind == "csv":
        return b"step,rollout/ep_return_mean,time/sps\n200,11.5,2600.0\n"
    if kind == "random-bytes":
        return np.random.default_rng(0).bytes(4096)
    size, cut = kind.split("-", 1)  # a checkpoint whose copy (scp, rsync) was interrupted
    data = _checkpoint_bytes(tmp_path, size)
    return data[: len(data) // 2] if cut == "half" else data[:-100]


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "five-bytes",
        "yaml",
        "csv",
        "random-bytes",
        "small-half",
        "small-minus-100",
        "larger-half",
        "larger-minus-100",
    ],
)
def test_corrupt_truncated_and_foreign_files_raise_value_error_naming_the_file(
    tmp_path, kind: str
) -> None:
    """Whatever torch's reader trips over, callers see the one documented exception."""
    path = tmp_path / "broken_latest.pt"
    path.write_bytes(_not_a_checkpoint(kind, tmp_path))
    with pytest.raises(ValueError, match="broken_latest.pt") as excinfo:
        load_checkpoint(path)
    assert excinfo.value.__cause__ is not None  # the reader's own error stays attached


def test_a_checkpoint_that_cannot_be_opened_is_not_reported_as_corrupt(
    tmp_path, monkeypatch
) -> None:
    _save(tmp_path / "c.pt")

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(checkpoint_module.torch, "load", denied)
    with pytest.raises(PermissionError):
        load_checkpoint(tmp_path / "c.pt")


# --------------------------------------------------------------------------- #
# rotation
# --------------------------------------------------------------------------- #


def _touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"x")


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_list_checkpoints_sorts_by_step_number(tmp_path) -> None:
    _touch(tmp_path, "ckpt_10000.pt", "ckpt_900.pt", "ckpt_200000.pt", "latest.pt", "ckpt_x.pt")
    assert [p.name for p in list_checkpoints(tmp_path)] == [
        "ckpt_900.pt",
        "ckpt_10000.pt",
        "ckpt_200000.pt",
    ]
    assert list_checkpoints(tmp_path / "missing") == []


def test_rotation_keeps_the_newest_by_step_and_spares_latest_and_best(tmp_path) -> None:
    _touch(
        tmp_path,
        "ckpt_900.pt",
        "ckpt_10000.pt",
        "ckpt_20000.pt",
        "ckpt_100000.pt",
        "latest.pt",
        "best.pt",
        "notes.txt",
        "ckpt_final.pt",
    )
    rotate_checkpoints(tmp_path, keep=2)
    assert _names(tmp_path) == [
        "best.pt",
        "ckpt_100000.pt",
        "ckpt_20000.pt",
        "ckpt_final.pt",
        "latest.pt",
        "notes.txt",
    ]
    rotate_checkpoints(tmp_path, keep=5)  # fewer files than `keep`: nothing happens
    assert len(_names(tmp_path)) == 6


def test_rotation_keep_zero_removes_all_numbered_and_negative_keeps_all(tmp_path) -> None:
    _touch(tmp_path, "ckpt_1.pt", "ckpt_2.pt", "latest.pt", "best.pt")
    rotate_checkpoints(tmp_path, keep=-1)
    assert _names(tmp_path) == ["best.pt", "ckpt_1.pt", "ckpt_2.pt", "latest.pt"]
    rotate_checkpoints(tmp_path, keep=0)
    assert _names(tmp_path) == ["best.pt", "latest.pt"]


def test_rotation_sweeps_stale_tmp_files_of_crashed_saves(tmp_path, monkeypatch) -> None:
    written = []
    real_replace = checkpoint_module.os.replace

    def crash_before_rename(src, dst) -> None:
        written.append(Path(src).name)
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoint_module.os, "replace", crash_before_rename)
    with pytest.raises(KeyboardInterrupt):
        _save(tmp_path / "ckpt_50.pt")
    monkeypatch.setattr(checkpoint_module.os, "replace", real_replace)

    _touch(tmp_path, "ckpt_10.pt", written[0])  # simulate the orphan a hard kill leaves behind
    rotate_checkpoints(tmp_path, keep=3)
    assert _names(tmp_path) == ["ckpt_10.pt"]


def test_rotation_on_a_missing_directory_is_a_no_op(tmp_path) -> None:
    rotate_checkpoints(tmp_path / "missing", keep=3)
