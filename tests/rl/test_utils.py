"""Seeding, device resolution, RNG state capture and schedules."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from mario_play.rl.utils import (
    get_rng_state,
    linear_schedule,
    resolve_device,
    set_rng_state,
    set_seed,
)


@pytest.fixture(autouse=True)
def _restore_global_torch_flags():
    yield
    torch.use_deterministic_algorithms(False)


def _draws() -> tuple:
    """One draw from every global RNG the framework uses (incl. the cached-gaussian paths)."""
    return (
        random.random(),
        random.gauss(0.0, 1.0),
        float(np.random.rand()),
        float(np.random.randn()),
        int(np.random.randint(0, 2**31 - 1)),
        torch.rand(3).tolist(),
        torch.randn(2).tolist(),
        int(torch.randint(0, 2**31 - 1, (1,)).item()),
    )


# --------------------------------------------------------------------------- #
# set_seed
# --------------------------------------------------------------------------- #


def test_set_seed_makes_all_global_rngs_reproducible() -> None:
    set_seed(123)
    first = _draws()
    set_seed(123)
    assert _draws() == first
    set_seed(124)
    assert _draws() != first


def test_set_seed_toggles_torch_determinism() -> None:
    set_seed(0, torch_deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    set_seed(0)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False


def test_deterministic_mode_only_warns_on_unsupported_ops() -> None:
    set_seed(0, torch_deterministic=True)
    assert torch.is_deterministic_algorithms_warn_only_enabled()


# --------------------------------------------------------------------------- #
# resolve_device
# --------------------------------------------------------------------------- #


def _fake_hardware(monkeypatch, cuda: bool, mps: bool, n_cuda: int = 1) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n_cuda if cuda else 0)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_resolve_device_cpu() -> None:
    device = resolve_device("cpu")
    assert isinstance(device, torch.device)
    assert device == torch.device("cpu")


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_resolve_device_auto_prefers_cuda_then_mps_then_cpu(
    monkeypatch, cuda: bool, mps: bool, expected: str
) -> None:
    _fake_hardware(monkeypatch, cuda, mps)
    assert resolve_device("auto").type == expected
    assert resolve_device().type == expected  # "auto" is the default


def test_resolve_device_explicit_available(monkeypatch) -> None:
    _fake_hardware(monkeypatch, cuda=True, mps=True, n_cuda=2)
    assert resolve_device("cuda") == torch.device("cuda")
    assert resolve_device("cuda:1") == torch.device("cuda", 1)
    assert resolve_device("mps") == torch.device("mps")
    assert resolve_device("CUDA") == torch.device("cuda")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_explicit_unavailable_raises_clear_error(monkeypatch) -> None:
    _fake_hardware(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="MPS is not available"):
        resolve_device("mps")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device(torch.device("cuda"))


def test_resolve_device_cuda_index_out_of_range(monkeypatch) -> None:
    _fake_hardware(monkeypatch, cuda=True, mps=False, n_cuda=1)
    with pytest.raises(RuntimeError, match="cuda:3"):
        resolve_device("cuda:3")


@pytest.mark.parametrize("name", ["tpu", "", "gpu0", "cuda:x"])
def test_resolve_device_unknown_name_raises(name: str) -> None:
    with pytest.raises(ValueError, match="device"):
        resolve_device(name)


# --------------------------------------------------------------------------- #
# RNG state
# --------------------------------------------------------------------------- #


def test_rng_state_round_trip_reproduces_subsequent_draws() -> None:
    set_seed(7)
    random.gauss(0.0, 1.0)  # leaves a cached second gaussian in python's RNG
    np.random.randn()  # ... and in numpy's legacy RNG
    state = get_rng_state()
    expected = [_draws() for _ in range(3)]
    set_seed(999)  # scramble everything
    _draws()
    set_rng_state(state)
    assert [_draws() for _ in range(3)] == expected


def test_get_rng_state_does_not_advance_any_rng() -> None:
    set_seed(3)
    expected = _draws()
    set_seed(3)
    get_rng_state()
    assert _draws() == expected


def _assert_weights_only_types(obj, path: str = "state") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert isinstance(key, str), f"{path}: key {key!r}"
            _assert_weights_only_types(value, f"{path}[{key!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _assert_weights_only_types(value, f"{path}[{i}]")
    else:
        assert obj is None or type(obj) in (int, float, bool, str, torch.Tensor), (
            f"{path}: {type(obj).__name__} is not weights_only-safe"
        )


def test_rng_state_holds_only_tensors_and_python_primitives() -> None:
    state = get_rng_state()
    assert {"python", "numpy", "torch"} <= set(state)
    _assert_weights_only_types(state)
    assert state["torch"].dtype == torch.uint8


def test_rng_state_survives_torch_save_and_weights_only_load(tmp_path) -> None:
    set_seed(11)
    random.gauss(0.0, 1.0)
    np.random.randn()
    torch.save(get_rng_state(), tmp_path / "rng.pt")
    expected = _draws()
    set_seed(5)
    set_rng_state(torch.load(tmp_path / "rng.pt", weights_only=True))
    assert _draws() == expected


def test_rng_state_skips_cuda_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "cuda" not in get_rng_state()


def test_set_rng_state_ignores_accelerator_states_it_cannot_apply(monkeypatch) -> None:
    """A checkpoint written on a CUDA box must still resume on a machine without CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    set_seed(2)
    state = get_rng_state()
    state["cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    expected = _draws()
    set_rng_state(state)
    assert _draws() == expected


def test_set_rng_state_rejects_a_foreign_numpy_bit_generator() -> None:
    state = get_rng_state()
    state["numpy"]["bit_generator"] = "PCG64"
    with pytest.raises(ValueError, match="MT19937"):
        set_rng_state(state)


# --------------------------------------------------------------------------- #
# linear_schedule
# --------------------------------------------------------------------------- #


def test_linear_schedule_endpoints_and_midpoint() -> None:
    assert linear_schedule(1.0, 0.05, 1000, 0) == pytest.approx(1.0)
    assert linear_schedule(1.0, 0.05, 1000, 1000) == pytest.approx(0.05)
    assert linear_schedule(1.0, 0.05, 1000, 500) == pytest.approx(0.525)
    assert linear_schedule(0.0, 2.0, 4, 1) == pytest.approx(0.5)  # increasing works too


def test_linear_schedule_clamps_outside_the_interval() -> None:
    assert linear_schedule(1.0, 0.05, 1000, -50) == pytest.approx(1.0)
    assert linear_schedule(1.0, 0.05, 1000, 10**9) == pytest.approx(0.05)


def test_linear_schedule_with_no_duration_is_already_finished() -> None:
    assert linear_schedule(1.0, 0.05, 0, 0) == pytest.approx(0.05)
    assert linear_schedule(1.0, 0.05, -5, 3) == pytest.approx(0.05)


def test_linear_schedule_returns_a_python_float() -> None:
    assert type(linear_schedule(1, 0, 10, np.int64(5))) is float
