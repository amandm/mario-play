"""Vectorized environments with same-step auto-reset.

`SyncVecEnv` steps its envs one after another in this process; `SubprocVecEnv`
gives every env its own worker process. Both return a `VecStep` whose `obs` is
what the agent acts on next (the first observation of the new episode where one
just ended) while `final_obs` and `infos` describe the transition that actually
happened - which is what truncation bootstrapping and replay buffers need.

This module is imported by every worker process, so it must stay free of torch:
a spawned worker that imports torch pays about a second and ~200 MB for nothing.
"""

from __future__ import annotations

import functools
import multiprocessing as mp
import pickle
import signal
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from typing import Any

import gymnasium as gym
import numpy as np

from mario_play.envs.factory import make_env
from mario_play.rl.config import EnvConfig
from mario_play.rl.types import VecStep

EnvFn = Callable[[], gym.Env]

# (obs, reward, terminated, truncated, info, final_obs); final_obs is None unless the episode ended
_StepResult = tuple[Any, float, bool, bool, dict[str, Any], Any]

_JOIN_TIMEOUT = 5.0  # seconds a worker gets to exit on its own before it is terminated


def _step_env(env: gym.Env, action: Any) -> _StepResult:
    """Step one env and, if its episode ended, reset it right away (no reseeding)."""
    obs, reward, terminated, truncated, info = env.step(action)
    terminated, truncated = bool(terminated), bool(truncated)
    final_obs = None
    if terminated or truncated:
        final_obs = obs
        obs, _ = env.reset()
    return obs, float(reward), terminated, truncated, info, final_obs


class VecEnv(ABC):
    """`n_envs` independent copies of one environment, stepped in lockstep.

    Attributes: `n_envs`, `single_observation_space`, `single_action_space` and
    `reset_infos` (the info dicts of the most recent `reset` call).
    """

    n_envs: int
    single_observation_space: gym.Space
    single_action_space: gym.Space
    reset_infos: list[dict[str, Any]]

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset every env and return the stacked observations `(n_envs, *obs_shape)`.

        With a seed, env `i` is reset with `seed + i`; with `None` the envs keep
        their current random streams.
        """

    @abstractmethod
    def step(self, actions: np.ndarray) -> VecStep:
        """Apply `actions[i]` to env `i`; finished envs are reset within the same call."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources. Idempotent; the vec env is unusable afterwards."""

    def __enter__(self) -> VecEnv:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -------------------------------------------------------- shared by both kinds

    def _init_spaces(self, spaces: Sequence[tuple[gym.Space, gym.Space]]) -> None:
        """Adopt env 0's spaces after checking that every env agrees with them."""
        obs_space, action_space = spaces[0]
        for i, (other_obs, other_action) in enumerate(spaces[1:], start=1):
            if other_obs != obs_space or other_action != action_space:
                raise ValueError(
                    f"env {i} has different spaces than env 0: observation space "
                    f"{other_obs} vs {obs_space}, action space {other_action} vs {action_space}"
                )
        self.n_envs = len(spaces)
        self.single_observation_space = obs_space
        self.single_action_space = action_space
        self.reset_infos = [{} for _ in spaces]

    def _split_actions(self, actions: np.ndarray) -> list[Any]:
        """Validate the action batch and return one env-ready action per env."""
        actions = np.asarray(actions)
        if actions.ndim == 0 or len(actions) != self.n_envs:
            raise ValueError(
                f"expected {self.n_envs} actions (one per env), got shape {actions.shape}"
            )
        if isinstance(self.single_action_space, gym.spaces.Discrete):
            return [int(a) for a in actions]
        return list(actions)

    def _stack_obs(self, observations: Sequence[Any]) -> np.ndarray:
        stacked = np.stack([np.asarray(o) for o in observations])
        dtype = getattr(self.single_observation_space, "dtype", None)
        return stacked if dtype is None else stacked.astype(dtype, copy=False)

    def _assemble(self, results: Sequence[_StepResult]) -> VecStep:
        """Build a `VecStep` of freshly allocated arrays from per-env results."""
        obs = self._stack_obs([r[0] for r in results])
        final_obs = obs.copy()
        for i, result in enumerate(results):
            if result[5] is not None:
                final_obs[i] = result[5]
        return VecStep(
            obs=obs,
            rewards=np.array([r[1] for r in results], dtype=np.float32),
            terminated=np.array([r[2] for r in results], dtype=np.bool_),
            truncated=np.array([r[3] for r in results], dtype=np.bool_),
            final_obs=final_obs,
            infos=[r[4] for r in results],
        )


class SyncVecEnv(VecEnv):
    """Runs all envs in the calling process, one after another.

    The right choice for cheap envs (grid observations, CartPole), where process
    round trips would cost more than the simulation itself. `envs` is public.
    """

    def __init__(self, env_fns: Sequence[EnvFn]) -> None:
        if len(env_fns) == 0:
            raise ValueError("need at least one env_fn")
        self._closed = False
        self.envs: list[gym.Env] = []
        try:
            for fn in env_fns:
                self.envs.append(fn())
            self._init_spaces([(e.observation_space, e.action_space) for e in self.envs])
        except BaseException:
            self.close()
            raise

    def reset(self, seed: int | None = None) -> np.ndarray:
        """See `VecEnv.reset`."""
        self._assert_open()
        observations, infos = [], []
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=None if seed is None else seed + i)
            observations.append(obs)
            infos.append(info)
        self.reset_infos = infos
        return self._stack_obs(observations)

    def step(self, actions: np.ndarray) -> VecStep:
        """See `VecEnv.step`."""
        self._assert_open()
        per_env = self._split_actions(actions)
        return self._assemble(
            [_step_env(env, a) for env, a in zip(self.envs, per_env, strict=True)]
        )

    def close(self) -> None:
        """Close every env (each at most once)."""
        if self._closed:
            return
        self._closed = True
        for env in self.envs:
            try:
                env.close()
            except Exception:  # closing must never mask the error that led here
                pass

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("SyncVecEnv is closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _worker(conn: Connection, env_fn: EnvFn) -> None:
    """Worker process main loop: build the env, then serve `reset`/`step` until `close`.

    Every reply is `("ok", payload)` or `("error", formatted_traceback)`. After an
    error the worker exits: its env is in an unknown state and the parent tears the
    whole vec env down anyway.
    """
    # Ctrl-C reaches the whole process group. Shutdown is the parent's job (it may
    # still want to write a checkpoint), so workers must not die on SIGINT by themselves.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = None
    try:
        env = env_fn()
        conn.send(("ok", (env.observation_space, env.action_space)))
        while True:
            command, data = conn.recv()
            if command == "step":
                conn.send(("ok", _step_env(env, data)))
            elif command == "reset":
                conn.send(("ok", env.reset(seed=data)))
            elif command == "close":
                break
            else:
                raise ValueError(f"unknown command {command!r}")
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass  # the parent is gone; nobody is left to report to
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


class SubprocVecEnv(VecEnv):
    """Runs every env in its own process; worth it when a single env step is expensive.

    Processes are started with the "spawn" method (the macOS default, and the only
    safe one once torch has started threads in the parent), hence each `env_fn`
    must be picklable: use `functools.partial(make_env, cfg, seed)`, not a lambda.
    Under spawn the workers also re-import the `__main__` module, so keep heavy
    imports such as torch out of a script's top level or behind `if __name__ == ...`.

    An exception inside a worker is re-raised here as `RuntimeError` carrying the
    worker's traceback, after all workers have been shut down. `close()` is
    idempotent and never hangs: workers that do not exit on request (or already
    died) are terminated. `processes` is public for diagnostics.
    """

    def __init__(self, env_fns: Sequence[EnvFn], start_method: str = "spawn") -> None:
        if len(env_fns) == 0:
            raise ValueError("need at least one env_fn")
        self._closed = False
        self.start_method = start_method
        self.processes: list[mp.process.BaseProcess] = []
        self._conns: list[Connection] = []
        if start_method != "fork":
            for i, fn in enumerate(env_fns):
                try:
                    pickle.dumps(fn)
                except Exception as exc:
                    raise TypeError(
                        f"env_fns[{i}] is not picklable, which the {start_method!r} start method "
                        "requires; use functools.partial over a module-level function instead "
                        f"of a lambda or closure ({exc})"
                    ) from exc

        ctx = mp.get_context(start_method)
        try:
            for i, fn in enumerate(env_fns):
                parent_conn, child_conn = ctx.Pipe()
                process = ctx.Process(
                    target=_worker, args=(child_conn, fn), name=f"vec-env-worker-{i}", daemon=True
                )
                try:
                    process.start()
                except BaseException:
                    parent_conn.close()
                    raise
                finally:
                    # The worker owns this end now. Closing our copy is what turns a dead
                    # worker into an EOFError here instead of a recv() that blocks forever.
                    child_conn.close()
                self.processes.append(process)
                self._conns.append(parent_conn)
            self._init_spaces(self._gather("while creating its env"))
        except BaseException:
            self.close()
            raise

    def reset(self, seed: int | None = None) -> np.ndarray:
        """See `VecEnv.reset`."""
        self._assert_open()
        seeds = [None if seed is None else seed + i for i in range(self.n_envs)]
        results = self._round_trip([("reset", s) for s in seeds], "during reset")
        self.reset_infos = [info for _, info in results]
        return self._stack_obs([obs for obs, _ in results])

    def step(self, actions: np.ndarray) -> VecStep:
        """See `VecEnv.step`."""
        self._assert_open()
        messages = [("step", action) for action in self._split_actions(actions)]
        return self._assemble(self._round_trip(messages, "during step"))

    def close(self) -> None:
        """Ask every worker to exit, join them, and terminate any that do not comply."""
        if self._closed:
            return
        self._closed = True
        for conn, process in zip(self._conns, self.processes, strict=True):
            if process.is_alive():
                try:
                    conn.send(("close", None))
                except (OSError, ValueError):
                    pass  # worker died in the meantime
        deadline = time.monotonic() + _JOIN_TIMEOUT
        for conn, process in zip(self._conns, self.processes, strict=True):
            # A worker may still be blocked sending a reply nobody waits for any more
            # (we are closing after a failure); reading it lets the worker reach "close".
            while process.is_alive() and time.monotonic() < deadline:
                try:
                    if conn.poll(0.05):
                        conn.recv()
                except Exception:  # EOF: the worker closed its end, i.e. it is exiting
                    break
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(_JOIN_TIMEOUT)
            if process.is_alive():
                process.kill()
                process.join()
        for conn in self._conns:
            conn.close()

    # ----------------------------------------------------------------- helpers

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("SubprocVecEnv is closed")

    def _round_trip(self, messages: Sequence[tuple[str, Any]], when: str) -> list[Any]:
        """Send `messages[i]` to worker `i` and collect every reply.

        Whatever goes wrong - a worker error, a dead worker, or a KeyboardInterrupt
        while waiting - the vec env is closed before the exception propagates: a half
        finished exchange leaves replies in the pipes, and every later call would
        silently read stale data.
        """
        try:
            for i, message in enumerate(messages):
                try:
                    self._conns[i].send(message)
                except (OSError, ValueError) as exc:
                    raise RuntimeError(
                        f"env worker {i} died unexpectedly {when} ({self._exit_status(i)})"
                    ) from exc
            return self._gather(when)
        except BaseException:
            self.close()
            raise

    def _gather(self, when: str) -> list[Any]:
        """Collect one reply per worker; raise `RuntimeError` for the first failure.

        All workers are read before raising, so the traceback of a failing worker is
        reported even if an earlier worker merely died.
        """
        payloads: list[Any] = []
        died: str | None = None
        errored: str | None = None
        for i, conn in enumerate(self._conns):
            try:
                status, payload = conn.recv()
            except (EOFError, OSError):
                died = died or f"env worker {i} died unexpectedly {when} ({self._exit_status(i)})"
                continue
            if status == "error":
                errored = errored or f"env worker {i} raised an exception {when}:\n{payload}"
            payloads.append(payload)
        if errored or died:
            raise RuntimeError(errored or died)
        return payloads

    def _exit_status(self, index: int) -> str:
        process = self.processes[index]
        process.join(0.5)  # let a process that is just dying report its exit code
        return f"exit code {process.exitcode}"

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_vec_env(cfg: EnvConfig, n_envs: int, seed: int, kind: str = "sync") -> VecEnv:
    """Build `n_envs` envs from `cfg`; env `i` is created by `make_env(cfg, seed + i)`.

    `kind` is `"sync"` or `"subproc"`. The envs are not reset here: call
    `venv.reset(seed)` to seed env `i` with `seed + i` and get the first observations.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    env_fns = [functools.partial(make_env, cfg, seed + i) for i in range(n_envs)]
    if kind == "sync":
        return SyncVecEnv(env_fns)
    if kind == "subproc":
        return SubprocVecEnv(env_fns)
    raise ValueError(f"unknown vec env kind {kind!r}; expected 'sync' or 'subproc'")
