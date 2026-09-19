"""SyncVecEnv / SubprocVecEnv: auto-reset semantics, seeding, errors, shutdown."""

from __future__ import annotations

import functools
import gc
import os
import signal
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from mario_play.envs.factory import make_env
from mario_play.rl.config import EnvConfig
from mario_play.rl.debug_envs import COUNTING_ENV_ID, CountingEnv, SeededNoiseEnv
from mario_play.rl.types import VecStep
from mario_play.rl.vec_env import SubprocVecEnv, SyncVecEnv, VecEnv, make_vec_env

pytestmark = pytest.mark.timeout(60)

KINDS = ["sync", "subproc"]


def build(kind: str, env_fns: list) -> VecEnv:
    return SyncVecEnv(env_fns) if kind == "sync" else SubprocVecEnv(env_fns)


@pytest.fixture(scope="module", params=KINDS)
def counting(request) -> VecEnv:
    """env 0 terminates at t=2, env 1 truncates at t=3 (shared: tests must `reset()` first)."""
    venv = build(
        request.param,
        [
            functools.partial(CountingEnv, terminate_at=2),
            functools.partial(CountingEnv, terminate_at=None, truncate_at=3),
        ],
    )
    yield venv
    venv.close()


@pytest.fixture(scope="module", params=KINDS)
def noise(request) -> VecEnv:
    venv = build(request.param, [functools.partial(SeededNoiseEnv, episode_length=4)] * 3)
    yield venv
    venv.close()


def zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


# --------------------------------------------------------------------------- #
# shapes, dtypes, spaces
# --------------------------------------------------------------------------- #


def test_spaces_and_n_envs(counting: VecEnv) -> None:
    reference = CountingEnv()
    assert isinstance(counting, VecEnv)
    assert counting.n_envs == 2
    assert counting.single_observation_space == reference.observation_space
    assert counting.single_action_space == reference.action_space


def test_reset_returns_stacked_first_observations(counting: VecEnv) -> None:
    obs = counting.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2, 1)
    assert obs.dtype == np.float32
    np.testing.assert_array_equal(obs, [[0.0], [0.0]])


def test_step_shapes_and_dtypes(counting: VecEnv) -> None:
    counting.reset(seed=0)
    step = counting.step(zeros(2))
    assert isinstance(step, VecStep)
    assert step.obs.shape == (2, 1) and step.obs.dtype == np.float32
    assert step.final_obs.shape == (2, 1) and step.final_obs.dtype == np.float32
    assert step.rewards.shape == (2,) and step.rewards.dtype == np.float32
    assert step.terminated.shape == (2,) and step.terminated.dtype == np.bool_
    assert step.truncated.shape == (2,) and step.truncated.dtype == np.bool_
    assert isinstance(step.infos, list) and len(step.infos) == 2
    assert all(isinstance(info, dict) for info in step.infos)
    np.testing.assert_array_equal(step.rewards, [1.0, 1.0])


def test_step_returns_fresh_arrays(counting: VecEnv) -> None:
    """The trainer keeps the previous `obs` around while stepping, so buffers must not be reused."""
    counting.reset(seed=0)
    first = counting.step(zeros(2))
    assert not np.shares_memory(first.obs, first.final_obs)
    kept = first.obs.copy()
    second = counting.step(zeros(2))
    np.testing.assert_array_equal(first.obs, kept)
    assert not np.shares_memory(first.obs, second.obs)


def test_actions_are_routed_to_the_matching_env(counting: VecEnv) -> None:
    counting.reset(seed=0)
    step = counting.step(np.array([1, 0], dtype=np.int64))
    assert [info["action"] for info in step.infos] == [1, 0]
    assert all(type(info["action"]) is int for info in step.infos)


def test_wrong_number_of_actions_raises(counting: VecEnv) -> None:
    counting.reset(seed=0)
    with pytest.raises(ValueError, match="2"):
        counting.step(zeros(3))


# --------------------------------------------------------------------------- #
# same-step auto-reset
# --------------------------------------------------------------------------- #


def test_auto_reset_obs_final_obs_and_terminal_info(counting: VecEnv) -> None:
    counting.reset(seed=0)
    assert [info["t"] for info in counting.reset_infos] == [0, 0]

    step1 = counting.step(zeros(2))
    np.testing.assert_array_equal(step1.obs, [[1.0], [1.0]])
    np.testing.assert_array_equal(step1.final_obs, step1.obs)
    assert not step1.terminated.any() and not step1.truncated.any()

    step2 = counting.step(zeros(2))  # env 0 reaches terminate_at=2
    np.testing.assert_array_equal(step2.terminated, [True, False])
    np.testing.assert_array_equal(step2.truncated, [False, False])
    np.testing.assert_array_equal(step2.obs, [[0.0], [2.0]])  # env 0: first obs of new episode
    np.testing.assert_array_equal(step2.final_obs, [[2.0], [2.0]])  # true successor
    assert step2.infos[0]["t"] == 2  # terminal info, not the reset info (t == 0)
    assert step2.infos[1]["t"] == 2
    np.testing.assert_array_equal(step2.rewards, [1.0, 1.0])

    step3 = counting.step(zeros(2))  # env 0 continues its new episode, env 1 truncates at 3
    np.testing.assert_array_equal(step3.obs, [[1.0], [0.0]])
    np.testing.assert_array_equal(step3.final_obs, [[1.0], [3.0]])
    assert step3.infos[0]["t"] == 1
    assert step3.infos[1]["t"] == 3


def test_terminated_and_truncated_are_kept_apart(counting: VecEnv) -> None:
    counting.reset(seed=0)
    steps = [counting.step(zeros(2)) for _ in range(3)]
    np.testing.assert_array_equal(steps[1].terminated, [True, False])
    np.testing.assert_array_equal(steps[1].truncated, [False, False])
    np.testing.assert_array_equal(steps[2].terminated, [False, False])
    np.testing.assert_array_equal(steps[2].truncated, [False, True])
    np.testing.assert_array_equal(steps[2].dones, [False, True])


def test_both_flags_survive_when_set_on_the_same_step() -> None:
    venv = SyncVecEnv([functools.partial(CountingEnv, terminate_at=1, truncate_at=1)])
    venv.reset()
    step = venv.step(zeros(1))
    assert step.terminated[0] and step.truncated[0]
    venv.close()


class _ImageEnv(gym.Env):
    """uint8 frames `(2, 4, 4)`; deliberately returns int64 arrays to check the dtype cast."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(0, 255, shape=(2, 4, 4), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.t = 0

    def _obs(self) -> np.ndarray:
        return np.full((2, 4, 4), 50 * self.t, dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), 0.5, self.t >= 2, False, {}


def test_image_observations_keep_the_space_dtype_and_shape() -> None:
    with SyncVecEnv([_ImageEnv, _ImageEnv, _ImageEnv]) as venv:
        obs = venv.reset(seed=0)
        assert obs.shape == (3, 2, 4, 4) and obs.dtype == np.uint8
        venv.step(zeros(3))
        step = venv.step(zeros(3))  # every env terminates here
        assert step.obs.dtype == np.uint8 and step.final_obs.dtype == np.uint8
        assert step.final_obs.shape == (3, 2, 4, 4)
        assert (step.final_obs == 100).all() and (step.obs == 0).all()
        assert step.rewards.dtype == np.float32 and step.rewards.tolist() == [0.5, 0.5, 0.5]


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #


def test_reset_seeds_env_i_with_seed_plus_i(noise: VecEnv) -> None:
    obs = noise.reset(seed=100)
    assert obs.shape == (3, 3) and obs.dtype == np.float32
    for i in range(3):
        expected, _ = SeededNoiseEnv(episode_length=4).reset(seed=100 + i)
        np.testing.assert_array_equal(obs[i], expected)
    assert [info["seed"] for info in noise.reset_infos] == [100, 101, 102]
    assert len({obs[i].tobytes() for i in range(3)}) == 3


def test_reset_without_seed_continues_the_stream(noise: VecEnv) -> None:
    seeded = noise.reset(seed=7)
    unseeded = noise.reset()
    assert not np.array_equal(seeded, unseeded)
    assert [info["seed"] for info in noise.reset_infos] == [None, None, None]
    np.testing.assert_array_equal(noise.reset(seed=7), seeded)
    np.testing.assert_array_equal(noise.reset(None), unseeded)


def test_matches_a_hand_written_single_env_loop(noise: VecEnv) -> None:
    """Auto-reset must call `reset()` without a seed, exactly like a manual episode loop."""
    actions = np.random.default_rng(0).integers(0, 2, size=(11, 3))
    obs = noise.reset(seed=40)
    steps = [noise.step(a) for a in actions]

    for i in range(3):
        env = SeededNoiseEnv(episode_length=4)
        o, _ = env.reset(seed=40 + i)
        np.testing.assert_array_equal(obs[i], o)
        for t, a in enumerate(actions[:, i]):
            o, r, term, trunc, _ = env.step(int(a))
            np.testing.assert_array_equal(steps[t].final_obs[i], o)
            assert steps[t].rewards[i] == np.float32(r)
            assert steps[t].terminated[i] == term and steps[t].truncated[i] == trunc
            if term or trunc:
                o, _ = env.reset()
            np.testing.assert_array_equal(steps[t].obs[i], o)
    assert sum(int(s.terminated.sum()) for s in steps) == 6  # 3 envs x 2 finished episodes


def _rollout(venv: VecEnv, seed: int, actions: np.ndarray) -> list:
    out = [venv.reset(seed=seed)]
    for a in actions:
        step = venv.step(a)
        out += [step.obs, step.rewards, step.terminated, step.truncated, step.final_obs]
    return out


def test_sync_and_subproc_produce_identical_trajectories() -> None:
    cfg = EnvConfig(id="CartPole-v1", max_episode_steps=15)
    actions = np.random.default_rng(1).integers(0, 2, size=(60, 3))
    rollouts = []
    for kind in KINDS:
        venv = make_vec_env(cfg, n_envs=3, seed=5, kind=kind)
        try:
            rollouts.append(_rollout(venv, 5, actions))
        finally:
            venv.close()
    for a, b in zip(*rollouts, strict=True):
        np.testing.assert_array_equal(a, b)
    terminated = np.stack(rollouts[0][3::5])
    truncated = np.stack(rollouts[0][4::5])
    assert terminated.any() and truncated.any()  # both kinds of episode end were exercised


# --------------------------------------------------------------------------- #
# make_vec_env
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", KINDS)
def test_make_vec_env_cartpole(kind: str) -> None:
    cfg = EnvConfig(id="CartPole-v1")
    with make_vec_env(cfg, n_envs=2, seed=3, kind=kind) as venv:
        assert isinstance(venv, SyncVecEnv if kind == "sync" else SubprocVecEnv)
        assert venv.n_envs == 2
        assert venv.single_observation_space.shape == (4,)
        assert venv.single_action_space == gym.spaces.Discrete(2)
        obs = venv.reset(seed=3)
        assert obs.shape == (2, 4) and obs.dtype == np.float32
        for i in range(2):
            expected, _ = gym.make("CartPole-v1").reset(seed=3 + i)
            np.testing.assert_array_equal(obs[i], expected)

        episodes = []
        for _ in range(200):
            step = venv.step(zeros(2))  # always pushing left ends an episode within ~10 steps
            for i in np.flatnonzero(step.dones):
                episodes.append(step.infos[i]["episode"])
                assert not np.array_equal(step.obs[i], step.final_obs[i])
            if len(episodes) >= 3:
                break
        assert len(episodes) >= 3  # RecordEpisodeStatistics info reaches the caller
        assert all(float(ep["l"]) == float(ep["r"]) for ep in episodes)


def test_make_vec_env_builds_env_i_with_seed_plus_i() -> None:
    cfg = EnvConfig(id="CartPole-v1")
    venv = make_vec_env(cfg, n_envs=3, seed=11)
    assert isinstance(venv, SyncVecEnv)  # default kind
    for i, env in enumerate(venv.envs):
        reference = make_env(cfg, 11 + i)
        samples = [env.action_space.sample() for _ in range(20)]
        assert samples == [reference.action_space.sample() for _ in range(20)]
        reference.close()
    venv.close()


def test_make_vec_env_builds_registered_debug_env_in_workers() -> None:
    cfg = EnvConfig(id=f"mario_play.rl.debug_envs:{COUNTING_ENV_ID}", kwargs={"terminate_at": 2})
    with make_vec_env(cfg, n_envs=2, seed=0, kind="subproc") as venv:
        venv.reset(seed=0)
        venv.step(zeros(2))
        step = venv.step(zeros(2))
        assert step.terminated.all()
        assert all("episode" in info for info in step.infos)


def test_make_vec_env_rejects_unknown_kind_and_bad_n_envs() -> None:
    with pytest.raises(ValueError, match="threaded"):
        make_vec_env(EnvConfig(id="CartPole-v1"), n_envs=2, seed=0, kind="threaded")
    with pytest.raises(ValueError, match="n_envs"):
        make_vec_env(EnvConfig(id="CartPole-v1"), n_envs=0, seed=0)


# --------------------------------------------------------------------------- #
# construction errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", [SyncVecEnv, SubprocVecEnv])
def test_empty_env_fns_raises(cls: type) -> None:
    with pytest.raises(ValueError, match="at least one"):
        cls([])


@pytest.mark.parametrize("kind", KINDS)
def test_mismatched_spaces_raise(kind: str) -> None:
    with pytest.raises(ValueError, match="space"):
        build(kind, [CountingEnv, SeededNoiseEnv])


def test_subproc_rejects_unpicklable_env_fn() -> None:
    with pytest.raises(TypeError, match="picklable"):
        SubprocVecEnv([lambda: CountingEnv()])


def test_subproc_constructor_error_carries_the_worker_traceback() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        SubprocVecEnv([CountingEnv, functools.partial(CountingEnv, terminate_at=0)])
    message = str(excinfo.value)
    assert "Traceback" in message
    assert "ValueError" in message and "terminate_at must be >= 1" in message
    assert "worker 1" in message


# --------------------------------------------------------------------------- #
# worker failures and shutdown
# --------------------------------------------------------------------------- #


def test_subproc_uses_the_spawn_start_method() -> None:
    venv = SubprocVecEnv([CountingEnv])
    try:
        assert venv.start_method == "spawn"
        assert all(type(p).__name__ == "SpawnProcess" for p in venv.processes)
        assert all(p.daemon for p in venv.processes)
    finally:
        venv.close()


def test_subproc_worker_exception_surfaces_with_traceback() -> None:
    venv = SubprocVecEnv([CountingEnv, functools.partial(CountingEnv, fail_at=2)])
    venv.reset(seed=0)
    venv.step(zeros(2))
    with pytest.raises(RuntimeError) as excinfo:
        venv.step(zeros(2))
    message = str(excinfo.value)
    assert "worker 1" in message
    assert "Traceback (most recent call last)" in message
    assert "CountingEnv failed on purpose at t=2" in message
    assert "debug_envs.py" in message
    # a failed vec env shuts itself down: no orphaned workers, further use is refused
    assert all(not p.is_alive() for p in venv.processes)
    with pytest.raises(RuntimeError, match="closed"):
        venv.step(zeros(2))
    venv.close()


def test_sync_env_exception_propagates_unchanged() -> None:
    venv = SyncVecEnv([functools.partial(CountingEnv, fail_at=1)])
    venv.reset()
    with pytest.raises(RuntimeError, match="failed on purpose"):
        venv.step(zeros(1))
    venv.close()


@pytest.mark.parametrize("kind", KINDS)
def test_close_is_idempotent_and_blocks_further_use(kind: str) -> None:
    venv = build(kind, [CountingEnv, CountingEnv])
    venv.reset(seed=0)
    venv.step(zeros(2))
    venv.close()
    venv.close()
    if kind == "subproc":
        assert all(not p.is_alive() for p in venv.processes)
        assert all(p.exitcode == 0 for p in venv.processes)  # joined after a clean exit
    with pytest.raises(RuntimeError, match="closed"):
        venv.step(zeros(2))
    with pytest.raises(RuntimeError, match="closed"):
        venv.reset()


def test_sync_close_closes_every_env() -> None:
    closed = []

    class Tracked(CountingEnv):
        def close(self) -> None:
            closed.append(id(self))

    venv = SyncVecEnv([Tracked, Tracked])
    venv.close()
    venv.close()
    assert len(closed) == 2


@pytest.mark.timeout(30)
def test_subproc_dead_worker_is_reported_and_close_does_not_hang() -> None:
    venv = SubprocVecEnv([CountingEnv, CountingEnv])
    venv.reset(seed=0)
    venv.processes[0].kill()
    venv.processes[0].join(10)
    with pytest.raises(RuntimeError, match="worker 0 died"):
        venv.step(zeros(2))
    venv.close()
    venv.close()
    assert all(not p.is_alive() for p in venv.processes)


@pytest.mark.timeout(30)
def test_subproc_close_with_a_dead_worker_and_no_prior_error() -> None:
    venv = SubprocVecEnv([CountingEnv, CountingEnv])
    venv.processes[1].kill()
    venv.processes[1].join(10)
    venv.close()
    assert all(not p.is_alive() for p in venv.processes)


@pytest.mark.timeout(30)
def test_subproc_workers_stop_when_the_vec_env_is_garbage_collected() -> None:
    venv = SubprocVecEnv([CountingEnv, CountingEnv])
    venv.reset(seed=0)
    processes = list(venv.processes)
    del venv
    gc.collect()
    for p in processes:
        p.join(10)
    assert all(not p.is_alive() for p in processes)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_subproc_workers_ignore_sigint() -> None:
    """Ctrl-C hits the whole process group; workers must survive so the trainer can checkpoint."""
    with SubprocVecEnv([CountingEnv, CountingEnv]) as venv:
        venv.reset(seed=0)
        for process in venv.processes:
            os.kill(process.pid, signal.SIGINT)
        time.sleep(0.2)
        step = venv.step(zeros(2))
        np.testing.assert_array_equal(step.obs, [[1.0], [1.0]])
        assert all(p.is_alive() for p in venv.processes)


@pytest.mark.timeout(30)
def test_subproc_close_with_unread_replies_is_fast_and_clean() -> None:
    """Replies too big for the pipe buffer block the worker until someone reads them."""
    env_fn = functools.partial(SeededNoiseEnv, obs_size=500_000)  # 2 MB per observation
    venv = SubprocVecEnv([env_fn, env_fn])
    venv.reset(seed=0)
    for i in range(2):
        venv._conns[i].send(("step", 0))  # noqa: SLF001 - a step whose reply is never read
    time.sleep(0.3)
    start = time.monotonic()
    venv.close()
    assert time.monotonic() - start < 3.0  # well below the terminate() fallback
    assert [p.exitcode for p in venv.processes] == [0, 0]  # exited on their own, not killed


class _InterruptingConnection:
    """Delegates to a real connection, but the first `recv` is hit by Ctrl-C."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._interrupted = False

    def recv(self):
        if not self._interrupted:
            self._interrupted = True
            raise KeyboardInterrupt
        return self._conn.recv()

    def __getattr__(self, name: str):
        return getattr(self._conn, name)


class _InterruptingSendConnection(_InterruptingConnection):
    """Ctrl-C arrives while the commands are still being sent out."""

    def send(self, message) -> None:
        if not self._interrupted:
            self._interrupted = True
            raise KeyboardInterrupt
        self._conn.send(message)


@pytest.mark.timeout(30)
def test_subproc_interrupt_while_sending_closes_as_well() -> None:
    venv = SubprocVecEnv([CountingEnv, CountingEnv])
    venv.reset(seed=0)
    venv._conns[1] = _InterruptingSendConnection(venv._conns[1])  # noqa: SLF001
    with pytest.raises(KeyboardInterrupt):
        venv.step(zeros(2))  # worker 0 already got its command, worker 1 did not
    assert all(not p.is_alive() for p in venv.processes)
    assert [p.exitcode for p in venv.processes] == [0, 0]
    with pytest.raises(RuntimeError, match="closed"):
        venv.reset()


@pytest.mark.timeout(30)
def test_subproc_interrupted_step_closes_instead_of_desynchronising() -> None:
    venv = SubprocVecEnv([CountingEnv, CountingEnv])
    venv.reset(seed=0)
    venv._conns[1] = _InterruptingConnection(venv._conns[1])  # noqa: SLF001
    with pytest.raises(KeyboardInterrupt):
        venv.step(zeros(2))
    assert all(not p.is_alive() for p in venv.processes)
    with pytest.raises(RuntimeError, match="closed"):
        venv.step(zeros(2))


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process semantics")
def test_subproc_workers_exit_when_the_parent_dies_without_cleanup() -> None:
    """A trainer that gets SIGKILLed must not leave orphaned env workers behind."""
    code = (
        "import os\n"
        "from mario_play.rl.debug_envs import CountingEnv\n"
        "from mario_play.rl.vec_env import SubprocVecEnv\n"
        "venv = SubprocVecEnv([CountingEnv, CountingEnv])\n"
        "venv.reset(seed=0)\n"
        "print(*[p.pid for p in venv.processes], flush=True)\n"
        "os._exit(0)\n"  # no atexit handlers, no close(): like a hard kill
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    pids = [int(pid) for pid in result.stdout.split()]
    assert len(pids) == 2

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    deadline = time.monotonic() + 20
    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(alive(pid) for pid in pids)


@pytest.mark.skipif(sys.platform == "win32", reason="forkserver is POSIX only")
def test_subproc_start_method_is_configurable() -> None:
    with SubprocVecEnv([CountingEnv], start_method="forkserver") as venv:
        assert venv.start_method == "forkserver"
        np.testing.assert_array_equal(venv.reset(seed=0), [[0.0]])


# --------------------------------------------------------------------------- #
# debug envs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "env_fn",
    [
        CountingEnv,
        functools.partial(CountingEnv, terminate_at=None, truncate_at=3),
        SeededNoiseEnv,
    ],
)
def test_debug_envs_pass_the_gymnasium_checker(env_fn) -> None:
    check_env(env_fn(), skip_render_check=True)


def test_debug_envs_are_registered_with_gymnasium() -> None:
    env = gym.make(f"mario_play.rl.debug_envs:{COUNTING_ENV_ID}", terminate_at=2)
    obs, info = env.reset(seed=0)
    np.testing.assert_array_equal(obs, [0.0])
    assert info["seed"] == 0
    env.step(0)
    obs, reward, terminated, truncated, info = env.step(1)
    assert (obs.tolist(), reward, terminated, truncated) == ([2.0], 1.0, True, False)
    assert info == {"t": 2, "action": 1}
    assert env.unwrapped.total_steps == 2 and env.unwrapped.episode_index == 0
    env.close()


def test_counting_env_validates_its_arguments() -> None:
    for kwargs in ({"terminate_at": 0}, {"truncate_at": -1}, {"fail_at": 0}):
        with pytest.raises(ValueError, match=">= 1"):
            CountingEnv(**kwargs)
    with pytest.raises(ValueError, match=">= 1"):
        SeededNoiseEnv(episode_length=0)


def test_env_side_modules_do_not_import_torch() -> None:
    """Spawn workers import these modules; pulling in torch would cost ~1 s and ~200 MB each."""
    code = (
        "import sys\n"
        "import mario_play.rl.vec_env, mario_play.rl.debug_envs\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
