"""evaluate(): episode accounting, seeding, Mario extras, frames; load_algorithm() round trip."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

import mario_play.rl.evaluate as evaluate_module
from mario_play.envs.factory import make_env
from mario_play.rl.algos.dqn import DQN
from mario_play.rl.algos.ppo import PPO
from mario_play.rl.checkpoint import save_checkpoint
from mario_play.rl.config import EnvConfig, config_to_dict
from mario_play.rl.debug_envs import COUNTING_ENV_ID
from mario_play.rl.evaluate import evaluate, load_algorithm
from mario_play.rl.utils import get_rng_state

CPU = torch.device("cpu")
COUNTING = f"mario_play.rl.debug_envs:{COUNTING_ENV_ID}"
CARTPOLE = EnvConfig(id="CartPole-v1")


def build_algo(cfg, n_envs: int = 1):
    env = make_env(cfg.env)
    cls = PPO if cfg.algo == "ppo" else DQN
    algo = cls(env.observation_space, env.action_space, cfg, CPU, n_envs)
    env.close()
    return algo


def save(algo, cfg, path, global_step: int = 123, best_eval: float | None = 4.5) -> None:
    save_checkpoint(
        path,
        algo_name=cfg.algo,
        algo_state=algo.state_dict(),
        config_dict=config_to_dict(cfg),
        global_step=global_step,
        best_eval=best_eval,
        rng_state=get_rng_state(),
    )


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


def test_result_keys_and_types_on_cartpole(tiny_cfg):
    algo = build_algo(tiny_cfg())
    result = evaluate(algo, CARTPOLE, episodes=3, seed=0)
    assert set(result) == {"mean_return", "std_return", "mean_length", "episodes"}
    assert result["episodes"] == 3
    for key in ("mean_return", "std_return", "mean_length"):
        assert type(result[key]) is float
    # CartPole pays 1 per step, so return and length must agree.
    assert result["mean_return"] == pytest.approx(result["mean_length"])
    assert result["mean_return"] >= 8.0


def test_episode_accounting_matches_a_known_env(constant_policy):
    env_cfg = EnvConfig(id=COUNTING, kwargs={"terminate_at": 5})
    policy = constant_policy(0)
    result = evaluate(policy, env_cfg, episodes=4, seed=0)
    assert result["mean_return"] == 5.0
    assert result["std_return"] == 0.0
    assert result["mean_length"] == 5.0
    assert result["episodes"] == 4
    # Episodes run one after another on a single env: every predict call sees a batch of one.
    assert len(policy.batch_shapes) == 4 * 5
    assert set(policy.batch_shapes) == {(1, 1)}


def test_truncated_episodes_end_too(constant_policy):
    env_cfg = EnvConfig(id=COUNTING, kwargs={"terminate_at": None, "truncate_at": 7})
    result = evaluate(constant_policy(1), env_cfg, episodes=2, seed=0)
    assert result["mean_length"] == 7.0


def test_max_steps_bounds_an_episode_that_never_ends(constant_policy):
    env_cfg = EnvConfig(id=COUNTING, kwargs={"terminate_at": None})
    result = evaluate(constant_policy(0), env_cfg, episodes=2, seed=0, max_steps=11)
    assert result["mean_length"] == 11.0
    assert result["mean_return"] == 11.0


def test_deterministic_flag_reaches_predict(constant_policy):
    env_cfg = EnvConfig(id=COUNTING, kwargs={"terminate_at": 2})
    greedy, sampled = constant_policy(0), constant_policy(0)
    evaluate(greedy, env_cfg, episodes=1, seed=0)
    evaluate(sampled, env_cfg, episodes=1, seed=0, deterministic=False)
    assert set(greedy.deterministic_flags) == {True}
    assert set(sampled.deterministic_flags) == {False}


def test_std_return_is_the_population_std_over_episodes(tiny_cfg):
    algo = build_algo(tiny_cfg())
    returns: list[float] = []

    class Spy:
        def predict(self, obs, deterministic=True):
            return algo.predict(obs, deterministic)

    # Re-run the episodes one by one with the seeds evaluate() uses (seed + episode index).
    for episode in range(4):
        returns.append(evaluate(Spy(), CARTPOLE, episodes=1, seed=50 + episode)["mean_return"])
    result = evaluate(algo, CARTPOLE, episodes=4, seed=50)
    assert result["mean_return"] == pytest.approx(np.mean(returns))
    assert result["std_return"] == pytest.approx(np.std(returns))


def test_same_seed_gives_the_same_result_and_other_seeds_differ(tiny_cfg):
    algo = build_algo(tiny_cfg())
    first = evaluate(algo, CARTPOLE, episodes=5, seed=7)
    again = evaluate(algo, CARTPOLE, episodes=5, seed=7)
    assert first == again
    lengths = {evaluate(algo, CARTPOLE, episodes=5, seed=s)["mean_length"] for s in (7, 8, 9, 10)}
    assert len(lengths) > 1


def test_mario_extras_are_reported_when_the_env_provides_them(flag_run_env_id, constant_policy):
    env_cfg = EnvConfig(id=flag_run_env_id)
    winner = evaluate(constant_policy(1), env_cfg, episodes=3, seed=0)
    assert winner["flag_rate"] == 1.0
    assert winner["mean_progress"] == 1.0
    assert winner["mean_return"] == 4.0
    loser = evaluate(constant_policy(0), env_cfg, episodes=3, seed=0)
    assert loser["flag_rate"] == 0.0
    assert loser["mean_progress"] == 0.0
    assert type(loser["flag_rate"]) is float


def test_extras_are_absent_for_envs_without_them(constant_policy):
    result = evaluate(constant_policy(0), CARTPOLE, episodes=1, seed=0)
    assert "flag_rate" not in result
    assert "mean_progress" not in result


def test_frame_callback_gets_one_frame_per_reset_and_step(constant_policy):
    frames: list[np.ndarray] = []
    result = evaluate(
        constant_policy(0), CARTPOLE, episodes=2, seed=0, frame_callback=frames.append
    )
    assert len(frames) == round(result["mean_length"] * 2) + 2
    assert all(frame.ndim == 3 and frame.shape[2] == 3 for frame in frames)
    assert frames[0].dtype == np.uint8


def test_frame_callback_cannot_be_combined_with_a_human_window(constant_policy):
    with pytest.raises(ValueError, match="rgb_array"):
        evaluate(
            constant_policy(0),
            CARTPOLE,
            episodes=1,
            seed=0,
            render_mode="human",
            frame_callback=lambda frame: None,
        )


@pytest.mark.parametrize("episodes", [0, -1])
def test_needs_at_least_one_episode(constant_policy, episodes):
    with pytest.raises(ValueError, match="episodes"):
        evaluate(constant_policy(0), CARTPOLE, episodes=episodes, seed=0)


def test_builds_its_own_env_and_closes_it_even_when_the_policy_fails(monkeypatch):
    built: list[gym.Env] = []
    closed: list[gym.Env] = []
    real_make_env = evaluate_module.make_env

    def spy_make_env(cfg, seed=None, render_mode=None):
        env = real_make_env(cfg, seed=seed, render_mode=render_mode)
        real_close = env.close
        env.close = lambda: (closed.append(env), real_close())[1]
        built.append(env)
        return env

    class Broken:
        def predict(self, obs, deterministic=True):
            raise RuntimeError("policy exploded")

    monkeypatch.setattr(evaluate_module, "make_env", spy_make_env)
    with pytest.raises(RuntimeError, match="policy exploded"):
        evaluate(Broken(), CARTPOLE, episodes=1, seed=0)
    assert len(built) == 1
    assert closed == built


def test_evaluation_does_not_touch_the_algorithms_training_state(tiny_cfg):
    cfg = tiny_cfg("dqn")
    algo = build_algo(cfg)
    before = algo.state_dict()
    evaluate(algo, CARTPOLE, episodes=2, seed=0, deterministic=False)
    after = algo.state_dict()
    assert before["rng"] == after["rng"]
    assert before["vector_steps"] == after["vector_steps"]
    assert len(algo.buffer) == 0


# --------------------------------------------------------------------------- #
# load_algorithm
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("algo_name", ["ppo", "dqn"])
def test_load_algorithm_rebuilds_the_policy_from_a_checkpoint_alone(tiny_cfg, tmp_path, algo_name):
    cfg = tiny_cfg(algo_name)
    torch.manual_seed(3)
    original = build_algo(cfg, n_envs=cfg.n_envs)
    path = tmp_path / "policy.pt"
    save(original, cfg, path)

    torch.manual_seed(99)  # a different init: equality below can only come from the checkpoint
    loaded, loaded_cfg = load_algorithm(path, device="cpu")
    assert type(loaded) is type(original)
    assert loaded_cfg == cfg
    assert loaded.device == CPU

    obs = np.random.default_rng(0).normal(size=(16, 4)).astype(np.float32)
    np.testing.assert_array_equal(loaded.predict(obs), original.predict(obs))
    assert evaluate(loaded, loaded_cfg.env, episodes=2, seed=5) == evaluate(
        original, cfg.env, episodes=2, seed=5
    )


def test_load_algorithm_builds_dqn_without_the_training_sized_replay_buffer(tiny_cfg, tmp_path):
    cfg = tiny_cfg("dqn", dqn={"buffer_size": 1_000_000, "batch_size": 8})
    path = tmp_path / "policy.pt"
    save(build_algo(tiny_cfg("dqn", dqn={"batch_size": 8})), cfg, path)
    loaded, loaded_cfg = load_algorithm(path, device="cpu")
    assert loaded.buffer.capacity == 8
    assert loaded_cfg.dqn.buffer_size == 1_000_000  # the returned config is the training config


def test_load_algorithm_accepts_a_torch_device(tiny_cfg, tmp_path):
    cfg = tiny_cfg()
    path = tmp_path / "policy.pt"
    save(build_algo(cfg, n_envs=cfg.n_envs), cfg, path)
    loaded, _ = load_algorithm(str(path), device=CPU)
    assert loaded.device == CPU


def test_load_algorithm_closes_its_throwaway_env(tiny_cfg, tmp_path, monkeypatch):
    cfg = tiny_cfg()
    path = tmp_path / "policy.pt"
    save(build_algo(cfg, n_envs=cfg.n_envs), cfg, path)

    closed: list[bool] = []
    real_make_env = evaluate_module.make_env

    def spy_make_env(cfg, seed=None, render_mode=None):
        env = real_make_env(cfg, seed=seed, render_mode=render_mode)
        real_close = env.close
        env.close = lambda: (closed.append(True), real_close())[1]
        return env

    monkeypatch.setattr(evaluate_module, "make_env", spy_make_env)
    load_algorithm(path, device="cpu")
    assert closed == [True]


def test_load_algorithm_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_algorithm(tmp_path / "nope.pt", device="cpu")
