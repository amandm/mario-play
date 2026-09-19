"""Offline proofs of the Taxi experiment's evaluation and PPO integration."""

from __future__ import annotations

import json
import runpy
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from mario_play.envs.factory import make_env
from mario_play.envs.taxi_jev import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    MODEL,
    SCHEMA_VERSION,
    request_for_state,
    specification_sha256,
)
from mario_play.rl.checkpoint import load_checkpoint
from mario_play.rl.utils import get_rng_state
from mario_play.rl.vec_env import SyncVecEnv


def runner():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train_taxi_jev.py"))


@pytest.fixture
def artificial_table(tmp_path):
    """This synthetic table exists only in pytest's temporary directory, never in a real run."""
    records = {}
    for state in range(500):
        probabilities = [((state + offset) % 17) / 16 for offset in range(4)]
        usage = {"input_tokens": 1, "output_tokens": 1}
        records[str(state)] = {
            "state_id": state,
            "request": request_for_state(state),
            "probabilities": probabilities,
            "usage": usage,
            "raw_response": {
                "model": MODEL,
                "answers": {
                    name: {"type": "noul", "noul": probability}
                    for name, probability in zip(FEATURE_NAMES, probabilities, strict=True)
                },
                "usage": usage,
            },
        }
    path = tmp_path / "ARTIFICIAL_TEST_TABLE.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "feature_schema": FEATURE_SCHEMA,
                "model": MODEL,
                "feature_names": list(FEATURE_NAMES),
                "spec_sha256": specification_sha256(),
                "coverage": {"complete": True},
                "records": records,
                "provenance": "SYNTHETIC OFFLINE TEST FIXTURE; NOT ACTUAL JEV OUTPUT",
            }
        )
    )
    return path


def tiny_config(tmp_path, table, condition="zeros", seed=0):
    cfg = runner()["make_config"](tmp_path, table, condition, seed, horizon=128, milestone_steps=64)
    cfg.n_envs = 2
    cfg.ppo.n_steps = 8
    cfg.ppo.n_epochs = cfg.ppo.n_minibatches = 1
    cfg.network.mlp_hidden = [16, 16]
    return cfg


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            same_state(lhs, rhs)
    else:
        assert left == right


def test_evaluation_dynamics_match_real_gym_for_every_state_action(tmp_path, artificial_table):
    cfg = tiny_config(tmp_path, artificial_table)
    evaluator = runner()["TaxiEvaluator"](cfg.env)
    env = make_env(cfg.env)
    try:
        assert len(evaluator.starts) == 300
        for state in range(500):
            for action in range(6):
                env.reset(seed=0, options={"state": state})
                observed, reward, terminated, truncated, info = env.step(action)
                assert reward == evaluator.reward[state, action]
                assert terminated == evaluator.done[state, action]
                assert not truncated
                np.testing.assert_array_equal(
                    observed, evaluator.observations[evaluator.next_state[state, action]]
                )
                assert "action_mask" not in info
    finally:
        env.close()


def test_batched_evaluation_matches_capped_real_episodes_and_only_passes_observations(
    tmp_path, artificial_table
):
    cfg = tiny_config(tmp_path, artificial_table, "rules")
    evaluator = runner()["TaxiEvaluator"](cfg.env)

    class PickupPolicy:
        def predict(self, observations, deterministic):
            assert deterministic and observations.shape[1:] == (71,)
            assert observations.dtype == np.float32
            for row in observations:
                assert np.any(np.all(evaluator.observations == row, axis=1))
            return np.full(len(observations), 4, dtype=np.int64)

    starts = np.array([461, 91, 244])
    result = evaluator.evaluate(PickupPolicy(), starts=starts, cap=7)
    env = make_env(cfg.env)
    try:
        for row in result["episode_results"]:
            env.reset(seed=7, options={"state": row["initial_state"]})
            total, illegal = 0.0, 0
            for _ in range(7):
                _, reward, terminated, _, _ = env.step(4)
                total += reward
                illegal += reward == -10
                assert not terminated
            assert row["return"] == total and row["illegal_pickup_dropoff"] == illegal
            assert row["steps"] == 7 and row["truncated"] and not row["success"]
        assert result["logical_feature_refreshes"] == 21
        assert result["physical_api_requests"] == 0
    finally:
        env.close()


def test_initial_weights_match_and_evaluation_does_not_change_rng(tmp_path, artificial_table):
    script = runner()
    trainers = [
        script["TaxiTrainer"](tiny_config(tmp_path, artificial_table, condition))
        for condition in script["CONDITIONS"]
    ]
    try:
        assert len({trainer.initial_model_sha256 for trainer in trainers}) == 1
        assert {trainer.venv.single_observation_space.shape for trainer in trainers} == {(71,)}
        original = get_rng_state()
        trainers[-1].evaluator.evaluate(trainers[-1].algo, cap=2)
        same_state(get_rng_state(), original)
        initial = load_checkpoint(trainers[-1].ckpt_dir / "initial.pt")
        assert initial["global_step"] == initial["extra"]["ppo_updates"] == 0
        assert initial["extra"]["elapsed"] == 0
    finally:
        for trainer in trainers:
            trainer.close()


def test_taxi_timeout_bootstraps_from_enriched_final_observation(
    tmp_path, artificial_table, monkeypatch
):
    cfg = tiny_config(tmp_path, artificial_table, "jev")
    cfg.env.max_episode_steps = 1
    script = runner()
    trainer = script["TaxiTrainer"](cfg)
    try:
        obs = trainer.venv.reset(seed=0)
        actions, extras = trainer.algo.select_actions(obs, 0)
        transition = trainer.venv.step(actions)
        assert transition.truncated.all() and not transition.terminated.any()
        assert not np.array_equal(transition.obs, transition.final_obs)
        seen = []

        def final_values(values):
            seen.append(values.detach().cpu().numpy().copy())
            return torch.full((len(values),), 7.0, dtype=torch.float32)

        monkeypatch.setattr(trainer.algo.model, "get_value", final_values)
        trainer.algo.observe(obs, actions, extras, transition)
        np.testing.assert_array_equal(seen[0], transition.final_obs)
        np.testing.assert_allclose(
            trainer.algo.buffer.rewards[0].numpy(), transition.rewards + cfg.ppo.gamma * 7
        )
        assert trainer.algo.buffer.dones[0].tolist() == [1.0, 1.0]
    finally:
        trainer.close()


def test_live_ppo_training_saves_each_milestone_with_actual_update_count(
    tmp_path, artificial_table
):
    script = runner()
    trainer = script["TaxiTrainer"](tiny_config(tmp_path, artificial_table))
    result = trainer.train()
    assert not result["interrupted"] and result["global_step"] == 128
    records = json.loads((trainer.run_dir / "milestones.json").read_text())
    assert [row["global_step"] for row in records] == [64, 128]
    assert [row["ppo_updates"] for row in records] == [4, 8]
    assert all(row["assessment"]["episodes"] == 300 for row in records)
    assert all(row["logical_feature_refreshes"] == 0 for row in records)
    assert load_checkpoint(trainer.ckpt_dir / "latest.pt")["extra"]["ppo_updates"] == 8


def test_wrapped_taxi_vector_final_observation_is_distinct_from_auto_reset(
    tmp_path, artificial_table
):
    cfg = tiny_config(tmp_path, artificial_table, "rules")
    cfg.env.max_episode_steps = 1
    with SyncVecEnv([partial(make_env, cfg.env)]) as vec:
        vec.reset(seed=0)
        result = vec.step(np.array([0]))
        assert result.truncated.tolist() == [True]
        assert result.obs.shape == result.final_obs.shape == (1, 71)
        assert not np.array_equal(result.obs, result.final_obs)


def test_report_auc_and_sustained_target_use_the_predeclared_metrics():
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/report_taxi_jev.py"))
    records = [
        {"global_step": step, "assessment": {"success_rate": rate}}
        for step, rate in ((20_000, 0.0), (40_000, 0.5), (60_000, 1.0))
    ]
    assert script["normalized_auc"](records) == pytest.approx(0.5)
    assert script["confirmed_target"](records) is None
    records.extend(
        [
            {"global_step": 80_000, "assessment": {"success_rate": 0.9}},
            {"global_step": 100_000, "assessment": {"success_rate": 0.95}},
        ]
    )
    assert script["confirmed_target"](records) == 100_000


def test_report_partial_run_never_claims_final_metrics(tmp_path):
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/report_taxi_jev.py"))
    (tmp_path / "experiment.json").write_text(
        json.dumps({"seeds": [0], "milestone_steps": 20_000, "horizon": 200_000})
    )
    directory = tmp_path / "zeros/seed_0"
    directory.mkdir(parents=True)
    record = {"global_step": 20_000, "assessment": {"success_rate": 0.1}, "ppo_updates": 19}
    (directory / "milestones.json").write_text(json.dumps([record]))
    data = script["collect"](tmp_path)
    assert not data["complete"]
    row = data["conditions"]["zeros"][0]
    assert not row["complete"]
    assert row["normalized_success_auc"] is None
    assert row["final_success_rate"] is None
    (directory / "milestones.json").write_text(json.dumps([record, record]))
    with pytest.raises(ValueError, match="invalid milestone sequence"):
        script["collect"](tmp_path)
