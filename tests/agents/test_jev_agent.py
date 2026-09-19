"""Offline proofs of Jev request boundaries, observation fidelity and failure handling."""

from __future__ import annotations

import copy
import io
import json
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from mario_play.agents.jev_agent import (
    ACTION_NAMES,
    API_URL,
    MODEL,
    JevAgent,
    JevBudgetExceeded,
    JevError,
    _NoRedirects,
    semantic_state,
)
from mario_play.envs.observations import GRID_PLANES, GRID_SHAPE, PLAYER_COLUMN


@pytest.fixture
def observation():
    grid = np.zeros(GRID_SHAPE, dtype=np.float32)
    grid[0, 13:, :] = 1
    grid[0, 10:13, 8:10] = 1
    grid[1, 10, 8] = 1
    grid[3, 12, 7:9] = 1
    grid[7, 12, PLAYER_COLUMN] = 1
    grid[8] = 0.5
    grid[9] = -0.25
    grid[10] = 1
    grid[12] = 0.3
    grid[13] = 1
    return grid


def answer(choice="right+run"):
    return {
        "model": MODEL,
        "answers": {
            "action": {
                "type": "choice",
                "choice": choice,
                "probabilities": {name: float(name == choice) for name in ACTION_NAMES},
                "confidence": 1.0,
            }
        },
        "usage": {"input_tokens": 321, "output_tokens": 42},
    }


class FakeOpener:
    def __init__(self, response=None, error=None):
        self.response = answer() if response is None else response
        self.error = error
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append((request, timeout))
        if self.error:
            raise self.error
        raw = (
            self.response
            if isinstance(self.response, bytes)
            else json.dumps(self.response).encode()
        )
        return io.BytesIO(raw)


def test_grid_geometry_survives_json_serialization(observation):
    before = observation.copy()
    state = json.loads(json.dumps(semantic_state(observation)))
    restored = np.zeros_like(observation[:8])
    for plane, name in enumerate(GRID_PLANES[:8]):
        for left, top, right, bottom in state["occupied_tile_rectangles"][name]:
            restored[plane, top : bottom + 1, left + PLAYER_COLUMN : right + PLAYER_COLUMN + 1] = 1
    np.testing.assert_array_equal(restored, observation[:8])
    np.testing.assert_array_equal(before, observation)
    assert state["player"]["vx_pixels_per_frame"] == 1.25
    assert state["player"]["vy_pixels_per_frame_clipped"] == -1.25
    assert state["player"]["x_offset_pixels"] == pytest.approx(4.8)
    assert state["player"]["on_ground"] is True
    assert "world_x" not in state and "reward" not in state


@pytest.mark.parametrize(
    "enemy_column,gap,relation",
    [
        (PLAYER_COLUMN + 2, 1, "1 empty tile columns"),
        (PLAYER_COLUMN + 1, 0, "adjacent"),
        (PLAYER_COLUMN, 0, "overlapping"),
    ],
)
def test_semantic_enemy_distance_preserves_tile_uncertainty(
    observation, enemy_column, gap, relation
):
    observation[3] = 0
    observation[3, 12, enemy_column] = 1
    facts = semantic_state(observation)["semantic_local_facts"]
    enemy = facts["nearest_forward_enemy"]
    assert enemy["body_rows_overlap"] is True
    assert enemy["empty_tile_columns_between"] == gap
    assert relation in enemy["occupied_columns_relation"]
    assert "exact" in facts["distance_precision"].lower()
    assert "grounded, moving right" in facts["player_motion"]
    assert facts["empty_columns_to_solid_at_body_height"] == 3
    assert facts["next_column_ahead_has_solid_below_body"] is True
    assert facts["first_visible_column_ahead_without_solid_below_body"] is None
    assert "action" not in facts and "jump_deadline" not in facts


def test_semantics_distinguish_elevated_enemy_and_visible_ground_gap(observation):
    observation[3] = 0
    observation[3, 8, PLAYER_COLUMN + 1] = 1
    observation[0, 13:, PLAYER_COLUMN + 1] = 0
    facts = semantic_state(observation)["semantic_local_facts"]
    assert facts["nearest_forward_enemy"]["body_rows_overlap"] is False
    assert "above the player's body" in facts["nearest_forward_enemy"]["description"]
    assert facts["next_column_ahead_has_solid_below_body"] is False
    assert facts["first_visible_column_ahead_without_solid_below_body"] == 1


def test_request_contract_and_real_choice_mapping(observation):
    key = "offline-test-secret"
    agent = JevAgent(key, max_calls=2, timeout=3, frame_skip=4)
    opener = FakeOpener(answer("right+run+jump"))
    agent._opener = opener
    assert agent.act(observation) == 4
    request, timeout = opener.requests[0]
    assert request.full_url == API_URL
    assert request.get_header("Authorization") == f"Bearer {key}"
    assert timeout == 3
    body = json.loads(request.data)
    assert set(body["questions"]) == {"action"}
    assert list(body["questions"]["action"]["criteria"]) == list(ACTION_NAMES)
    assert body["state"]["recent_actions_oldest_first"] == []
    assert body["state"]["semantic_local_facts"]["previous_jump_button"] == (
        "unknown at episode start"
    )
    assert agent.last_decision["choice"] == "right+run+jump"
    assert agent.last_decision["call_number"] == 1
    assert agent.last_decision["latency_ms"] >= 0
    assert key not in json.dumps(agent.last_decision)
    assert key not in json.dumps(agent.build_request(observation))
    assert agent.build_request(observation)["state"]["recent_actions_oldest_first"] == [
        "right+run+jump"
    ]
    assert agent.usage == {"input_tokens": 321, "output_tokens": 42}
    assert (
        agent.build_request(observation)["state"]["semantic_local_facts"]["previous_jump_button"]
        == "held during the previous action"
    )


def test_reset_cannot_replenish_budget(observation):
    agent = JevAgent("offline", max_calls=1)
    opener = FakeOpener()
    agent._opener = opener
    agent.act(observation)
    agent.reset()
    assert agent.last_decision is None
    assert agent.build_request(observation)["state"]["recent_actions_oldest_first"] == []
    with pytest.raises(JevBudgetExceeded):
        agent.act(observation)
    assert len(opener.requests) == 1
    assert agent.calls == 1 and agent.usage["input_tokens"] == 321


def test_two_decimal_api_rounding_preserves_reported_distribution(observation):
    response = answer()
    response["answers"]["action"]["probabilities"] = {name: 0.14 for name in ACTION_NAMES}
    agent = JevAgent("offline")
    agent._opener = FakeOpener(response)
    assert agent.act(observation) == 3
    assert agent.last_decision["probabilities_sum"] == pytest.approx(0.98)
    assert agent.last_decision["probabilities"]["right+run"] == 0.14


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "missing",
        "nan",
        "negative",
        "sum",
        "argmax",
        "confidence",
        "model",
        "usage",
        "type",
        "malformed",
    ],
)
def test_invalid_response_never_becomes_an_action(observation, fault):
    response = answer()
    action = response["answers"]["action"]
    if fault == "unknown":
        action["choice"] = "teleport"
    elif fault == "missing":
        del action["probabilities"]["left"]
    elif fault == "nan":
        action["probabilities"]["left"] = float("nan")
    elif fault == "negative":
        action["probabilities"]["left"] = -0.1
    elif fault == "sum":
        action["probabilities"]["left"] = 0.1
    elif fault == "argmax":
        action["choice"] = "left"
    elif fault == "confidence":
        action["confidence"] = True
    elif fault == "model":
        response["model"] = "jev-latest"
    elif fault == "usage":
        response["usage"]["input_tokens"] = -1
    elif fault == "type":
        action["type"] = "noul"
    else:
        response["answers"] = []
    agent = JevAgent("offline", max_calls=1)
    agent._opener = FakeOpener(response)
    with pytest.raises(JevError):
        agent.act(observation)
    assert agent.last_decision is None and agent.calls == 1
    if fault != "usage":
        assert agent.usage["input_tokens"] == 321
    with pytest.raises(JevBudgetExceeded):
        agent.act(observation)


@pytest.mark.parametrize(
    "error",
    [
        URLError("sensitive-response-content"),
        HTTPError(API_URL, 401, "sensitive-response-content", {}, io.BytesIO(b"sensitive-body")),
        TimeoutError("sensitive-response-content"),
    ],
)
def test_http_failures_are_sanitized_and_never_retried(observation, error):
    agent = JevAgent("offline-test-secret", max_calls=2)
    opener = FakeOpener(error=error)
    agent._opener = opener
    with pytest.raises(JevError) as caught:
        agent.act(observation)
    assert "sensitive" not in str(caught.value)
    assert "offline-test-secret" not in str(caught.value)
    assert len(opener.requests) == 1 and agent.calls == 1
    assert agent.usage_missing_calls == 1


@pytest.mark.parametrize("raw", [b"not json", b"[]", b"x" * 1_048_577])
def test_malformed_http_body_is_rejected(observation, raw):
    agent = JevAgent("offline")
    agent._opener = FakeOpener(raw)
    with pytest.raises(JevError):
        agent.act(observation)
    assert agent.usage_missing_calls == 1


def test_bad_observation_does_not_spend_a_request(observation):
    agent = JevAgent("offline")
    opener = FakeOpener()
    agent._opener = opener
    bad = copy.deepcopy(observation)
    bad[8, 0, 0] = 0.3
    with pytest.raises(ValueError):
        agent.act(bad)
    assert not opener.requests and agent.calls == 0


def test_close_removes_key_and_redirects_are_disabled(observation):
    agent = JevAgent("offline")
    agent.close()
    agent.close()
    assert agent._api_key == ""
    with pytest.raises(JevError, match="closed"):
        agent.act(observation)
    assert _NoRedirects().redirect_request(None, None, 302, "", {}, "https://elsewhere") is None
