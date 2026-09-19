"""Taxi-v4 observations and frozen Jev assessment features for three matched PPO arms.

Every arm receives 19 one-hot decoded state facts, 40 fixed wall-edge bits,
eight fixed landmark coordinates, and four auxiliary values: zeros, exact
observable-state rules, or actual cached Jev Noul probabilities. No action masks,
policy labels, rewards, history, path search or future simulation enter features.
These deliberately simple observable facts test Jev extraction against equivalent
cheap coded features; they do not constitute a new planning algorithm.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from gymnasium.envs.toy_text.taxi import MAP

MODEL = "jev-1.13.0"
SCHEMA_VERSION = 1
FEATURE_SCHEMA = "taxi-observable-relations-v1"
GYM_ID = "TaxiJev-v0"
FEATURE_NAMES = (
    "pickup_ready",
    "dropoff_ready",
    "active_goal_within_two_manhattan_steps",
    "same_row_direct_route_blocked",
)
FEATURE_COUNT = len(FEATURE_NAMES)
LANDMARKS = ((0, 0), (0, 4), (4, 0), (4, 3))
LANDMARK_NAMES = ("R", "G", "Y", "B")
HORIZONTAL_WALLS = tuple(
    (row, col) for row in range(5) for col in range(4) if MAP[row + 1][2 * col + 2] == "|"
)
RAW_FEATURE_COUNT = 19 + 40 + 8
OBSERVATION_SIZE = RAW_FEATURE_COUNT + FEATURE_COUNT
QUESTIONS = {
    "pickup_ready": {
        "type": "noul",
        "instructions": "The passenger is currently outside the taxi, and the taxi occupies "
        "the passenger's current landmark location. Is this statement true? Assess the "
        "present state only; do not recommend an action.",
    },
    "dropoff_ready": {
        "type": "noul",
        "instructions": "The passenger is currently inside the taxi, and the taxi occupies "
        "the passenger's destination landmark. Is this statement true? Assess the present "
        "state only; do not recommend an action.",
    },
    "active_goal_within_two_manhattan_steps": {
        "type": "noul",
        "instructions": "Define the active goal as the passenger's current landmark when "
        "the passenger is outside the taxi, or the destination landmark when aboard. "
        "Ignoring walls, is the absolute row difference plus absolute column difference "
        "between taxi and active goal at most two? This is a geometry assessment, "
        "not a recommendation or shortest-path calculation.",
    },
    "same_row_direct_route_blocked": {
        "type": "noul",
        "instructions": "Define the active goal as the passenger's current landmark when "
        "outside the taxi, or the destination landmark when aboard. Are BOTH statements "
        "true: taxi and active goal share a row, AND at least one listed blocked horizontal "
        "edge lies strictly along the column segment between their cells? If they share "
        "the same cell, or have different rows, answer false. Assess current map geometry "
        "only, without recommending an action or planning a route.",
    },
}


def canonical_json(value: Any) -> bytes:
    """Serialize credential-free specification data deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode_state(state: int) -> tuple[int, int, int, int]:
    """Decode Gymnasium's public Taxi state ID without consulting simulator internals."""
    if isinstance(state, bool) or not isinstance(state, (int, np.integer)) or not 0 <= state < 500:
        raise ValueError("Taxi state must be an integer from 0 through 499")
    state = int(state)
    destination = state % 4
    state //= 4
    passenger = state % 5
    state //= 5
    return state // 5, state % 5, passenger, destination


def encode_state(row: int, col: int, passenger: int, destination: int) -> int:
    """Encode four public Taxi state facts, rejecting out-of-range coordinates."""
    values = (row, col, passenger, destination)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 <= value < limit
        for value, limit in zip(values, (5, 5, 5, 4), strict=True)
    ):
        raise ValueError("Invalid decoded Taxi state")
    return int(((row * 5 + col) * 5 + passenger) * 4 + destination)


def valid_start_states() -> tuple[int, ...]:
    """The 300 default starts: passenger outside taxi and distinct from destination."""
    return tuple(
        state
        for state in range(500)
        if decode_state(state)[2] < 4 and decode_state(state)[2] != decode_state(state)[3]
    )


def static_map_facts() -> dict[str, Any]:
    """The same fixed topology and landmarks encoded in every arm's raw inputs."""
    return {
        "rows": 5,
        "columns": 5,
        "coordinate_convention": "row increases south; column east",
        "landmarks": {
            name: {"row": row, "column": col}
            for name, (row, col) in zip(LANDMARK_NAMES, LANDMARKS, strict=True)
        },
        "blocked_horizontal_edges": [
            {"row": row, "left_column": col, "right_column": col + 1}
            for row, col in HORIZONTAL_WALLS
        ],
        "blocked_vertical_edges": [],
        "boundaries": "Movement beyond the 5-by-5 grid is blocked.",
    }


def state_facts(state: int) -> dict[str, Any]:
    """Only decoded current facts; derived assessment answers are deliberately absent."""
    row, col, passenger, destination = decode_state(state)
    return {
        "taxi": {"row": row, "column": col},
        "passenger_location": "inside_taxi" if passenger == 4 else LANDMARK_NAMES[passenger],
        "destination": LANDMARK_NAMES[destination],
        "map": static_map_facts(),
    }


def request_for_state(state: int) -> dict[str, Any]:
    """Four independent Noul assessments in a single request; no credentials included."""
    return {
        "model": MODEL,
        "state": state_facts(state),
        "questions": {name: dict(question) for name, question in QUESTIONS.items()},
    }


def specification_sha256() -> str:
    """Hash fixed assessment prompts, topology, state encoding and observation layout."""
    return hashlib.sha256(
        canonical_json(
            {
                "feature_schema": FEATURE_SCHEMA,
                "questions": QUESTIONS,
                "map": static_map_facts(),
                "state_encoding": "((row*5+column)*5+passenger)*4+destination",
                "raw_layout": "one-hot row5,column5,passenger5,destination4; horizontal walls20; "
                "vertical walls20; landmarks R,G,Y,B row,column divided by4",
                "auxiliary_order": FEATURE_NAMES,
            }
        )
    ).hexdigest()


def rule_features(state: int) -> np.ndarray:
    """Exact truth of the four requested quantities using only public state and map facts."""
    row, col, passenger, destination = decode_state(state)
    aboard = passenger == 4
    goal_row, goal_col = LANDMARKS[destination if aboard else passenger]
    pickup = not aboard and (row, col) == LANDMARKS[passenger]
    dropoff = aboard and (row, col) == LANDMARKS[destination]
    nearby = abs(row - goal_row) + abs(col - goal_col) <= 2
    blocked = row == goal_row and any(
        (row, edge) in HORIZONTAL_WALLS for edge in range(min(col, goal_col), max(col, goal_col))
    )
    return np.asarray((pickup, dropoff, nearby, blocked), dtype=np.float32)


def raw_observation(state: int) -> np.ndarray:
    """Encode identical current-state and map information for all experimental arms."""
    row, col, passenger, destination = decode_state(state)
    result = np.zeros(RAW_FEATURE_COUNT, dtype=np.float32)
    result[row] = result[5 + col] = result[10 + passenger] = result[15 + destination] = 1
    result[19:39] = [(r, c) in HORIZONTAL_WALLS for r in range(5) for c in range(4)]
    # All twenty vertical neighbor edges are open in the fixed default Taxi map.
    result[59:67] = np.asarray(LANDMARKS, dtype=np.float32).reshape(-1) / 4
    return result


def validate_response(response: Any) -> tuple[list[float], dict[str, int]]:
    """Validate all four actual outputs and reported usage without echoing response content."""
    try:
        if response["model"] != MODEL or set(response["answers"]) != set(FEATURE_NAMES):
            raise ValueError("Taxi Jev response model or question coverage differs")
        probabilities = []
        for name in FEATURE_NAMES:
            answer = response["answers"][name]
            value = answer["noul"]
            if (
                answer["type"] != "noul"
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Taxi Jev output is not a valid Noul probability")
            probabilities.append(float(value))
        usage = response["usage"]
        if any(
            isinstance(usage[key], bool) or not isinstance(usage[key], int) or usage[key] < 0
            for key in ("input_tokens", "output_tokens")
        ):
            raise ValueError("Taxi Jev token usage is invalid")
    except (KeyError, TypeError):
        raise ValueError("Taxi Jev returned a malformed response") from None
    return probabilities, {key: usage[key] for key in ("input_tokens", "output_tokens")}


@dataclass(frozen=True)
class TaxiJevTable:
    """A complete direct cache of actual Jev outputs, with immutable backing bytes."""

    sha256: str
    probabilities: np.ndarray

    def features(self, state: int) -> np.ndarray:
        """Read four cached assessments; no network call or invented fallback is possible."""
        decode_state(state)
        return self.probabilities[int(state)]


_TABLE_CACHE: dict[str, TaxiJevTable] = {}


def load_table(path: str | Path, expected_sha256: str | None = None) -> TaxiJevTable:
    """Verify current bytes before reusing any cached immutable table."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Taxi Jev table SHA256 differs from the configured artifact")
    if digest in _TABLE_CACHE:
        return _TABLE_CACHE[digest]
    data = json.loads(raw)
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("feature_schema") != FEATURE_SCHEMA
        or data.get("model") != MODEL
        or data.get("feature_names") != list(FEATURE_NAMES)
        or data.get("spec_sha256") != specification_sha256()
        or data.get("coverage", {}).get("complete") is not True
    ):
        raise ValueError("Taxi Jev table schema, prompts or completeness do not match")
    records = data.get("records", {})
    if not isinstance(records, dict) or set(records) != {str(state) for state in range(500)}:
        raise ValueError("Taxi Jev table must cover all 500 encoded states")
    values = []
    for state in range(500):
        record = records[str(state)]
        probabilities, usage = validate_response(record["raw_response"])
        if (
            record.get("request") != request_for_state(state)
            or record.get("state_id") != state
            or record.get("probabilities") != probabilities
            or record.get("usage") != usage
        ):
            raise ValueError("Taxi Jev cached values differ from their actual request/response")
        values.append(probabilities)
    immutable = np.frombuffer(np.asarray(values, dtype=np.float32).tobytes(), dtype=np.float32)
    table = TaxiJevTable(digest, immutable.reshape(500, FEATURE_COUNT))
    _TABLE_CACHE[digest] = table
    return table


def make_observation_table(feature_mode: str, table: TaxiJevTable | None = None) -> np.ndarray:
    """Precompute all 500 equally sized observations for training and batched evaluation."""
    if feature_mode not in {"zeros", "rules", "jev"}:
        raise ValueError("Taxi feature_mode must be zeros, rules or jev")
    if feature_mode == "jev" and table is None:
        raise ValueError("The Jev arm requires a complete verified table")
    rows = []
    for state in range(500):
        auxiliary = (
            np.zeros(FEATURE_COUNT, dtype=np.float32)
            if feature_mode == "zeros"
            else rule_features(state)
            if feature_mode == "rules"
            else table.features(state)
        )
        rows.append(np.concatenate((raw_observation(state), auxiliary)))
    return np.frombuffer(np.asarray(rows, dtype=np.float32).tobytes(), dtype=np.float32).reshape(
        500, OBSERVATION_SIZE
    )


class TaxiJevEnv(gym.Wrapper):
    """Matched observation wrapper; default Taxi transitions, rewards and time limit persist."""

    def __init__(
        self,
        env: gym.Env | None = None,
        *,
        feature_mode: str = "zeros",
        table_path: str | Path | None = None,
        table_sha256: str | None = None,
        render_mode: str | None = None,
    ) -> None:
        base = env if env is not None else gym.make("Taxi-v4", render_mode=render_mode)
        super().__init__(base)
        if (
            getattr(base.unwrapped, "is_rainy", False)
            or getattr(base.unwrapped, "fickle_passenger", False)
            or np.asarray(base.unwrapped.desc).tolist()
            != np.asarray([list(row) for row in MAP], dtype="c").tolist()
        ):
            raise ValueError("Taxi comparison requires the fixed default deterministic map")
        self.feature_mode = feature_mode
        self.feature_table = None
        if feature_mode == "jev":
            if table_path is None or table_sha256 is None:
                raise ValueError("The Jev arm requires table_path and its frozen table_sha256")
            self.feature_table = load_table(table_path, table_sha256)
        self.observation_table = make_observation_table(feature_mode, self.feature_table)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.feature_refreshes = 0
        self.physical_api_calls = 0

    def encode_observation(self, state: int) -> np.ndarray:
        """Look up public state features; preserve immutable precomputation against caller edits."""
        decode_state(state)
        return self.observation_table[int(state)].copy()

    @staticmethod
    def _without_mask(info: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in info.items() if key != "action_mask"}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Allow an explicit evaluation state after normal reset, preserving TimeLimit reset."""
        options = dict(options or {})
        fixed_state = options.pop("state", None)
        if fixed_state is not None:
            decode_state(fixed_state)
        state, info = self.env.reset(seed=seed, options=options)
        if fixed_state is not None:
            state = self.env.unwrapped.s = int(fixed_state)
            self.env.unwrapped.lastaction = None
        if self.feature_mode != "zeros":
            self.feature_refreshes += 1
        return self.encode_observation(state), self._without_mask(info)

    def step(self, action):
        """Execute the unmodified Gymnasium transition; no policy restrictions or reward shaping."""
        state, reward, terminated, truncated, info = self.env.step(action)
        if self.feature_mode != "zeros":
            self.feature_refreshes += 1
        return (
            self.encode_observation(state),
            reward,
            terminated,
            truncated,
            self._without_mask(info),
        )


def make_taxi_jev_env(**kwargs) -> TaxiJevEnv:
    """Gymnasium factory used by the existing PPO trainer through EnvConfig.kwargs."""
    return TaxiJevEnv(**kwargs)


if GYM_ID not in registry:
    register(id=GYM_ID, entry_point="mario_play.envs.taxi_jev:make_taxi_jev_env")
