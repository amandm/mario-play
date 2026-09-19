"""Frozen Jev risk assessments derived exclusively from the current grid observation.

The exhaustive bank contains 476 cases: enemy 260, obstacle 80, gap 100 and
overhead 32, plus one no-visible-player case per feature. It directly caches
actual Jev Noul answers, and is not a learned
surrogate. Cases contain no selected actions, rewards, history or simulator state.

Bucket specification (tile occupancy is approximate hitbox geometry):
* Enemy ahead: absent, or horizontal overlap / adjacent / 1-2 empty columns /
  >=3 empty columns, crossed with body-height / above / below = 13 geometries.
* Body-height solid ahead: absent / adjacent-or-overlapping / 1-2 / >=3 empty cols.
* Unsupported visible ground: none / under player / next column / 1-2 supported
  columns before the gap / >=3 supported columns before the gap.
* Overhead solid: absent / adjacent / 1-2 / >=3 empty rows above player's body.
* Motion: still (|normalized vx| <= 1e-6), left/right slow (<=0.6), left/right fast.
* Phase: grounded, rising, falling, airborne level; grounded takes precedence.

The first three assessments cross geometry with five motion and four phase
buckets. Overhead crosses geometry, phase and the observed small/big flag.
All original grid channels remain available to the policy through its wrapper.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

FEATURE_NAMES = (
    "enemy_side_contact_risk",
    "forward_obstacle_risk",
    "visible_gap_risk",
    "overhead_clearance_risk",
)
FEATURE_COUNT = len(FEATURE_NAMES)
MODEL = "jev-1.13.0"
SCHEMA_VERSION = 1
FEATURE_SCHEMA = "jev-current-grid-risk-v1"
GRID_SHAPE = (14, 15, 16)

MOTIONS = ("still", "left_slow", "left_fast", "right_slow", "right_fast")
PHASES = ("grounded", "rising", "falling", "airborne_level")
DISTANCES = ("overlapping", "adjacent", "near", "far")
ENEMY_GEOMETRIES = ("none",) + tuple(
    f"{distance}_{height}"
    for distance, height in itertools.product(DISTANCES, ("body", "above", "below"))
)
OBSTACLE_GEOMETRIES = ("none", "adjacent", "near", "far")
GAP_GEOMETRIES = ("none", "under_player", "next_column", "near", "far")
OVERHEAD_GEOMETRIES = ("none", "adjacent", "near", "far")
QUESTIONS = {
    FEATURE_NAMES[0]: (
        "Is side contact with the described visible enemy an immediate danger for the player, "
        "considering the current separation, body height and observed motion? Assess danger, "
        "not which controller action should be chosen."
    ),
    FEATURE_NAMES[1]: (
        "Is the described solid obstacle an immediate obstruction to the player's current "
        "horizontal movement at body height? Assess obstruction, not a preferred action."
    ),
    FEATURE_NAMES[2]: (
        "Does the visible lack of solid support present an immediate falling danger for the "
        "player, considering its location and the player's current motion? Assess danger, "
        "not a preferred controller action."
    ),
    FEATURE_NAMES[3]: (
        "Does the described overhead solid present an immediate headroom obstruction to "
        "the player's current vertical motion? Assess obstruction, not a preferred action."
    ),
}
MOTION_TEXT = {
    "still": "Horizontally still.",
    "left_slow": "Moving left, speed greater than zero and at most 1.5 pixels per frame.",
    "left_fast": "Moving left, speed greater than 1.5 and at most 2.5 pixels per frame.",
    "right_slow": "Moving right, speed greater than zero and at most 1.5 pixels per frame.",
    "right_fast": "Moving right, speed greater than 1.5 and at most 2.5 pixels per frame.",
}
PHASE_TEXT = {
    "grounded": "Grounded.",
    "rising": "Airborne and rising.",
    "falling": "Airborne and falling.",
    "airborne_level": "Airborne with no vertical motion.",
}
DISTANCE_TEXT = {
    "overlapping": "Occupied horizontal tile columns overlap.",
    "adjacent": "Occupied tile bounds are adjacent, with no completely empty tile between.",
    "near": "One or two empty tile columns separate occupied bounds.",
    "far": "At least three empty tile columns separate occupied bounds, inside the visible window.",
}


def canonical_json(value: Any) -> bytes:
    """Stable JSON bytes for case identities and provenance hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _geometry_text(feature: str, geometry: str) -> str:
    if geometry == "player_not_visible":
        return (
            "The current observation contains no visible player occupancy, so local "
            "geometry relative to the player cannot be determined."
        )
    if feature == FEATURE_NAMES[0]:
        if geometry == "none":
            return "No enemy is observed ahead of or horizontally overlapping the player."
        distance, height = geometry.split("_")
        height_text = {
            "body": "overlaps the player's body rows",
            "above": "is above the player",
            "below": "is below the player",
        }[height]
        return (
            f"The nearest observed enemy ahead or horizontally overlapping {height_text}. "
            + (DISTANCE_TEXT[distance])
        )
    if feature == FEATURE_NAMES[1]:
        if geometry == "none":
            return "No solid tile is observed ahead at the player's occupied body rows."
        return "There is a solid obstacle ahead at body height. " + DISTANCE_TEXT[geometry]
    if feature == FEATURE_NAMES[2]:
        return {
            "none": "All visible columns under and ahead of the player have solid below the body.",
            "under_player": "At least one occupied player column has no visible solid "
            "below the body.",
            "next_column": "The next tile column beyond the player's occupied right edge has no "
            "visible solid below the body.",
            "near": "One or two supported tile columns separate the player's occupied right edge "
            "from a column with no visible solid below the body.",
            "far": "At least three supported tile columns separate the player's occupied right "
            "edge from a visible column with no solid below the body.",
        }[geometry]
    return {
        "none": "No solid is visible above the player's occupied horizontal columns.",
        "adjacent": "A solid tile is immediately above the player's occupied top row.",
        "near": "One or two empty tile rows separate the player's top occupied row "
        "from solid above.",
        "far": "At least three empty rows separate the player's top occupied row from solid above.",
    }[geometry]


@dataclass(frozen=True)
class FeatureCase:
    """One finite assessment context; its API payload contains no credentials."""

    feature: str
    geometry: str
    phase: str
    motion: str | None = None
    size: str | None = None

    @property
    def key(self) -> str:
        return "|".join((self.feature, self.geometry, self.motion or self.size or "", self.phase))

    def context(self) -> dict[str, str]:
        result = {"geometry": self.geometry, "phase": self.phase}
        if self.motion is not None:
            result["motion"] = self.motion
        if self.size is not None:
            result["size"] = self.size
        return result

    def request(self) -> dict[str, Any]:
        state = {
            "scene": "An original side-scrolling platform game; one tile is 16 pixels. "
            "Coordinates increase rightward and downward. Side contact with enemies "
            "and falling into unsupported space are hazards; solid tiles block movement.",
            "observed_relation": _geometry_text(self.feature, self.geometry),
            "vertical_state": PHASE_TEXT.get(self.phase, "Not observed in this assessment."),
            "precision": "These are buckets of the current observed tile grid, not exact "
            "hitboxes or a simulation. No future action, outcome, reward or history "
            "is provided. Assess only the described local risk.",
        }
        if self.motion is not None:
            state["horizontal_motion"] = MOTION_TEXT[self.motion]
        if self.size is not None:
            state["player_size"] = self.size
        return {
            "model": MODEL,
            "state": state,
            "questions": {"risk": {"type": "noul", "instructions": QUESTIONS[self.feature]}},
        }


def case_bank() -> tuple[FeatureCase, ...]:
    """Every possible runtime lookup, including uncommon but representable observations."""
    cases = []
    for feature, geometries in zip(
        FEATURE_NAMES[:3], (ENEMY_GEOMETRIES, OBSTACLE_GEOMETRIES, GAP_GEOMETRIES), strict=True
    ):
        cases.extend(
            FeatureCase(feature, geometry, phase, motion=motion)
            for geometry, motion, phase in itertools.product(geometries, MOTIONS, PHASES)
        )
    cases.extend(
        FeatureCase(FEATURE_NAMES[3], geometry, phase, size=size)
        for geometry, size, phase in itertools.product(
            OVERHEAD_GEOMETRIES, ("small", "big"), PHASES
        )
    )
    cases.extend(
        FeatureCase(feature, "player_not_visible", "not_observed") for feature in FEATURE_NAMES
    )
    return tuple(cases)


def specification_sha256() -> str:
    """Hash both the exhaustive bucket cases and their exact assessment prompts."""
    return hashlib.sha256(
        canonical_json(
            [
                {"key": case.key, "context": case.context(), "request": case.request()}
                for case in case_bank()
            ]
        )
    ).hexdigest()


def _distance(empty: int) -> str:
    return "adjacent" if empty <= 0 else "near" if empty <= 2 else "far"


def cases_for_observation(obs: np.ndarray) -> tuple[FeatureCase, ...]:
    """Quantize only the supplied grid; never query the environment or synthesize a score."""
    grid = np.asarray(obs)
    if grid.shape != GRID_SHAPE or grid.dtype.kind not in "fiu":
        raise ValueError("Jev features require a numeric (14, 15, 16) grid")
    if not np.isfinite(grid).all() or np.any(grid < -1) or np.any(grid > 1):
        raise ValueError("Jev grid values must be finite and within [-1, 1]")
    if not np.isin(grid[:8], (0, 1)).all():
        raise ValueError("Jev spatial grid channels must be binary")
    scalars = grid[8:, 0, 0]
    if not np.all(grid[8:] == scalars[:, None, None]):
        raise ValueError("Jev grid scalar channels must be spatially constant")
    rows, cols = np.nonzero(grid[7])
    if not len(rows):
        return tuple(
            FeatureCase(feature, "player_not_visible", "not_observed") for feature in FEATURE_NAMES
        )
    top, bottom, left, right = map(int, (rows.min(), rows.max(), cols.min(), cols.max()))
    vx, vy = float(scalars[0]), float(scalars[1])
    motion = (
        "still"
        if abs(vx) <= 1e-6
        else ("left" if vx < 0 else "right")
        + ("_slow" if abs(vx) <= float(np.float32(0.6)) else "_fast")
    )
    phase = (
        "grounded"
        if scalars[2] > 0.5
        else "rising"
        if vy < -1e-6
        else "falling"
        if vy > 1e-6
        else "airborne_level"
    )
    enemies = [
        (int(row), int(col)) for row, col in zip(*np.nonzero(grid[3]), strict=True) if col >= left
    ]
    enemy = "none"
    if enemies:
        row, col = min(
            enemies,
            key=lambda cell: (max(0, cell[1] - right), max(top - cell[0], cell[0] - bottom, 0)),
        )
        distance = "overlapping" if col <= right else _distance(col - right - 1)
        height = "above" if row < top else "below" if row > bottom else "body"
        enemy = f"{distance}_{height}"
    solids = np.flatnonzero(np.any(grid[0, top : bottom + 1], axis=0))
    solids = [int(col) for col in solids if col >= right]
    obstacle = _distance(solids[0] - right - 1) if solids else "none"
    unsupported = [
        col for col in range(left, grid.shape[2]) if not np.any(grid[0, bottom + 1 :, col])
    ]
    gap = "none"
    if unsupported:
        col = unsupported[0]
        gap = (
            "under_player"
            if col <= right
            else "next_column"
            if col == right + 1
            else "near"
            if col - right - 1 <= 2
            else "far"
        )
    above = np.flatnonzero(np.any(grid[0, :top, left : right + 1], axis=1))
    overhead = _distance(top - int(above[-1]) - 1) if len(above) else "none"
    return (
        FeatureCase(FEATURE_NAMES[0], enemy, phase, motion=motion),
        FeatureCase(FEATURE_NAMES[1], obstacle, phase, motion=motion),
        FeatureCase(FEATURE_NAMES[2], gap, phase, motion=motion),
        FeatureCase(FEATURE_NAMES[3], overhead, phase, size="big" if scalars[3] > 0.5 else "small"),
    )


def validate_response(response: Any) -> tuple[float, dict[str, int]]:
    """Validate one real response without including response contents in errors."""
    try:
        answer = response["answers"]["risk"]
        probability = answer["noul"]
        usage = response["usage"]
        if response["model"] != MODEL or answer["type"] != "noul":
            raise ValueError("Jev response has an unexpected model or answer type")
        if isinstance(probability, bool) or not isinstance(probability, (float, int)):
            raise ValueError("Jev probability must be numeric")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Jev probability must be finite and within [0, 1]")
        if any(
            isinstance(usage[key], bool) or not isinstance(usage[key], int) or usage[key] < 0
            for key in ("input_tokens", "output_tokens")
        ):
            raise ValueError("Jev token usage must be nonnegative integers")
    except (KeyError, TypeError):
        raise ValueError("Jev returned a malformed risk response") from None
    return float(probability), {key: usage[key] for key in ("input_tokens", "output_tokens")}


@dataclass(frozen=True)
class JevFeatureTable:
    """Immutable direct lookup of verified Jev outputs, identified by file SHA256."""

    sha256: str
    probabilities: Any

    def features(self, obs: np.ndarray) -> np.ndarray:
        """Return four actual cached Jev probabilities; missing cases never receive defaults."""
        return np.asarray(
            [self.probabilities[case.key] for case in cases_for_observation(obs)], dtype=np.float32
        )


_LOADED_TABLES: dict[str, JevFeatureTable] = {}


def load_table(path: str | Path, expected_sha256: str | None = None) -> JevFeatureTable:
    """Verify current file bytes on every load; reuse parsing only for identical content."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Jev table SHA256 does not match the configured artifact")
    if digest in _LOADED_TABLES:
        return _LOADED_TABLES[digest]
    data = json.loads(raw)
    bank = case_bank()
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("model") != MODEL
        or data.get("feature_schema") != FEATURE_SCHEMA
        or data.get("feature_names") != list(FEATURE_NAMES)
        or data.get("spec_sha256") != specification_sha256()
    ):
        raise ValueError("Jev table schema, model or specification does not match")
    if data.get("coverage", {}).get("complete") is not True:
        raise ValueError("Jev table is incomplete")
    records = data.get("records", {})
    if not isinstance(records, dict) or set(records) != {case.key for case in bank}:
        raise ValueError("Jev table must cover every declared case exactly once")
    probabilities = {}
    for case in bank:
        record = records[case.key]
        probability, usage = validate_response(record["raw_response"])
        if (
            record.get("context") != case.context()
            or record.get("feature") != case.feature
            or record.get("request") != case.request()
            or record.get("probability") != probability
            or record.get("usage") != usage
        ):
            raise ValueError("Jev table record does not match its request and raw response")
        probabilities[case.key] = probability
    table = JevFeatureTable(digest, MappingProxyType(probabilities))
    _LOADED_TABLES[digest] = table
    return table


load_feature_table = load_table


def features_for_observation(obs: np.ndarray, table: JevFeatureTable) -> np.ndarray:
    """Compatibility function for callers that prefer a standalone feature adapter."""
    return table.features(obs)
