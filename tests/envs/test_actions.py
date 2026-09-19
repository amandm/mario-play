"""Discrete action sets (spec 4): sizes, exact index order, nesting and lookup."""

from __future__ import annotations

import pytest

from mario_play.envs.actions import ACTION_SETS, action_names, get_action_set
from mario_play.game.engine import Buttons

NOOP = Buttons()
RIGHT = Buttons(right=True)
RIGHT_JUMP = Buttons(right=True, jump=True)
RIGHT_RUN = Buttons(right=True, run=True)
RIGHT_RUN_JUMP = Buttons(right=True, run=True, jump=True)
JUMP = Buttons(jump=True)
LEFT = Buttons(left=True)
LEFT_JUMP = Buttons(left=True, jump=True)
LEFT_RUN = Buttons(left=True, run=True)
LEFT_RUN_JUMP = Buttons(left=True, run=True, jump=True)


def test_the_three_sets_and_their_sizes():
    assert set(ACTION_SETS) == {"right_only", "simple", "complex"}
    assert [len(ACTION_SETS[name]) for name in ("right_only", "simple", "complex")] == [5, 7, 10]


def test_exact_index_order():
    right_only = [NOOP, RIGHT, RIGHT_JUMP, RIGHT_RUN, RIGHT_RUN_JUMP]
    simple = [*right_only, JUMP, LEFT]
    complex_ = [*simple, LEFT_JUMP, LEFT_RUN, LEFT_RUN_JUMP]
    assert ACTION_SETS["right_only"] == right_only
    assert ACTION_SETS["simple"] == simple
    assert ACTION_SETS["complex"] == complex_


@pytest.mark.parametrize("name", ["right_only", "simple", "complex"])
def test_index_zero_is_noop_and_actions_are_unique_buttons(name):
    buttons = ACTION_SETS[name]
    assert buttons[0] == NOOP
    assert all(isinstance(b, Buttons) for b in buttons)
    assert len(set(buttons)) == len(buttons)


def test_no_action_holds_left_and_right_together():
    assert not any(b.left and b.right for buttons in ACTION_SETS.values() for b in buttons)


def test_get_action_set_returns_a_private_copy():
    first = get_action_set("simple")
    assert first == ACTION_SETS["simple"]
    first.append(NOOP)
    assert len(ACTION_SETS["simple"]) == 7
    assert len(get_action_set("simple")) == 7


def test_unknown_action_set_raises_value_error_naming_the_choices():
    with pytest.raises(ValueError, match="right_only"):
        get_action_set("everything")


def test_action_names_are_readable_and_aligned_with_the_buttons():
    assert action_names("simple") == [
        "noop",
        "right",
        "right+jump",
        "right+run",
        "right+run+jump",
        "jump",
        "left",
    ]
    assert len(action_names("complex")) == 10
    assert action_names("complex")[-1] == "left+run+jump"
