"""Discrete action sets: each action index stands for one `Buttons` combination.

The sets are nested (`right_only` is a prefix of `simple`, `simple` of `complex`),
so an index keeps its meaning across sets and index 0 is always "no buttons".
"""

from __future__ import annotations

from mario_play.game.engine import Buttons

_RIGHT_ONLY: tuple[Buttons, ...] = (
    Buttons(),
    Buttons(right=True),
    Buttons(right=True, jump=True),
    Buttons(right=True, run=True),
    Buttons(right=True, run=True, jump=True),
)
_SIMPLE: tuple[Buttons, ...] = (
    *_RIGHT_ONLY,
    Buttons(jump=True),
    Buttons(left=True),
)
_COMPLEX: tuple[Buttons, ...] = (
    *_SIMPLE,
    Buttons(left=True, jump=True),
    Buttons(left=True, run=True),
    Buttons(left=True, run=True, jump=True),
)

ACTION_SETS: dict[str, list[Buttons]] = {
    "right_only": list(_RIGHT_ONLY),
    "simple": list(_SIMPLE),
    "complex": list(_COMPLEX),
}
"""Action set name -> buttons per action index (5, 7 and 10 actions)."""


def get_action_set(name: str) -> list[Buttons]:
    """A new list with the buttons of action set `name`; `ValueError` for an unknown name."""
    try:
        return list(ACTION_SETS[name])
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown action set {name!r}; choose one of {', '.join(ACTION_SETS)}"
        ) from None


def button_name(buttons: Buttons) -> str:
    """Readable name of a button combination: `"noop"`, `"right+run+jump"`, ..."""
    held = [
        name
        for name, down in (
            ("left", buttons.left),
            ("right", buttons.right),
            ("run", buttons.run),
            ("jump", buttons.jump),
        )
        if down
    ]
    return "+".join(held) or "noop"


def action_names(name: str) -> list[str]:
    """Readable names of the actions of set `name`, aligned with the action indices."""
    return [button_name(buttons) for buttons in get_action_set(name)]
