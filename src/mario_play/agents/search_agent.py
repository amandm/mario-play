"""`SearchAgent`: receding-horizon search over macro-plans, simulated on `Game.clone()` copies.

At every decision the agent copies the live game and plays each plan of a small
fixed library ("run right", "run and hold jump for j steps", "wait, then jump",
"step back, run up and jump", "run on, then stop", ...) on its own copy for
`horizon` env steps. Reaching the flag beats everything, otherwise further right
is better (and sooner a little better than later). A plan that dies is worth the
last spot on the ground before its end from which some library plan survives -
the agent stops there and plans again - and a plain death if there is none. The
scripted part of the winner (at least `commit` actions) is executed, then the
agent plans again; the unexecuted rest of the winner competes in that decision.

When no plan reaches new ground (a wide pit and no speed, a wall that needs an
exact take-off, plans that undo each other) a breadth-first search over short
macros looks for a way to stand safely a little further right than ever before.

The copies are stepped exactly like `MarioEnv.step` steps the live game (the
action's buttons held for `frame_skip` frames, stopping when the game ends), and
the game is deterministic, so a plan's simulated outcome is what the real env
will do. Nothing here draws random numbers and ties go to the earlier plan, so
the agent is deterministic too. The live game is only ever read.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from mario_play.envs.actions import button_name
from mario_play.game.engine import Buttons, Game

Segment = tuple[str, int]
"""`(button combination name as in `button_name`, duration in units of `PLAN_UNIT_FRAMES`)`."""

PLAN_UNIT_FRAMES = 4
"""Game frames per unit of a `Segment` duration (one env step at the default `frame_skip`)."""
DEFAULT_HORIZON_FRAMES = 120
DEFAULT_COMMIT_FRAMES = 16
DEFAULT_SETTLE_FRAMES = 96
"""Defaults are game frames, converted to env steps with the env's `frame_skip`."""

COAST_UNITS = 12
"""Plan units a surviving plan gets to come to rest after its horizon (it has to survive that)."""

CHECKPOINT_TRIES = 3
CHECKPOINT_STRIDE = 3
"""A deadly plan: how many of its grounded steps are checked for a way out, and how far apart."""

MIN_PROGRESS_PX = 8.0
"""A decision that promises less new ground than this triggers the detour search."""
STALL_FRAMES = 480
"""This long without new ground also triggers the detour search."""
DETOUR_GOAL_PX = 16.0
"""The detour search looks for safe ground this far beyond the furthest x so far."""
DETOUR_MAX_NODES = 6000
DETOUR_MAX_GLIDE = 30  # plan units a detour jump may stay in the air
DETOUR_COOLDOWN = 2  # decisions without detour search after a fruitless one; doubles each time

_RUN = "right+run"
_RUN_JUMP = "right+run+jump"
_WALK = "right"
_WALK_JUMP = "right+jump"
_NOOP = "noop"
_JUMP = "jump"
_LEFT = "left"

_WIN = 1e9
_WIN_THRESHOLD = 0.5e9
_DEATH = -1e9
_DOOMED_PENALTY = 8.0  # px: a plan that ends well beats an equally far one that ends badly
_EARLY_WEIGHT = 0.25  # weight of the average x along the plan next to the final x
_SURVIVAL_WEIGHT = 1e3  # among deadly plans, the one that lives longest wins


Macro = tuple[str, int, str | None]
"""One edge of the detour search: hold a combination for some units; a jump macro then
holds its third entry until the player has landed."""

_DETOUR_MACROS: tuple[Macro, ...] = (
    (_RUN, 2, None),
    (_LEFT, 2, None),
    (_WALK, 2, None),
    (_NOOP, 2, None),
    (_RUN_JUMP, 2, _RUN),
    (_RUN_JUMP, 4, _RUN),
    (_RUN_JUMP, 7, _RUN),
    (_RUN_JUMP, 7, _NOOP),
    (_WALK_JUMP, 4, _WALK),
    (_WALK_JUMP, 7, _WALK),
    (_JUMP, 7, _NOOP),
    (_JUMP, 7, _LEFT),
)


def default_plan_library() -> list[list[Segment]]:
    """The macro-plans tried at every decision; each is padded to the horizon with its filler.

    The last segment of a plan is its filler: it is held until the horizon ends.
    Order matters only for ties (the earlier plan wins).
    """
    plans: list[list[Segment]] = [[(_RUN, 1)]]
    jumps = (1, 2, 3, 4, 6, 8, 10, 14)
    for hold in jumps:
        plans.append([(_RUN_JUMP, hold), (_RUN, 1)])
    # Jump a little later: the right take-off point is rarely "now".
    for delay in (1, 2, 3, 5, 8):
        for hold in (2, 5, 9, 14):
            plans.append([(_RUN, delay), (_RUN_JUMP, hold), (_RUN, 1)])
    # Run (or fly) on for a while, then stop: islands are short.
    for run in (3, 6, 10, 15, 22):
        plans.append([(_RUN, run), (_NOOP, 1)])
    for hold in (4, 8, 14):
        plans.append([(_RUN_JUMP, hold), (_RUN, 6), (_NOOP, 1)])
    # Slow and precise.
    plans.append([(_WALK, 1)])
    for hold in (2, 5, 9, 14):
        plans.append([(_WALK_JUMP, hold), (_WALK, 1)])
    # Shape the flight: let go, or pull back, to land short of trouble.
    for hold in (3, 6, 10, 14):
        for brake in (4, 8):
            plans.append([(_RUN_JUMP, hold), (_NOOP, brake), (_RUN, 1)])
            plans.append([(_RUN_JUMP, hold), (_LEFT, brake), (_RUN, 1)])
    # Let trouble come to us, or pass below us.
    for wait in (2, 4, 8, 12, 20):
        plans.append([(_NOOP, wait), (_RUN, 1)])
        plans.append([(_NOOP, wait), (_RUN_JUMP, 10), (_RUN, 1)])
    for hold in (4, 10, 14):
        plans.append([(_JUMP, hold), (_RUN, 1)])
        plans.append([(_JUMP, hold), (_NOOP, 6), (_RUN, 1)])
    # Make room for a run-up.
    for back in (3, 6, 10, 16):
        for hold in (6, 14):
            plans.append([(_LEFT, back), (_RUN_JUMP, hold), (_RUN, 1)])
            plans.append([(_LEFT, back), (_RUN, 6), (_RUN_JUMP, hold), (_RUN, 1)])
    return plans


def advance(game: Game, buttons: Buttons, frame_skip: int) -> None:
    """Step `game` the way one `MarioEnv.step` does: `buttons` held for `frame_skip` frames."""
    for _ in range(frame_skip):
        game.step(buttons)
        if game.over:
            break


class SearchAgent:
    """Deterministic lookahead planner; see the module docstring.

    Args:
        env: A `MarioEnv`, bare or wrapped (`env.unwrapped` is used).
        horizon: Env steps every candidate plan is simulated for.
        commit: Fewest actions of the winning plan executed before planning again (the
            scripted part of a plan is always executed as a whole).
        settle_steps: A plan that ends in mid-air is simulated for up to this many
            more env steps, until the player lands, so that a jump into a pit is
            recognised as deadly even when the horizon ends before the fall does.
        plans: Replaces `default_plan_library()`. Plans that need a button
            combination missing from the env's action set are dropped.

    `horizon`, `commit` and `settle_steps` are env steps; their defaults are 120, 16 and
    96 game frames converted with the env's `frame_skip`.

    Attributes: `plans` (the compiled library: action indices per env step, each
    `horizon` long), `last_score` (score of the last decision), `replans` (decisions
    since `reset`) and `detours` (how many of them needed the detour search).
    """

    def __init__(
        self,
        env: gym.Env,
        horizon: int | None = None,
        commit: int | None = None,
        settle_steps: int | None = None,
        plans: list[list[Segment]] | None = None,
    ) -> None:
        base = env.unwrapped
        for attribute in ("game", "action_buttons", "frame_skip"):
            if not hasattr(base, attribute):
                raise TypeError(f"SearchAgent needs a MarioEnv, but {base!r} has no {attribute!r}")
        self._env = base
        self._buttons: list[Buttons] = list(base.action_buttons)
        self._frame_skip = int(base.frame_skip)
        self.horizon = self._steps(horizon, DEFAULT_HORIZON_FRAMES, "horizon", 1)
        self.commit = min(self._steps(commit, DEFAULT_COMMIT_FRAMES, "commit", 1), self.horizon)
        self.settle_steps = self._steps(settle_steps, DEFAULT_SETTLE_FRAMES, "settle_steps", 0)
        index = {button_name(b): i for i, b in reversed(list(enumerate(self._buttons)))}
        self.plans: list[tuple[int, ...]] = []
        self._scripted: list[int] = []  # per plan: steps before its filler takes over
        self._compile(default_plan_library() if plans is None else plans, index)
        self._noop: Buttons | None = self._buttons[index[_NOOP]] if _NOOP in index else None
        self._coast_steps = self._units(COAST_UNITS)
        if not self.plans:
            raise ValueError("no plan can be expressed with the env's action set")
        self._macros: list[tuple[int, int, int | None]] = [
            (index[hold], self._units(units), None if glide is None else index[glide])
            for hold, units, glide in _DETOUR_MACROS
            if hold in index and (glide is None or glide in index)
        ]
        self._queue: list[tuple[int, tuple[Any, ...]]] = []
        self.last_score = 0.0
        self.replans = 0
        self.detours = 0
        self._max_x = float("-inf")
        self._max_x_frame = 0
        self._last_frame = -1
        self._detour_cooldown = 0
        self._detour_failures = 0
        self._carry: tuple[int, ...] = ()
        self._viable: dict[tuple[Any, ...], bool] = {}

    # --- agent API -------------------------------------------------------------------------------

    def reset(self) -> None:
        """Drop the rest of the current plan; the next `act` plans from scratch."""
        self._queue.clear()
        self.last_score = 0.0
        self.replans = 0
        self.detours = 0
        self._max_x = float("-inf")
        self._max_x_frame = 0
        self._last_frame = -1
        self._detour_cooldown = 0
        self._detour_failures = 0
        self._carry: tuple[int, ...] = ()

    def act(self, obs: Any = None) -> int:
        """The next action for the env's live game; `obs` is ignored (the state is read)."""
        game: Game = self._env.game
        # The queue is only valid for the states it was planned through: after an env reset
        # or somebody else's action the fingerprint differs and the agent plans anew.
        if self._queue and self._queue[0][1] != _fingerprint(game):
            self._queue.clear()
            self._carry = ()  # what was planned has nothing to do with this state
        if not self._queue:
            self._queue = self._plan(game)
        return self._queue.pop(0)[0]

    # --- planning --------------------------------------------------------------------------------

    def simulate(self, game: Game, plan: tuple[int, ...]) -> float:
        """Play `plan` on `game` (pass a clone: it is mutated) and score where that leads.

        A deadly plan gets its optimistic score here: the ground it covers before the
        end, as if the agent could always still save itself from there. `act` checks that.
        """
        return self._evaluate(game, plan)[0]

    def _evaluate(self, game: Game, plan: tuple[int, ...]) -> tuple[float, float, list[int], int]:
        """`(score, x, checkpoints, frames)` of `plan` played on `game`.

        `checkpoints` is empty for a plan that survives (or wins): all of it may
        be executed, and `x` is where it ends. For a deadly plan it lists the steps after
        which the player stood on the ground, and `score` and `x` belong to the last of them.
        A deadly plan without any checkpoint scores as the death it is. `frames` is how long
        the player lived (or took to win).
        """
        buttons = self._buttons
        frame_skip = self._frame_skip
        player = game.player
        start_frame = game.frame
        x_sum = 0.0
        grounded: list[int] = []
        grounded_x = player.x
        step = 0
        for action in plan:
            advance(game, buttons[action], frame_skip)
            if game.over:
                break
            step += 1
            x_sum += player.x
            if player.on_ground:
                grounded.append(step)
                grounded_x = player.x
        if not game.over and self.settle_steps:
            filler = buttons[plan[-1]]
            for _ in range(self.settle_steps):
                if player.on_ground or game.over:
                    break
                advance(game, filler, frame_skip)
        if game.won:
            return _WIN - (game.frame - start_frame), player.x, [], game.frame - start_frame
        end_x = player.x
        if not game.over and self._noop is not None:
            # Alive is not enough: the plan must be able to come to rest. Sliding off an edge
            # or into an enemy right after the horizon is as deadly as doing it before.
            for _ in range(self._coast_steps):
                if player.on_ground and player.vx == 0.0:
                    break
                advance(game, self._noop, frame_skip)
                if game.over:
                    break
        # Mostly "how far right does this end", but sooner beats later: without the
        # average, waiting a little longer always looks just as good as acting now.
        early = _EARLY_WEIGHT * x_sum / len(plan)
        frames = game.frame - start_frame
        if not game.over or game.won:  # (won: while coming to rest)
            return end_x + early, end_x, [], frames
        if grounded:
            return grounded_x + early - _DOOMED_PENALTY, grounded_x, grounded, frames
        return _DEATH + _SURVIVAL_WEIGHT * frames + player.x, player.x, [], frames

    def _plan(self, game: Game) -> list[tuple[int, tuple[Any, ...]]]:
        self.replans += 1
        self._viable.clear()
        x_now = game.player.x
        if game.frame <= self._last_frame or x_now > self._max_x:  # new episode / new ground
            self._max_x = x_now
            self._max_x_frame = game.frame
        self._last_frame = game.frame
        score, actions, x, self._carry = self._best_plan(game, self._carry)
        stalled = game.frame - self._max_x_frame > STALL_FRAMES
        if score < _WIN_THRESHOLD and (stalled or x - self._max_x < MIN_PROGRESS_PX):
            # The library does not get anywhere new from here (typically: a wide pit and no
            # speed, or plans that undo each other). Search a way to new ground step by step.
            if self._detour_cooldown > 0:
                self._detour_cooldown -= 1
            else:
                self.detours += 1
                detour = self._detour(game, self._max_x + DETOUR_GOAL_PX)
                if detour:
                    score, actions = self._max_x + DETOUR_GOAL_PX, detour
                    self._carry = ()
                    self._detour_failures = 0
                else:  # expensive and probably fruitless again soon: back off
                    self._detour_failures += 1
                    self._detour_cooldown = DETOUR_COOLDOWN * 2**self._detour_failures
        self.last_score = score
        # Replay the committed actions once more to know the state each of them expects.
        twin = game.clone()
        queue: list[tuple[int, tuple[Any, ...]]] = []
        for action in actions:
            queue.append((action, _fingerprint(twin)))
            advance(twin, self._buttons[action], self._frame_skip)
            if twin.over:
                break
        return queue

    def _detour(self, game: Game, goal_x: float) -> tuple[int, ...]:
        """Breadth-first search over short macros for a way to stand safely at `goal_x` or
        beyond; `()` when `DETOUR_MAX_NODES` states do not contain one."""
        buttons = self._buttons
        frame_skip = self._frame_skip
        max_glide = self._units(DETOUR_MAX_GLIDE)
        seen = {_detour_key(game)}
        frontier: list[tuple[Game, tuple[int, ...]]] = [(game, ())]
        nodes = 0
        while frontier and nodes < DETOUR_MAX_NODES:
            next_frontier: list[tuple[Game, tuple[int, ...]]] = []
            for state, path in frontier:
                for hold, steps, glide in self._macros:
                    twin = state.clone()
                    player = twin.player
                    taken = [hold] * steps
                    for _ in range(steps):
                        advance(twin, buttons[hold], frame_skip)
                    if glide is not None:
                        while not (player.on_ground or twin.over) and len(taken) < max_glide:
                            advance(twin, buttons[glide], frame_skip)
                            taken.append(glide)
                    nodes += 1
                    new_path = path + tuple(taken)
                    if twin.won:
                        return new_path
                    if twin.over:
                        continue
                    key = _detour_key(twin)
                    if key in seen:
                        continue
                    seen.add(key)
                    if player.on_ground and player.x >= goal_x and self._is_viable(twin):
                        return new_path
                    next_frontier.append((twin, new_path))
            frontier = next_frontier
        return ()

    def _best_plan(
        self, game: Game, carry: tuple[int, ...] = ()
    ) -> tuple[float, tuple[int, ...], float, tuple[int, ...]]:
        """`(score, actions to commit to, x reached, rest of the plan)` of the best plan.

        `carry`, the rest of the plan chosen last time, competes with the library (and wins
        ties): what made that plan good lies partly behind its committed part, and the
        library alone may not contain a way to get there from the middle of it.
        """
        plans = self.plans
        scripted = self._scripted
        if carry:
            plans = [(carry + carry[-1:] * self.horizon)[: self.horizon], *plans]
            scripted = [0, *scripted]
        candidates = []
        frames: dict[int, int] = {}
        for i, plan in enumerate(plans):
            score, x, checkpoints, frames[i] = self._evaluate(game.clone(), plan)
            candidates.append((-score, i, x, checkpoints))
            if score >= _WIN_THRESHOLD:
                break  # the flag within the horizon: good enough, stop searching
        candidates.sort()  # best first; ties: the earlier plan

        best: tuple[float, tuple[int, ...], float, tuple[int, ...]] | None = None
        for negated, i, x, checkpoints in candidates:
            score = -negated
            if best is not None and best[0] >= score:
                break  # scores of deadly plans only go down when they are checked
            plan = plans[i]
            if not checkpoints:
                # The scripted part of a plan is executed as a whole: a run-up or a wait only
                # pays off with the jump that follows it, and planning again half-way could
                # choose to start over, for ever.
                n = max(self.commit, scripted[i], 1)
                best = (score, plan[:n], x, plan[n:])
                break
            # A deadly plan is worth the last checkpoint from which the agent can still save
            # itself; a few are tried, going back in time.
            early = score - x
            for step in checkpoints[
                : -1 - CHECKPOINT_TRIES * CHECKPOINT_STRIDE : -CHECKPOINT_STRIDE
            ]:
                twin = game.clone()
                for action in plan[:step]:
                    advance(twin, self._buttons[action], self._frame_skip)
                if self._is_viable(twin):
                    checked = twin.player.x + early
                    if best is None or checked > best[0]:
                        best = (checked, plan[:step], twin.player.x, plan[step:])
                    break
        if best is None:  # every plan dies, whatever is done: live as long as possible
            i = max(range(len(candidates)), key=lambda k: (frames[candidates[k][1]], -k))
            negated, plan_index, x, _ = candidates[i]
            best = (
                _DEATH + _SURVIVAL_WEIGHT * frames[plan_index],
                plans[plan_index][: self.commit],
                x,
                plans[plan_index][self.commit :],
            )
        return best

    def _is_viable(self, game: Game) -> bool:
        """Does at least one library plan started in this state survive its horizon?"""
        key = _fingerprint(game)
        known = self._viable.get(key)
        if known is None:
            known = False
            for plan in self.plans:
                twin = game.clone()
                score, _, checkpoints, _ = self._evaluate(twin, plan)
                if not checkpoints and score > _DEATH / 2:
                    known = True
                    break
            self._viable[key] = known
        return known

    def _steps(self, value: int | None, default_frames: int, name: str, minimum: int) -> int:
        """`value` env steps, or the default number of frames converted to env steps."""
        if value is None:
            return max(minimum, 1, round(default_frames / self._frame_skip))
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        return value

    def _units(self, units: int) -> int:
        """Plan units (`PLAN_UNIT_FRAMES` frames each) as env steps of this env."""
        return max(1, round(units * PLAN_UNIT_FRAMES / self._frame_skip))

    def _compile(self, plans: list[list[Segment]], index: dict[str, int]) -> None:
        seen: set[tuple[int, ...]] = set()
        for segments in plans:
            if not segments or any(name not in index for name, _ in segments):
                continue
            actions: list[int] = []
            for name, units in segments[:-1]:
                actions.extend([index[name]] * self._units(units))
            scripted = min(len(actions), self.horizon)
            filler = index[segments[-1][0]]
            plan = tuple((actions + [filler] * self.horizon)[: self.horizon])
            if plan not in seen:
                seen.add(plan)
                self.plans.append(plan)
                self._scripted.append(scripted)


def _detour_key(game: Game) -> tuple[Any, ...]:
    """States that are the same for the purposes of the detour search."""
    player = game.player
    return (
        int(player.x) // 2,
        int(player.y) // 2,
        round(player.vx * 2),
        round(player.vy),
        player.on_ground,
        player.jump_armed,
    )


def _fingerprint(game: Game) -> tuple[Any, ...]:
    player = game.player
    return (game.frame, player.x, player.y, player.vx, player.vy, game.over)
