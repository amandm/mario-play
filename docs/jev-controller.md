# Jev controller and PPO recording

This is the earlier direct-controller experiment. It does **not** measure
whether Jev accelerates PPO training. The subsequent experiment trains matched
PPO policies from scratch; see its [protocol](jev-assisted-experiment.md) and
[learning explanation](jev-assisted-learning.md).

The Jev controller is a new inference algorithm around TypeSafe's pretrained
decision model. It does not train Jev, distill its outputs, or change our PPO
checkpoint. PPO learned a neural policy from gameplay rewards; Jev interprets
the current game state and selects an action through a hosted API.

## Decision loop

1. Receive the same 14-plane grid observation used by PPO.
2. Describe its visible terrain, enemies, player motion, and recent controls as
   compact JSON. This does not inspect unseen map tiles or simulate future paths.
3. Ask `jev-1.13.0` one Choice question over the existing seven controller actions.
4. Apply the selected action for the environment's normal four game frames.
5. Repeat until death, completion, stalling, or the fixed decision limit.

The game pauses while the API responds. Recordings use simulation time, so API
waiting time is excluded from playback and reported separately. A model response
with an invalid action or a failed API request stops the run; it must not silently
substitute another controller.

The observation description and instructions are part of the Jev algorithm.
The current version also states relationships explicitly: whether the nearest
visible enemy overlaps the player's body rows, how many empty tile columns
separate their occupied bounds, and whether visible ground supports the path.
These are measurements of the observed grid, not recommended actions or
predicted future collisions.
The instructions explain game controls, including releasing jump before another
jump and holding it to extend ascent. This is domain guidance, not a learned
physics model. Both controllers use greedy choices, with Jev choosing its
highest-probability action as defined by the API.

## What the visual comparison means

Use the selected one-million-step PPO checkpoint, level `1-1`, seed `4000000`,
the same action mapping, reward function, frame skip, and maximum decisions.
Record the first episode from each controller. Retain failures and stop reasons;
do not search for a successful take. One pair of recordings is an illustration
of behavior, not a statistical estimate of which method is better.

This comparison also does not match training compute: PPO received one million
local environment interactions, while Jev's vendor training is unspecified.
Jev receives an engineered description, its previous four actions, and previous
jump-button state. PPO consumes only the current numeric grid. This history
gives Jev extra information about jump re-arming, so the information available
to the controllers is not identical even though the current grid is shared.

## Development history

The first implementation used tile rectangles and scalar observations. In its
first episode, Jev repeatedly ran toward an approaching enemy and died after
26 decisions at 7.5% progress. Those artifacts and the original source are kept
in `runs/jev-vs-ppo-v1/`.

The second implementation adds explicit spatial relationships after inspecting
that failure. Its run lives in `runs/jev-vs-ppo-v2/` and reuses the same PPO
recording. This is a development iteration on the same level, not an untouched
or held-out evaluation. No model weights were updated in either implementation.
It chose walking and jumping actions, but still died at the first enemy after
37 decisions, at 6.8% progress. The saved PPO policy completed the level after
333 decisions. These are individual episode outcomes, not success-rate estimates.

## Credentials and allowance

Run from the repository root after checking the allowance:

```bash
uv run python scripts/compare_jev_ppo.py --agent both --out runs/jev-comparison-new --max-calls 1000
```

The script reads `.env` without executing it. Use a new output folder for a new
comparison; it refuses to repeat an existing agent recording. `--agent ppo`
records only the local checkpoint without requiring API access. Both episodes
use the selected decision cap. Results include separate `gameplay.gif` files,
`side-by-side.gif`, per-decision logs, configuration, and `report.md`.

Place `TYPESAFE_API_KEY` in the project-root `.env`, as shown in `.env.example`.
The real file is git-ignored and should have owner-only permissions. Do not copy
the key into reports, notebooks, shell history, screenshots, or source code.

Existing TypeSafe allowance was verified before making requests, and no new
purchases were made. An authenticated model-list request succeeded. A single
action-choice probe also succeeded with model `jev-1.13.0`, using 459 input
tokens and 71 output tokens. The key is omitted from the saved access-check
artifact.

Credit availability can change. Before another run, verify the balance and keep
the call limit below the existing allowance. Never add funds, enable recharge,
or buy a subscription to run this example. The documented input price is $0.042
per million tokens, with free output tokens; token-based estimates are not an
invoice. [TypeSafe model reference](https://docs.typesafe.ai/models)

The request and response formats follow the official
[HTTP API reference](https://docs.typesafe.ai/api).
