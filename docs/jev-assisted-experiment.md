# Does more frequent Jev advice help PPO learn sooner?

Protocol defined before training on 19 September 2026.

The experiment trains three new PPO policies from scratch. Jev supplies frozen
assessments as observation features; it does not choose the actions, provide
expert-action targets, change the reward, or receive weight updates.

| Condition | Current grid | Four auxiliary feature planes | Refresh interval |
| --- | --- | --- | --- |
| Control | Original 14 planes | Zeros | Never |
| Infrequent advice | Original 14 planes | Jev risk probabilities | Every 16 actions |
| Frequent advice | Original 14 planes | Jev risk probabilities | Every action |

All networks therefore have 18 input channels and identical initial weights.
They use the same level `1-1`, training seed, seven actions, four simulation
frames per action, reward function, PPO settings, and evaluation schedule. The
old trained PPO checkpoint is a historical reference, never an initialization.

The auxiliary features assess enemy side-contact, forward obstacles, visible
gaps, and overhead clearance. Their contexts come only from the current grid,
with explicit coarse categories. There is no simulator lookahead, unseen-map
access, action history, or reward information in the Jev requests. Infrequent
advice remains stale between refreshes, including across state changes; both
assisted conditions refresh at episode reset.

## What is actually measured

The primary target is the earlier PPO milestone: at least 12 flags in 20 sampled
evaluation episodes and completion in one greedy episode. Evaluate after each
100,000 game interactions per condition, up to one million. Use sampled seed
`2000000` and greedy seed `4000000`, with a 6,000-decision episode cap. Record the
first evaluated milestone meeting both requirements; the crossing time has
100,000-interaction resolution.

Keep the PPO learning-rate schedule's horizon at five million interactions,
matching the earlier training configuration even if this experiment ends sooner.
Run the three conditions in matched stages and preserve each condition's own
RNG and partial rollouts. Do not select a different training seed after results
are visible. Save every evaluated checkpoint and retain failures.

Report game interactions, PPO updates, validation completion/progress, advice
refreshes, actual API requests, input tokens, and wall time separately. A method
can require fewer game interactions while taking more preparation or compute.

## Caching and interpretation

Before training, query pinned `jev-1.13.0` for the finite set of abstract hazard
contexts and save its actual probabilities and provenance. Both assisted
conditions use this same immutable table. At training time, a refresh retrieves
the matching Jev assessments; it does not resend an identical API request.
There is no learned surrogate or imitation objective.

Consequently, this tests **frequency of access to Jev-derived advice**, not a
claim that increasing duplicate network requests improves training. Actual API
request counts belong to preparation of the shared table and must not be
confused with its potentially millions of lookups during training.

This is one training-seed pilot on one known level. A monotonic result across
the three conditions would support the hypothesis for this setup; it would
not prove a universal relationship between Jev calls and sample efficiency.
An improvement over zeros also includes the effect of engineering the
observation features; it does not isolate Jev from a hand-built feature control.
If no condition reaches the target by the cap, report progress and state that
time-to-target was not observed.

Final gameplay clips show the first fixed-seed greedy episode of each selected
checkpoint. Selection uses validation performance and the earliest successful
milestone, not a search for an attractive recording.

Use only existing Colab Pro units and existing TypeSafe credits. No purchases,
top-ups, subscription changes, or automatic recharge are part of this experiment.
Credentials stay in the local ignored `.env`; Colab receives source and the
credential-free frozen feature table.
