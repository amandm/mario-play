# Taxi: a replicated PPO test of frozen Jev features

This protocol is fixed before the comparison is trained. A separate throughput
check uses an excluded seed and is used only to estimate the CPU runtime.

## Question and controls

Does adding frozen Jev state assessments improve PPO learning beyond ordinary
state encoding or inexpensive task-specific features?

Train three PPO conditions from scratch on Gymnasium 1.3.0's default deterministic
`Taxi-v4`, with the usual reward and 200-action episode limit:

| Condition | Shared raw observation | Four auxiliary inputs |
| --- | --- | --- |
| Zeros | Decoded state and fixed map | Zeros |
| Rules | Identical | Exact deterministic task features |
| Jev | Identical | Frozen Jev probabilities for the same four questions |

The shared observation contains one-hot taxi row, column, passenger location,
and destination, plus fixed wall and landmark information. Every condition has
the same 71-dimensional observation and network architecture. The four questions
concern pickup readiness, dropoff readiness, whether the current task goal is
within two Manhattan steps, and whether a same-row direct route is blocked.
No action mask is consumed by any policy. No expert actions, transition model,
future outcomes, rewards, or hidden simulator information enter the features.

Jev's weights remain fixed. A complete table is prepared once before training;
local lookups do not issue additional API requests. PPO chooses every action and
learns only from actual environment rewards. The rules control is needed because
an improvement over zeros could otherwise come from feature engineering alone.

## Fixed training plan

- Five independent training seeds: **0, 1, 2, 3, 4**, paired across conditions.
- **200,000 interactions per run**, 15 runs total; no performance-based extension.
- Eight environments; 128 decisions each per rollout, giving 1,024 interactions.
- PPO: learning rate 0.0003, linearly annealed over the fixed 200,000 interactions;
  four epochs and four minibatches, gamma 0.99, GAE lambda 0.95, clipping 0.2,
  entropy coefficient 0.01, value coefficient 0.5, gradient norm limit 0.5.
- One shared policy/value network with a 64-by-64 MLP; identical initial weights
  within each paired seed, verified by tensor hashes. All runs use one CPU thread.
- Evaluate and save a checkpoint every **20,000 interactions**; retain all results.
  Partial PPO rollouts continue through evaluations. No best-checkpoint selection
  is used for the primary final result.

At the final budget, there are 195 completed PPO updates (3,120 optimizer steps)
and 320 interactions in an incomplete rollout. Those interactions still count
toward the identical budget. Per-checkpoint logs retain actual update counts.

## Evaluation and outcomes

Evaluate the greedy policy from **all 300 valid initial Taxi states** after each
milestone, with a 200-action cap. This exhaustively assesses the task's existing
start distribution; it is not an unseen-task or held-out-generalization test.
Policies only receive the same public observation representation used in training.
The evaluator batches states using Gym's deterministic transition table, checked
against real Gym transitions; that table is not given to the learner.

The primary measures are normalized trapezoidal area under the success curve
over **20,000 through 200,000 interactions**, and final success at 200,000. Also
report return, episode length including capped failures, and illegal pickup or
dropoff counts. A secondary measure is the checkpoint at which at least 90%
success has been observed for three consecutive evaluations; report the third
checkpoint as the confirmation time.

The replication unit is the training seed. Show every seed curve plus the mean
and seed range. Compare Jev against both zeros and rules using paired seed
differences. Three hundred starts do not constitute 300 independent training
runs, and a five-seed pilot does not establish universal superiority.

Fixed gameplay illustrations use seed-0 policies at the final budget and encoded
initial states **461, 91, and 244**, chosen before training. All three conditions
receive those same starts; failed recordings are retained. These clips illustrate
behavior and never select checkpoints or seeds.

## Reproduction

Prepare the frozen table using the separately documented bounded API workflow.
Keep its SHA-256 and exact raw responses. Then run locally:

```bash
uv run python scripts/train_taxi_jev.py \
  --run-dir runs/taxi-jev-v1 --table path/to/verified-taxi-table.json

uv run --with matplotlib python scripts/report_taxi_jev.py \
  --run-dir runs/taxi-jev-v1
```

The training command makes no API requests. Completed runs can be reused only
when their configuration and checkpoint hashes match. Interrupted partial runs
are preserved for inspection rather than silently restarted or mixed with the
fixed experiment. Tables, checkpoints, and recordings remain local run artifacts.
