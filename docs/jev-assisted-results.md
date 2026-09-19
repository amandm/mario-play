# Jev-assisted PPO: the first learning experiment

Completed on a Colab Tesla T4 on 19 September 2026. Three fresh policies each
received one million game interactions. The experiment took 41.4 minutes of
training, evaluation, and recording, plus 55 seconds to prepare the shared Jev
table. No additional purchases were made.

**This pilot did not show that more Jev advice reduces the training needed to
reach the predefined target.** The control reached it first. Frequent advice
did produce the earliest single greedy completion, which is a different and
weaker milestone.

The predefined target was at least 12 completions in 20 sampled evaluation
episodes **and** one greedy completion at the same checkpoint. Evaluations ran
every 100,000 interactions. All three policies started with identical weights,
18 input channels, the same PPO settings, and training seed 0 on level 1-1.

| Condition | First greedy completion | First combined target | PPO updates at target | Final sampled completions | Final greedy completion |
| --- | ---: | ---: | ---: | ---: | --- |
| PPO, zero advice | 600,000 | **600,000** | 585 | 15/20 | Yes |
| PPO + Jev every 16 actions | 900,000 | Not reached by 1,000,000 | — | 9/20 | Yes |
| PPO + Jev every action | **500,000** | 900,000 | 878 | 11/20 | No |

At its first target checkpoint, the control completed 14/20 sampled episodes;
frequent advice completed 16/20 at its own first target checkpoint. Frequent
advice therefore needed 50% more game interactions to meet that target. These
are first observed crossings, not stable convergence: all three runs had
regressions. The selected successful control and frequent-advice checkpoints
remain available at 600,000 and 900,000 interactions respectively.

## What Jev contributed

Jev remained frozen. It supplied probabilities for enemy side-contact risk,
forward obstacles, gaps, and overhead clearance. PPO selected every action and
learned from the unchanged game rewards. No Jev action labels, action imitation,
or Jev weight updates were used.

The exhaustive cache contains **476 actual Jev API responses**, shared by both
assisted conditions: 220,119 input tokens and 9,520 output tokens. Training used
65,738 advice refreshes in the sparse condition and 1,005,497 in the frequent
condition, including episode-reset refreshes. Those refreshes were local
lookups, not extra API requests; each refresh obtained four risk features.

This tests access frequency to cached Jev-derived features. It does not test
whether more physical API calls cause better learning. The comparison with
zeros also includes engineered geometry extraction, so it cannot isolate Jev's
reasoning from the feature design. One training seed on one known level is a
pilot, not a general verdict about Jev or reinforcement learning.

## Saved artifacts

- [Public curves, milestone data, frozen tables, and final 2M gameplay](research/README.md)
- [Two-million-interaction continuation results](jev-mario-extension-results.md)
- [Fixed experiment protocol](jev-assisted-experiment.md)
- [How the learning loop works](jev-assisted-learning.md)

The local run folder retains every evaluated checkpoint, individual episode outcomes,
configuration, exact frozen Jev outputs, source manifest, and playback metadata.
The original source archive is in `runs/colab/jev-assisted-v1-tools/source.zip`.
The raw run artifacts are intentionally git-ignored.

For a local run with saved checkpoints, regenerate equal-budget playback with:

```bash
uv run python scripts/record_jev_comparison.py \
  --run-dir runs/colab/jev-assisted-v1 --steps 1000000 --device cpu
```

The portable loader verifies the adjacent frozen table's hash before using it.
Regenerate the chart and report with:

```bash
uv run --no-project --with matplotlib scripts/report_jev_assisted.py \
  --run-dir runs/colab/jev-assisted-v1
```
