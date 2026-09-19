# Mario Jev-assisted PPO: two-million-transition continuation

The extension continues each of the three original policies from its **latest
one-million-transition checkpoint** to exactly two million total transitions.
It adds one million transitions per condition; it is not a new initialization.
The original `runs/colab/jev-assisted-v1/` artifacts remain unchanged. New
artifacts belong to `runs/colab/jev-assisted-v2/`.

```bash
uv run python scripts/train_jev_assisted_ppo.py \
  --run-dir /content/jev-assisted-ppo-v2 \
  --table /content/jev-assisted-ppo-v2/table.json \
  --resume --fixed-budget --max-steps 2000000 --device cuda
```

`--fixed-budget` prevents the earlier criterion crossings from stopping this
continuation. Without that flag, the runner retains its original criterion-based
stopping behavior. The learning-rate horizon stays at five million transitions;
the environment, seed, reward, PPO hyperparameters, advice intervals, frozen
table and evaluation schedule are unchanged.

Resume restores each policy, optimizer, PPO update counter and independent RNG
state. All environments start fresh episodes. Each original checkpoint contains
976 completed PPO updates and 576 transitions from an incomplete rollout. Those
576 transitions are not checkpointed as a rollout and are discarded equally for
all conditions on restart. A single uninterrupted continuation to two million
therefore ends at **1,952 actual PPO updates per condition**, with another 576
pending transitions. This differs from the 1,953 updates implied by simply
flooring two million divided by 1,024. Actual counters and discarded/pending
transition counts are saved in progress, milestones and checkpoint metadata.

`continuation.json` records source checkpoint hashes, the parent artifact hashes,
the frozen table hash, source archive hash and the restart boundary. Source is
based on commit `af51e51`; the runner patch changes only stopping and accounting.
Core game, environment, PPO and configuration sources are checked against the
original run's manifest before allocation. No new Jev API requests are needed.

The training source archive was frozen before a local metadata wording fix:
the CLI now labels resumed initialization as continuation. That wording change
does not alter the live archived runner or its training. The final experiment
metadata clarifies continuation and retains the ancestor's initialization text;
the frozen training manifest records the exact code that ran.

Evaluation remains 20 sampled episodes from seed 2,000,000 plus one greedy
episode at seed 4,000,000 after every 100,000 transitions. Report the entire
learning curve and final two-million results even if performance regresses.
The old selected checkpoints remain useful examples but are not initialization
sources or substitutes for a matched-budget comparison. Final gameplay uses
the first fixed-seed greedy episode of each two-million checkpoint.
