# Longer Mario training: all three policies complete the level

The [fixed continuation](jev-mario-extension.md) finished at **2,000,000 training
interactions per policy**. All three final policies completed **20/20 sampled
episodes** and the separate greedy episode on level `1-1`. These results use the
final checkpoints, rather than the earlier selected examples.

| PPO inputs | Sampled flags at 1M | Sampled flags at 2M | First original target crossing |
| --- | ---: | ---: | ---: |
| Zero auxiliary features | 15/20 | 20/20 | 600,000 |
| Jev features every 16 decisions | 9/20 | 20/20 | 1,100,000 |
| Jev features every decision | 11/20 | 20/20 | 900,000 |

The original target was at least 12/20 sampled completions **and** a greedy
completion at the same evaluated checkpoint. Baseline PPO reached that target
first in this seed. Longer training helped all three policies reach a strong
final result; it did not reverse the observed ordering for that predefined
learning-speed measure.

The full curves matter. Frequent Jev first scored 20/20 at 1.3M, then dropped to
15/20 at 1.4M. All three scored 20/20 at 1.6M, but the infrequent condition fell
to 14/20 at 1.7M and baseline fell to 16/20 at 1.8M. A perfect checkpoint did not
establish stable performance. These are raw evaluations of **one training seed
on one level**, with 20 sampled episodes per checkpoint and reused evaluation
seeds. They do not establish a reliable general benefit or harm from Jev, nor
performance on other levels.

## What was extended

Each condition resumed its latest 1M checkpoint, preserving the policy,
optimizer, independent random state, original settings, frozen Jev table, and
5M learning-rate horizon. Each restarted its environments and discarded 576
incomplete-rollout transitions. Final counters verified **1,952 PPO updates**
per condition. The original experiment's 147 artifacts remain byte-for-byte
unchanged. The extension collected 3M additional interactions across the three
policies and made **zero new Jev API requests**.

Total training feature refreshes across the full 2M histories were zero for
baseline, 129,863 for infrequent Jev, and 2,008,873 for frequent Jev. These are
cached feature accesses, not physical API calls. Jev supplies observations;
PPO still chooses every action and learns from the original game rewards.

## Artifacts and verification

The [public research package](research/README.md) contains the final comparison
GIF, complete learning curves, all milestone and episode data, and the frozen
Jev table used by both assisted conditions.

Local artifacts are in `runs/colab/jev-assisted-v2/`: `report.md`,
`learning_curve.png`, `matched_2000000.gif`, all milestone checkpoints,
`continuation.json`, `artifact_verification.json`,
`cpu_replay_verification.json`, and `runtime_release.json`.

The final GIF compares the same greedy evaluation seed and training budget.
Local CPU playback reproduced the remote evaluation exactly: baseline reached
the flag in 314 decisions, infrequent Jev in 312, and frequent Jev in 303.
Gameplay episode length is a different measure from training sample efficiency.
The Colab runtime was released after verification.

The separate [five-seed Taxi experiment](taxi-jev-results.md) was inconclusive
about assistance: every condition improved penalty avoidance but failed to
learn deliveries within its fixed budget.
