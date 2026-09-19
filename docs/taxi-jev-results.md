# Taxi pilot: penalty avoidance learned, delivery not learned

The fixed Taxi experiment completed all 15 runs: three observation conditions,
five paired training seeds, and 200,000 interactions per run. PPO with zero
auxiliaries, exact-rule auxiliaries, and frozen Jev auxiliaries all had **zero
greedy deliveries** at every evaluated milestone. This is a floor effect that
does not establish whether Jev can improve a functioning delivery learner.

The [predeclared protocol](taxi-jev-protocol.md) was preserved. There were no
action masks, reward changes, selected winning seeds, or extensions based on
results. All policies had the same architecture and matched initial weights
within each seed. The 15 runs took 113.75 seconds of local CPU wall time, with
3,000,000 total training interactions and no training-time API requests. Each
run completed 195 PPO updates and 3,120 optimizer steps; the final 320 collected
interactions remained in an incomplete rollout.

| Condition | Seeds | Success AUC, 20k–200k | Final deliveries | Final return | Final actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO + zeros | 5 | 0.000 | 0/300 in every seed | -200 | 200 |
| PPO + exact rules | 5 | 0.000 | 0/300 in every seed | -200 | 200 |
| PPO + Jev probabilities | 5 | 0.000 | 0/300 in every seed | -200 | 200 |

All 150 milestone evaluations covered the same 300 valid initial states.
Failures reached the 200-action cap. No run reached the secondary sustained
90% success target. The paired success differences are exactly zero, but the
absence of any successful control policy limits what that equality means.
These are in-domain starts, not an unseen-task generalization test.

## What did change during learning?

A separately labeled, post-hoc diagnostic evaluated the saved untrained
checkpoints on all 300 starts. This does not change the primary AUC interval.
Across the five seeds, the average greedy return and illegal-action counts
improved as follows:

| Condition | Initial return | Final return | Initial illegal pickup/dropoff per episode | Final illegal actions |
| --- | ---: | ---: | ---: | ---: |
| Zeros | -793.04 | -200.00 | 65.89 | 0.00 |
| Exact rules | -839.48 | -200.00 | 71.05 | 0.00 |
| Jev | -868.33 | -200.00 | 74.26 | 0.00 |

Both initial and final delivery success were zero. Thus there is evidence of
learning to avoid expensive illegal actions, but no evidence of learning the
pickup-and-delivery task. Sparse successful experience and early suppression
of pickup/dropoff are plausible explanations; this experiment does not isolate
their causal roles. Longer training alone is not guaranteed to overcome this
behavior.

## The Jev input was imperfect

Jev supplied four continuous probabilities for each of all 500 encoded states.
A 0.5 threshold was used only for the following diagnostic; PPO received the
original probabilities without thresholding.

| Feature | Positive states | True positives | False positives | False negatives | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pickup ready | 16 | 4 | 0 | 12 | 100.0% | 25.0% |
| Dropoff ready | 4 | 4 | 12 | 0 | 25.0% | 100.0% |
| Active goal within two Manhattan steps | 130 | 79 | 27 | 51 | 74.5% | 60.8% |
| Same-row direct route blocked | 60 | 55 | 44 | 5 | 55.6% | 91.7% |

These counts cover all 500 encoded states, including states outside the normal
start distribution. Overall accuracy alone would hide the class imbalance.
Feature diagnostic accuracy does not prove an effect on reinforcement learning;
the exact-rule condition also failed to learn deliveries here.

The frozen bank used 500 actual successful API calls, 411,200 input tokens, and
43,000 output tokens. Precomputation took 61.60 seconds with concurrent requests.
There were zero unaccounted calls. Local feature lookups during training and
evaluation did not make further requests. Jev itself was not fine-tuned.

## Artifacts and interpretation

The [public research package](research/README.md) includes the curves, a gameplay
comparison, all seed and milestone metrics, per-start outcomes, the initial-policy
diagnostic, and the frozen Jev table.

Local artifacts under `runs/taxi-jev-v1/` include the full `report.md`,
`learning-curves.png`, `summary.json`, all 150 milestone checkpoints, all initial
and final checkpoints, the verified frozen table, per-start outcomes, and
`initial_policy_diagnostic.json`. All four evaluation curves overlap exactly
across conditions and seeds. The three GIFs retain the failures from the
preselected seed-0 final policies and starts 461, 91, and 244.

Training source, protocol, exact configurations, and table hashes were frozen
before the runs. Report-only improvements to incomplete-result handling and
diagnostic presentation are documented in `analysis_source_manifest.json`, with
the original source snapshot and the revised analysis script both retained.

This pilot tests these four features, Jev 1.13, and this PPO/Taxi configuration.
It provides no evidence of faster delivery learning with Jev. A useful next
experiment would first establish a delivery-learning PPO baseline using separate
development seeds, then freeze the setup and repeat all three conditions on new
paired seeds. That is a new experiment; the present result remains unchanged.
