# Following one PPO agent's learning

Train one PPO policy on level `1-1`. Save checkpoints and `metrics.csv` so its
behavior can be inspected as learning progresses. The checkpoint contains the
network weights, training configuration and interaction count. Training steps
count interactions across all parallel environments, not completed games.

On the prepared Colab runtime, start the learning run with:

```bash
python scripts/train_ppo_agent.py --run-dir /content/mario-ppo-agent --device cuda
```

This uses seed 0 and a fixed five-million-step learning-rate schedule. It saves
checkpoints every 100,000 interactions and checks behavior every 500,000.
Training stops early if at least 18 of 20 sampled validation episodes finish
and the greedy policy also finishes. A separate final assessment uses 100
sampled episodes and one greedy episode. Success here means completing the
chosen level; the checks do not cover other levels.

If interrupted, restore the downloaded directory to the runtime and continue
with `--resume /content/mario-ppo-agent --device cuda`. Use the existing Pro
allowance only, download artifacts before shutdown, and stop the runtime when
finished. Never purchase additional units or upgrade.

After downloading a run from Colab, generate a report and an actual gameplay GIF:

```bash
uv run python scripts/explain_ppo_run.py --run-dir runs/my_ppo_run
```

The helper reads `checkpoints/selected.pt` (falling back to `latest.pt`), reuses
`final_assessment.json` only if its checkpoint hash matches, and records one
greedy episode with the fixed seed `4,000,000`. It writes `report.md`,
`evaluation.json` and `gameplay.gif` under
`runs/my_ppo_run/learning/step_<checkpoint-step>_greedy/`. The recording is not
selected for success. Its caption states the level, checkpoint step, action
mode, episode seed, actual completion result and final progress.

To inspect an earlier checkpoint from the same run, add
`--checkpoint runs/my_ppo_run/checkpoints/ckpt_250000.pt` (use an existing file).
Each checkpoint gets its own output folder. To watch the policy without saving a
GIF, use:

```bash
uv run mario-play watch --checkpoint runs/my_ppo_run/checkpoints/selected.pt --episodes 1
```

`--mode sampled` follows PPO's learned action probabilities. The default greedy
mode always takes the most likely action; it does not train a different algorithm.
Use one chosen mode consistently when following learning. Since the level is
fixed, repeated greedy episodes can be identical. Evaluation seeds do not
generate different maps.

Read the report in this order: level completions, distance reached, then reward.
A higher training reward does not by itself mean the agent can finish reliably.
The milestone table contains logged evaluations from this one run; small
batches can fluctuate while PPO explores. The training runner's final evaluation
uses a separate seed (`3,000,000`), with 100 sampled episodes and one greedy
episode. The helper reuses these saved results instead of repeating them.
The recorded episode has a 6,000-decision cap. These are checks
of behavior on the chosen level, not evidence of generalization to other levels.

For an early checkpoint without a matching final assessment, the report clearly
labels its score as the single recorded episode. To request an additional local
evaluation explicitly, add `--evaluate --episodes 20`. This evaluates the selected
playback mode and does not change the checkpoint or continue training.
