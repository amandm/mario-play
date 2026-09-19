# Training guide

Practical advice for training agents with mario-play: what to expect on a laptop,
how to size a run, how to train on a rented GPU or Colab across several sessions,
where to start with hyperparameters, what to watch in TensorBoard and what
usually goes wrong.

> **Read this first.** No full-length training run has been done with this
> repository yet. The framework is verified by tests, by CartPole convergence, by
> the search agent finishing every level and by short smoke runs - but the
> hyperparameters in `configs/` are standard starting values, not tuned results,
> and nobody has yet watched a learned policy reach the flag of `1-1`. Everything
> below that sounds like experience with *this* game is either a measurement from
> a short run (and says so) or general PPO / DQN practice. Treat your first long
> run as an experiment and please record what you find.

All commands run from the repository root. The numbers were measured on an Apple
M4 laptop while other jobs were running; take them as orders of magnitude.

## 1. Training locally (CPU / Apple MPS)

**Use the grid observation.** `configs/ppo_grid.yaml` is the laptop path: the
`(14, 15, 16)` grid needs no renderer, an env step costs a fraction of a pixel
step, and the network is a small conv stack (about 1.1M parameters). Pixels
(`ppo_pixels.yaml`, `dqn_pixels.yaml`) work on a laptop but are meant for a GPU.

```bash
# 1. a one-minute plumbing check on the obstacle-free level
uv run mario-play train --config configs/ppo_grid.yaml env.level=flat total_timesteps=30000 \
    log_interval=10000 eval.interval=10000 checkpoint_interval=10000 run_name=flat_smoke

# 2. the real run
uv run mario-play train --config configs/ppo_grid.yaml run_name=ppo_grid_1-1

# in a second terminal
uv run tensorboard --logdir runs
```

`flat` proves that the pipeline works, not that learning works: it has no
obstacles, and a policy that mostly presses right already reaches the flag (the
smoke run shows `flag_rate 1.00` from the first log line on).

**Device.** `device: auto` picks CUDA, then MPS, then CPU. Whether MPS beats the
CPU depends on the network, because every MPS call has a fixed overhead that a
tiny network cannot amortise:

| Workload (short smoke runs, M4) | `device=cpu` | `device=mps` |
|---|---:|---:|
| PPO, `CartPole-v1`, 64x64 MLP (`ppo_cartpole.yaml`) | ~16,000 steps/s | ~2,200 steps/s |
| PPO, grid, shipped network (`ppo_grid.yaml`) | ~500-1,000 steps/s | ~1,400-1,900 steps/s |
| DQN, grid, shipped network (`dqn_grid.yaml`), once updates have started | ~200-270 steps/s | ~650 steps/s |

So: MLPs and shrunken grid networks (small `network.hidden_size`, as the tests
use) belong on the CPU - the CartPole configs pin `device: cpu` for that reason -
while the shipped grid and pixel networks were faster on MPS here. Do not guess:
run the same 20k-step command twice and compare the `sps` column.

```bash
uv run mario-play train --config configs/ppo_grid.yaml device=cpu      # force the CPU
uv run mario-play train --config configs/ppo_grid.yaml device=mps      # force MPS (error if unavailable)
```

An explicitly requested device that is not available is an error, never a silent
fallback. `eval`, `watch` and `record` run the policy on the CPU unless
`--device` says otherwise; checkpoints are device-independent (train on CUDA or
MPS, evaluate or resume anywhere, e.g. `--resume runs/x device=cpu`).

**Vector envs.** For the grid keep `vec_env: sync`: the env is so cheap that
process round trips cost more than they save (measured: 8 sync envs ~39k steps/s,
8 subprocess envs ~34k). For pixels use `vec_env: subproc`: rendering and resizing
dominate a step, and one process per env pays off (one env ~5.7k steps/s, 8
subprocess envs ~12k).

**Interrupting.** Ctrl-C finishes the current step, writes `latest.pt` and prints
the resume command. Closing the laptop lid is fine too, as long as the process
survives; otherwise you lose at most `checkpoint_interval` steps.

## 2. Sizing a run with `mario-play bench`

```bash
uv run mario-play bench                                                  # grid, 8 sync envs
uv run mario-play bench --obs-mode pixels --n-envs 8 --vec subproc --steps 4000
```

```
bench: level 1-1 | obs grid (14, 15, 16) float32 | frame_skip 4 | random actions
setup              |  env steps |  seconds |    steps/s |   frames/s
single env         |     10,000 |     0.22 |     44,494 |    177,978
8 sync envs        |     10,000 |     0.26 |     38,899 |    155,596
```

`bench` measures the **env alone** with random actions: an upper bound that no
training run can beat. The number that sizes a run is the `sps` column of a real
training log, which includes action selection and the updates:

1. Run the config you intend to use for a minute or two (`total_timesteps=50000`)
   and read `sps` from the second log line on (the first includes warm-up).
2. `wall-clock ~= total_timesteps / sps`. Example: the 5M steps of `ppo_grid.yaml`
   at ~1,500 steps/s are about 55 minutes; at ~700 steps/s about 2 hours. Periodic
   evaluation and checkpoints are excluded from `sps` but are small.
3. Compare with `bench`:
   - training `sps` far below `bench` steps/s (the grid case: ~1-2k against ~40k):
     the **learner** is the bottleneck. More envs or subprocesses will not help; a
     faster device, a smaller network, fewer `ppo.n_epochs` or a larger rollout
     per update will.
   - training `sps` close to `bench` (likely for pixels on a GPU): the **envs**
     are the bottleneck. Raise `n_envs` with `vec_env=subproc` up to the number of
     physical cores and re-run `bench` with the same `--n-envs` to see where it
     stops scaling.

Units: every step count in the configs (`total_timesteps`, intervals,
`learning_starts`, `eps_decay_steps`, `target_update_interval`) is in env
transitions summed over all envs. One transition is `frame_skip` = 4 game frames,
so 5M steps are about 93 hours of game time. `dqn.train_freq` alone is in vector
steps.

Memory: PPO needs little. DQN's replay stores obs **and** next obs: 100k grid
transitions are ~2.7 GB, 100k pixel transitions ~5.6 GB, allocated lazily while
the buffer fills. Lower `dqn.buffer_size` on a small machine.

## 3. Cloud GPU / Colab: training across sessions

The trainer is built for preemptible machines: everything needed to continue
lives in `runs/<run_name>/`, and `--resume` picks up `latest.pt`.

### A rented GPU box (SSH)

```bash
git clone <this repository> mario-play && cd mario-play
curl -LsSf https://astral.sh/uv/install.sh | sh        # if uv is not installed yet
uv sync                                                # Linux: installs the CUDA build of torch

uv run mario-play bench --obs-mode pixels --n-envs 16 --vec subproc   # does this box have the cores?
uv run mario-play train --config configs/ppo_pixels.yaml run_name=ppo_pixels_1-1
```

Run it inside `tmux` or `screen` so that a dropped SSH connection does not kill
it. When the session ends or the instance is preempted:

```bash
uv run mario-play train --resume runs/ppo_pixels_1-1                        # same config, continues to total_timesteps
uv run mario-play train --resume runs/ppo_pixels_1-1 total_timesteps=2e7    # ... or train longer
```

`--resume` needs no `--config`: the checkpoint carries its config, and trailing
`key=value` overrides still apply. Logs, TensorBoard curves and checkpoints
continue in the same directory. What a resume does and does not restore is spelled
out in [architecture.md](architecture.md#checkpoints-and-resume); in short:
weights, optimizer, schedules, RNG and the best score are restored; environments
start new episodes, and DQN's replay buffer starts empty and refills before
updates continue.

### Where checkpoints live, and getting them off the box

```
runs/<run_name>/checkpoints/latest.pt      newest state (every checkpoint_interval, at the end, on Ctrl-C)
runs/<run_name>/checkpoints/best.pt        best periodic evaluation so far
runs/<run_name>/checkpoints/ckpt_<step>.pt the newest keep_checkpoints numbered ones
```

A PPO grid checkpoint is about 13 MB (weights plus Adam moments). `runs/` is
git-ignored, so copy it explicitly **before the instance goes away**:

```bash
tar czf ppo_pixels_1-1.tgz runs/ppo_pixels_1-1          # on the box
scp user@box:mario-play/ppo_pixels_1-1.tgz .           # from your machine
tar xzf ppo_pixels_1-1.tgz                              # restores runs/ppo_pixels_1-1
uv run mario-play watch --checkpoint runs/ppo_pixels_1-1
```

`rsync -av user@box:mario-play/runs/ runs/` in a loop or a cron job is the
low-tech way to keep a preemptible instance backed up. A run directory can be
moved freely: resuming continues in whatever directory the checkpoint sits in.
If you only want the policy, `best.pt` alone is enough for `eval`, `watch` and
`record` - it contains the config.

### Colab

Colab sessions end after a few hours and the local disk is wiped, so write runs to
Google Drive and resume from there. In notebook cells:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone <this repository> mario-play
%cd mario-play
!pip install uv && uv sync
!uv run mario-play train --config configs/ppo_grid.yaml run_dir=/content/drive/MyDrive/mario-runs run_name=ppo_grid_1-1
```

Next session: mount Drive, clone and `uv sync` again, then

```bash
!uv run mario-play train --resume /content/drive/MyDrive/mario-runs/ppo_grid_1-1
```

Colab notes: free instances have two CPU cores, so many subprocess envs will not
scale - check with `bench` and lower `n_envs` for the pixel configs. Keep
`checkpoint_interval` small enough that a disconnect costs little. Writing
TensorBoard events to Drive is slow but works; point TensorBoard at the Drive
folder (`%load_ext tensorboard`, `%tensorboard --logdir /content/drive/MyDrive/mario-runs`).

## 4. Hyperparameter starting points

The shipped configs are the starting points; every value can be overridden on the
command line (`ppo.ent_coef=0.02 env.action_set=right_only`).

**PPO (`ppo_grid.yaml`)**: 8 envs x 128 steps = 1,024 transitions per update, 4
epochs x 4 minibatches, `lr` 2.5e-4 annealed linearly to 0, `gamma` 0.99,
`gae_lambda` 0.95, `clip_coef` 0.2, `ent_coef` 0.01, `vf_coef` 0.5, grad-norm 0.5.
The pixel config differs in `n_envs` 16, `clip_coef` 0.1 and the NatureCNN.

What to try first, in this order:

1. **Curriculum by level.** `env.level=flat` (seconds to solve) -> `1-1`. For
   generalisation train on a list - `"env.level=[1-1,1-2,1-3]"` draws one level per
   episode - and evaluate on each level separately with `eval --level`.
2. **Exploration.** If entropy collapses before the agent has learned to jump,
   raise `ppo.ent_coef` (0.02-0.05). `env.action_set=right_only` (5 actions, no
   left, no standing jump) shrinks the search space considerably; `simple` is the
   default.
3. **Batch size.** More envs (`n_envs=16`/`32`) give steadier gradients for the
   same wall-clock when the learner is the bottleneck. Keep
   `n_steps * n_envs / n_minibatches` >= 256.
4. **Step size.** If `approx_kl` runs hot: `ppo.lr=1e-4`, fewer `ppo.n_epochs`, or
   `ppo.target_kl=0.03` as a safety net.
5. **Horizon.** `ppo.gamma=0.99` sees about 100 decisions (~7 s of game time). The
   flag bonus is far away for most of the level; progress reward carries the
   agent there. `ppo.gamma=0.995` is worth a try once it survives long enough.
6. **Reward.** Defaults: +1 per tile, -0.01 per step, -15 death, +50 flag.
   `env.reward.clip=5.0` (what the DQN configs use) or a smaller `death_penalty`
   tame the value targets if the critic struggles. Rewarding coins or score
   (`coin_bonus`, `score_weight`) changes the task: the agent may farm instead of
   finishing.
7. **`env.stall_steps`** (150 = 10 s of game time) truncates episodes that stopped
   making progress. Without it an agent stuck at a pipe burns 2,400 steps until
   the game clock kills it.

**DQN (`dqn_grid.yaml`)**: 8 envs, `train_freq` 1 with `gradient_steps` 2 (one
gradient step per 4 transitions), batch 32, `lr` 1e-4, replay 100k,
`learning_starts` 20k, hard target sync every 10k transitions, epsilon 1.0 -> 0.05
over the first 10% of the run, Double DQN + dueling head, reward clipped to +/-5.
First knobs: `dqn.eps_decay_steps` (longer if it never finds the first jump),
`dqn.target_update_interval`, `dqn.buffer_size` (RAM), `dqn.lr`. PPO is the better
first choice here: it is cheaper per step and has fewer ways to go wrong.

**When resuming with a larger `total_timesteps`,** remember that PPO's learning
rate is `lr * (1 - global_step / total_timesteps)` (with `global_step` taken at
the start of each rollout): extending a finished run raises the learning rate
again from ~0. That is usually what you want, but expect a visible kink in the
curves. `ppo.anneal_lr=false` avoids it.

## 5. What to watch in TensorBoard

```bash
uv run tensorboard --logdir runs
```

The same scalars are in `runs/<run>/metrics.csv`. `rollout/*` are means over the
last 100 finished *training* episodes (sampled actions); `eval/*` come from the
periodic evaluation (greedy by default); `train/*` are averaged between two log
lines.

| Scalar | Healthy | Trouble |
|---|---|---|
| `rollout/mean_progress` | The main curve early on: climbs in steps, each plateau is an obstacle the agent has not yet learned to pass. | Flat for millions of steps: see failure modes. A plateau value tells you *where* - watch the agent to see *what*. |
| `rollout/flag_rate`, `eval/flag_rate` | 0 for a long time, then rising quickly once the last obstacle falls. With greedy evaluation (`eval.deterministic=true`, the DQN configs) `eval/flag_rate` is 0 or 1 per level, because game and greedy policy are deterministic; the PPO configs evaluate the sampled policy over 5 seeded episodes. | `rollout` high but a greedy `eval` at 0: the argmax fails where the sampled policy often succeeds - keep training, or evaluate with `eval.deterministic=false`. |
| `rollout/ep_return_mean` | Tracks progress (+1 per tile) minus deaths; reference: random ~20, heuristic ~80, search agent ~234 on `1-1`. | Falling while progress rises: deaths (-15) dominate. |
| `train/approx_kl` | ~0.003-0.02 per update. | Sustained > 0.03-0.05 or spikes: policy steps too large -> lower `lr`, `n_epochs` or `clip_coef`, set `target_kl`. ~0: nothing is being learned. |
| `train/clip_frac` | ~0.05-0.25: some samples hit the clip, most do not. | > 0.3: too aggressive (same fixes as KL). ~0 together with KL ~0: learning rate annealed away or advantages vanished. |
| `train/explained_variance` | Starts near 0 (the smoke runs show ~0.00-0.01), should climb toward 0.5-0.9 as the critic learns. | Stuck near 0 or negative for long: the critic is not predicting returns - look at `value_loss` scale, consider `env.reward.clip`, a larger batch, or a higher `vf_coef`. |
| `train/entropy` | Starts at ln(n_actions): 1.95 for `simple`, 1.61 for `right_only`, 2.30 for `complex`; declines slowly as the policy commits. | Collapses toward 0 early while progress is low: premature convergence -> raise `ent_coef`. Never moves: not learning. |
| `train/value_loss`, `train/policy_loss` | Value loss is large at first (returns of +-50 squared) and falls; policy loss hovers around 0 and is not a progress signal. | Value loss growing without bound. |
| `train/q_mean` (DQN) | Rises smoothly, then levels off at the scale of plausible discounted returns. | Runaway growth or oscillation: divergence -> lower `lr`, sync the target less often, keep `reward.clip`. |
| `train/epsilon`, `train/buffer_size` (DQN) | Follow their schedules; after a resume `buffer_size` restarts at 0. | |
| `time/sps` | Steady. | Sudden drop: thermal throttling, swap (replay too large), another job. |

Look at behaviour, not just curves:

```bash
uv run mario-play watch --checkpoint runs/ppo_grid_1-1
uv run mario-play record --checkpoint runs/ppo_grid_1-1/checkpoints/best.pt --out best.gif
uv run mario-play eval --checkpoint runs/ppo_grid_1-1 --episodes 10 --stochastic
```

## 6. Common failure modes

- **Stuck at the first obstacle; `mean_progress` plateaus early.** Clearing a pipe
  or a pit needs a *held* jump (jump height depends on how long the button stays
  down) started at the right distance, often with run. With random exploration
  that is a rare event. Help it: higher `ent_coef`, `env.action_set=right_only`
  (every action moves right; two of five jump), more envs, more steps. Check that
  episodes end by `stall_steps` truncation rather than by the game clock.
- **Entropy collapse onto "right + run".** Running right is rewarded immediately,
  jumping only pays later. If entropy is near 0 while `flag_rate` is 0, restart
  with a larger `ent_coef` - a collapsed policy rarely recovers.
- **Dies at the same spot forever.** Watch it. If it jumps too late, the death
  penalty may be teaching it to fear the approach: try a smaller
  `env.reward.death_penalty` (e.g. -5).
- **KL spikes / performance collapses after looking good.** The classic PPO
  failure: lower `lr`, set `target_kl=0.03`, use more envs. Resume from an older
  `ckpt_<step>.pt` rather than from `latest.pt` (later checkpoints - numbered,
  `best.pt` and `latest.pt` - are set aside as `*.superseded.pt`, not deleted;
  `latest.pt` becomes the checkpoint you resumed from).
- **Learning rate is ~0.** With `anneal_lr` the rate falls linearly towards 0 at
  `total_timesteps` (the last update still runs at `lr / n_updates`); the last
  10% of a run learn little. Plan `total_timesteps` generously or disable
  annealing.
- **"Evaluation never changes" / "eval is far below the rollout numbers."** One
  greedy episode on a deterministic level is a single number that moves in jumps,
  and the argmax of a still-uncertain policy can be much worse than the policy
  itself (a 300k-step PPO smoke run on `1-1`: sampled progress 0.41, greedy 0.07).
  The shipped PPO configs therefore evaluate the sampled policy
  (`eval.deterministic: false`, `eval.episodes: 5`); sampled evaluation is seeded
  by `eval.seed`, reproducible, and leaves the training RNG untouched. Switch to
  `eval.deterministic=true eval.episodes=1` once the entropy is low, or train on
  several levels.
- **`best.pt` is missing.** It is written by periodic evaluation; with
  `eval.interval=0` only the final evaluation can write it. Use `latest.pt`.
  After a resume, look for `best.superseded.pt`: a `best.pt` from beyond the
  resumed step is set aside, and a new one appears once the resumed run beats
  the best score stored in the checkpoint it resumed from.
- **MPS is slower than the CPU.** Small networks: force `device=cpu` (section 1).
- **Subprocess envs are slower than sync.** Expected for the grid; use `sync`.
- **DQN eats all RAM / the machine swaps.** `dqn.buffer_size=50000` halves the
  replay memory. Watch `time/sps`.
- **DQN stalls after `--resume`.** The replay buffer is not checkpointed: it
  refills for `learning_starts` transitions before updates continue, and those
  steps count toward `total_timesteps`.
- **A new run landed in `runs/<name>_2`.** A non-empty run directory is never
  reused. To continue a run use `--resume runs/<name>`, not `--config` with the
  same `run_name`.
- **`error: bad config: ... unknown config key(s)`.** Config keys are strict; the
  message lists the valid ones. Overrides are dotted paths into the YAML tree
  (`ppo.lr=1e-4`, `env.reward.clip=5.0`, `env.level=[1-1,1-2]`).
- **Pixels on a laptop crawl.** That is what the grid observation is for. Size the
  pixel run with `bench` and a short training run before committing to it, and
  prefer a GPU box.
