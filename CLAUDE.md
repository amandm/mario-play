# CLAUDE.md

Guide for AI and human contributors. mario-play is an original Mario-style
platformer (pure Python + numpy), a Gymnasium env around it (`MarioPlay-v0`) and a
from-scratch PyTorch RL framework (PPO + Double DQN). PPO policies trained on
Colab have demonstrated completion of level `1-1`. The user subsequently
requested matched Jev-assisted PPO comparisons, a longer Mario continuation,
and a replicated Taxi experiment. Preserve the original learning demonstration
in `runs/colab/ppo-agent-v1/` and the separate comparison artifacts described
below. Performance on other platformer levels has not been established.

## Commands

Run everything from the repo root. `uv run X` and `.venv/bin/X` are equivalent
once the environment exists.

```bash
uv sync                                   # set up .venv (runtime + dev group); Python 3.11 pinned, >= 3.10 supported

uv run pytest -m "not slow"               # fast suite (~1 min) - run before every commit
uv run pytest -m slow                     # ~1.5 min: CartPole convergence, search agent on hard levels, long fuzzing
uv run pytest tests/game -q               # one subsystem: tests/{game,envs,agents,rl}, tests/test_cli.py, tests/test_integration.py
uv run ruff check . && uv run ruff format --check .    # lint; `uv run ruff format .` to fix formatting (line length 100)

uv run mario-play levels                  # CLI; also `python -m mario_play`
uv run mario-play play --level 1-1
uv run mario-play eval --agent search --level 1-1 --episodes 1
uv run mario-play train --config configs/ppo_grid.yaml env.level=flat total_timesteps=30000 run_name=smoke
uv run mario-play train --resume runs/smoke total_timesteps=60000
uv run mario-play eval --checkpoint runs/smoke --episodes 2
uv run mario-play bench
uv run python scripts/render_preview.py --level 1-1 --out preview.png   # look at what the renderer draws
```

Tests never open a window: `tests/conftest.py` sets `SDL_VIDEODRIVER=dummy` and
`SDL_AUDIODRIVER=dummy`. Set them yourself when running `play` / `watch` headless
(`--max-frames N` / `--episodes N --max-steps N` make those commands terminate).

## Architecture map

```
src/mario_play/
  game/      headless simulation + numpy renderer      engine.py (Game.step, clone, snapshot), physics.py,
             level.py (ASCII levels), entities.py, tiles.py, constants.py, sprites.py, font.py,
             renderer.py ((240,256,3) uint8 frame), human.py (pygame window, lazy import)
  levels/    flat.txt 1-1.txt 1-2.txt 1-3.txt          package data; every *.txt here is a bundled level
  envs/      Gymnasium layer                            mario_env.py, actions.py, rewards.py, observations.py
             (grid obs (14,15,16) float32), wrappers.py (GrayscaleResize, FrameStack), factory.py (make_env)
  agents/    non-learning baselines                     random_agent, heuristic_agent, search_agent (plans on Game.clone()), runner
  rl/        config.py (dataclasses = YAML schema), types.py (VecStep), vec_env.py (sync/subproc, same-step
             auto-reset), networks.py, algos/{base,ppo,dqn}.py, buffers/{rollout,replay}.py, trainer.py,
             evaluate.py, checkpoint.py, logger.py, utils.py, debug_envs.py
  cli.py     play | train | eval | watch | record | bench | levels
configs/     YAML mirrors of rl/config.py (ppo|dqn x grid|pixels, plus CartPole correctness configs)
tests/       mirrors src/; marker `slow` for long-running proofs
```

Dependency direction: `game` <- `envs` <- `agents`; `rl` reaches envs only through
`envs.factory.make_env`; `cli` imports everything, lazily inside handlers. One
trainer loop drives every algorithm: `select_actions -> venv.step -> observe ->
(ready_to_update -> update)`. Details, step order, `VecStep` semantics and
checkpoint/resume guarantees: [docs/architecture.md](docs/architecture.md).
Training practice: [docs/training-guide.md](docs/training-guide.md).

## Hard rules

- **Learning focus: PPO and understandable training evidence.** Explain
  progress through behavior, rewards, and task completion. Use Colab for the
  requested Mario runs and the bounded experiment protocols below. Do not add
  other algorithms, hardware comparisons, or extra training seeds beyond the
  user's requested scope. Do not claim that a completed budget proves convergence.
- **Jev comparison is now requested.** The user explicitly authorized a hosted
  Jev controller and a visual comparison with the saved PPO checkpoint on the
  same game setup. Label Jev as pretrained inference, preserve the PPO weights,
  and keep API calls bounded within verified existing/free allowance. Credentials
  belong only in the ignored `.env`; never print or include them in artifacts.
- **Current experiment: Jev-assisted PPO learning.** Train fresh matched PPO
  policies with no advice, advice every 16 actions, and advice every action.
  Jev supplies frozen risk assessments as inputs; PPO chooses all actions and
  learns only from unchanged game rewards. Match architecture, initialization,
  training budgets, and evaluation conditions. Count game interactions, advice
  refreshes, unique API calls, and wall time separately. Reuse identical cached
  Jev assessments; do not claim cache lookups are new API calls or that more
  advice is guaranteed to improve learning.
- **Longer Mario and replicated Taxi experiments are now requested.** Preserve
  the completed Mario pilot and extend each latest one-million-step checkpoint
  to a fixed two-million-step total, with the original five-million-step
  learning-rate horizon. Report continuation resets and actual PPO updates.
  Separately compare Taxi-v4 PPO with zero auxiliary inputs, exact coded
  features, and Jev estimates of those same features across five paired seeds.
  Keep observations, action-mask handling, architecture, budgets, and evaluation
  conditions matched. Freeze the protocol before training, report every seed,
  and preserve negative or inconclusive results.
- **No new purchases.** Never purchase compute, top up credits, upgrade, or
  start a paid subscription. The user has authorized their existing Colab Pro
  subscription and its available compute units for requested training jobs.
  Check the balance before allocation and the runtime's consumption rate before
  training; size bounded runs to fit with a reserve and stop before exhaustion.
  Save results and stop runtimes when finished. If existing allowance is
  insufficient, use verified free resources or local CPU / MPS.
- **`mario_play.game` must not import torch, gymnasium or pygame.** Only
  `game/human.py` may import pygame, lazily inside functions. `rl/vec_env.py`
  must stay torch-free (subprocess workers import it). `mario_play/__init__.py`
  and `cli.py` import nothing heavy at module level.
- **float32 everywhere** in tensors and float observations - MPS has no float64.
  uint8 image observations stay uint8 until the encoder scales them.
- **Checkpoints must stay `torch.load(weights_only=True)`-loadable**: `state_dict()`
  of an algorithm and everything passed to `save_checkpoint` hold tensors and
  plain Python containers / primitives only (no numpy scalars, dataclasses, Paths).
- **Import from concrete modules** (`from mario_play.game.engine import Game`),
  never from package-level re-exports. Package `__init__.py` files are lazy
  contract files: keep them free of eager imports (the only intended edit is
  registering a new algorithm in `rl/algos/__init__.py`).
- **No Nintendo assets**: no ROM data, sprites, level layouts, music or names. All
  art and levels are original; code uses neutral names (`Player`, `Walker`,
  `Turtle`, `Mushroom`).
- **Determinism**: the game uses no global RNG - only `Game.rng`. Same level +
  seed + buttons => identical `snapshot()` per frame, also across `clone()`.
  Agents and tools never mutate the live game when planning; they use clones.
- **`terminated` vs `truncated`** must be preserved end to end: death/flag
  terminate; stall and step limits truncate and are bootstrapped from
  `VecStep.final_obs` (`VecStep.obs` of a finished env is the *next* episode).
- **Config is strict**: every tunable is a dataclass field in `rl/config.py`;
  YAML and `key=value` overrides reject unknown keys. Add the field first.
- Tests: deterministic, no real window, fast files < 10 s; anything longer gets
  `@pytest.mark.slow`. Bug fixes land with a regression test.
- Style: type hints and docstrings on public API, comments only for the
  non-obvious *why*, ruff-clean (`E,F,W,I,UP,B`, line length 100).
- Docs: every command shown in README / docs must actually work. Distinguish
  measured pilot results from full-length training and evidence of convergence.
- Training artefacts (`runs/`, `*.pt`, GIFs outside `docs/`) are git-ignored; do
  not commit them.

## Where things are written down

- Design spec (authoritative for intended behaviour):
  [docs/superpowers/specs/2026-09-19-mario-rl-design.md](docs/superpowers/specs/2026-09-19-mario-rl-design.md)
- Implementation plan with per-task contracts and global constraints:
  [docs/superpowers/plans/2026-09-19-mario-rl.md](docs/superpowers/plans/2026-09-19-mario-rl.md)
- Where code and spec disagree on a signature, the code wins; note the deviation.
- CI: [.github/workflows/ci.yml](.github/workflows/ci.yml) - ruff, fast suite on
  Python 3.10 and 3.12 with CPU-only torch, slow suite on 3.12.
