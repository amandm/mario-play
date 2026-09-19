# Mario-Play — Design Spec

**Date:** 2026-09-19 · **Status:** approved · **Owner:** Aman Deep Meena

An original Mario-style platformer, a Gymnasium environment around it, and a
from-scratch PyTorch reinforcement-learning framework (PPO + Double DQN) that can
train an agent to play it. Full-scale training is deferred until compute is
available; everything else — game, env, algorithms, training/eval tooling, tests —
is complete and verified now.

## 1. Goals and non-goals

**Goals**

1. A playable, deterministic, headless-capable platformer with classic mechanics.
2. A Gymnasium env (`MarioPlay-v0`) with pixel and compact grid observations.
3. An RL framework written from scratch in PyTorch: PPO and Double DQN behind one
   `Algorithm` interface, vectorized envs, config-driven trainer, full-resume
   checkpoints, TensorBoard/CSV logging, evaluation, recording.
4. Proof of correctness without a full training run (section 9).
5. A clean GitHub repo: `uv` project, ruff, pytest, GitHub Actions CI, docs.

**Non-goals (deferred):** fire flower, sound, moving platforms, warp pipes,
procedural level generator, recurrent policies, distributed training, prioritized
replay. No Nintendo assets, ROMs, sprites, or level layouts — all code, art and
levels are original. Code uses neutral names: `Player`, `Walker` (goomba-like),
`Turtle` (koopa-like).

## 2. Architecture

Headless-first: the simulation is pure Python + numpy with no display dependency.
One renderer draws the view into a `(240, 256, 3)` uint8 RGB numpy array. The
pygame window only upscales that array and reads the keyboard, so the agent sees
exactly what a human sees and training needs no display.

```
mario-play/
├── pyproject.toml  README.md  LICENSE  CLAUDE.md  .gitignore  .python-version
├── .github/workflows/ci.yml
├── configs/                  ppo_grid.yaml ppo_pixels.yaml dqn_grid.yaml dqn_pixels.yaml
│                             ppo_cartpole.yaml dqn_cartpole.yaml
├── docs/                     architecture.md training-guide.md superpowers/{specs,plans}/
├── src/mario_play/
│   ├── __init__.py           light: version only (never imports torch/pygame)
│   ├── game/                 constants tiles level entities physics engine
│   │                         sprites font renderer human
│   ├── levels/               flat.txt 1-1.txt 1-2.txt 1-3.txt
│   ├── envs/                 actions rewards observations mario_env wrappers factory
│   ├── agents/               random_agent heuristic_agent search_agent
│   ├── rl/                   config types utils vec_env networks logger checkpoint
│   │   ├── buffers/          rollout replay
│   │   ├── algos/            base ppo dqn (+ registry in __init__)
│   │   ├── evaluate.py
│   │   └── trainer.py
│   └── cli.py                play | train | eval | watch | record | bench | levels
└── tests/                    game/ envs/ agents/ rl/ test_cli.py   (marker: slow)
```

Dependency direction: `game` ← `envs` ← `agents`; `rl` depends on `envs.factory`
only; `cli` depends on everything. `game` must not import gymnasium, torch or
pygame (except `game/human.py`, which imports pygame lazily inside functions).

## 3. Game engine (`mario_play.game`)

### 3.1 Units and constants (`constants.py`)

- `TILE = 16`, `VIEW_W = 256`, `VIEW_H = 240`, `VIEW_TILES_W = 16`,
  `LEVEL_H_TILES = 15`, `FPS = 60`. Every level is exactly 15 tiles tall.
- World coordinates are pixels, floats, origin top-left, +y down. One `step` = one
  frame = 1/60 s. Velocities are px/frame.
- Default physics (tunable, but the invariants below are tested):
  walk max 1.5, run max 2.5, ground accel 0.07 (walk) / 0.10 (run), release
  decel 0.08, skid decel 0.18, air accel 0.06; jump impulse −5.0 (−5.4 when
  `|vx| > 2.0`), gravity 0.20 while rising with jump held, 0.50 otherwise,
  terminal fall speed 4.5; stomp bounce −3.5 (−5.0 with jump held).
- **Invariants (unit-tested):** full-hold standing jump apex is 3.5–4.5 tiles;
  tap jump apex ≤ 2 tiles; a running full-hold jump covers ≥ 6 tiles
  horizontally; player can never tunnel through a solid tile at any speed.
- Hitboxes: small player 12×15, big player 12×30, walker 14×14, turtle 14×22,
  shell 14×14, mushroom 14×14. `x, y` are the hitbox top-left.
- Timer: `time_left` starts at the level's `time` (default 400) and decrements
  once every 24 frames. At 0 the player dies with cause `"timeout"`.

### 3.2 Tiles (`tiles.py`)

```python
class Tile(IntEnum):
    EMPTY = 0
    GROUND = 1
    HARD = 2
    BRICK = 3
    QUESTION_COIN = 4
    QUESTION_MUSHROOM = 5
    USED = 6
    PIPE_TL = 7
    PIPE_TR = 8
    PIPE_L = 9
    PIPE_R = 10
    COIN = 11
    FLAGPOLE = 12
    FLAG_TOP = 13


SOLID: frozenset[int]  # GROUND HARD BRICK QUESTION_* USED PIPE_*
```

### 3.3 Levels (`level.py`, `mario_play/levels/*.txt`)

ASCII, one char per tile. Lines starting with `;` are metadata (`; time=400`).
Fewer than 15 rows → padded with empty rows on top. All rows right-padded to the
longest row. Width ≥ 16.

| char | meaning | char | meaning |
|---|---|---|---|
| ` ` `.` | empty | `o` | coin |
| `#` | ground | `[` `]` | pipe left/right (top lip auto-detected) |
| `X` | hard block | `g` | walker spawn |
| `B` | brick | `k` | turtle spawn |
| `?` | ?-block (coin) | `S` | player start (feet tile) |
| `M` | ?-block (mushroom) | `F` | flagpole tile (topmost becomes `FLAG_TOP`) |

```python
@dataclass
class Spawn: kind: str; col: int; row: int          # kind: "walker" | "turtle"

class Level:
    name: str
    tiles: np.ndarray            # (15, width_tiles) int8, tiles[row, col], row 0 = top
    width_tiles: int; width_px: int
    player_start: tuple[int, int]   # (col, row) of the tile the player's feet occupy
    spawns: list[Spawn]
    time: int
    flag_col: int
    @classmethod
    def from_string(cls, text: str, name: str = "custom") -> "Level"
    def copy(self) -> "Level"
    def solid_at(self, col: int, row: int) -> bool   # out of bounds: left/right = solid, above/below = empty

def load_level(name_or_path: str) -> Level      # bundled name or filesystem path
def list_levels() -> list[str]                  # ["1-1", "1-2", "1-3", "flat"]
```

Parsing errors (no `S`, no `F`, unknown char, >15 rows) raise `ValueError` with
line/column. Bundled levels: `flat` (≈60 tiles wide, no enemies, no pits — sanity
level), `1-1` (≈200 wide, gentle intro: walkers, pipes, a few pits ≤ 3 wide,
staircase, flag), `1-2` and `1-3` (harder: wider pits ≤ 4, turtles, denser
enemies). No corridor may require a 1-tile-high gap (no crouch). Every bundled
level must be completable — proven by `SearchAgent` in tests.

### 3.4 Entities and rules (`entities.py`, `physics.py`)

```python
class Entity:            # x, y, w, h, vx, vy, alive, kind, facing(+1/-1)
class Player(Entity):    # kind="player"; big, on_ground, invuln_frames, dead
class Walker(Entity):    # kind="walker"; squished_frames (>0 = flat corpse, harmless)
class Turtle(Entity):    # kind="turtle"; state: "walk" | "shell" | "shell_moving"
class Mushroom(Entity):  # kind="mushroom"
```

- Movement resolves X then Y against the tile grid (AABB), sub-stepped so no
  tunnelling. Enemies walk at 0.5 px/frame, reverse at walls, fall off ledges,
  and reverse when bumping each other. Moving shells travel at 3.5 px/frame.
- Enemies are dormant until activated: a spawn activates (the entity is created,
  facing left) the first time `spawn_x < camera_x + 384`, i.e. half a screen
  before it scrolls into view. Entities that fall below the level are removed.
- Stomp: player moving down and player's bottom above the enemy's vertical
  midpoint at contact → walker squished / turtle becomes shell / moving shell
  stops; player bounces. Touching a stationary shell kicks it away from the
  player. Any other enemy contact hurts: big → small with 120 invulnerable
  frames; small → death (`"enemy"`). Moving shells kill other enemies.
- Head-bump on a tile from below: `QUESTION_COIN` → +1 coin, tile `USED`;
  `QUESTION_MUSHROOM` → spawns a mushroom above, tile `USED`; `BRICK` → breaks
  (tile `EMPTY`) if big, otherwise nothing.
- `COIN` tiles are collected on overlap. Mushroom makes the player big (feet stay
  put). Falling below the level kills (`"pit"`). Overlapping `FLAGPOLE`/`FLAG_TOP`
  wins immediately.
- Camera: `camera_x = clamp(player.x − 112, prev_camera_x, width_px − 256)` — never
  scrolls left; the left screen edge is a wall for the player.
- Score: coin 200, stomp 100, brick 50, mushroom 1000, flag 1000 + 10·time_left.

### 3.5 Engine (`engine.py`)

```python
@dataclass(frozen=True)
class Buttons: left: bool = False; right: bool = False; jump: bool = False; run: bool = False

@dataclass
class StepEvents:
    coins: int = 0; stomps: int = 0; bricks: int = 0; powerups: int = 0
    hurt: bool = False; died: bool = False; won: bool = False
    death_cause: str | None = None      # "pit" | "enemy" | "timeout"
    score_delta: int = 0

class Game:
    def __init__(self, level: Level | str = "1-1", seed: int | None = None)
    level: Level                 # private mutable copy
    player: Player
    entities: list[Entity]       # active non-player entities
    frame: int; time_left: int; score: int; coins: int; camera_x: float
    over: bool; won: bool; death_cause: str | None
    rng: random.Random           # the only randomness source
    def reset(self, seed: int | None = None) -> None
    def step(self, buttons: Buttons) -> StepEvents     # no-op returning empty events once over
    def clone(self) -> "Game"                          # independent deep copy
    def snapshot(self) -> dict    # x, y, vx, vy, on_ground, big, coins, score, time_left, over, won, death_cause, frame
```

Jump is edge-triggered: a new jump needs `jump` released since the last jump and
`on_ground`. Determinism: same level + seed + button sequence ⇒ identical
`snapshot()` at every frame, also across `clone()`.

### 3.6 Rendering (`sprites.py`, `font.py`, `renderer.py`, `human.py`)

- `sprites.py`: original pixel art as palette-indexed strings → `(h, w, 4)` uint8
  RGBA arrays. `get_sprite(name: str) -> np.ndarray`; names include tiles
  (`"ground"`, `"brick"`, `"question"`, `"used"`, `"hard"`, `"pipe_tl"` …,
  `"coin"`, `"flagpole"`, `"flag_top"`), player (`"player_small_stand|walk1|walk2|jump"`,
  `"player_big_…"`), enemies (`"walker_1|2|flat"`, `"turtle_1|2"`, `"shell"`),
  `"mushroom"`, decorations. `blit(dst_rgb, sprite_rgba, x, y, flip=False)`
  clips to bounds and honours alpha (binary).
- `font.py`: 5×7 bitmap font (A–Z, 0–9, `-`, `×`, space); `draw_text(dst, text, x, y, color)`.
- `renderer.py`:

```python
class Renderer:
    def __init__(self, hud: bool = True)
    def render(self, game: Game) -> np.ndarray   # new (240, 256, 3) uint8 array each call
```

  Pre-renders the whole level's static background once; each call diffs
  `game.level.tiles` against its cached copy (repaints changed tiles; full
  re-render if shape/level differs), crops the camera window, blits entities and
  player (blink while invulnerable), draws the HUD (score, coins, time). Works
  with any `Game`, including clones and after `reset`.
- `human.py`: `play(level: str, scale: int = 3, fps: int = 60)` — pygame window.
  Arrows/WASD move, Z/Space jump, X/Shift run, R restart, P pause, Esc quit.
  On death or win shows an overlay then restarts. pygame is imported lazily.

## 4. Environment (`mario_play.envs`)

```python
ACTION_SETS: dict[str, list[Buttons]]
#  "right_only" (5):  noop, right, right+jump, right+run, right+run+jump
#  "simple"     (7):  right_only + jump, left
#  "complex"    (10): simple + left+jump, left+run, left+run+jump
#  Index order is exactly as listed; index 0 is always noop.

@dataclass
class RewardConfig:
    progress_weight: float = 1 / 16     # per pixel of Δx → +1 per tile moved right
    time_penalty: float = -0.01         # per env step
    death_penalty: float = -15.0
    flag_bonus: float = 50.0
    coin_bonus: float = 0.0
    score_weight: float = 0.0
    clip: float | None = None           # symmetric clip of the per-step reward

class MarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    def __init__(self, level: str | Sequence[str] = "1-1", obs_mode: str = "pixels",
                 action_set: str = "simple", frame_skip: int = 4,
                 reward: RewardConfig | dict | None = None,
                 stall_steps: int | None = None, hud: bool = True,
                 render_mode: str | None = None)
    game: Game        # public, for agents/tests (env.unwrapped.game)
```

- `level` may be a list: one is sampled per episode with `self.np_random`.
- An env step repeats the action for `frame_skip` frames (stops early when the
  game is over) and sums the reward. `terminated` = death or flag. `truncated` =
  `stall_steps` consecutive env steps without a new max x (if set). Time limits
  by step count use Gymnasium's `TimeLimit` via `max_episode_steps` in the factory.
- `obs_mode="pixels"`: `Box(0, 255, (240, 256, 3), uint8)`.
- `obs_mode="grid"`: `Box(-1, 1, (C, 15, 16), float32)`, window horizontally
  egocentric (player in window column 4; columns outside the level read as solid),
  all 15 rows. Planes: 0 solid, 1 breakable/question, 2 coin, 3 enemy (walker,
  turtle, shell), 4 moving shell, 5 mushroom, 6 goal (flagpole), 7 player body;
  then scalar planes broadcast over the grid: 8 `vx/2.5`, 9 `vy/5`, 10 on_ground,
  11 big, 12 sub-tile x offset `(x mod 16)/16`, 13 time fraction. `C = 14`.
- `info`: `x_pos`, `max_x`, `progress` (0–1 toward the flag), `coins`, `score`,
  `time_left`, `flag_get`, `death_cause`, `level`.
- `render()`: `rgb_array` → frame; `human` → pygame window showing the frame.
- Registered as `MarioPlay-v0` on `import mario_play.envs`. Must pass
  `gymnasium.utils.env_checker.check_env`.

Wrappers (`wrappers.py`): `GrayscaleResize(env, size=(84, 84), grayscale=True)` →
`(H, W)` uint8 when grayscale, else channel-first `(3, H, W)` uint8 (Pillow
resize); `FrameStack(env, k)` → stacks on a new leading axis for 2-D obs
(`(k, H, W)`), or concatenates along axis 0 for `(C, H, W)` obs (`(k·C, H, W)`);
on reset the first frame is repeated. The factory always applies `FrameStack` in
pixel mode (so pixel obs are always `(C, H, W)`), and in grid mode only if
`frame_stack > 1`.

Factory (`factory.py`, written as shared contract code):

```python
def make_env(cfg: EnvConfig, seed: int | None = None, render_mode: str | None = None) -> gym.Env
```

`MarioPlay-v0` → `MarioEnv` + wrappers per config; any other id →
`gym.make(cfg.id, **cfg.kwargs)`. Always wraps with `RecordEpisodeStatistics`
(so finished episodes carry `info["episode"] = {"r", "l", "t"}`).

## 5. Baseline agents (`mario_play.agents`)

All take a `MarioEnv` (unwrapped access to `env.game`) and expose
`act(obs) -> int`. `RandomAgent`; `HeuristicAgent` (run right, jump for
obstacles/pits/enemies from game state); `SearchAgent` (lookahead over
`Game.clone()` with macro-actions; picks the action maximizing progress while
surviving). Purpose: prove levels are completable, sanity-check env/reward,
provide non-learning baselines for `eval`/`watch`/`record`.

## 6. RL framework (`mario_play.rl`)

### 6.1 Config (`config.py`, shared contract code)

Dataclasses `EnvConfig`, `NetworkConfig`, `PPOConfig`, `DQNConfig`, `TrainConfig`
(fields in the source are authoritative). `load_config(path, overrides)` reads
YAML mirroring the dataclass tree and applies dotted overrides
(`ppo.lr=1e-4 env.level=flat`); unknown keys raise. `config_to_dict` /
`config_from_dict` round-trip for checkpoints.

### 6.2 Vectorized envs (`types.py`, `vec_env.py`)

```python
@dataclass
class VecStep:
    obs: np.ndarray          # (n, *obs_shape) next obs; for finished envs: first obs of the NEW episode
    rewards: np.ndarray      # (n,) float32
    terminated: np.ndarray   # (n,) bool
    truncated: np.ndarray    # (n,) bool
    final_obs: np.ndarray    # (n, *obs_shape) true successor obs of this transition (== obs where not done)
    infos: list[dict]        # per-env info of this step (terminal info for finished envs)

class VecEnv(ABC):
    n_envs: int; single_observation_space: gym.Space; single_action_space: gym.Space
    def reset(self, seed: int | None = None) -> np.ndarray     # env i seeded with seed + i
    def step(self, actions: np.ndarray) -> VecStep             # same-step auto-reset
    def close(self) -> None
class SyncVecEnv(VecEnv): ...
class SubprocVecEnv(VecEnv): ...   # multiprocessing "spawn"; env_fns must be picklable (functools.partial)
def make_vec_env(cfg: EnvConfig, n_envs: int, seed: int, kind: str = "sync") -> VecEnv
```

Worker errors propagate to the parent with traceback; `close()` is idempotent and
never hangs.

### 6.3 Networks (`networks.py`)

`build_encoder(obs_space, net_cfg) -> (nn.Module, feature_dim)` chooses:
uint8 image `(C, H, W)` → `NatureCNN` (scales by 1/255, flatten size computed
dynamically); float `(C, 15, 16)`-like grids → `GridEncoder` (small conv stack);
1-D vectors → `MLPEncoder`. `ActorCritic(obs_space, n_actions, net_cfg)` with
`get_value`, `get_action_and_value(obs, action=None)`; `QNetwork(obs_space,
n_actions, net_cfg, dueling)`. Orthogonal init (√2 hidden, 0.01 policy head,
1.0 value head).

### 6.4 Algorithm interface (`algos/base.py`, shared contract code)

```python
class Algorithm(ABC):
    def __init__(self, obs_space, action_space, cfg: TrainConfig, device: torch.device, n_envs: int)
    def select_actions(self, obs: np.ndarray, global_step: int) -> tuple[np.ndarray, dict]
    def observe(self, obs: np.ndarray, actions: np.ndarray, extras: dict, step: VecStep) -> None
    def ready_to_update(self, global_step: int) -> bool
    def update(self, global_step: int, progress: float) -> dict[str, float]
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray
    def state_dict(self) -> dict
    def load_state_dict(self, state: dict) -> None
```

The trainer drives one loop for both on- and off-policy algorithms:
`select_actions → venv.step → observe → (ready_to_update → update)`.

- **PPO** (`ppo.py`, `buffers/rollout.py`): rollout of `n_steps × n_envs`;
  GAE(λ); truncation bootstrapping (`r += γ·V(final_obs)` when truncated and not
  terminated); clipped surrogate, optional clipped value loss, entropy bonus,
  advantage normalization, minibatch epochs, grad-norm clip, linear LR anneal
  via `progress`, optional `target_kl` early stop. Metrics: policy_loss,
  value_loss, entropy, approx_kl, clip_frac, explained_variance, lr.
- **DQN** (`dqn.py`, `buffers/replay.py`): uniform replay storing obs in their
  native dtype (uint8 frames stay uint8) with `final_obs` as successor and
  `terminated` (not truncated) as the done flag; ε-greedy with linear schedule on
  `global_step`; Double DQN targets; optional dueling head; Huber loss; hard
  target sync every `target_update_interval` env steps (or Polyak if `tau < 1`).
  Replay contents are not checkpointed (documented): on resume the buffer refills.

### 6.5 Trainer, checkpoints, logging, evaluation

- `Trainer(cfg, resume: str | None)`. Run dir `runs/<run_name>/` holds
  `config.yaml`, `metrics.csv`, `tb/`, `checkpoints/{ckpt_<step>.pt, latest.pt, best.pt}`.
- Device `auto` → CUDA, else MPS, else CPU. Seeding of python/numpy/torch.
- Checkpoint = model + optimizer + algorithm counters + `global_step` + RNG states
  + best eval score + config dict; contains only tensors and Python primitives so
  it loads with `torch.load(weights_only=True)`. Atomic write (tmp + rename).
  Keeps the last `keep_checkpoints`.
- Resume continues `global_step`, LR schedule, ε schedule, logging (append).
- `evaluate(algo, env_cfg, episodes, seed, deterministic) -> dict` with
  `mean_return`, `std_return`, `mean_length`, and for Mario `flag_rate`,
  `mean_progress`. Periodic eval in training updates `best.pt`.
- Logger: console line every `log_interval` steps (steps, SPS, mean return/length
  over last 100 episodes, flag rate, progress, losses), CSV, TensorBoard.
- `SIGINT` → saves `latest.pt` and exits cleanly.

## 7. CLI (`mario-play`, also `python -m mario_play`)

```
mario-play play   [--level 1-1] [--scale 3]
mario-play train  --config configs/ppo_grid.yaml [--resume PATH] [key=value ...]
mario-play eval   (--checkpoint PATH | --agent random|heuristic|search) [--episodes N] [--level L] [--stochastic]
mario-play watch  (--checkpoint PATH | --agent ...) [--level L] [--scale 3]
mario-play record (--checkpoint PATH | --agent ...) --out run.gif [--level L]
mario-play bench  [--obs-mode grid|pixels] [--n-envs N] [--vec sync|subproc] [--steps N]
mario-play levels
```

## 8. Tooling

`uv` project (src layout, hatchling), Python ≥ 3.10 (3.11 pinned locally). Deps:
numpy, gymnasium ≥ 1.0, torch ≥ 2.2, pygame-ce, pyyaml, tensorboard, pillow.
Dev: pytest, pytest-timeout, ruff. Ruff lint + format (line length 100).
GitHub Actions: lint + `pytest -m "not slow"` with CPU-only torch on Python 3.10
and 3.12; separate job for slow tests. MIT license. `CLAUDE.md` documents
commands and architecture for future sessions.

## 9. Verification without a full training run

1. **Unit tests** — physics invariants, collision, stomp/hurt/power-up rules,
   level parsing errors, determinism and clone independence, renderer output
   shape/diffing, `check_env`, wrappers, reward accounting, VecEnv auto-reset
   and `final_obs` semantics, GAE vs hand-computed values, buffers, config
   overrides, checkpoint round-trip and resume continuity.
2. **Algorithm correctness (slow)** — PPO and DQN each solve `CartPole-v1`
   (mean return ≥ 195 PPO / ≥ 150 DQN over 20 eval episodes) on CPU.
3. **Levels are beatable** — `SearchAgent` finishes every bundled level.
4. **End-to-end smoke** — short PPO run, grid obs, `flat` level on the M4: reward
   should rise; results reported honestly. Random and heuristic agents as
   reference points.
5. **Throughput** — `mario-play bench` reports env steps/s for sizing training.

## 10. Build process

Spec → implementation plan → multi-agent build: shared contract code first, then
parallel subsystem agents with disjoint file ownership, then adversarial review,
fixes, integration verification, and finally the private GitHub repo + CI.
