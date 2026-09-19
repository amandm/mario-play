"""The `mario-play` command line (also `python -m mario_play`).

    mario-play play   [--level 1-1] [--scale 3]
    mario-play train  --config configs/ppo_grid.yaml [--resume PATH] [key=value ...]
    mario-play eval   (--checkpoint PATH | --agent random|heuristic|search) [--episodes N] [--json]
    mario-play watch  (--checkpoint PATH | --agent ...) [--level L] [--scale 3]
    mario-play record (--checkpoint PATH | --agent ...) --out run.gif [--level L]
    mario-play bench  [--obs-mode grid|pixels] [--n-envs N] [--vec sync|subproc] [--steps N]
    mario-play levels

This module imports nothing heavy at the top: torch, gymnasium and pygame are
imported inside the command handlers, so `--help` and `levels` answer at once and
the worker processes of a subprocess vec env (which re-import the main module)
start light.

Exit codes: 0 success, 2 user error (one `error: ...` line on stderr, no
traceback), 130 training interrupted with Ctrl-C (after `latest.pt` was saved).
Anything else that goes wrong is a bug and keeps its traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mario_play import __version__

if TYPE_CHECKING:
    import gymnasium as gym
    import numpy as np

    from mario_play.rl.config import EnvConfig, TrainConfig

AGENTS = ("random", "heuristic", "search")
VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm")
DEFAULT_LEVEL = "1-1"
DEFAULT_EVAL_SEED = 10_000  # `EvalConfig.seed`: baselines and policies face the same episodes
EXIT_OK, EXIT_USER_ERROR, EXIT_INTERRUPTED = 0, 2, 130
_MIN_GIF_DELAY_MS = 20  # browsers play shorter delays at 100 ms instead
_WATCH_PAUSE_S = 0.5  # the last frame of an episode stays up this long before the next one

FrameSink = Callable[["np.ndarray"], None]


class CLIError(Exception):
    """A mistake of the user (bad path, bad value, missing extra): reported as one line, exit 2."""


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    """The options `eval`, `watch` and `record` share: who plays, where, and for how long."""
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="a trained policy: a checkpoint file, or a run directory (its checkpoints/latest.pt)",
    )
    source.add_argument("--agent", choices=AGENTS, help="a non-learning baseline agent")
    parser.add_argument(
        "--level",
        metavar="L",
        help=f"bundled level name or level file (default: the checkpoint's level; "
        f"{DEFAULT_LEVEL} for --agent)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions from the trained policy instead of taking its best one",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=f"episode k is reset with seed + k (default: the checkpoint's eval seed; "
        f"{DEFAULT_EVAL_SEED} for --agent)",
    )
    parser.add_argument(
        "--max-steps", type=int, metavar="N", help="cut every episode off after N env steps"
    )
    parser.add_argument(
        "--device", default="cpu", help="torch device of the policy (default: cpu; auto, cuda, mps)"
    )


def build_parser() -> argparse.ArgumentParser:
    """The argparse tree of the `mario-play` command."""
    parser = argparse.ArgumentParser(
        prog="mario-play",
        description="An original Mario-style platformer, its Gymnasium env and an RL framework.",
    )
    parser.add_argument("--version", action="version", version=f"mario-play {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    play = commands.add_parser("play", help="play a level with the keyboard")
    play.add_argument("--level", default=DEFAULT_LEVEL, help="bundled level name or level file")
    play.add_argument("--scale", type=int, default=3, help="window magnification (default: 3)")
    play.add_argument("--fps", type=int, default=60, help="frame rate cap (default: 60)")
    play.add_argument("--max-frames", type=int, metavar="N", help="quit after N frames")
    play.set_defaults(handler=_cmd_play)

    train = commands.add_parser(
        "train",
        help="train (or resume) a policy from a YAML config",
        epilog="examples:  mario-play train --config configs/ppo_grid.yaml ppo.lr=1e-4 "
        "env.level=flat  |  mario-play train --resume runs/my_run total_timesteps=2e7",
    )
    train.add_argument("--config", metavar="PATH", help="YAML training config (see configs/)")
    train.add_argument(
        "--resume",
        metavar="PATH",
        help="checkpoint file or run directory to continue; without --config the config "
        "stored in the checkpoint is used",
    )
    train.add_argument(
        "overrides", nargs="*", metavar="key=value", help="dotted config overrides, applied last"
    )
    train.set_defaults(handler=_cmd_train)

    evaluate = commands.add_parser("eval", help="score a trained policy or a baseline agent")
    _add_policy_arguments(evaluate)
    evaluate.add_argument("--episodes", type=int, default=5, help="episodes to play (default: 5)")
    evaluate.add_argument("--json", action="store_true", help="print the result as JSON only")
    evaluate.set_defaults(handler=_cmd_eval)

    watch = commands.add_parser("watch", help="watch a policy or a baseline agent play in a window")
    _add_policy_arguments(watch)
    watch.add_argument("--scale", type=int, default=3, help="window magnification (default: 3)")
    watch.add_argument(
        "--episodes", type=int, help="stop after N episodes (default: until the window is closed)"
    )
    watch.set_defaults(handler=_cmd_watch)

    record = commands.add_parser("record", help="record episodes as a GIF (or MP4 with imageio)")
    _add_policy_arguments(record)
    record.add_argument(
        "--out", required=True, metavar="FILE", help=".gif, or .mp4 (needs the 'video' extra)"
    )
    record.add_argument("--episodes", type=int, default=1, help="episodes to record (default: 1)")
    record.add_argument(
        "--scale", type=int, default=1, help="integer upscaling of the 256x240 frame"
    )
    record.add_argument(
        "--fps", type=float, help="playback rate (default: the env's real-time rate)"
    )
    record.set_defaults(handler=_cmd_record)

    bench = commands.add_parser("bench", help="measure env throughput with random actions")
    bench.add_argument("--obs-mode", choices=("grid", "pixels"), default="grid")
    bench.add_argument("--n-envs", type=int, default=8, help="size of the vec env (default: 8)")
    bench.add_argument("--vec", choices=("sync", "subproc"), default="sync")
    bench.add_argument(
        "--steps", type=int, default=10_000, help="env steps per measurement (default: 10000)"
    )
    bench.add_argument("--level", default=DEFAULT_LEVEL, help="bundled level name or level file")
    bench.set_defaults(handler=_cmd_bench)

    levels = commands.add_parser("levels", help="list the bundled levels")
    levels.set_defaults(handler=_cmd_levels)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the process exit code (see the module docstring)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse has printed --help (0) or a usage error (2) already
        return exc.code if isinstance(exc.code, int) else EXIT_USER_ERROR
    try:
        return int(args.handler(args))
    except CLIError as exc:
        print(f"error: {' '.join(str(exc).split())}", file=sys.stderr)
        return EXIT_USER_ERROR


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _positive(name: str, value: int | float | None, optional: bool = False) -> None:
    if value is None and optional:
        return
    if value is None or value < 1:
        raise CLIError(f"{name} must be at least 1, got {value}")


def _check_level(level: str) -> None:
    """Fail early, and in one line, for a level that does not exist or does not parse."""
    from mario_play.game.level import load_level

    try:
        load_level(level)
    except (ValueError, OSError) as exc:
        raise CLIError(str(exc)) from None


def _check_levels(env_cfg: EnvConfig) -> None:
    from mario_play.rl.config import MARIO_ENV_ID

    if env_cfg.id != MARIO_ENV_ID:
        return
    levels = [env_cfg.level] if isinstance(env_cfg.level, str) else list(env_cfg.level)
    if not levels:
        raise CLIError("env.level must name at least one level")
    for level in levels:
        _check_level(level)


def _checkpoint_file(path: str) -> Path:
    """`path` itself, or `latest.pt` of the run directory / checkpoints directory it names."""
    target = Path(path)
    if target.is_dir():
        for candidate in (target / "latest.pt", target / "checkpoints" / "latest.pt"):
            if candidate.is_file():
                return candidate
        raise CLIError(f"no latest.pt in {target} or {target / 'checkpoints'}")
    if not target.is_file():
        raise CLIError(f"checkpoint not found: {target}")
    return target


def _unreadable(path: Path, exc: Exception) -> CLIError:
    return CLIError(f"cannot load checkpoint {path}: {type(exc).__name__}: {exc}")


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    from mario_play.rl.checkpoint import load_checkpoint

    try:
        return load_checkpoint(path, map_location="cpu")
    except Exception as exc:  # a foreign or truncated file fails in many ways inside the unpickler
        raise _unreadable(path, exc) from None


def _resolve_device(name: str) -> None:
    from mario_play.rl.utils import resolve_device

    try:
        resolve_device(name)
    except (ValueError, RuntimeError) as exc:
        raise CLIError(str(exc)) from None


class _PolicyActor:
    """A trained `Algorithm` behind the baseline agents' `act` / `reset` interface."""

    def __init__(self, algo: Any, deterministic: bool) -> None:
        self._algo = algo
        self._deterministic = deterministic

    def reset(self) -> None:
        """Feed-forward policies carry nothing from one episode to the next."""

    def act(self, obs: Any) -> int:
        import numpy as np

        batch = np.asarray(obs)[None]
        return int(np.asarray(self._algo.predict(batch, deterministic=self._deterministic))[0])


class _Player:
    """Who plays (`--checkpoint` or `--agent`) and in which env: shared by eval / watch / record."""

    def __init__(self, args: argparse.Namespace) -> None:
        from mario_play.rl.config import MARIO_ENV_ID, EnvConfig

        _positive("--max-steps", args.max_steps, optional=True)
        self.max_steps: int | None = args.max_steps
        self._algo: Any = None
        self._agent_name: str | None = args.agent
        self._deterministic = not args.stochastic

        if args.checkpoint is not None:
            from mario_play.rl.evaluate import load_algorithm

            self.checkpoint: Path | None = _checkpoint_file(args.checkpoint)
            _resolve_device(args.device)
            try:
                self._algo, cfg = load_algorithm(self.checkpoint, device=args.device)
            except Exception as exc:  # not a checkpoint, or one this code cannot rebuild
                raise _unreadable(self.checkpoint, exc) from None
            self.env_cfg: EnvConfig = cfg.env
            self.seed: int = cfg.eval.seed if args.seed is None else args.seed
            self.name = f"{cfg.algo} {self.checkpoint}"
            if args.level is not None and self.env_cfg.id != MARIO_ENV_ID:
                raise CLIError(
                    f"--level needs a {MARIO_ENV_ID} policy, this one plays {cfg.env.id}"
                )
        else:
            if args.stochastic:
                raise CLIError("--stochastic samples from a trained policy: it needs --checkpoint")
            self.checkpoint = None
            # Baseline agents read the game state, not the observation: the cheap grid will do.
            self.env_cfg = EnvConfig(level=DEFAULT_LEVEL, obs_mode="grid")
            self.seed = DEFAULT_EVAL_SEED if args.seed is None else args.seed
            self.name = str(args.agent)
        if args.level is not None:
            self.env_cfg.level = args.level
        _check_levels(self.env_cfg)

    @property
    def level(self) -> str | None:
        """The level(s) played, for display; None for a non-Mario env."""
        from mario_play.rl.config import MARIO_ENV_ID

        if self.env_cfg.id != MARIO_ENV_ID:
            return None
        level = self.env_cfg.level
        return level if isinstance(level, str) else ",".join(level)

    def make_env(self, render_mode: str | None = None) -> gym.Env:
        """A fresh env for this player."""
        from mario_play.envs.factory import make_env

        return make_env(self.env_cfg, seed=self.seed, render_mode=render_mode)

    def actor(self, env: gym.Env) -> Any:
        """The `act(obs) -> int` / `reset()` object that plays in `env`."""
        if self._algo is not None:
            if not self._deterministic:
                import torch

                torch.manual_seed(self.seed)  # sampled actions come from torch's global generator
            return _PolicyActor(self._algo, self._deterministic)
        if self._agent_name == "random":
            from mario_play.agents.random_agent import RandomAgent

            return RandomAgent(env, seed=self.seed)
        if self._agent_name == "heuristic":
            from mario_play.agents.heuristic_agent import HeuristicAgent

            return HeuristicAgent(env)
        from mario_play.agents.search_agent import SearchAgent

        return SearchAgent(env)

    def episodes(
        self, env: gym.Env, count: int | None, on_frame: FrameSink | None = None
    ) -> Iterator[dict[str, Any]]:
        """Play `count` episodes (None: forever) and yield one summary dict after each.

        Episode `k` is reset with `seed + k`, exactly like `mario_play.rl.evaluate.evaluate`
        does, so the numbers are comparable with the evaluations of a training run.
        `on_frame` receives `env.render()` after every reset and step.
        """
        actor = self.actor(env)
        index = 0
        while count is None or index < count:
            obs, info = env.reset(seed=self.seed + index)
            actor.reset()
            if on_frame is not None:
                on_frame(env.render())
            total, length = 0.0, 0
            while self.max_steps is None or length < self.max_steps:
                obs, reward, terminated, truncated, info = env.step(actor.act(obs))
                total += float(reward)
                length += 1
                if on_frame is not None:
                    on_frame(env.render())
                if terminated or truncated:
                    break
            yield _episode_summary(index, total, length, info)
            index += 1


def _as_float(value: Any) -> float:
    """Python float from a Python/numpy scalar or a one-element array (as found in env infos)."""
    import numpy as np

    return float(np.asarray(value).reshape(-1)[0])


def _episode_summary(index: int, total: float, length: int, info: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"episode": index, "return": total, "length": length}
    if "flag_get" in info:
        summary["flag_get"] = bool(_as_float(info["flag_get"]))
    if "progress" in info:
        summary["progress"] = _as_float(info["progress"])
    if "death_cause" in info:
        summary["death_cause"] = info["death_cause"]
    return summary


def _episode_line(summary: dict[str, Any]) -> str:
    parts = [
        f"episode {summary['episode'] + 1}",
        f"return {summary['return']:.2f}",
        f"{summary['length']} steps",
    ]
    if "progress" in summary:
        parts.append(f"progress {summary['progress']:.3f}")
    if summary.get("flag_get"):
        parts.append("flag")
    elif summary.get("death_cause"):
        parts.append(f"died ({summary['death_cause']})")
    return " | ".join(parts)


def _format_number(value: Any) -> str:
    return str(value) if isinstance(value, int) else f"{value:.3f}"


def _format_metrics(metrics: dict[str, Any], separator: str) -> str:
    return separator.join(f"{key} {_format_number(value)}" for key, value in metrics.items())


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The keys of `mario_play.rl.evaluate.evaluate`, computed the same way."""
    import numpy as np

    summary: dict[str, Any] = {
        "mean_return": float(np.mean([r["return"] for r in results])),
        "std_return": float(np.std([r["return"] for r in results])),
        "mean_length": float(np.mean([r["length"] for r in results])),
        "episodes": len(results),
    }
    flags = [float(r["flag_get"]) for r in results if "flag_get" in r]
    progress = [r["progress"] for r in results if "progress" in r]
    if flags:
        summary["flag_rate"] = float(np.mean(flags))
    if progress:
        summary["mean_progress"] = float(np.mean(progress))
    return summary


# --------------------------------------------------------------------------- #
# play / levels
# --------------------------------------------------------------------------- #


def _cmd_play(args: argparse.Namespace) -> int:
    _positive("--scale", args.scale)
    _positive("--max-frames", args.max_frames, optional=True)
    _check_level(args.level)
    from mario_play.game.human import CONTROLS, play

    print(CONTROLS)
    play(level=args.level, scale=args.scale, fps=args.fps, max_frames=args.max_frames)
    return EXIT_OK


def _cmd_levels(args: argparse.Namespace) -> int:
    from mario_play.game.level import list_levels, load_level

    print(f"{'level':<8}{'width':>7}{'time':>6}{'walkers':>9}{'turtles':>9}")
    for name in list_levels():
        level = load_level(name)
        kinds = [spawn.kind for spawn in level.spawns]
        print(
            f"{name:<8}{level.width_tiles:>7}{level.time:>6}"
            f"{kinds.count('walker'):>9}{kinds.count('turtle'):>9}"
        )
    return EXIT_OK


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #


def _train_config(args: argparse.Namespace) -> TrainConfig:
    """The config of this run: `--config` (or the resumed checkpoint's) plus the overrides."""
    import yaml

    from mario_play.rl.config import apply_overrides, config_from_dict, load_config

    try:
        if args.config is not None:
            if not Path(args.config).is_file():
                raise CLIError(f"config file not found: {args.config}")
            return load_config(args.config, args.overrides)
        payload = _load_checkpoint_payload(_checkpoint_file(args.resume))
        return config_from_dict(apply_overrides(payload["config"], args.overrides))
    except (ValueError, yaml.YAMLError) as exc:
        raise CLIError(f"bad config: {exc}") from None


def _config_lines(args: argparse.Namespace, cfg: TrainConfig) -> list[str]:
    source = args.config if args.config is not None else f"stored in {args.resume}"
    if args.overrides:
        source += f" + {' '.join(args.overrides)}"
    env = cfg.env
    env_parts = [env.id]
    if _is_mario(cfg):
        level = env.level if isinstance(env.level, str) else ",".join(env.level)
        env_parts += [
            f"level {level}",
            f"obs {env.obs_mode}",
            f"actions {env.action_set}",
            f"frame_skip {env.frame_skip}",
            f"stall_steps {env.stall_steps}",
        ]
    if env.max_episode_steps:
        env_parts.append(f"max_episode_steps {env.max_episode_steps}")
    return [
        f"config   {source}",
        f"env      {' | '.join(env_parts)}",
        f"training {cfg.algo} | {cfg.total_timesteps:,} steps | {cfg.n_envs} {cfg.vec_env} envs | "
        f"device {cfg.device} | seed {cfg.seed}",
    ]


def _is_mario(cfg: TrainConfig) -> bool:
    from mario_play.rl.config import MARIO_ENV_ID

    return cfg.env.id == MARIO_ENV_ID


def _cmd_train(args: argparse.Namespace) -> int:
    if args.config is None and args.resume is None:
        raise CLIError("train needs --config PATH, or --resume PATH to continue a run")
    if args.resume is not None:
        _checkpoint_file(args.resume)  # a missing checkpoint is reported before anything is built
    cfg = _train_config(args)
    _check_levels(cfg.env)
    _resolve_device(cfg.device)

    from mario_play.rl.trainer import Trainer

    try:
        trainer = Trainer(cfg, resume=args.resume)
    except (ValueError, FileNotFoundError) as exc:  # what a config can get wrong past its types
        raise CLIError(str(exc)) from None
    for line in _config_lines(args, trainer.cfg):
        print(line, flush=True)

    result = trainer.train()  # announces the run directory, then logs as it goes

    checkpoints = Path(result["run_dir"]) / "checkpoints"
    if result["interrupted"]:
        print(f"resume with: mario-play train --resume {result['run_dir']}")
        return EXIT_INTERRUPTED
    best = "n/a" if result["best_eval"] is None else f"{result['best_eval']:.2f}"
    print(f"finished {result['global_step']:,} steps | best eval return {best}")
    if result["final_eval"]:
        print(f"final eval: {_format_metrics(result['final_eval'], ' | ')}")
    policy = checkpoints / ("best.pt" if (checkpoints / "best.pt").is_file() else "latest.pt")
    print(f"next: mario-play eval --checkpoint {policy}   (or watch / record)")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# eval / watch / record
# --------------------------------------------------------------------------- #


def _cmd_eval(args: argparse.Namespace) -> int:
    _positive("--episodes", args.episodes)
    player = _Player(args)
    env = player.make_env()
    try:
        results = list(player.episodes(env, args.episodes))
    finally:
        env.close()

    summary = _aggregate(results)
    if args.json:
        report = {"policy": player.name, "level": player.level, "seed": player.seed, **summary}
        report["episode_results"] = results
        print(json.dumps(report, indent=2))
        return EXIT_OK

    where = f" on level {player.level}" if player.level is not None else ""
    print(f"{player.name}{where} | {args.episodes} episode(s) from seed {player.seed}")
    for result in results:
        print(f"  {_episode_line(result)}")
    for key, value in summary.items():
        print(f"{key:<14}{_format_number(value):>10}")
    return EXIT_OK


def _cmd_watch(args: argparse.Namespace) -> int:
    _positive("--scale", args.scale)
    _positive("--episodes", args.episodes, optional=True)
    player = _Player(args)
    env = player.make_env(render_mode="human")
    if hasattr(env.unwrapped, "window_scale"):
        env.unwrapped.window_scale = args.scale
    print(f"watching {player.name} - P pauses, Esc or closing the window quits")
    try:
        for result in player.episodes(env, args.episodes):
            print(_episode_line(result), flush=True)
            if args.episodes is None or result["episode"] + 1 < args.episodes:
                time.sleep(_WATCH_PAUSE_S)
    except KeyboardInterrupt:  # how the env's human mode reports Esc / a closed window
        pass
    finally:
        env.close()
    return EXIT_OK


def _cmd_record(args: argparse.Namespace) -> int:
    _positive("--episodes", args.episodes)
    _positive("--scale", args.scale)
    if args.fps is not None and not 0 < args.fps <= 1000 / _MIN_GIF_DELAY_MS:
        raise CLIError(f"--fps must be in (0, {1000 // _MIN_GIF_DELAY_MS}], got {args.fps}")
    out = Path(args.out)
    suffix = out.suffix.lower()
    if suffix != ".gif" and suffix not in VIDEO_SUFFIXES:
        raise CLIError(
            f"cannot write {suffix or 'a file without extension'}: "
            f"use .gif, or {' '.join(VIDEO_SUFFIXES)} with imageio"
        )
    if suffix in VIDEO_SUFFIXES:
        _import_imageio()  # before anything is played

    player = _Player(args)
    env = player.make_env(render_mode="rgb_array")
    frames: list[np.ndarray] = []
    try:
        fps = args.fps or float(env.unwrapped.metadata.get("render_fps") or 15)
        for result in player.episodes(env, args.episodes, on_frame=frames.append):
            print(_episode_line(result), flush=True)
    finally:
        env.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".gif":
        _write_gif(out, frames, fps, args.scale)
    else:
        _write_video(out, frames, fps, args.scale)
    height, width = frames[0].shape[:2]
    print(
        f"wrote {out}: {len(frames)} frames, {width * args.scale}x{height * args.scale}, "
        f"{fps:g} fps, {out.stat().st_size / 1024:.0f} KiB"
        if out.is_file()
        else f"wrote {out}: {len(frames)} frames"
    )
    return EXIT_OK


def _upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour magnification: pixel art must stay crisp."""
    if scale == 1:
        return frame
    return frame.repeat(scale, axis=0).repeat(scale, axis=1)


def gif_durations(n_frames: int, fps: float) -> list[int]:
    """Per-frame delays in ms for a GIF that plays `n_frames` at `fps`.

    GIF stores delays in hundredths of a second, so 15 fps (66.7 ms) cannot be
    written as such. Rounding the *cumulative* time instead of every delay mixes
    60 and 70 ms frames and keeps the clip's length - and so its speed - right.
    """
    durations: list[int] = []
    shown = 0
    for index in range(1, n_frames + 1):
        target = round(index * 100 / fps) * 10
        delay = max(_MIN_GIF_DELAY_MS, target - shown)
        durations.append(delay)
        shown += delay
    return durations


def _write_gif(path: Path, frames: list[np.ndarray], fps: float, scale: int) -> None:
    from PIL import Image

    def convert(frame: np.ndarray) -> Image.Image:
        # No dithering: the game has few flat colours, which an adaptive palette keeps exact.
        image = Image.fromarray(_upscale(frame, scale))
        return image.quantize(colors=256, dither=Image.Dither.NONE)

    first = convert(frames[0])
    first.save(
        path,
        save_all=True,
        append_images=(convert(frame) for frame in frames[1:]),  # lazily: one big frame at a time
        duration=gif_durations(len(frames), fps),
        loop=0,
        optimize=False,
    )


def _import_imageio() -> Any:
    try:
        import imageio
    except ImportError:
        raise CLIError(
            "writing video needs imageio, which is not installed: "
            'pip install "imageio[ffmpeg]" (the "video" extra of mario-play), or record a .gif'
        ) from None
    return imageio


def _write_video(path: Path, frames: list[np.ndarray], fps: float, scale: int) -> None:
    imageio = _import_imageio()
    try:
        writer = imageio.get_writer(str(path), fps=fps)
    except (ImportError, ValueError) as exc:  # imageio without its ffmpeg plugin
        raise CLIError(f"imageio cannot write {path.suffix}: {exc}") from None
    try:
        for frame in frames:
            writer.append_data(_upscale(frame, scale))
    finally:
        writer.close()


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #


def _bench_single(env_cfg: EnvConfig, steps: int) -> float:
    """Seconds for `steps` random-action steps of one env (resets included)."""
    from mario_play.envs.factory import make_env

    env = make_env(env_cfg, seed=0)
    try:
        env.reset(seed=0)
        started = time.perf_counter()
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()
        return time.perf_counter() - started
    finally:
        env.close()


def _bench_vec(env_cfg: EnvConfig, n_envs: int, kind: str, vector_steps: int) -> float:
    """Seconds for `vector_steps` random-action steps of a vec env (worker start-up excluded)."""
    import numpy as np

    from mario_play.rl.vec_env import make_vec_env

    rng = np.random.default_rng(0)
    venv = make_vec_env(env_cfg, n_envs, seed=0, kind=kind)
    try:
        venv.reset(seed=0)
        n_actions = int(venv.single_action_space.n)
        started = time.perf_counter()
        for _ in range(vector_steps):
            venv.step(rng.integers(n_actions, size=n_envs))
        return time.perf_counter() - started
    finally:
        venv.close()


def _cmd_bench(args: argparse.Namespace) -> int:
    _positive("--steps", args.steps)
    _positive("--n-envs", args.n_envs)
    _check_level(args.level)
    from mario_play.envs.factory import make_env
    from mario_play.rl.config import EnvConfig

    # The pixel pipeline as trained: 84x84 grayscale, 4 stacked frames.
    env_cfg = EnvConfig(
        level=args.level, obs_mode=args.obs_mode, frame_stack=4 if args.obs_mode == "pixels" else 1
    )
    probe = make_env(env_cfg)
    space = probe.observation_space
    probe.close()
    print(
        f"bench: level {args.level} | obs {args.obs_mode} {space.shape} {space.dtype} | "
        f"frame_skip {env_cfg.frame_skip} | random actions"
    )

    vector_steps = max(1, args.steps // args.n_envs)
    measurements = [
        ("single env", args.steps, _bench_single(env_cfg, args.steps)),
        (
            f"{args.n_envs} {args.vec} envs",
            vector_steps * args.n_envs,
            _bench_vec(env_cfg, args.n_envs, args.vec, vector_steps),
        ),
    ]
    print(
        f"{'setup':<18} | {'env steps':>10} | {'seconds':>8} | {'steps/s':>10} | {'frames/s':>10}"
    )
    for name, steps, seconds in measurements:
        rate = steps / max(seconds, 1e-9)
        print(
            f"{name:<18} | {steps:>10,} | {seconds:>8.2f} | {rate:>10,.0f} | "
            f"{rate * env_cfg.frame_skip:>10,.0f}"
        )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
