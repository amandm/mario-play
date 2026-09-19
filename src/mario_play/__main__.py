"""Allow `python -m mario_play ...` as an alias for the `mario-play` command."""

from mario_play.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
