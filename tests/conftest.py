"""Shared pytest setup: headless SDL and single-threaded torch for fast, stable tests."""

import os

# Must be set before pygame / torch are imported anywhere in the test session.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
