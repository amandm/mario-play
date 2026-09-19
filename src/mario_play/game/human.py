"""Human play: a pygame window around the numpy renderer, and a keyboard-driven game loop.

The window does no drawing of its own. `Renderer` produces the same ``(240, 256, 3)`` frame an
agent observes; `Window` upscales it with nearest-neighbour sampling and reads the keyboard. It
is also what the Gymnasium env's ``human`` render mode and the CLI's ``watch`` command show
their frames in, so every method works without the `play` loop.

Three layers, from pure to stateful:

* `buttons_from_keys` and `draw_overlay` are pure functions;
* `PlaySession` holds the rules of a play session (pause, restart, the end-of-run overlay)
  and needs no window, so it can be driven frame by frame;
* `Window` and `play` talk to pygame.

pygame is imported lazily, inside functions only: importing this module needs no display stack
and the rest of `mario_play.game` stays free of pygame.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from functools import lru_cache
from types import MappingProxyType, ModuleType, TracebackType
from typing import Any, TypedDict

import numpy as np

from mario_play.game.constants import FPS, VIEW_H, VIEW_W
from mario_play.game.engine import Buttons, Game
from mario_play.game.font import GLYPH_H, Color, draw_text, text_width
from mario_play.game.level import Level
from mario_play.game.renderer import Renderer
from mario_play.game.sprites import get_sprite

CONTROLS = "Arrows/WASD move | Z/Space/Up/W jump | X/Shift run | R restart | P pause | Esc quit"
"""One-line summary of the key bindings, for help texts."""

OVERLAY_FRAMES = 90
"""How long the end-of-run message stays up before the level restarts."""

WIN_TEXT = "COURSE CLEAR"
LOSE_TEXT = "TRY AGAIN"
PAUSE_TEXT = "PAUSED"
_PAUSE_HINT = "PRESS P TO RESUME"

_TEXT: Color = (252, 252, 252)
_TEXT_WIN: Color = (252, 216, 96)
_SHADOW: Color = (30, 24, 48)
_TITLE_SCALE = 2
_SUBTITLE_GAP = 6  # rows between the title and the subtitle
_BAND_PADDING = 8  # darkened rows above and below the text

# pygame constant names per action; resolved by `key_bindings` once pygame is loaded.
_KEY_NAMES: dict[str, tuple[str, ...]] = {
    "left": ("K_LEFT", "K_a"),
    "right": ("K_RIGHT", "K_d"),
    "jump": ("K_z", "K_SPACE", "K_UP", "K_w"),
    "run": ("K_x", "K_LSHIFT", "K_RSHIFT"),
    "restart": ("K_r",),
    "pause": ("K_p",),
    "quit": ("K_ESCAPE",),
}

PressedKeys = Mapping[int, Any] | Sequence[Any] | AbstractSet[int]
"""Key state indexable by pygame key constants, or simply the set of keys that are down."""


class Inputs(TypedDict):
    """What `Window.poll` reports. `restart` and `pause` are true once per key press."""

    quit: bool
    buttons: Buttons
    restart: bool
    pause: bool


def _import_pygame() -> ModuleType:
    """Import pygame on first use; without its banner, which would end up in CLI output."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    return pygame


# --- keyboard ------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def key_bindings() -> Mapping[str, tuple[int, ...]]:
    """The pygame key constants bound to each action (read-only).

    Actions: ``left right jump run`` (held, see `buttons_from_keys`) and ``restart pause quit``
    (acted on once per key press, see `Window.poll`).
    """
    pygame = _import_pygame()
    return MappingProxyType(
        {action: tuple(getattr(pygame, n) for n in names) for action, names in _KEY_NAMES.items()}
    )


def _any_down(pressed: PressedKeys, keys: tuple[int, ...]) -> bool:
    if isinstance(pressed, AbstractSet):
        return not pressed.isdisjoint(keys)
    for key in keys:
        try:
            if pressed[key]:
                return True
        except (KeyError, IndexError):  # a partial mapping or a short sequence: not pressed
            continue
    return False


def buttons_from_keys(pressed: PressedKeys) -> Buttons:
    """Translate keyboard state into the game's `Buttons`.

    `pressed` is anything indexable by pygame key constants - `pygame.key.get_pressed()`, a
    dict, a sequence - where missing keys count as released; or a set of the keys that are
    down. Left and right may both be set: the engine lets them cancel out.
    """
    bindings = key_bindings()
    return Buttons(
        left=_any_down(pressed, bindings["left"]),
        right=_any_down(pressed, bindings["right"]),
        jump=_any_down(pressed, bindings["jump"]),
        run=_any_down(pressed, bindings["run"]),
    )


# --- overlay -------------------------------------------------------------------------------------


def draw_overlay(
    frame: np.ndarray, text: str, subtitle: str | None = None, color: Color = _TEXT
) -> None:
    """Write `text` (and a smaller `subtitle`) on a darkened band across the middle of `frame`.

    Draws in place into an ``(H, W, 3)`` uint8 frame of any size; rows outside the band are
    left untouched and everything is clipped to the frame.
    """
    height, width = frame.shape[:2]
    title_height = GLYPH_H * _TITLE_SCALE
    content_height = title_height + (_SUBTITLE_GAP + GLYPH_H if subtitle else 0)
    band_height = content_height + 2 * _BAND_PADDING
    top = max((height - band_height) // 2, 0)
    frame[top : top + band_height] >>= 2  # a quarter of the brightness keeps the scene readable

    y = top + _BAND_PADDING
    x = (width - text_width(text, _TITLE_SCALE)) // 2
    draw_text(frame, text, x, y, color, scale=_TITLE_SCALE, shadow=_SHADOW)
    if subtitle:
        x = (width - text_width(subtitle)) // 2
        draw_text(frame, subtitle, x, y + title_height + _SUBTITLE_GAP, _TEXT, shadow=_SHADOW)


# --- session -------------------------------------------------------------------------------------


class PlaySession:
    """One human play session on one level: the game plus pause, restart and the run-over message.

    Windowless, so it can be driven frame by frame: call `update` once per frame with the
    player's input, then `render` for the picture to show. When a run ends - flag or death - the
    game holds still under a message for `OVERLAY_FRAMES` frames and then restarts by itself.

    Attributes:
        game: The running `Game`.
        paused: While true, `update` changes nothing (apart from handling restart and pause).
        overlay_frames: Frames the end-of-run message still has to stay up; 0 during play.
    """

    def __init__(self, level: Level | str | os.PathLike[str] = "1-1", hud: bool = True) -> None:
        """`level`: a `Level`, a bundled level name or a level file path (see `Game`)."""
        self.game = Game(level)
        self.paused = False
        self.overlay_frames = 0
        self._renderer = Renderer(hud=hud)

    @property
    def message(self) -> str | None:
        """The text currently laid over the picture, or None during normal play."""
        if self.paused:
            return PAUSE_TEXT
        if self.game.over:
            return WIN_TEXT if self.game.won else LOSE_TEXT
        return None

    def restart(self) -> None:
        """Start the level over, also out of a pause or an end-of-run message."""
        self.game.reset()
        self.paused = False
        self.overlay_frames = 0

    def update(self, buttons: Buttons, restart: bool = False, pause: bool = False) -> None:
        """Advance the session by one frame.

        `restart` starts the level over instead of stepping; `pause` toggles the pause before
        anything else happens, so the game already moves again in the frame that resumes it.
        """
        if restart:
            self.restart()
            return
        if pause:
            self.paused = not self.paused
        if self.paused:
            return
        game = self.game
        if game.over:
            self.overlay_frames -= 1
            if self.overlay_frames <= 0:
                self.restart()
            return
        game.step(buttons)
        if game.over:
            self.overlay_frames = OVERLAY_FRAMES

    def render(self) -> np.ndarray:
        """The current picture, message included: a new ``(240, 256, 3)`` uint8 array.

        Under a message the player is always drawn, invulnerable or not.
        """
        message = self.message
        # Under a message the game holds still, and so does the invulnerability blink: without
        # this, a pause that starts in a hidden window shows a level with no player on it.
        frame = self._renderer.render(self.game, blink=message is None)
        if message == PAUSE_TEXT:
            draw_overlay(frame, PAUSE_TEXT, subtitle=_PAUSE_HINT)
        elif message is not None:
            draw_overlay(frame, message, color=_TEXT_WIN if self.game.won else _TEXT)
        return frame


# --- window --------------------------------------------------------------------------------------


def _icon(pygame: ModuleType) -> Any:
    """The small standing player as a 32x32 window icon."""
    sprite = get_sprite("player_small_stand")
    height, width = sprite.shape[:2]
    surface = pygame.image.frombuffer(sprite.tobytes(), (width, height), "RGBA")
    return pygame.transform.scale(surface, (2 * width, 2 * height))  # a copy that owns its pixels


class Window:
    """A pygame window that shows game frames upscaled by an integer factor.

    `show`, `poll`, `tick` and `close` are independent of each other: a viewer that needs no
    keyboard can just call `show` and `tick`. Usable as a context manager.

    pygame has a single display, which all windows of a process share. A window therefore
    reclaims the display whenever it finds it resized or closed by somebody else, and closing
    one window takes the display away from the others until they are used again.

    Args:
        scale: Integer magnification of the 256x240 view (3: a 768x720 window).
        title: Caption of the window.
    """

    def __init__(self, scale: int = 3, title: str = "mario-play") -> None:
        if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)) or scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale!r}")
        self.scale = int(scale)
        self.title = str(title)
        self.size = (VIEW_W * self.scale, VIEW_H * self.scale)
        self._closed = False
        self._source: Any = None  # a surface of the frame's size in the display's pixel format
        self._pygame = _import_pygame()
        self._clock = self._pygame.time.Clock()
        try:
            self._screen()
        except BaseException:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        """True once `close` has been called."""
        return self._closed

    def __enter__(self) -> Window:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _screen(self) -> Any:
        """The display surface at this window's size, (re)opened if somebody else changed it."""
        pygame = self._pygame
        if not pygame.display.get_init():
            pygame.display.init()  # the display only: no mixer, no joysticks
        screen = pygame.display.get_surface()
        if screen is None or screen.get_size() != self.size:
            pygame.display.set_icon(_icon(pygame))  # has to precede set_mode on some platforms
            screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption(self.title)
            self._source = None
        return screen

    def show(self, frame: np.ndarray) -> None:
        """Display an ``(H, W, 3)`` uint8 RGB frame, stretched over the whole window.

        Meant for the game's ``(240, 256, 3)`` frames, which come out magnified by exactly
        `scale` with crisp pixels. Also keeps the window responsive for callers that never
        `poll`. Raises `RuntimeError` on a closed window.
        """
        if self._closed:
            raise RuntimeError("the window is closed")
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            found = (
                f"{frame.shape} {frame.dtype}"
                if isinstance(frame, np.ndarray)
                else type(frame).__name__
            )
            raise ValueError(f"frame must be a non-empty (H, W, 3) uint8 array, got {found}")
        pygame = self._pygame
        screen = self._screen()
        height, width = frame.shape[:2]
        source = self._source
        if source is None or source.get_size() != (width, height):
            # In the display's format, which lets `scale` write straight into the display.
            source = self._source = pygame.Surface((width, height), 0, screen)
        pygame.surfarray.blit_array(source, frame.swapaxes(0, 1))  # surfarray is (W, H, 3)
        pygame.transform.scale(source, self.size, screen)
        pygame.display.flip()
        pygame.event.pump()  # leaves the events in the queue for `poll`

    def poll(self) -> Inputs:
        """Consume pending window events and read the keyboard.

        Returns ``{"quit", "buttons", "restart", "pause"}``: `quit` on Esc or when the window's
        close button was clicked; `restart` (R) and `pause` (P) once per key press; `buttons`
        from the keys held right now, plus any key tapped so briefly that it was pressed and
        released between two polls. A closed window reports `quit`.
        """
        if self._closed:
            return Inputs(quit=True, buttons=Buttons(), restart=False, pause=False)
        pygame = self._pygame
        self._screen()
        bindings = key_bindings()
        quit_requested = restart = pause = False
        tapped: set[int] = set()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                key = getattr(event, "key", None)
                if key in bindings["quit"]:
                    quit_requested = True
                elif key in bindings["restart"]:
                    restart = True
                elif key in bindings["pause"]:
                    pause = True
                elif key is not None:
                    tapped.add(key)
        held = buttons_from_keys(pygame.key.get_pressed())
        if tapped:
            brief = buttons_from_keys(tapped)
            held = Buttons(
                left=held.left or brief.left,
                right=held.right or brief.right,
                jump=held.jump or brief.jump,
                run=held.run or brief.run,
            )
        return Inputs(quit=quit_requested, buttons=held, restart=restart, pause=pause)

    def tick(self, fps: int) -> None:
        """Wait so that consecutive calls are at least ``1 / fps`` seconds apart.

        ``fps <= 0`` means no limit. Works on a closed window too.
        """
        self._clock.tick(max(fps, 0))

    def close(self) -> None:
        """Close the window and shut pygame's display down. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._source = None
        self._pygame.display.quit()


# --- the game loop -------------------------------------------------------------------------------


def play(level: str = "1-1", scale: int = 3, fps: int = FPS, max_frames: int | None = None) -> None:
    """Play `level` with the keyboard in a window until Esc or the close button.

    Keys: see `CONTROLS`. A finished run - flag or death - shows a message for `OVERLAY_FRAMES`
    frames, then the level starts over.

    Args:
        level: A bundled level name or a level file path.
        scale: Integer magnification of the 256x240 view.
        fps: Frame rate cap; the game runs at one simulation step per frame (``<= 0``: no cap).
        max_frames: Return after this many displayed frames (paused ones count); None: no limit.
    """
    session = PlaySession(level)  # a bad level fails here, before any window opens
    window = Window(scale=scale, title=f"mario-play [{session.game.level.name}]")
    try:
        shown = 0
        while max_frames is None or shown < max_frames:
            inputs = window.poll()
            if inputs["quit"]:
                break
            session.update(inputs["buttons"], restart=inputs["restart"], pause=inputs["pause"])
            window.show(session.render())
            window.tick(fps)
            shown += 1
    finally:
        window.close()
