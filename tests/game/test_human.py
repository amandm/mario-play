"""Human play: key mapping, the pygame window, the play session rules and the `play` loop.

Everything runs under SDL's dummy video driver (see tests/conftest.py): no real window opens.
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict

import numpy as np
import pygame
import pytest

from mario_play.game import human
from mario_play.game.constants import VIEW_H, VIEW_W
from mario_play.game.engine import Buttons
from mario_play.game.human import (
    OVERLAY_FRAMES,
    PlaySession,
    Window,
    buttons_from_keys,
    draw_overlay,
    key_bindings,
    play,
)
from mario_play.game.level import Level
from mario_play.game.renderer import Renderer

# The player starts two tiles left of the flagpole: walking right wins within half a second.
WIN_LEVEL = "\n".join([" S F".ljust(16), "#" * 16])
# One unit of time is 24 frames, so standing still dies of "timeout" on frame 24.
TIMEOUT_LEVEL = "\n".join(["; time=1", " S" + " " * 13 + "F", "#" * 16])
FRAMES_TO_TIMEOUT = 24

RIGHT = Buttons(right=True)
IDLE = Buttons()


@pytest.fixture(autouse=True)
def _display_is_quit_after_each_test():
    yield
    pygame.display.quit()


def _noise_frame(height: int = VIEW_H, width: int = VIEW_W, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _displayed() -> np.ndarray:
    """What the pygame display currently shows, as an (H, W, 3) array."""
    return pygame.surfarray.array3d(pygame.display.get_surface()).swapaxes(0, 1)


def _upscaled(frame: np.ndarray, factor: int) -> np.ndarray:
    return frame.repeat(factor, axis=0).repeat(factor, axis=1)


def _press(*keys: int) -> None:
    for key in keys:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))


# --- key mapping ---------------------------------------------------------------------------------


def test_no_keys_pressed_is_a_noop():
    assert buttons_from_keys({}) == Buttons()
    assert buttons_from_keys(set()) == Buttons()
    assert buttons_from_keys([]) == Buttons()


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (pygame.K_LEFT, Buttons(left=True)),
        (pygame.K_a, Buttons(left=True)),
        (pygame.K_RIGHT, Buttons(right=True)),
        (pygame.K_d, Buttons(right=True)),
        (pygame.K_z, Buttons(jump=True)),
        (pygame.K_SPACE, Buttons(jump=True)),
        (pygame.K_UP, Buttons(jump=True)),
        (pygame.K_w, Buttons(jump=True)),
        (pygame.K_x, Buttons(run=True)),
        (pygame.K_LSHIFT, Buttons(run=True)),
        (pygame.K_RSHIFT, Buttons(run=True)),
    ],
)
def test_every_bound_key_maps_to_its_button(key, expected):
    assert buttons_from_keys({key: True}) == expected
    assert buttons_from_keys({key}) == expected


@pytest.mark.parametrize(
    "key",
    [pygame.K_DOWN, pygame.K_s, pygame.K_r, pygame.K_p, pygame.K_ESCAPE, pygame.K_RETURN],
)
def test_keys_without_a_button_are_ignored(key):
    assert buttons_from_keys({key: True}) == Buttons()


def test_key_combinations_set_several_buttons():
    pressed = {pygame.K_RIGHT: True, pygame.K_x: True, pygame.K_SPACE: True}
    assert buttons_from_keys(pressed) == Buttons(right=True, jump=True, run=True)
    everything = {pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_LSHIFT}
    assert buttons_from_keys(everything) == Buttons(left=True, right=True, jump=True, run=True)


def test_a_key_that_is_present_but_released_does_not_count():
    assert buttons_from_keys({pygame.K_RIGHT: False, pygame.K_z: 0}) == Buttons()


def test_accepts_mappings_sequences_and_pygames_own_key_state():
    held = defaultdict(bool, {pygame.K_LEFT: True})
    assert buttons_from_keys(held) == Buttons(left=True)

    # A plain sequence is too short for the arrow-key constants: those count as released.
    sequence = [False] * 512
    sequence[pygame.K_z] = True
    sequence[pygame.K_d] = True
    assert buttons_from_keys(sequence) == Buttons(right=True, jump=True)
    assert buttons_from_keys(np.asarray(sequence)) == Buttons(right=True, jump=True)

    pygame.display.init()
    assert buttons_from_keys(pygame.key.get_pressed()) == Buttons()


def test_key_bindings_cover_every_action_and_are_read_only():
    bindings = key_bindings()
    assert set(bindings) == {"left", "right", "jump", "run", "restart", "pause", "quit"}
    assert bindings["restart"] == (pygame.K_r,)
    assert bindings["pause"] == (pygame.K_p,)
    assert bindings["quit"] == (pygame.K_ESCAPE,)
    all_keys = [key for keys in bindings.values() for key in keys]
    assert len(all_keys) == len(set(all_keys)), "one key is bound to two actions"
    with pytest.raises(TypeError):
        bindings["left"] = ()


def test_importing_the_module_does_not_import_pygame():
    code = (
        "import sys\n"
        "import mario_play.game.human\n"
        "assert 'pygame' not in sys.modules, 'pygame was imported eagerly'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- window --------------------------------------------------------------------------------------


def test_window_opens_at_the_scaled_view_size_with_its_title():
    window = Window(scale=2, title="a test window")
    try:
        assert window.size == (2 * VIEW_W, 2 * VIEW_H)
        assert pygame.display.get_surface().get_size() == (2 * VIEW_W, 2 * VIEW_H)
        assert pygame.display.get_caption()[0] == "a test window"
        assert not window.closed
    finally:
        window.close()


def test_window_defaults_match_the_contract():
    window = Window()
    try:
        assert window.scale == 3
        assert window.title == "mario-play"
        assert window.size == (768, 720)
    finally:
        window.close()


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_show_upscales_the_frame_pixel_perfectly(scale):
    frame = _noise_frame()
    window = Window(scale=scale)
    try:
        window.show(frame)
        assert np.array_equal(_displayed(), _upscaled(frame, scale))
    finally:
        window.close()


def test_show_accepts_views_and_replaces_the_previous_frame():
    window = Window(scale=1)
    try:
        window.show(_noise_frame(seed=1))
        mirrored = _noise_frame(seed=2)[:, ::-1]  # a non-contiguous view
        window.show(mirrored)
        assert np.array_equal(_displayed(), mirrored)
    finally:
        window.close()


def test_show_stretches_a_frame_of_another_size_over_the_window():
    small = _noise_frame(height=VIEW_H // 2, width=VIEW_W // 2)
    window = Window(scale=1)
    try:
        window.show(small)
        window.show(_noise_frame(seed=3))  # and back to the usual size
        window.show(small)
        assert np.array_equal(_displayed(), _upscaled(small, 2))
    finally:
        window.close()


def test_show_does_not_modify_the_frame():
    frame = _noise_frame()
    before = frame.copy()
    window = Window(scale=1)
    try:
        window.show(frame)
    finally:
        window.close()
    assert np.array_equal(frame, before)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((VIEW_H, VIEW_W), dtype=np.uint8),
        np.zeros((VIEW_H, VIEW_W, 4), dtype=np.uint8),
        np.zeros((VIEW_H, VIEW_W, 3), dtype=np.float32),
        np.zeros((0, VIEW_W, 3), dtype=np.uint8),
        [[0, 0, 0]],
    ],
)
def test_show_rejects_anything_but_an_rgb_uint8_image(bad):
    window = Window(scale=1)
    try:
        with pytest.raises(ValueError, match="uint8"):
            window.show(bad)
    finally:
        window.close()


@pytest.mark.parametrize("scale", [0, -1, 1.5, "3", True, None])
def test_window_rejects_a_bad_scale_before_opening_anything(scale):
    with pytest.raises(ValueError, match="scale"):
        Window(scale=scale)
    assert not pygame.display.get_init()


def test_window_accepts_a_numpy_integer_scale():
    with Window(scale=np.int64(2)) as window:
        assert window.scale == 2 and type(window.scale) is int
        assert window.size == (2 * VIEW_W, 2 * VIEW_H)


def test_a_window_that_fails_to_open_leaves_no_display_behind(monkeypatch):
    def no_video(size):
        raise pygame.error("No available video device")

    monkeypatch.setattr(pygame.display, "set_mode", no_video)
    with pytest.raises(pygame.error, match="No available video device"):
        Window(scale=1)
    assert not pygame.display.get_init()


def test_close_is_idempotent_and_quits_the_display():
    window = Window(scale=1)
    window.close()
    assert window.closed
    assert not pygame.display.get_init()
    window.close()
    window.close()
    assert window.closed


def test_a_closed_window_refuses_frames_and_reports_quit():
    window = Window(scale=1)
    window.close()
    with pytest.raises(RuntimeError, match="closed"):
        window.show(_noise_frame())
    assert window.poll() == {"quit": True, "buttons": Buttons(), "restart": False, "pause": False}
    window.tick(0)  # harmless
    assert not pygame.display.get_init(), "a closed window must not reopen the display"


def test_window_is_a_context_manager():
    with Window(scale=1) as window:
        window.show(_noise_frame())
        assert not window.closed
    assert window.closed
    assert not pygame.display.get_init()


def test_poll_without_input_reports_nothing():
    with Window(scale=1) as window:
        result = window.poll()
        assert result == {"quit": False, "buttons": Buttons(), "restart": False, "pause": False}
        assert list(result) == ["quit", "buttons", "restart", "pause"]
        assert isinstance(result["buttons"], Buttons)
        assert all(type(result[name]) is bool for name in ("quit", "restart", "pause"))


def test_poll_reports_closing_the_window_and_escape_as_quit():
    with Window(scale=1) as window:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        assert window.poll()["quit"] is True
        assert window.poll()["quit"] is False, "events are consumed by the poll that reports them"
        _press(pygame.K_ESCAPE)
        assert window.poll()["quit"] is True


def test_poll_reports_restart_and_pause_once_per_key_press():
    with Window(scale=1) as window:
        _press(pygame.K_r)
        assert window.poll() == {"quit": False, "buttons": IDLE, "restart": True, "pause": False}
        _press(pygame.K_p)
        assert window.poll() == {"quit": False, "buttons": IDLE, "restart": False, "pause": True}
        assert window.poll() == {"quit": False, "buttons": IDLE, "restart": False, "pause": False}


def test_poll_counts_a_key_tapped_between_two_polls_as_pressed():
    with Window(scale=1) as window:
        _press(pygame.K_RIGHT, pygame.K_z)
        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_z))
        assert window.poll()["buttons"] == Buttons(right=True, jump=True)
        assert window.poll()["buttons"] == Buttons()


def test_poll_survives_events_without_a_key():
    with Window(scale=1) as window:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN))
        pygame.event.post(pygame.event.Event(pygame.USEREVENT, note="not for the window"))
        assert window.poll()["quit"] is False


def test_tick_limits_the_frame_rate_only_when_asked_to():
    with Window(scale=1) as window:
        window.tick(0)
        window.tick(0)  # uncapped: returns at once
        window.tick(1000)
        assert window.tick(60) is None


def test_window_survives_another_window_closing_the_shared_display():
    first = Window(scale=1)
    second = Window(scale=2)
    try:
        second.close()  # pygame has one display: this takes the first window's surface away
        frame = _noise_frame()
        first.show(frame)
        assert np.array_equal(_displayed(), frame)
        assert first.poll()["quit"] is False
    finally:
        first.close()


def test_poll_also_reopens_a_display_that_another_window_closed():
    first = Window(scale=1)
    Window(scale=1).close()
    assert not pygame.display.get_init()
    try:
        assert first.poll() == {"quit": False, "buttons": IDLE, "restart": False, "pause": False}
        assert pygame.display.get_surface().get_size() == first.size
        _press(pygame.K_r)
        assert first.poll()["restart"] is True
    finally:
        first.close()


def test_window_takes_the_display_back_after_another_window_resized_it():
    first = Window(scale=1)
    second = Window(scale=2)
    try:
        frame = _noise_frame()
        assert pygame.display.get_surface().get_size() == second.size
        second.show(frame)
        assert np.array_equal(_displayed(), _upscaled(frame, 2))
        first.show(frame)
        assert np.array_equal(_displayed(), frame)
        second.show(frame)
        assert np.array_equal(_displayed(), _upscaled(frame, 2))
    finally:
        second.close()
        first.close()


def test_show_presents_the_frame_and_keeps_the_event_queue_alive(monkeypatch):
    calls = []
    flip, pump = pygame.display.flip, pygame.event.pump
    monkeypatch.setattr(pygame.display, "flip", lambda: (calls.append("flip"), flip())[1])
    monkeypatch.setattr(pygame.event, "pump", lambda: (calls.append("pump"), pump())[1])
    with Window(scale=1) as window:
        window.show(_noise_frame())
        assert calls == ["flip", "pump"]
        _press(pygame.K_r)
        window.show(_noise_frame())
        assert window.poll()["restart"] is True, "showing a frame must not swallow input"


# --- overlay -------------------------------------------------------------------------------------


def test_draw_overlay_only_touches_a_band_across_the_middle():
    frame = np.full((VIEW_H, VIEW_W, 3), 200, dtype=np.uint8)
    draw_overlay(frame, "TRY AGAIN")
    changed_rows = np.flatnonzero((frame != 200).any(axis=(1, 2)))
    assert changed_rows.size > 0
    top, bottom = changed_rows[0], changed_rows[-1]
    assert VIEW_H // 4 < top and bottom < 3 * VIEW_H // 4
    assert top < VIEW_H // 2 < bottom
    assert abs((top + bottom + 1) / 2 - VIEW_H / 2) <= 1, "the band is vertically centred"
    # The band is darkened edge to edge, the text on it is bright.
    band = frame[top : bottom + 1]
    assert (band[:, 0] < 200).all() and (band[:, -1] < 200).all()
    assert (band > 200).any()


def test_draw_overlay_centres_the_text_horizontally():
    frame = np.full((VIEW_H, VIEW_W, 3), 100, dtype=np.uint8)
    draw_overlay(frame, "COURSE CLEAR", color=(255, 255, 255))
    columns = np.flatnonzero((frame == 255).all(axis=2).any(axis=0))
    assert abs((columns[0] + columns[-1] + 1) / 2 - VIEW_W / 2) <= 1


def test_draw_overlay_with_a_subtitle_draws_a_taller_band():
    plain = np.full((VIEW_H, VIEW_W, 3), 200, dtype=np.uint8)
    titled = plain.copy()
    draw_overlay(plain, "PAUSED")
    draw_overlay(titled, "PAUSED", subtitle="PRESS P TO RESUME")
    rows_plain = (plain != 200).any(axis=(1, 2)).sum()
    rows_titled = (titled != 200).any(axis=(1, 2)).sum()
    assert rows_titled > rows_plain


def test_draw_overlay_is_safe_on_tiny_frames():
    tiny = np.full((8, 8, 3), 200, dtype=np.uint8)
    draw_overlay(tiny, "COURSE CLEAR", subtitle="WELL DONE")
    assert tiny.shape == (8, 8, 3)


# --- play session --------------------------------------------------------------------------------


def test_session_steps_the_game_with_the_given_buttons():
    session = PlaySession("flat")
    start_x = session.game.player.x
    for _ in range(30):
        session.update(RIGHT)
    assert session.game.frame == 30
    assert session.game.player.x > start_x
    assert session.message is None


def test_session_accepts_a_level_object():
    session = PlaySession(Level.from_string(WIN_LEVEL, name="tiny"))
    assert session.game.level.name == "tiny"


def test_session_frame_is_the_renderers_frame_while_playing():
    session = PlaySession("flat")
    session.update(RIGHT)
    frame = session.render()
    assert frame.shape == (VIEW_H, VIEW_W, 3) and frame.dtype == np.uint8
    assert np.array_equal(frame, Renderer().render(session.game))
    assert frame is not session.render(), "a new array per call"


def test_death_shows_try_again_for_ninety_frames_then_restarts():
    assert OVERLAY_FRAMES == 90
    session = PlaySession(Level.from_string(TIMEOUT_LEVEL, name="timeout"))
    for _ in range(FRAMES_TO_TIMEOUT - 1):
        session.update(IDLE)
    assert not session.game.over and session.message is None
    session.update(IDLE)
    assert session.game.over and session.game.death_cause == "timeout"

    shown = 0
    while session.game.over:
        assert session.message == "TRY AGAIN"
        assert not np.array_equal(session.render(), Renderer().render(session.game))
        shown += 1
        session.update(RIGHT)
    assert shown == OVERLAY_FRAMES
    assert session.game.frame == 0 and session.game.time_left == 1
    assert session.message is None
    session.update(RIGHT)
    assert session.game.frame == 1, "play continues after the automatic restart"


def test_win_shows_course_clear_then_restarts():
    session = PlaySession(Level.from_string(WIN_LEVEL, name="tiny"))
    for _ in range(120):
        session.update(RIGHT)
        if session.game.over:
            break
    assert session.game.won
    assert session.message == "COURSE CLEAR"
    overlay = session.render()
    assert not np.array_equal(overlay, Renderer().render(session.game))
    for _ in range(OVERLAY_FRAMES - 1):
        session.update(IDLE)
    assert session.game.won and session.message == "COURSE CLEAR"
    session.update(IDLE)
    assert not session.game.over and session.game.frame == 0 and session.game.score == 0


def test_restart_resets_the_game_at_once():
    session = PlaySession("flat")
    start_x = session.game.player.x
    for _ in range(40):
        session.update(RIGHT)
    session.update(RIGHT, restart=True)
    assert session.game.frame == 0
    assert session.game.player.x == start_x


def test_restart_cuts_the_end_of_run_overlay_short():
    session = PlaySession(Level.from_string(TIMEOUT_LEVEL, name="timeout"))
    for _ in range(FRAMES_TO_TIMEOUT + 5):
        session.update(IDLE)
    assert session.message == "TRY AGAIN"
    session.update(IDLE, restart=True)
    assert not session.game.over and session.message is None
    # A fresh run gets the full overlay again when it ends.
    for _ in range(FRAMES_TO_TIMEOUT):
        session.update(IDLE)
    assert session.overlay_frames == OVERLAY_FRAMES


def test_pause_freezes_the_game_until_toggled_again():
    session = PlaySession("flat")
    for _ in range(10):
        session.update(RIGHT)
    before = session.game.snapshot()

    session.update(RIGHT, pause=True)
    assert session.paused and session.message == "PAUSED"
    for _ in range(25):
        session.update(RIGHT)
    assert session.game.snapshot() == before
    assert not np.array_equal(session.render(), Renderer().render(session.game))

    session.update(RIGHT, pause=True)
    assert not session.paused and session.message is None
    assert session.game.frame == before["frame"] + 1


def test_pause_also_holds_the_end_of_run_overlay():
    session = PlaySession(Level.from_string(TIMEOUT_LEVEL, name="timeout"))
    for _ in range(FRAMES_TO_TIMEOUT + 10):
        session.update(IDLE)
    remaining = session.overlay_frames
    session.update(IDLE, pause=True)
    for _ in range(200):
        session.update(IDLE)
    assert session.game.over and session.overlay_frames == remaining
    assert session.message == "PAUSED"
    session.update(IDLE, pause=True)
    assert session.message == "TRY AGAIN"


def _render_at_every_invuln(session: PlaySession) -> list[np.ndarray]:
    """`session.render()` for every value the invulnerability counter can have (1..120)."""
    frames = []
    for invuln in range(1, 121):
        session.game.player.invuln_frames = invuln
        frames.append(session.render())
    return frames


def test_the_player_never_blinks_away_under_the_pause_message():
    # Regression: pausing freezes `invuln_frames`, so a pause that began in a hidden window of
    # the invulnerability blink showed a level without a player until the game was resumed.
    session = PlaySession("flat")
    for _ in range(10):
        session.update(RIGHT)
    session.update(IDLE, pause=True)
    steady = session.render()
    assert session.game.player.invuln_frames == 0

    for invuln, frame in enumerate(_render_at_every_invuln(session), start=1):
        assert np.array_equal(frame, steady), f"player hidden while paused at invuln={invuln}"
    assert session.game.player.invuln_frames == 120, "rendering must not use up invulnerability"


def test_the_player_never_blinks_away_under_the_end_of_run_message():
    session = PlaySession(Level.from_string(WIN_LEVEL, name="tiny"))
    for _ in range(120):
        session.update(RIGHT)
        if session.game.over:
            break
    assert session.message == "COURSE CLEAR"
    steady = session.render()

    for invuln, frame in enumerate(_render_at_every_invuln(session), start=1):
        assert np.array_equal(frame, steady), f"player hidden under the message at invuln={invuln}"


def test_the_player_still_blinks_during_play():
    session = PlaySession("flat")
    session.update(IDLE)
    steady = session.render()

    hidden = [not np.array_equal(f, steady) for f in _render_at_every_invuln(session)]

    assert 56 <= sum(hidden) <= 64, "hidden in every other 4-frame window"
    assert np.array_equal(session.render(), Renderer().render(session.game))


def test_restart_also_resumes_a_paused_session():
    session = PlaySession("flat")
    for _ in range(10):
        session.update(RIGHT)
    session.update(IDLE, pause=True)
    session.update(IDLE, restart=True)
    assert not session.paused and session.game.frame == 0
    session.update(RIGHT)
    assert session.game.frame == 1


# --- play ----------------------------------------------------------------------------------------


def test_play_runs_for_max_frames_and_cleans_up():
    assert play("flat", scale=1, fps=0, max_frames=30) is None
    assert not pygame.display.get_init()


def test_play_at_the_default_scale_and_frame_rate():
    play(max_frames=3)
    assert not pygame.display.get_init()


def test_play_stops_when_the_window_is_closed(monkeypatch):
    shown = []
    original = Window.show
    monkeypatch.setattr(
        Window, "show", lambda self, frame: (shown.append(frame), original(self, frame))[1]
    )
    pygame.display.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    play("flat", scale=1, fps=0, max_frames=100_000)
    assert shown == []
    assert not pygame.display.get_init()


def test_play_rejects_an_unknown_level_before_opening_a_window():
    with pytest.raises(ValueError, match="no-such-level"):
        play("no-such-level", max_frames=1)
    assert not pygame.display.get_init()


class _ScriptedWindow:
    """Stands in for `Window`: replays scripted input and records what `play` does with it."""

    script: list[dict] = []
    fail_on_show = False
    last: _ScriptedWindow | None = None

    def __init__(self, scale: int = 3, title: str = "mario-play") -> None:
        self.scale, self.title = scale, title
        self.pending = list(type(self).script)
        self.frames: list[np.ndarray] = []
        self.ticks: list[int] = []
        self.close_calls = 0
        type(self).last = self

    def poll(self) -> dict:
        neutral = {"quit": False, "buttons": Buttons(), "restart": False, "pause": False}
        return {**neutral, **(self.pending.pop(0) if self.pending else {})}

    def show(self, frame: np.ndarray) -> None:
        if type(self).fail_on_show:
            raise RuntimeError("the display broke")
        self.frames.append(frame)

    def tick(self, fps: int) -> None:
        self.ticks.append(fps)

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def scripted_window(monkeypatch):
    fake = type("ScriptedWindow", (_ScriptedWindow,), {})
    monkeypatch.setattr(human, "Window", fake)
    return fake


def test_play_feeds_the_keyboard_to_the_game_and_shows_every_frame(scripted_window):
    scripted_window.script = [{"buttons": RIGHT}] * 40
    play("flat", scale=2, fps=50, max_frames=40)
    window = scripted_window.last
    assert window.scale == 2 and "flat" in window.title
    assert len(window.frames) == 40 and window.ticks == [50] * 40
    assert all(f.shape == (VIEW_H, VIEW_W, 3) and f.dtype == np.uint8 for f in window.frames)
    assert not np.array_equal(window.frames[0], window.frames[-1]), "the player moved"
    assert window.close_calls == 1


def test_play_quits_on_request_without_showing_another_frame(scripted_window):
    scripted_window.script = [{}, {}, {}, {"quit": True}]
    play("flat", max_frames=None)
    window = scripted_window.last
    assert len(window.frames) == 3
    assert window.close_calls == 1


def test_play_passes_restart_and_pause_on_to_the_session(scripted_window):
    scripted_window.script = [{"buttons": RIGHT}] * 30 + [{"pause": True}] + [{}] * 9
    play("flat", max_frames=40)
    paused = scripted_window.last.frames[-9:]
    assert all(np.array_equal(paused[0], frame) for frame in paused[1:]), "paused: a still picture"

    scripted_window.script = [{"buttons": RIGHT}] * 30 + [{"restart": True}]
    play("flat", max_frames=31)
    frames = scripted_window.last.frames
    fresh = Renderer().render(PlaySession("flat").game)
    assert np.array_equal(frames[-1], fresh)
    assert not np.array_equal(frames[-2], fresh)


def test_play_closes_the_window_when_something_goes_wrong(scripted_window):
    scripted_window.fail_on_show = True
    with pytest.raises(RuntimeError, match="display broke"):
        play("flat", max_frames=5)
    assert scripted_window.last.close_calls == 1


def test_play_with_no_frames_to_show_returns_at_once(scripted_window):
    play("flat", max_frames=0)
    assert scripted_window.last.frames == []
    assert scripted_window.last.close_calls == 1
