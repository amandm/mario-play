"""Sprite sheet contract (names, sizes, read-only caching) and the clipping alpha blitter."""

import subprocess
import sys

import numpy as np
import pytest

from mario_play.game import sprites
from mario_play.game.sprites import SKY_COLOR, blit, get_sprite, sprite_names

SOLID_BLOCKS = ["ground", "hard", "brick", "question", "used"]
TILES = SOLID_BLOCKS + [
    "pipe_tl",
    "pipe_tr",
    "pipe_l",
    "pipe_r",
    "coin",
    "flagpole",
    "flag_top",
    "flag",
]
PLAYER_SMALL = [f"player_small_{pose}" for pose in ("stand", "walk1", "walk2", "jump")]
PLAYER_BIG = [f"player_big_{pose}" for pose in ("stand", "walk1", "walk2", "jump")]
DECORATIONS = ["cloud", "bush", "hill"]

# name -> (width, height) demanded by the plan.
EXPECTED_SIZES: dict[str, tuple[int, int]] = {
    **{name: (16, 16) for name in TILES},
    **{name: (16, 16) for name in PLAYER_SMALL},
    **{name: (16, 32) for name in PLAYER_BIG},
    "walker_1": (16, 16),
    "walker_2": (16, 16),
    "walker_flat": (16, 16),
    "turtle_1": (16, 24),
    "turtle_2": (16, 24),
    "shell": (16, 16),
    "mushroom": (16, 16),
}
CONTRACT_NAMES = sorted([*EXPECTED_SIZES, *DECORATIONS])
# Sprites the renderer anchors bottom-centre on a hitbox: they must not float.
GROUNDED = [
    "player_small_stand",
    "player_small_walk1",
    "player_small_walk2",
    "player_big_stand",
    "player_big_walk1",
    "player_big_walk2",
    "walker_1",
    "walker_2",
    "walker_flat",
    "turtle_1",
    "turtle_2",
    "shell",
    "mushroom",
    "bush",
    "hill",
]


def _opaque(sprite: np.ndarray) -> np.ndarray:
    return sprite[..., 3] > 0


def _reference_blit(dst, sprite, x, y, flip=False):
    """Obviously-correct per-pixel blit used as the oracle for the fast implementation."""
    out = dst.copy()
    src = sprite[:, ::-1] if flip else sprite
    for sy in range(src.shape[0]):
        for sx in range(src.shape[1]):
            dx, dy = x + sx, y + sy
            if 0 <= dx < out.shape[1] and 0 <= dy < out.shape[0] and src[sy, sx, 3] > 0:
                out[dy, dx] = src[sy, sx, :3]
    return out


def _noise(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _test_sprite():
    """A 3x4 sprite with a distinct colour per pixel and a transparent diagonal."""
    sprite = np.zeros((3, 4, 4), dtype=np.uint8)
    for row in range(3):
        for col in range(4):
            sprite[row, col] = (10 + row, 20 + col, 30 + row * 4 + col, 0 if row == col else 255)
    return sprite


# ---------------------------------------------------------------- sprite sheet contract


def test_every_contract_name_is_listed():
    assert set(CONTRACT_NAMES) <= set(sprite_names())


def test_sprite_names_is_a_sorted_fresh_list_of_loadable_names():
    names = sprite_names()
    assert names == sorted(names)
    assert len(names) == len(set(names))
    names.clear()
    assert sprite_names(), "mutating the returned list must not affect the module"
    for name in sprite_names():
        assert get_sprite(name).ndim == 3


@pytest.mark.parametrize("name", sorted(EXPECTED_SIZES))
def test_fixed_size_sprites_have_the_contract_shape(name):
    width, height = EXPECTED_SIZES[name]
    sprite = get_sprite(name)
    assert sprite.shape == (height, width, 4)
    assert sprite.dtype == np.uint8


@pytest.mark.parametrize("name", DECORATIONS)
def test_decorations_are_rgba_and_big_enough_to_read(name):
    sprite = get_sprite(name)
    assert sprite.dtype == np.uint8
    assert sprite.ndim == 3 and sprite.shape[2] == 4
    assert sprite.shape[0] >= 8 and sprite.shape[1] >= 16


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_sprites_are_cached_and_read_only(name):
    sprite = get_sprite(name)
    assert get_sprite(name) is sprite
    assert sprite.flags.writeable is False
    with pytest.raises(ValueError):
        sprite[0, 0, 0] = 1


@pytest.mark.parametrize("name", ["", "nope", "Ground", "player_small", "player_big_walk3"])
def test_unknown_name_raises_key_error(name):
    with pytest.raises(KeyError):
        get_sprite(name)


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_alpha_is_binary_and_something_is_visible(name):
    alpha = get_sprite(name)[..., 3]
    assert set(np.unique(alpha).tolist()) <= {0, 255}
    assert (alpha == 255).any()


@pytest.mark.parametrize("name", SOLID_BLOCKS)
def test_solid_blocks_are_fully_opaque(name):
    assert _opaque(get_sprite(name)).all()


@pytest.mark.parametrize("name", sorted(set(CONTRACT_NAMES) - set(SOLID_BLOCKS)))
def test_everything_else_has_transparent_pixels(name):
    assert not _opaque(get_sprite(name)).all()


def test_sky_color_is_an_rgb_tuple_no_sprite_disappears_into():
    assert isinstance(SKY_COLOR, tuple) and len(SKY_COLOR) == 3
    assert all(isinstance(c, int) and 0 <= c <= 255 for c in SKY_COLOR)
    for name in sprite_names():
        sprite = get_sprite(name)
        colours = {tuple(c) for c in sprite[_opaque(sprite)][:, :3].tolist()}
        assert SKY_COLOR not in colours, name


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_palettes_stay_small(name):
    sprite = get_sprite(name)
    colours = np.unique(sprite[_opaque(sprite)][:, :3], axis=0)
    assert 2 <= len(colours) <= 5, f"{name} uses {len(colours)} colours"


@pytest.mark.parametrize("name", GROUNDED)
def test_grounded_sprites_touch_their_bottom_row(name):
    assert _opaque(get_sprite(name))[-1].any()


def test_animation_frames_differ():
    for group in (PLAYER_SMALL, PLAYER_BIG, ["walker_1", "walker_2"], ["turtle_1", "turtle_2"]):
        for i, first in enumerate(group):
            for second in group[i + 1 :]:
                assert not np.array_equal(get_sprite(first), get_sprite(second)), (first, second)


def test_flat_walker_is_squashed_onto_the_ground():
    opaque = _opaque(get_sprite("walker_flat"))
    assert not opaque[:8].any()
    assert opaque[-1].sum() >= 10


def test_characters_fill_roughly_their_hitbox_height():
    # Hitboxes: small player 15 px, big player 30 px, walker/shell/mushroom 14 px, turtle 22 px.
    for name, min_rows in [
        ("player_small_stand", 14),
        ("player_big_stand", 28),
        ("walker_1", 13),
        ("turtle_1", 21),
        ("shell", 12),
        ("mushroom", 13),
    ]:
        rows = _opaque(get_sprite(name)).any(axis=1).sum()
        assert rows >= min_rows, name


def test_pole_and_pipe_segments_stack_seamlessly():
    pole = _opaque(get_sprite("flagpole"))
    assert (pole == pole[0]).all(), "every flagpole row must be identical so segments tile"
    assert np.array_equal(_opaque(get_sprite("flag_top"))[-1], pole[0])
    for name in ("pipe_l", "pipe_r"):
        body = get_sprite(name)
        assert (body == body[0]).all(), name
    # The lip is one 32 px piece: both halves mirror each other's silhouette.
    lip_left, lip_right = _opaque(get_sprite("pipe_tl")), _opaque(get_sprite("pipe_tr"))
    assert np.array_equal(lip_left, lip_right[:, ::-1])
    assert np.array_equal(_opaque(get_sprite("pipe_l")), _opaque(get_sprite("pipe_r"))[:, ::-1])


def test_optional_grass_capped_ground_is_a_full_tile():
    sprite = get_sprite("ground_top")
    assert sprite.shape == (16, 16, 4)
    assert _opaque(sprite).all()
    assert not np.array_equal(sprite, get_sprite("ground"))


@pytest.mark.parametrize(
    ("size", "palette", "rows", "message"),
    [
        ((3, 2), {"A": (1, 2, 3, 255)}, ["AAA", "AA"], "row 1"),  # ragged row
        ((3, 2), {"A": (1, 2, 3, 255)}, ["AAA", "AAAA"], "row 1"),
        ((4, 2), {"A": (1, 2, 3, 255)}, ["AAA", "AAA"], "expected 4x2"),  # wrong size
        ((3, 1), {"A": (1, 2, 3, 255)}, ["ABA"], "undefined"),  # typo'd symbol
        ((3, 1), {"A": (1, 2, 3, 255), "B": (4, 5, 6, 255)}, ["A.A"], "never uses"),
        ((3, 1), {"A": (1, 2, 3, 128)}, ["AAA"], "opaque"),  # alpha must stay binary
        ((3, 1), {"A": (1, 2, 300, 255)}, ["AAA"], "opaque"),
        ((3, 1), {".": (1, 2, 3, 255)}, ["..."], "palette key"),
        ((0, 0), {}, [], "no pixels"),
    ],
)
def test_art_typos_fail_loudly_when_compiled(size, palette, rows, message):
    with pytest.raises(ValueError, match=message) as excinfo:
        sprites._compile("typo", size, palette, rows)
    assert "typo" in str(excinfo.value)


def test_compiling_art_maps_symbols_to_colours_and_dots_to_transparency():
    palette = {"A": (1, 2, 3, 255), "B": (4, 5, 6, 255)}
    sprite = sprites._compile("ok", (2, 2), palette, ["A.", "BA"])
    assert sprite.shape == (2, 2, 4) and sprite.dtype == np.uint8
    assert sprite[0, 0].tolist() == [1, 2, 3, 255]
    assert sprite[1, 0].tolist() == [4, 5, 6, 255]
    assert sprite[0, 1, 3] == 0
    assert sprite.flags.writeable is False


def test_a_sprite_name_cannot_be_registered_twice():
    before = sprite_names()
    with pytest.raises(ValueError, match="twice"):
        sprites._sprite("coin", (1, 1), {"A": (1, 2, 3, 255)}, ["A"])
    assert sprite_names() == before
    assert get_sprite("coin").shape == (16, 16, 4)


def test_importing_sprites_and_font_pulls_in_no_heavy_or_sibling_modules():
    code = (
        "import sys\n"
        "import mario_play.game.sprites, mario_play.game.font\n"
        "banned = ['pygame', 'torch', 'gymnasium', 'PIL', 'mario_play.game.engine',\n"
        "          'mario_play.game.level', 'mario_play.game.entities']\n"
        "loaded = [m for m in banned if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------- blit


def test_blit_copies_opaque_pixels_and_keeps_dst_under_transparent_ones():
    dst = _noise(8, 9)
    before = dst.copy()
    sprite = _test_sprite()
    assert blit(dst, sprite, 2, 3) is None
    region, opaque = dst[3:6, 2:6], _opaque(sprite)
    assert np.array_equal(region[opaque], sprite[opaque][:, :3])
    assert np.array_equal(region[~opaque], before[3:6, 2:6][~opaque])
    outside = np.ones((8, 9), dtype=bool)
    outside[3:6, 2:6] = False
    assert np.array_equal(dst[outside], before[outside])


@pytest.mark.parametrize("flip", [False, True])
def test_blit_matches_the_reference_at_every_offset_around_the_frame(flip):
    sprite = _test_sprite()
    base = _noise(6, 7, seed=1)
    for y in range(-5, 9):
        for x in range(-6, 10):
            dst = base.copy()
            blit(dst, sprite, x, y, flip=flip)
            assert np.array_equal(dst, _reference_blit(base, sprite, x, y, flip)), (x, y)


@pytest.mark.parametrize(
    ("x", "y"),
    [(-16, 4), (20, 4), (4, -16), (4, 12), (-100, -100), (10_000, 10_000), (-17, 30)],
)
def test_blit_fully_off_screen_is_a_no_op(x, y):
    dst = _noise(12, 20, seed=2)
    before = dst.copy()
    blit(dst, get_sprite("walker_1"), x, y)
    blit(dst, get_sprite("walker_1"), x, y, flip=True)
    assert np.array_equal(dst, before)


@pytest.mark.parametrize(
    ("x", "y", "rows", "cols"),
    [
        (-10, 4, slice(4, 20), slice(0, 6)),  # left edge
        (30, 4, slice(4, 20), slice(30, 40)),  # right edge
        (12, -9, slice(0, 7), slice(12, 28)),  # top edge
        (12, 22, slice(22, 30), slice(12, 28)),  # bottom edge
        (-3, -5, slice(0, 11), slice(0, 13)),  # corner
    ],
)
def test_blit_clips_real_sprites_at_each_edge(x, y, rows, cols):
    base = _noise(30, 40, seed=3)
    sprite = get_sprite("brick")
    dst = base.copy()
    blit(dst, sprite, x, y)
    assert np.array_equal(dst, _reference_blit(base, sprite, x, y))
    changed = np.zeros(base.shape[:2], dtype=bool)
    changed[rows, cols] = True
    assert np.array_equal(dst[~changed], base[~changed])
    assert not np.array_equal(dst[changed], base[changed])


def test_blit_never_writes_outside_a_dst_view():
    guard = np.full((40, 48, 3), 7, dtype=np.uint8)
    dst = guard[12:28, 16:32]  # a 16x16 window inside a larger buffer
    sprite = get_sprite("player_big_stand")
    for y in range(-34, 20, 3):
        for x in range(-18, 20, 3):
            blit(dst, sprite, x, y, flip=bool((x + y) % 2))
    outside = np.ones(guard.shape[:2], dtype=bool)
    outside[12:28, 16:32] = False
    assert (guard[outside] == 7).all()
    assert (dst != 7).any()


@pytest.mark.parametrize("name", ["player_small_walk1", "turtle_1", "cloud"])
def test_flip_mirrors_cached_sprites(name):
    sprite = get_sprite(name)
    mirrored = np.ascontiguousarray(sprite[:, ::-1])
    base = _noise(40, 80, seed=4)
    flipped, manual = base.copy(), base.copy()
    blit(flipped, sprite, 5, 3, flip=True)
    blit(manual, mirrored, 5, 3)
    assert np.array_equal(flipped, manual)
    assert np.array_equal(flipped, _reference_blit(base, sprite, 5, 3, flip=True))
    unflipped = base.copy()
    blit(unflipped, sprite, 5, 3)
    assert not np.array_equal(flipped, unflipped)


def test_blit_sees_in_place_edits_of_caller_owned_sprites():
    sprite = _test_sprite()
    dst = np.zeros((3, 4, 3), dtype=np.uint8)
    blit(dst, sprite, 0, 0)
    sprite[0, 1] = (200, 201, 202, 255)
    sprite[2, 3, 3] = 0
    dst2 = np.zeros((3, 4, 3), dtype=np.uint8)
    blit(dst2, sprite, 0, 0)
    assert dst2[0, 1].tolist() == [200, 201, 202]
    assert dst2[2, 3].tolist() == [0, 0, 0]


def test_blit_accepts_numpy_integer_and_float_coordinates():
    base = _noise(20, 20, seed=5)
    expected = _reference_blit(base, get_sprite("coin"), 3, -2)
    for x, y in [(np.int64(3), np.int32(-2)), (3.0, -2.0), (3.9, -1.1)]:
        dst = base.copy()
        blit(dst, get_sprite("coin"), x, y)
        assert np.array_equal(dst, expected), (x, y)  # floats floor, like world -> screen


def test_blit_on_a_full_frame_leaves_shape_and_dtype_alone():
    frame = np.empty((240, 256, 3), dtype=np.uint8)
    frame[:] = SKY_COLOR
    for i, name in enumerate(sprite_names()):
        blit(frame, get_sprite(name), (i * 37) % 300 - 20, (i * 53) % 280 - 20, flip=bool(i % 2))
    assert frame.shape == (240, 256, 3) and frame.dtype == np.uint8
    assert (frame != np.array(SKY_COLOR, dtype=np.uint8)).any()


@pytest.mark.parametrize(
    ("dst", "sprite"),
    [
        (np.zeros((8, 8), dtype=np.uint8), np.zeros((2, 2, 4), dtype=np.uint8)),
        (np.zeros((8, 8, 3), dtype=np.float32), np.zeros((2, 2, 4), dtype=np.uint8)),
        (np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)),
        (np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)),
    ],
)
def test_blit_rejects_malformed_arrays(dst, sprite):
    with pytest.raises(ValueError):
        blit(dst, sprite, 0, 0)
