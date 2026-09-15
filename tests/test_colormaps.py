import numpy as np

from maskfits.colormaps import (
    COLORMAP_LUTS,
    COLORMAP_NAMES,
    ISOPY_COLOR_NAMES,
    ISOPY_NAME,
    _ISOPY_COLOR_RGB,
    auto_mask_tint_for,
    build_isopy_lut,
    mask_tint_for,
)


def test_isopy_name_registered_but_not_precomputed():
    assert ISOPY_NAME in COLORMAP_NAMES
    # unlike every other colormap, IsoPy's LUT is data/header-dependent and
    # built per-image (see gui.Entry.isopy_cuts_and_lut) - it should NOT
    # have a static entry here.
    assert ISOPY_NAME not in COLORMAP_LUTS


def test_build_isopy_lut_endpoints_match_black_and_purple():
    stops = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]
    lut = build_isopy_lut(stops)
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    assert tuple(int(c) for c in lut[0]) == (0, 0, 0)  # black
    expected_purple = tuple(round(c * 255) for c in _ISOPY_COLOR_RGB["purple"])
    assert tuple(int(c) for c in lut[-1]) == expected_purple


def test_build_isopy_lut_sorts_defensively():
    # Deliberately out-of-order positions - build_isopy_lut must not crash
    # or raise, even though this never happens for real ZP/pixel-scale
    # values (isopy_cuts_and_stops always returns increasing positions).
    stops = [0.5, 0.0, 0.9, 0.1, 0.2, 0.3, 0.4, 1.0]
    lut = build_isopy_lut(stops)
    assert lut.shape == (256, 3)


def test_isopy_tint_colors_absent_from_its_own_palette():
    manual = mask_tint_for(ISOPY_NAME, np.zeros((256, 3), dtype=np.uint8))
    auto = auto_mask_tint_for(ISOPY_NAME, np.zeros((256, 3), dtype=np.uint8))
    assert manual != auto
    isopy_palette_rgb = {tuple(round(c * 255) for c in rgb) for rgb in _ISOPY_COLOR_RGB.values()}
    assert manual not in isopy_palette_rgb
    assert auto not in isopy_palette_rgb


def test_isopy_color_names_and_rgb_table_match():
    assert set(ISOPY_COLOR_NAMES) == set(_ISOPY_COLOR_RGB.keys())
    assert len(ISOPY_COLOR_NAMES) == 8
