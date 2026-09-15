"""Display color palettes: 256-entry RGB lookup tables built from a few control points."""

import colorsys

import numpy as np

from maskfits.theme import ACCENT, BLUE, hex_to_rgb

# (t, r, g, b) control points, t and colors in [0, 1]. Viridis/Inferno are close
# perceptual approximations (a handful of anchor points interpolated), not exact
# reproductions of the matplotlib tables.
_STOPS: dict[str, list[tuple[float, float, float, float]]] = {
    "Grayscale": [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
    ],
    "Viridis": [
        (0.00, 0.267, 0.005, 0.329),
        (0.13, 0.283, 0.141, 0.458),
        (0.25, 0.254, 0.265, 0.530),
        (0.38, 0.207, 0.372, 0.553),
        (0.50, 0.164, 0.471, 0.558),
        (0.63, 0.128, 0.567, 0.551),
        (0.75, 0.135, 0.659, 0.518),
        (0.88, 0.267, 0.749, 0.441),
        (1.00, 0.993, 0.906, 0.144),
    ],
    "Inferno": [
        (0.00, 0.001, 0.000, 0.014),
        (0.20, 0.259, 0.038, 0.408),
        (0.40, 0.578, 0.148, 0.404),
        (0.60, 0.865, 0.317, 0.226),
        (0.80, 0.988, 0.645, 0.039),
        (1.00, 0.988, 0.998, 0.645),
    ],
}

# Jan-Niklas's MIDAS-style colormap, originally a standalone matplotlib
# ListedColormap + BoundaryNorm utility (midascmap.py). Copied here as plain
# data rather than importing that module, so the GUI doesn't pick up a hard
# matplotlib dependency just to reuse two literal lists.
#
# This is a discrete/stepped colormap by design (each color is a flat band
# between two boundaries, not a gradient) - that banding is what makes
# isophotes stand out, so it gets its own step-function LUT builder below
# instead of the smooth interpolation the other colormaps use. Verified to
# reproduce the original ListedColormap + BoundaryNorm output exactly.
_MIDAS_COLORS = [
    "#000000", "#1500e6", "#fd02ff", "#ffb500", "#fcff0e", "#00ffaa", "#58caff",
    "#ffadff", "#fffeff", "#fedbf9", "#ffaefd", "#d370fe", "#a19dfe", "#5bc9fe",
    "#00ffff", "#00ffad", "#00fd0c", "#99fe00", "#feff01", "#ffd800", "#ffc500",
    "#ffb600", "#ff6e01", "#ff4e00", "#fe0000", "#ff00b6", "#ff00fe", "#c500ee",
    "#9800e8", "#8500e5", "#1500e5", "#1301d3", "#0f00bb", "#000124", "#000000",
]
_MIDAS_BOUNDARIES_PCT = [
    0.0, 0.25157233, 0.62893082, 1.00628931, 1.3836478, 1.88679245, 2.26415094,
    2.64150943, 3.01886792, 3.39622642, 4.1509434, 5.40880503, 6.5408805,
    8.17610063, 9.3081761, 11.32075472, 13.20754717, 15.59748428, 17.61006289,
    20.75471698, 24.27672956, 24.65408805, 28.17610063, 32.0754717,
    33.33333333, 37.61006289, 43.52201258, 50.56603774, 57.98742138,
    59.24528302, 65.91194969, 76.10062893, 77.35849057, 87.54716981,
    89.05660377, 100.0,
]

MIDAS_NAME = "Midas Rainbow"

# IsoPy: pysophotes' own output.png diagnostic-plot colorbar - a
# LinearSegmentedColormap built from named matplotlib colors at
# data/header-dependent stop positions (see imagedata.isopy_cuts_and_stops).
# The color SEQUENCE below is fixed; only the stop positions vary per image
# (they depend on that image's ZP/pixel scale), so unlike every other
# colormap here this can't be precomputed once into a static COLORMAP_LUTS
# entry - gui.py calls build_isopy_lut() itself, per image, and caches the
# result instead.
ISOPY_NAME = "IsoPy"
ISOPY_COLOR_NAMES = ["black", "lightgray", "blue", "green", "yellow", "orange", "red", "purple"]
# The standard basic/CSS-named-color RGB values (0-1) matplotlib itself
# resolves these exact names to - matching pysophotes' own color list.
_ISOPY_COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "black": (0.0, 0.0, 0.0),
    "lightgray": (0.827, 0.827, 0.827),
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 0.502, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "orange": (1.0, 0.647, 0.0),
    "red": (1.0, 0.0, 0.0),
    "purple": (0.502, 0.0, 0.502),
}

COLORMAP_NAMES = [*_STOPS.keys(), MIDAS_NAME, ISOPY_NAME]


def _build_lut(stops: list[tuple[float, float, float, float]]) -> np.ndarray:
    ts = [s[0] for s in stops]
    x = np.linspace(0.0, 1.0, 256)
    r = np.interp(x, ts, [s[1] for s in stops])
    g = np.interp(x, ts, [s[2] for s in stops])
    b = np.interp(x, ts, [s[3] for s in stops])
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255).astype(np.uint8)


def _build_stepped_lut(colors_hex: list[str], boundaries_pct: list[float]) -> np.ndarray:
    """colors_hex[i] fills the band [boundaries_pct[i], boundaries_pct[i+1])."""
    colors_rgb = np.array([hex_to_rgb(c) for c in colors_hex], dtype=np.uint8)
    boundaries = np.array(boundaries_pct, dtype=np.float64)
    x_pct = np.linspace(0.0, 100.0, 256)
    idx = np.searchsorted(boundaries, x_pct, side="right") - 1
    idx = np.clip(idx, 0, len(colors_hex) - 1)
    return colors_rgb[idx]


COLORMAP_LUTS: dict[str, np.ndarray] = {name: _build_lut(stops) for name, stops in _STOPS.items()}
COLORMAP_LUTS[MIDAS_NAME] = _build_stepped_lut(_MIDAS_COLORS, _MIDAS_BOUNDARIES_PCT)
# Deliberately no COLORMAP_LUTS[ISOPY_NAME] entry - see ISOPY_NAME's comment above.


def build_isopy_lut(stops_t: list[float]) -> np.ndarray:
    """256-entry RGB LUT for the IsoPy colormap from its 8 fractional stop
    positions (see imagedata.isopy_cuts_and_stops), matching
    ISOPY_COLOR_NAMES in order. Sorted defensively by position before
    interpolating - normal ZP/pixel-scale values always produce already-
    increasing positions, but np.interp requires strictly increasing x
    regardless, and a pathological header shouldn't be able to crash this."""
    stops = [(t, *_ISOPY_COLOR_RGB[name]) for t, name in sorted(zip(stops_t, ISOPY_COLOR_NAMES))]
    return _build_lut(stops)


def _hue_rotated_rgb(rgb: tuple[int, int, int], turns: float) -> tuple[int, int, int]:
    """Rotate rgb's hue by `turns` full rotations (0.5 = 180 degrees), keeping
    saturation/value fixed."""
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + turns) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(round(c * 255)) for c in (r2, g2, b2))


def _complementary_rgb(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Hue-rotate 180 degrees (same saturation/value) - the classic color-wheel opposite."""
    return _hue_rotated_rgb(rgb, 0.5)


def mask_tint_for(name: str, lut: np.ndarray) -> tuple[int, int, int]:
    """Mask overlay color for a colormap, given its (possibly inverted) active LUT.

    Grayscale and Midas Rainbow are special-cased to a fixed color rather than
    the generic midpoint-complement: grayscale has no hue to complement, and
    plain white reads best against Midas Rainbow's own busy palette. Both stay
    fixed regardless of inversion. Everything else uses the color-wheel
    complement of the LUT's current midpoint, so an inverted colormap gets a
    correspondingly different (still-contrasting) tint automatically.
    """
    if name == "Grayscale":
        return hex_to_rgb(ACCENT)
    if name == MIDAS_NAME:
        return 255, 0, 0
    if name == ISOPY_NAME:
        # Cyan sits outside IsoPy's own black/gray/blue/green/yellow/
        # orange/red/purple sequence entirely, so it reads clearly against
        # any part of that palette.
        return 0, 255, 255
    mid = tuple(int(c) for c in lut[len(lut) // 2])
    return _complementary_rgb(mid)


def auto_mask_tint_for(name: str, lut: np.ndarray) -> tuple[int, int, int]:
    """Auto Mask preview overlay color for a colormap - deliberately a
    DIFFERENT hue from mask_tint_for's manual-mask tint for the same
    colormap/LUT, so a pending (not-yet-confirmed) auto-mask preview never
    reads as the same color as an already-applied manual mask.

    Grayscale and Midas Rainbow are special-cased for the same reason
    mask_tint_for special-cases them (no hue to complement / a busy palette
    that swallows most single hues), picked specifically to sit far from
    their manual-mask tint (crimson, and red respectively). Everything else
    uses a quarter-turn (90 degree) hue rotation off the LUT's midpoint,
    versus mask_tint_for's half-turn (180 degree) complement - a different
    hue, adapting the same way to colormap inversion.
    """
    if name == "Grayscale":
        return hex_to_rgb(BLUE)
    if name == MIDAS_NAME:
        return 255, 255, 255
    if name == ISOPY_NAME:
        # Magenta, for the same reason mask_tint_for picks cyan above - also
        # outside IsoPy's own palette, and clearly distinct from that cyan.
        return 255, 0, 255
    mid = tuple(int(c) for c in lut[len(lut) // 2])
    return _hue_rotated_rgb(mid, 0.25)


# Default (non-inverted) mask tint per colormap - kept for convenience/reference;
# the GUI computes this dynamically via mask_tint_for() against whichever LUT
# (inverted or not) is actually active.
MASK_TINT_COLORS: dict[str, tuple[int, int, int]] = {
    name: mask_tint_for(name, lut) for name, lut in COLORMAP_LUTS.items()
}
