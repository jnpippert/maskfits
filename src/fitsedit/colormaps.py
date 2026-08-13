"""Display color palettes: 256-entry RGB lookup tables built from a few control points."""

import numpy as np

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
    "Hot": [
        (0.00, 0.0, 0.0, 0.0),
        (0.33, 1.0, 0.0, 0.0),
        (0.66, 1.0, 1.0, 0.0),
        (1.00, 1.0, 1.0, 1.0),
    ],
    "Cool": [
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
    ],
}

COLORMAP_NAMES = list(_STOPS.keys())


def _build_lut(stops: list[tuple[float, float, float, float]]) -> np.ndarray:
    ts = [s[0] for s in stops]
    x = np.linspace(0.0, 1.0, 256)
    r = np.interp(x, ts, [s[1] for s in stops])
    g = np.interp(x, ts, [s[2] for s in stops])
    b = np.interp(x, ts, [s[3] for s in stops])
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255).astype(np.uint8)


COLORMAP_LUTS: dict[str, np.ndarray] = {name: _build_lut(stops) for name, stops in _STOPS.items()}
