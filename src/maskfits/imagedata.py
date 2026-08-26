"""Loading FITS images and computing display stretch levels."""

import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, LinearStretch, LogStretch, ZScaleInterval
from astropy.wcs import WCS, FITSFixedWarning

# Display stretch functions (map cut-normalized [0, 1] values through a non-linear
# curve before colormapping), independent of the cut levels themselves. astropy's
# LogStretch default (a=1000) is tuned for raw high-dynamic-range data and badly
# washes out a zscale-normalized [0, 1] range (background ends up ~85% gray); a=2
# still compresses bright sources and lifts faint signal, without blowing out the
# background.
STRETCHES = {
    "linear": LinearStretch(),
    "log": LogStretch(a=2),
    "asinh": AsinhStretch(a=0.1),
}
STRETCH_NAMES = ["linear", "log", "asinh"]


@dataclass
class FitsImage:
    """data/mask are in "working" orientation: transposed once at load time if the
    original was portrait (taller than wide), so the longer side displays
    horizontally. header/wcs always describe the ORIGINAL (untransposed) file -
    rotated tells callers a transpose is needed to translate between the two
    (WCS lookups, and detransposing the mask back before writing it out).

    Everything else (rendering, coordinate math, ellipse angles, undo) just
    operates on data/mask directly with no rotation-aware math at all: the one
    transpose already happened here, once, instead of on every render/edit.
    """

    path: str
    data: np.ndarray
    header: "fits.Header"
    wcs: WCS | None = None
    rotated: bool = False
    mask: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mask is None:
            self.mask = np.zeros(self.data.shape, dtype=bool)


def load_fits_image(path: str) -> FitsImage:
    with fits.open(path) as hdul:
        hdu = None
        for candidate in hdul:
            if candidate.data is not None and candidate.data.ndim >= 2:
                hdu = candidate
                break
        if hdu is None:
            raise ValueError(f"No 2D image data found in {path}")

        data = np.asarray(hdu.data)
        if data.ndim > 2:
            data = data[tuple([0] * (data.ndim - 2))]
        header = hdu.header.copy()

        wcs: WCS | None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FITSFixedWarning)
                candidate_wcs = WCS(header)
            wcs = candidate_wcs if candidate_wcs.has_celestial else None
        except Exception:
            wcs = None

    data = data.astype(np.float64)
    ny, nx = data.shape
    rotated = ny > nx
    if rotated:
        data = np.ascontiguousarray(data.T)

    return FitsImage(path=path, data=data, header=header, wcs=wcs, rotated=rotated)


def minmax_cuts(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def zscale_cuts(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = ZScaleInterval().get_limits(finite)
    lo, hi = float(lo), float(hi)
    return (lo, hi) if hi > lo else minmax_cuts(data)


# ds9-style percentile cut presets: e.g. "99.5%" shows the middle 99.5% of pixel
# values, symmetrically excluding (100-p)/2 from each tail.
PERCENTILE_PRESETS = [99.5, 99.0, 98.0, 95.0, 90.0]


def percentile_cuts(data: np.ndarray, percent: float) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    tail = (100.0 - percent) / 2.0
    lo, hi = np.percentile(finite, [tail, 100.0 - tail])
    lo, hi = float(lo), float(hi)
    return (lo, hi) if hi > lo else minmax_cuts(data)
