"""Loading FITS images and computing display stretch levels."""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, LinearStretch, LogStretch, ZScaleInterval
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import gaussian_filter

# Display stretch functions (map cut-normalized [0, 1] values through a non-linear
# curve before colormapping), independent of the cut levels themselves. astropy's
# LogStretch default (a=1000) is tuned for raw high-dynamic-range data and badly
# washes out a zscale-normalized [0, 1] range (background ends up ~85% gray); a=2
# still compresses bright sources and lifts faint signal, without blowing out the
# background.
STRETCHES = {
    "linear": LinearStretch(),
    "log": LogStretch(),
    "asinh": AsinhStretch(),
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


def list_image_extensions(path: str) -> list[tuple[int, str]]:
    """Every HDU index in `path` that holds usable 2D+ image data, paired
    with a display label ("<index> - <OBJECT>", or "<index> - NO HDU NAME"
    if that HDU has no OBJECT keyword) - for the extension picker next to
    the filename when a multi-extension FITS has more than one to choose
    from."""
    extensions: list[tuple[int, str]] = []
    with fits.open(path) as hdul:
        for i, hdu in enumerate(hdul):
            if hdu.data is None or hdu.data.ndim < 2:
                continue
            name = hdu.header.get("OBJECT")
            label = f"{i} - {name}" if name else f"{i} - NO HDU NAME"
            extensions.append((i, label))
    return extensions


def load_fits_image(path: str, ext: Optional[int] = None) -> FitsImage:
    """Load `path`'s image data. `ext` picks a specific HDU index; left as
    None, the first HDU with usable 2D+ image data is used (matching the
    historical single-extension behavior)."""
    with fits.open(path) as hdul:
        hdu = None
        if ext is not None:
            if ext < 0 or ext >= len(hdul) or hdul[ext].data is None or hdul[ext].data.ndim < 2:
                raise ValueError(f"Extension {ext} has no 2D image data in {path}")
            hdu = hdul[ext]
        else:
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


# IsoPy colormap: the fixed 22-27 mag/arcsec^2 surface-brightness display
# range pysophotes' own output.png diagnostic figure uses (its
# default_sb_cutlevels(), minus the K-band special case - not something a
# generic viewer like this one has a "filter" concept to key off of).
ISOPY_MIN_SB = 22.0
ISOPY_MAX_SB = 27.0
ISOPY_DEFAULT_ZP = 30.0
ISOPY_DEFAULT_PXSCALE = 1.0


def _pixel_scale_arcsec(header: "fits.Header", wcs: Optional[WCS]) -> float:
    """Pixel scale in arcsec/pixel: the header's own PXSCALE keyword if
    present (assumed already in arcsec/pixel), else derived from a celestial
    WCS if there is one, else the default (1.0 "/px)."""
    pxscale = header.get("PXSCALE")
    if pxscale is not None:
        try:
            return abs(float(pxscale))
        except (TypeError, ValueError):
            pass
    if wcs is None or not wcs.has_celestial:
        return ISOPY_DEFAULT_PXSCALE
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(wcs.celestial)
        return abs(float(np.mean(scales))) * 3600.0
    except Exception:
        return ISOPY_DEFAULT_PXSCALE


def isopy_cuts_and_stops(header: "fits.Header", wcs: Optional[WCS]) -> tuple[float, float, list[float]]:
    """Surface-brightness-based cut levels (vmin, vmax, in raw flux units)
    and the 8 fractional [0, 1] stop positions for the IsoPy colormap (see
    colormaps.build_isopy_lut and its ISOPY_COLOR_NAMES) - ported from
    pysophotes' own output.png diagnostic plot: same fixed 22-27
    mag/arcsec^2 display range, same per-magnitude tick levels between them.

    Positions are computed in plain linear flux, not through pysophotes' own
    log-like display stretch - maskfits already has its own separate lin/
    log/asinh scale-function choice (see gui.py's scale_function) that
    layers on top of whatever lowcut/highcut ends up active, isopy's
    included, so reproducing that second stretch here would be redundant
    rather than more faithful.

    ZP comes from the header's ZP keyword (default 30.0 if absent). Pixel
    scale comes from the header's own PXSCALE keyword if present, else from
    `wcs` in arcsec/pixel, else the default (1.0 "/px) - see
    _pixel_scale_arcsec.
    """
    zp = float(header.get("ZP", ISOPY_DEFAULT_ZP))
    pxscale = _pixel_scale_arcsec(header, wcs)

    minflux = 10 ** (-0.4 * (ISOPY_MAX_SB - zp)) * pxscale**2
    vmin = -minflux
    vmax = 10 ** (-0.4 * (ISOPY_MIN_SB - zp)) * pxscale**2
    span = max(vmax - vmin, 1e-30)

    cticklabels = np.arange(ISOPY_MIN_SB, ISOPY_MAX_SB + 1, 1.0)
    cticks = 10 ** (-0.4 * (cticklabels - zp)) * pxscale**2
    colorloc = (cticks - vmin) / span
    bg_frac = -vmin / span

    # Order matches colormaps.ISOPY_COLOR_NAMES: black, lightgray (at the
    # zero/background crossing), then blue/green/yellow/orange/red at the
    # faintest-to-brightest per-magnitude tick levels, purple at vmax.
    stops = [0.0, bg_frac, colorloc[5], colorloc[4], colorloc[3], colorloc[2], colorloc[1], 1.0]
    stops = [min(max(float(t), 0.0), 1.0) for t in stops]
    return float(vmin), float(vmax), stops


def gaussian_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """Blur the image with a Gaussian kernel, without letting non-finite pixels
    (NaN/inf) bleed into their neighbors - standard normalized-convolution trick:
    smooth the data with holes zeroed out alongside a 0/1 "how much real data is
    here" weight map, then divide one by the other so each output pixel is a
    weighted average of only the finite pixels near it."""
    if sigma <= 0:
        return data
    finite = np.isfinite(data)
    if finite.all():
        return gaussian_filter(data, sigma=sigma)
    filled = np.where(finite, data, 0.0)
    weights = gaussian_filter(finite.astype(np.float64), sigma=sigma)
    smoothed = gaussian_filter(filled, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = smoothed / weights
    result[weights == 0] = np.nan
    return result
