"""Loading FITS images and computing display stretch levels."""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy import units as u
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
    # Set only when this extension's data is a genuine 3D cube (e.g. an IFU
    # datacube) - the full (n_slices, ny, nx) array, already transposed
    # consistently with `data`/`rotated` if the frame is portrait. `data` is
    # always `cube[slice_index]`; switching slices (see gui.switch_slice)
    # just re-indexes this in memory, no reload from disk.
    cube: Optional[np.ndarray] = None
    slice_index: int = 0

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
        is_cube = data.ndim == 3
        if data.ndim > 3:
            # Higher-dimensional data (rare) - fall back to the first 2D
            # slice, same as before; no cube/slice UI for this case.
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
    cube: Optional[np.ndarray] = None
    if is_cube:
        _n_slices, ny, nx = data.shape
        rotated = ny > nx
        if rotated:
            data = np.ascontiguousarray(data.transpose(0, 2, 1))
        cube = data
        data = cube[0]
    else:
        ny, nx = data.shape
        rotated = ny > nx
        if rotated:
            data = np.ascontiguousarray(data.T)

    return FitsImage(path=path, data=data, header=header, wcs=wcs, rotated=rotated, cube=cube)


def safe_span(lowcut: float, highcut: float) -> float:
    """A minimum-clamped highcut-lowcut span, for use as a color-
    normalization divisor or display-margin size. Floored relative to the
    cut values' own magnitude rather than a fixed absolute constant, so it
    stays a true no-op for ordinary "counts"-scale data while still
    preventing a divide-by-zero for physical-flux-unit data (e.g. an IFU
    cube in erg/s/cm2/Angstrom), whose meaningful span can be many orders
    of magnitude smaller - a fixed ~1e-12-style floor would swallow a
    ~1e-20 span whole and flatten every pixel to the same normalized
    value (rendering solid black), the bug this fixes. Falls back to a
    tiny absolute floor only in the fully degenerate case (both cuts
    exactly 0), matching the previous fixed-floor behavior there."""
    span = highcut - lowcut
    magnitude = max(abs(lowcut), abs(highcut))
    floor = magnitude * 1e-9 if magnitude > 0 else 1e-12
    return span if span > floor else floor


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


# CTYPE3 prefix -> what the cube's third axis measures.
_SPECTRAL_QUANTITIES = {"WAVE": "wavelength", "AWAV": "wavelength", "FREQ": "frequency",
                        "ENER": "energy", "WAVN": "wavenumber", "VRAD": "velocity",
                        "VOPT": "velocity", "VELO": "velocity", "ZOPT": "redshift"}


@dataclass(frozen=True)
class SpectralAxis:
    """Where each slice of a cube sits along its third axis.

    `values` has one entry per slice; `quantity` names the axis
    ("wavelength", "frequency", ... or "slice" when the header doesn't
    describe it); `labels` are per-slice band names (e.g. "u", "g") when
    the header gives them, else empty strings."""

    values: np.ndarray
    quantity: str
    unit: str
    labels: tuple[str, ...]


def cube_spectral_axis(header: "fits.Header", n_slices: int) -> SpectralAxis:
    """The cube's spectral axis from its header. Checked in order:
    - Per-slice WAVE<i>/BAND<i> keywords (0-based i) - for a few discrete,
      unevenly spaced bands (e.g. SDSS ugriz). Unit from CUNIT3, else
      Angstrom. Where only some slices have one, those override the WCS
      axis below for just those slices.
    - The cube's third WCS axis (CTYPE3/CRVAL3/CRPIX3/CDELT3 or CD3_3/PC3_3,
      log axes included) via astropy. WCSLIB hands recognized spectral axes
      back in SI units (Angstrom -> m), so the values are converted back to
      the header's own CUNIT3. The quantity comes from CTYPE3,
      case-insensitively (real data uses e.g. "Wave", which astropy itself
      doesn't recognize as spectral).
    Otherwise just the 1-based slice numbers."""
    labels = tuple(str(header.get(f"BAND{i}", "")).strip() for i in range(n_slices))
    waves = [_header_float(header, (f"WAVE{i}",)) for i in range(n_slices)]
    if n_slices and all(w is not None for w in waves):
        unit = str(header.get("CUNIT3", "Angstrom")).strip()
        return SpectralAxis(np.array(waves, dtype=float), "wavelength", unit, labels)
    axis = _wcs_spectral_axis(header, n_slices, labels)
    if any(w is not None for w in waves):
        values = axis.values.copy()
        for i, wave in enumerate(waves):
            if wave is not None:
                values[i] = wave
        unit = str(header.get("CUNIT3", "Angstrom")).strip()
        return SpectralAxis(values, "wavelength", unit, labels)
    return axis


def _wcs_spectral_axis(header: "fits.Header", n_slices: int, labels: tuple[str, ...]) -> SpectralAxis:
    slices = SpectralAxis(np.arange(1, n_slices + 1, dtype=float), "slice", "", labels)
    if "CRVAL3" not in header and "CTYPE3" not in header:
        return slices
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            axis = WCS(header).sub([3])
            values = np.asarray(axis.pixel_to_world_values(np.arange(n_slices)), dtype=float)
        wcs_unit = axis.wcs.cunit[0]
    except Exception:
        return slices
    if values.shape != (n_slices,) or not np.all(np.isfinite(values)):
        return slices

    unit = str(header.get("CUNIT3", "")).strip()
    if unit:
        try:
            values = (values * wcs_unit).to_value(u.Unit(unit))
        except (ValueError, TypeError, u.UnitsError):
            unit = wcs_unit.to_string()
    elif wcs_unit != u.dimensionless_unscaled:
        unit = wcs_unit.to_string()
    ctype = str(header.get("CTYPE3", "")).strip().upper()
    quantity = _SPECTRAL_QUANTITIES.get(ctype[:4], "wavelength" if unit else "slice")
    return SpectralAxis(values, quantity, unit, labels)


FLUX_UNIT_KEYWORDS = ("BUNIT", "FLUXUNIT")


def header_flux_unit(header: "fits.Header") -> str:
    """The data's flux unit as the header states it (BUNIT, else
    FLUXUNIT), or "" when it doesn't say - never guessed."""
    for key in FLUX_UNIT_KEYWORDS:
        value = str(header.get(key, "")).strip()
        if value:
            return value
    return ""


# Header keywords that commonly hold the photometric zero point (mag) and
# the pixel scale (arcsec/pixel), checked in this order.
ZP_KEYWORDS = ("ZP", "ZEROPT", "ZEROPOINT", "ZPT", "MAGZP", "MAGZPT", "MAGZERO", "PHOTZP")
PXSCALE_KEYWORDS = ("PXSCALE", "PIXSCALE", "PIXSCAL", "PIXSCAL1", "PIXELSCL", "PIXELSCALE", "SECPIX", "SECPIX1")


def _header_float(header: "fits.Header", keywords: tuple[str, ...]) -> Optional[float]:
    for key in keywords:
        try:
            value = float(header[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def header_zeropoint(header: "fits.Header") -> Optional[float]:
    """The photometric zero point (mag) from the first of ZP_KEYWORDS
    present in the header, or None."""
    return _header_float(header, ZP_KEYWORDS)


def header_pixel_scale(header: "fits.Header", wcs: Optional[WCS]) -> Optional[float]:
    """Pixel scale in arcsec/pixel: the first of PXSCALE_KEYWORDS present in
    the header (assumed already in arcsec/pixel), else derived from a
    celestial WCS if there is one, else None."""
    value = _header_float(header, PXSCALE_KEYWORDS)
    if value is not None and value != 0:
        return abs(value)
    if wcs is None or not wcs.has_celestial:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        scale = abs(float(np.mean(proj_plane_pixel_scales(wcs.celestial)))) * 3600.0
    except Exception:
        return None
    return scale if np.isfinite(scale) and scale > 0 else None


def _pixel_scale_arcsec(header: "fits.Header", wcs: Optional[WCS]) -> float:
    """header_pixel_scale, or the IsoPy default (1.0 "/px) if unknown."""
    scale = header_pixel_scale(header, wcs)
    return scale if scale is not None else ISOPY_DEFAULT_PXSCALE


def sb_calibration(header: "fits.Header", wcs: Optional[WCS]) -> Optional[tuple[float, float]]:
    """(zero point, pixel scale in arcsec/pixel) if both are known for this
    image - see header_zeropoint/header_pixel_scale - else None. No
    defaults: a surface brightness is only shown when it's real."""
    zp = header_zeropoint(header)
    if zp is None:
        return None
    pxscale = header_pixel_scale(header, wcs)
    if pxscale is None:
        return None
    return zp, pxscale


def surface_brightness(flux: float, zp: float, pxscale: float) -> Optional[float]:
    """Surface brightness in mag/arcsec^2 of a pixel value, or None where it
    is undefined (zero, negative or non-finite flux). Binning averages
    pixels and smoothing preserves flux per pixel, so the original pixel
    scale applies to the displayed data either way."""
    if not np.isfinite(flux) or flux <= 0:
        return None
    return float(zp - 2.5 * np.log10(flux / pxscale**2))


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

    ZP comes from the header (see header_zeropoint; default 30.0 if
    absent). Pixel scale comes from the header if present, else from `wcs`
    in arcsec/pixel, else the default (1.0 "/px) - see header_pixel_scale.
    """
    zp = header_zeropoint(header)
    zp = ISOPY_DEFAULT_ZP if zp is None else zp
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
