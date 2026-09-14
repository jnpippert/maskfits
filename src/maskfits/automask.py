"""Threshold-based automatic source masking: background level + iterative
sigma-clipping to isolate background-only pixels, then a kappa-sigma cut
above that background flags source pixels for masking.
"""

from typing import Optional

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, label
from scipy.optimize import curve_fit

ERROR_METHODS = ["sigma", "sem"]
BG_METHODS = ["constant"]


def valid_pixels(data: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels eligible for background/threshold work at all -
    excludes NaN and exact-zero (no-data/padding) pixels."""
    return np.isfinite(data) & (data != 0)


def sigma_clip_mask(data: np.ndarray, valid: np.ndarray, kappa: float, max_iter: int = 10) -> np.ndarray:
    """Iteratively sigma-clip `data` (restricted to `valid`) at `kappa` sigma
    around the surviving set's mean, converging (or stopping at max_iter) once
    a pass removes nothing more - this is the "clean up" clip that strips
    bright source pixels out so only background-like pixels remain.

    Returns a boolean mask of the surviving ("background candidate") pixels.
    """
    kept = valid.copy()
    if kappa <= 0:
        return kept
    for _ in range(max_iter):
        vals = data[kept]
        if vals.size == 0:
            break
        mean = float(vals.mean())
        std = float(vals.std())
        if std == 0:
            break
        lo, hi = mean - kappa * std, mean + kappa * std
        new_kept = valid & (data >= lo) & (data <= hi)
        if int(new_kept.sum()) == int(kept.sum()):
            kept = new_kept
            break
        kept = new_kept
    return kept


def background_stats(data: np.ndarray, kept: np.ndarray, error_method: str) -> tuple[float, float]:
    """Mean background level and its error (plain sigma, or the standard
    error of the mean = sigma / sqrt(n)) over the surviving `kept` pixels."""
    vals = data[kept]
    if vals.size == 0:
        return 0.0, 0.0
    bg = float(vals.mean())
    sigma = float(vals.std())
    if error_method == "sem":
        err = sigma / np.sqrt(vals.size) if vals.size > 0 else 0.0
    else:
        err = sigma
    return bg, err


def _gaussian(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_to_histogram(
    values: np.ndarray, bins: int, hist_range: tuple[float, float]
) -> Optional[tuple[float, float, float]]:
    """Least-squares gaussian fit (amplitude, mu, sigma) to a histogram of
    `values` binned over `hist_range` - for overlaying a fitted curve on the
    auto-mask background histogram.

    Falls back to the sample mean/std (a cruder, but still valid, gaussian
    estimate via moment-matching) if the least-squares fit fails to
    converge or lands on a degenerate result; returns None only when there's
    nothing at all to fit.
    """
    if values.size < 3:
        return None
    counts, edges = np.histogram(values, bins=bins, range=hist_range)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mean = float(values.mean())
    std = float(values.std()) or 1.0
    try:
        popt, _ = curve_fit(
            _gaussian, centers, counts.astype(np.float64),
            p0=[float(counts.max()), mean, std], maxfev=2000,
        )
        amplitude, mu, sigma = float(popt[0]), float(popt[1]), abs(float(popt[2]))
        if sigma <= 0 or not np.isfinite(amplitude) or not np.isfinite(mu):
            raise ValueError("degenerate gaussian fit")
    except Exception:
        amplitude, mu, sigma = float(counts.max()), mean, std
    return amplitude, mu, sigma


def auto_mask_preview(data: np.ndarray, valid: np.ndarray, bg: float, bg_err: float, kappa: float) -> np.ndarray:
    """Boolean mask of pixels to flag: `valid` pixels whose value exceeds
    bg + kappa * bg_err. `valid` decides what's even eligible to be flagged -
    pass valid_pixels(data) intersected with anything else that should be
    excluded (e.g. pixels already masked manually - see AutoMaskWindow)."""
    threshold = bg + kappa * bg_err
    return valid & (data > threshold)


def _disk_footprint(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return (xx ** 2 + yy ** 2) <= radius ** 2 + 1e-9


# How far apart (in pixels) two flagged specks can be and still count as
# "the same object" for group-size filtering - see filter_by_group_size.
# Not user-facing: it's just large enough to bridge the sub-threshold noise
# gaps a real, noisy extended source leaves in its own thresholding, while
# staying far too small to merge genuinely separate point sources.
GROUP_BRIDGE_RADIUS = 4


def filter_by_group_size(mask: np.ndarray, max_size: int) -> np.ndarray:
    """Drop any connected group of flagged pixels larger than `max_size` -
    keeps compact, point-like detections (small background sources) while
    dropping extended ones (satellite trails, big galaxies, artifacts).
    Applied to the RAW threshold flags, before expand_mask pads whatever
    survives - padding first would inflate every group's size and defeat the
    point of this filter. max_size <= 0 disables filtering (nothing dropped).

    Group membership is measured on a version of `mask` with small gaps
    closed (binary_closing, radius=GROUP_BRIDGE_RADIUS) first, NOT on the raw
    mask directly. A real extended source (a galaxy, a trail) practically
    never thresholds into one solid blob - sky noise pushes individual
    pixels within it below the cut too, fragmenting it into many small,
    disconnected specks that would each individually slip under any size
    limit on their own. Closing small gaps first recognizes that scattered
    cluster of specks as the one big object it actually is, without merging
    genuinely separate, well-spaced point sources (which stay distinct
    groups, since they're farther apart than the bridging radius).
    """
    if max_size <= 0 or not mask.any():
        return mask
    structure = np.ones((3, 3), dtype=bool)
    bridged = binary_closing(mask, structure=_disk_footprint(GROUP_BRIDGE_RADIUS))
    labeled, num = label(bridged, structure=structure)
    if num == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    too_big = np.nonzero(sizes > max_size)[0]
    if too_big.size == 0:
        return mask
    return mask & ~np.isin(labeled, too_big)


def expand_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    """Pad each flagged region out by `radius` pixels (a disk-shaped dilation)
    so a thin/eroded detection still covers a source's faint wings. radius <=
    0 leaves the mask unchanged."""
    r = int(round(radius))
    if r <= 0:
        return mask
    return binary_dilation(mask, structure=_disk_footprint(r))


def neural_network_mask(data: np.ndarray, model_path: str) -> np.ndarray:
    """Placeholder interface for a future neural-network-based source/
    satellite masker (e.g. a model trained to flag sources or satellite
    trails directly from pixel data). Not implemented yet - deliberately not
    wired to any inference framework here, so this project doesn't pick up a
    heavy ML dependency just for a stub. Fixing the interface now (a model
    path in, a boolean mask matching `data`'s shape out) means a real
    implementation can be dropped in later - by this project or another user
    - without any GUI-side changes: AutoMaskWindow already calls this and
    handles the NotImplementedError gracefully.
    """
    raise NotImplementedError(
        "Neural network masking is not implemented yet - this is a placeholder interface "
        "for a future model."
    )
