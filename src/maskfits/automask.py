"""Threshold-based automatic source masking: background level + iterative
sigma-clipping to isolate background-only pixels, then a kappa-sigma cut
above that background flags source pixels for masking.
"""

from typing import Optional

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, label
from scipy.optimize import curve_fit

# How the (constant) background level is estimated from the background
# pixels left after sigma clipping - see background_stats.
BG_METHODS = ["median", "mode"]
MODE_BINS = 100


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


def background_stats(data: np.ndarray, kept: np.ndarray, method: str = "median") -> tuple[float, float]:
    """Constant background level and its error (the standard deviation)
    over the surviving `kept` pixels. The level is their median, or their
    mode: the center of the most populated bin of a MODE_BINS histogram
    over their range."""
    if method not in BG_METHODS:
        raise ValueError(f"unknown background method: {method!r}")
    vals = data[kept]
    if vals.size == 0:
        return 0.0, 0.0
    if method == "mode":
        lo, hi = float(vals.min()), float(vals.max())
        if hi > lo:
            counts, edges = np.histogram(vals, bins=MODE_BINS, range=(lo, hi))
            peak = int(np.argmax(counts))
            bg = float((edges[peak] + edges[peak + 1]) / 2)
        else:
            bg = lo
    else:
        bg = float(np.median(vals))
    return bg, float(vals.std())


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


def _bridged_groups(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Labels the connected groups of `mask` after closing small gaps (see
    filter_by_group_size for why), 8-connected. OR'd with the mask itself:
    closing can erode pixels at the image border, which would otherwise
    belong to no group at all."""
    bridged = mask | binary_closing(mask, structure=_disk_footprint(GROUP_BRIDGE_RADIUS))
    return label(bridged, structure=np.ones((3, 3), dtype=bool))


def group_elongation(labeled: np.ndarray, num: int) -> np.ndarray:
    """Per-group elongation, indexed by label (index 0, the background, is
    unused): sqrt of the ratio of the largest to the smallest variance of
    the group's pixel positions - its length-to-width ratio. About 1 for a
    round blob, large for a thin streak. Each pixel counts as a filled unit
    square (+1/12 variance on both axes), so single pixels and short runs
    stay finite. Only a few sums over the labeled pixels - no fitting."""
    ys, xs = np.nonzero(labeled)
    groups = labeled[ys, xs]
    n = np.maximum(np.bincount(groups, minlength=num + 1), 1)

    def mean(values: np.ndarray) -> np.ndarray:
        return np.bincount(groups, weights=values, minlength=num + 1) / n

    mx, my = mean(xs.astype(float)), mean(ys.astype(float))
    # Centered per group before squaring, to keep precision on large images.
    dx, dy = xs - mx[groups], ys - my[groups]
    cxx, cyy, cxy = mean(dx * dx) + 1 / 12, mean(dy * dy) + 1 / 12, mean(dx * dy)
    half_trace = (cxx + cyy) / 2
    spread = np.sqrt(np.maximum(half_trace**2 - (cxx * cyy - cxy**2), 0.0))
    return np.sqrt((half_trace + spread) / (half_trace - spread))


# Elongation at or above which a group counts as a line (satellite trail,
# diffraction spike, plane), below which as round (star, compact galaxy).
LINE_ELONGATION = 3.0
SHAPE_FILTERS = ("lines", "circles")


def filter_by_shape(mask: np.ndarray, keep: str, threshold: float = LINE_ELONGATION) -> np.ndarray:
    """Keep only the line-like groups (keep="lines": elongation >=
    threshold) or only the round ones (keep="circles": below it) - see
    group_elongation. Groups are measured like filter_by_group_size's
    (small gaps closed first), so a streak that noise broke into specks
    still counts as one line. A star with crossing diffraction spikes is
    one symmetric group and reads as round."""
    if keep not in SHAPE_FILTERS:
        raise ValueError(f"unknown shape filter: {keep!r}")
    if not mask.any():
        return mask
    labeled, num = _bridged_groups(mask)
    elongation = group_elongation(labeled, num)
    is_line = elongation >= threshold
    keep_group = is_line if keep == "lines" else ~is_line
    keep_group[0] = False
    return mask & keep_group[labeled]


def filter_by_group_size(mask: np.ndarray, max_size: int = 0, min_size: int = 0) -> np.ndarray:
    """Keep only connected groups of flagged pixels whose size is within
    [min_size, max_size] - e.g. drop extended detections (satellite trails,
    big galaxies, artifacts) above max_size and single hot pixels / noise
    specks below min_size. max_size <= 0 means no upper limit, min_size <= 1
    no lower limit. Applied to the RAW threshold flags, before expand_mask
    pads whatever survives - padding first would inflate every group's size
    and defeat the point of this filter.

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
    if (max_size <= 0 and min_size <= 1) or not mask.any():
        return mask
    labeled, num = _bridged_groups(mask)
    if num == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    drop = np.zeros(sizes.shape, dtype=bool)
    if max_size > 0:
        drop |= sizes > max_size
    if min_size > 1:
        drop |= sizes < min_size
    drop[0] = False  # the background, not a group
    if not drop.any():
        return mask
    return mask & ~drop[labeled]


def expand_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    """Pad each flagged region out by `radius` pixels (a disk-shaped dilation)
    so a thin/eroded detection still covers a source's faint wings. radius <=
    0 leaves the mask unchanged."""
    r = int(round(radius))
    if r <= 0:
        return mask
    return binary_dilation(mask, structure=_disk_footprint(r))
