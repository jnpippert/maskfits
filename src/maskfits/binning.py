import numpy as np


def binned_extent(shape: tuple, bin_factor: int) -> tuple:
    """The largest (rows, cols) of `shape` that divide evenly by bin_factor -
    the region binning actually covers. Any remainder rows/columns past this
    are left untouched by binning."""
    m, n = shape
    return m - (m % bin_factor), n - (n % bin_factor)


def bin_func(data: np.ndarray, bin_factor: int) -> np.ndarray:
    """NxN block-mean downsampling (bin_factor=3 -> 3x3 pixel binning): each
    output pixel is the average of its source block. Trims any remainder
    rows/columns that don't divide evenly by bin_factor. A block with any NaN
    pixel becomes NaN - a single bad source pixel taints the whole binned one.
    """
    if bin_factor <= 1:
        return data
    m, n = data.shape
    mcut, ncut = binned_extent((m, n), bin_factor)
    cropped = data[0:mcut, 0:ncut]
    reshaped = cropped.reshape(mcut // bin_factor, bin_factor, ncut // bin_factor, bin_factor)
    # Plain mean, no NaN masking: a single NaN in the source block should make
    # the whole binned pixel NaN, and unmasked numpy math already does that.
    return reshaped.mean(axis=(1, 3))


def bin_mask(mask: np.ndarray, bin_factor: int) -> np.ndarray:
    """Downsample a boolean mask: a binned pixel is masked if any pixel in its
    source block was masked."""
    if bin_factor <= 1:
        return mask.copy()
    m, n = mask.shape
    mcut, ncut = binned_extent((m, n), bin_factor)
    cropped = mask[0:mcut, 0:ncut]
    reshaped = cropped.reshape(mcut // bin_factor, bin_factor, ncut // bin_factor, bin_factor)
    return reshaped.any(axis=(1, 3))


def unbin_mask(binned_mask: np.ndarray, bin_factor: int, base: np.ndarray) -> np.ndarray:
    """Upsample a binned mask back to full resolution by repeating each binned
    pixel into its bin_factor x bin_factor source block, over a copy of `base`
    (the pre-binning full-res mask) - so any bottom/right remainder binning
    had to trim off keeps whatever mask it had before binning, and the binned
    region is fully replaced by the (possibly just-edited) binned mask rather
    than merged, since what's visible/edited at binned resolution is meant to
    be the whole story for that region.
    """
    bin_factor = max(int(bin_factor), 1)
    if bin_factor <= 1:
        return binned_mask.copy()
    m, n = base.shape
    mcut, ncut = binned_extent((m, n), bin_factor)
    result = base.copy()
    result[0:mcut, 0:ncut] = np.repeat(np.repeat(binned_mask, bin_factor, axis=0), bin_factor, axis=1)
    return result
