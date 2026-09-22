"""Switching extensions (via the Up/Down hotkeys or the extension combo -
both go through MaskFitsApp.switch_extension) carries the painted mask over
when the new extension has the same shape as the current one, letting the
user mask against whichever representation (science frame, weight map, ...)
of the same pixel grid shows a feature most clearly. A different shape has
nothing to line the old mask's pixels up with, so it's discarded completely,
the same as opening a new file."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _hdu(cls, shape, tag, rng):
    h = cls(rng.normal(100.0, 5.0, size=shape).astype(np.float32))
    h.header["OBJECT"] = tag
    return h


def _write_same_shape_fits(path):
    rng = np.random.default_rng(0)
    fits.HDUList([
        _hdu(fits.PrimaryHDU, (32, 32), "EXT0", rng),
        _hdu(fits.ImageHDU, (32, 32), "EXT1", rng),
    ]).writeto(path, overwrite=True)


def _write_different_shape_fits(path):
    rng = np.random.default_rng(0)
    fits.HDUList([
        _hdu(fits.PrimaryHDU, (32, 32), "EXT0", rng),
        _hdu(fits.ImageHDU, (20, 40), "EXT1", rng),
    ]).writeto(path, overwrite=True)


def _write_transposed_shape_fits(path):
    """Both extensions end up with the same *working* (post-rotation)
    shape, since load_fits_image transposes a portrait array so the longer
    side is horizontal - but their raw NAXIS1/NAXIS2 are swapped, i.e. not
    actually the same pixel grid. A shape check that only compares the final
    working shape (not also the rotated flag) would wrongly treat these as
    compatible."""
    rng = np.random.default_rng(0)
    fits.HDUList([
        _hdu(fits.PrimaryHDU, (80, 40), "EXT0", rng),  # portrait -> rotated, working shape (40, 80)
        _hdu(fits.ImageHDU, (40, 80), "EXT1", rng),    # landscape -> not rotated, working shape (40, 80)
    ]).writeto(path, overwrite=True)


def test_mask_carries_over_between_same_shape_extensions(qapp, tmp_path):
    path = tmp_path / "same.fits"
    _write_same_shape_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.image.mask[5:10, 5:10] = True
    painted = win.image.mask.copy()

    win.switch_extension(1)

    assert win.entry.ext == 1
    assert win.image.mask.shape == painted.shape
    assert np.array_equal(win.image.mask, painted)


def test_mask_is_discarded_between_different_shape_extensions(qapp, tmp_path):
    path = tmp_path / "diff.fits"
    _write_different_shape_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.image.mask[5:10, 5:10] = True

    win.switch_extension(1)

    assert win.entry.ext == 1
    assert win.image.mask.shape == (20, 40)
    assert not win.image.mask.any()


def test_mask_is_discarded_when_only_the_working_shape_matches(qapp, tmp_path):
    """A same-final-shape-but-transposed pair (see
    _write_transposed_shape_fits) must NOT be treated as compatible."""
    path = tmp_path / "transposed.fits"
    _write_transposed_shape_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    assert win.image.data.shape == (40, 80)
    win.image.mask[0:5, 0:5] = True

    win.switch_extension(1)

    assert win.entry.ext == 1
    assert win.image.data.shape == (40, 80)  # same working shape...
    assert not win.image.mask.any()  # ...but still discarded (raw shapes differ)


def test_mask_carries_over_via_the_up_down_hotkeys_too(qapp, tmp_path):
    path = tmp_path / "same.fits"
    _write_same_shape_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.image.mask[0:3, 0:3] = True
    painted = win.image.mask.copy()

    win.next_extension()

    assert win.entry.ext == 1
    assert np.array_equal(win.image.mask, painted)

    win.prev_extension()

    assert win.entry.ext == 0
    assert np.array_equal(win.image.mask, painted)


def test_carried_over_mask_is_independent_of_the_old_entry_state(qapp, tmp_path):
    """switch_extension() must copy the mask, not just hand over the same
    array reference - painting on the new extension shouldn't retroactively
    change what was captured from the old one (not that anything still
    references it, but this pins the copy() as intentional, not incidental)."""
    path = tmp_path / "same.fits"
    _write_same_shape_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.image.mask[5:10, 5:10] = True
    old_mask_object = win.image.mask

    win.switch_extension(1)
    win.image.mask[20:25, 20:25] = True  # paint more on the new extension

    assert not np.array_equal(old_mask_object, win.image.mask)


def test_mask_is_dropped_for_an_unloaded_entry(qapp, tmp_path):
    """switch_extension() on an entry that never successfully loaded an
    image (entry.image is None) must not blow up trying to read its mask."""
    win = MaskFitsApp([], settings=Settings())
    win.show()
    qapp.processEvents()
    assert win.entry.image is None
    # No extensions to switch to - just confirms no crash/AttributeError.
    win.switch_extension(1)
