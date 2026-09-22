"""Switching extensions (Up/Down or the extension combo, both routed through
MaskFitsApp.switch_extension) keeps the current zoom and pan - unlike
prev_image/next_image, which reset the view because they load an entirely
different file. An extension is still the same file, so the framing the user
was just looking at (e.g. to compare a science frame against its weight map)
should carry over regardless of whether the two extensions even share a
shape - this is about the view, not the mask (see
test_extension_mask_carryover.py for the shape-dependent mask behavior)."""

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


def _write_two_files(dir_path):
    rng = np.random.default_rng(0)
    a = dir_path / "a.fits"
    b = dir_path / "b.fits"
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(32, 32)).astype(np.float32)).writeto(a, overwrite=True)
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(32, 32)).astype(np.float32)).writeto(b, overwrite=True)
    return a, b


def _nudge_view(win):
    win.zoom_mult = 3.5
    win.view_cx, win.view_cy = 7.0, 11.0
    win._update_zoom_label()


def test_switch_extension_keeps_zoom_and_pan_for_same_shape(qapp, tmp_path):
    path = tmp_path / "same.fits"
    _write_same_shape_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    _nudge_view(win)
    win.switch_extension(1)

    assert win.entry.ext == 1
    assert win.zoom_mult == 3.5
    assert (win.view_cx, win.view_cy) == (7.0, 11.0)


def test_switch_extension_keeps_zoom_and_pan_even_for_different_shape(qapp, tmp_path):
    """Unchanged even when the mask itself gets discarded (shape mismatch)
    - the view and the mask are independent concerns here."""
    path = tmp_path / "diff.fits"
    _write_different_shape_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    _nudge_view(win)
    win.switch_extension(1)

    assert win.entry.ext == 1
    assert win.zoom_mult == 3.5
    assert (win.view_cx, win.view_cy) == (7.0, 11.0)


def test_extension_hotkeys_keep_zoom_and_pan(qapp, tmp_path):
    path = tmp_path / "same.fits"
    _write_same_shape_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    _nudge_view(win)
    win.next_extension()
    assert win.zoom_mult == 3.5
    assert (win.view_cx, win.view_cy) == (7.0, 11.0)

    win.prev_extension()
    assert win.zoom_mult == 3.5
    assert (win.view_cx, win.view_cy) == (7.0, 11.0)


def test_next_image_still_resets_the_view(qapp, tmp_path):
    """Switching to a different FILE (not just another extension of the same
    one) should still reset the view, same as before this change."""
    a, b = _write_two_files(tmp_path)
    win = MaskFitsApp([str(a), str(b)], settings=Settings())
    win.show()
    qapp.processEvents()

    _nudge_view(win)
    win.next_image()

    assert win.zoom_mult == 1.0


def test_prev_image_still_resets_the_view(qapp, tmp_path):
    a, b = _write_two_files(tmp_path)
    win = MaskFitsApp([str(a), str(b)], settings=Settings())
    win.show()
    qapp.processEvents()
    win.next_image()

    _nudge_view(win)
    win.prev_image()

    assert win.zoom_mult == 1.0
