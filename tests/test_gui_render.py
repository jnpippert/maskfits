"""Regression coverage for a real segfault (SIGSEGV, exit 139): render()
handed QImage the full-resolution crop's width/height/stride instead of the
actual (downsampled) rgb buffer's own - QImage has no way to cross-check
those against the real buffer size, so it read past the end of it, a native
out-of-bounds read rather than a catchable Python exception. Only shows up
once the displayed crop is actually downsampled (step_x/step_y > 1 in
render()), which needs an image too large to show 1:1 in the window - a
small image like tests/data/test1.fits never exercises that path.

Needs a Qt platform plugin to construct real widgets; QT_QPA_PLATFORM is
forced to "offscreen" (no real display required) before importing PySide6/
maskfits.gui, unless the environment already set one.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write_large_fits(path, size=2000):
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)
    fits.PrimaryHDU(data).writeto(path, overwrite=True)


def test_render_large_image_does_not_crash(qapp, tmp_path):
    """The exact reproduction: an image large enough that fit-to-window
    zoom is well below 1:1, so render() takes the downsampled (step > 1)
    path - this used to segfault the whole process instead of raising."""
    path = tmp_path / "large.fits"
    _write_large_fits(path)

    win = MaskFitsApp([str(path)])
    win.resize(800, 600)
    win.show()
    qapp.processEvents()

    assert win._base_pixmap is not None
    # If render() had passed the wrong (full-resolution) width/height/stride
    # to QImage again, this process would have already crashed above with a
    # native segfault - reaching this assertion at all is most of the test.
    pixmap_size = win._base_pixmap.size()
    assert pixmap_size.width() > 0
    assert pixmap_size.height() > 0


def test_render_pixmap_size_matches_disp_dimensions(qapp, tmp_path):
    """More precisely: the rendered pixmap's size should match disp_w/disp_h
    (the intended on-screen size after zoom), not crop_w/crop_h (the
    full-resolution source span that caused the out-of-bounds read)."""
    path = tmp_path / "large2.fits"
    _write_large_fits(path, size=3000)

    win = MaskFitsApp([str(path)])
    win.resize(700, 500)
    win.show()
    qapp.processEvents()

    ny, nx = win.image.data.shape
    ix0, iy0 = win.canvas_to_img(0, 0)
    ix1, iy1 = win.canvas_to_img(win.canvas_w, win.canvas_h)
    x0 = max(int(np.floor(min(ix0, ix1))), 0)
    x1 = min(int(np.ceil(max(ix0, ix1))), nx)
    y0 = max(int(np.floor(min(iy0, iy1))), 0)
    y1 = min(int(np.ceil(max(iy0, iy1))), ny)
    crop_h, crop_w = y1 - y0, x1 - x0
    disp_w = max(int(round(crop_w * win.zoom)), 1)
    disp_h = max(int(round(crop_h * win.zoom)), 1)

    pixmap_size = win._base_pixmap.size()
    assert pixmap_size.width() == disp_w
    assert pixmap_size.height() == disp_h
    # crop_w/crop_h (the buggy width/height) differ from disp_w/disp_h
    # whenever the image needed downsampling at all - true here since a
    # 3000px image can't fit a 700px-wide canvas at 1:1.
    assert (crop_w, crop_h) != (disp_w, disp_h)
