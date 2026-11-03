"""Coverage for the surface brightness readouts: SB of the cut levels under
the histogram and SB next to the cursor's pixel value - shown only when the
image has a zero point and a pixel scale."""

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


def _window(tmp_path, header=None, value=100.0, name="img.fits"):
    path = tmp_path / name
    fits.PrimaryHDU(np.full((40, 40), value, dtype=np.float32), header=fits.Header(header or {})).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    return win


def _hover(win, ix=20.5, iy=20.5):
    win._on_motion(*win.img_to_canvas(ix, iy))


def test_hidden_without_calibration(qapp, tmp_path):
    win = _window(tmp_path)
    _hover(win)
    assert win.sb_cuts_label.isHidden()
    assert win.readout["value"].text() == "100"


def test_hidden_with_only_a_zero_point(qapp, tmp_path):
    win = _window(tmp_path, {"ZP": 30.0})
    assert win.sb_cuts_label.isHidden()


def test_pixel_value_shows_its_surface_brightness(qapp, tmp_path):
    win = _window(tmp_path, {"ZP": 30.0, "PXSCALE": 1.0})
    _hover(win)
    assert win.readout["value"].text() == "100  (SB 25.00)"


def test_cut_levels_show_their_surface_brightness(qapp, tmp_path):
    win = _window(tmp_path, {"MAGZP": 30.0, "PIXSCALE": 1.0})
    win.cuts_histogram.set_cuts(1.0, 100.0)
    assert not win.sb_cuts_label.isHidden()
    assert win.sb_cuts_label.text() == "SB lowcut: 30.00   highcut: 25.00  mag/arcsec²"


def test_undefined_surface_brightness_shows_minus_infinity(qapp, tmp_path):
    win = _window(tmp_path, {"ZP": 30.0, "PXSCALE": 1.0}, value=-3.0)
    _hover(win)
    assert win.readout["value"].text() == "-3  (SB -\u221e)"
    win.cuts_histogram.set_cuts(-5.0, 100.0)
    assert win.sb_cuts_label.text().startswith("SB lowcut: -\u221e ")


def test_cut_label_updates_live_while_dragging(qapp, tmp_path):
    """set_cuts is what each step of a histogram handle drag calls."""
    win = _window(tmp_path, {"ZP": 30.0, "PXSCALE": 1.0})
    win.cuts_histogram.set_cuts(1.0, 10.0)
    assert "highcut: 27.50" in win.sb_cuts_label.text()
    win.cuts_histogram.set_cuts(1.0, 100.0)
    assert "highcut: 25.00" in win.sb_cuts_label.text()


def test_label_hides_again_for_an_uncalibrated_next_image(qapp, tmp_path):
    calibrated = tmp_path / "a.fits"
    plain = tmp_path / "b.fits"
    fits.PrimaryHDU(np.ones((40, 40), dtype=np.float32), header=fits.Header({"ZP": 30.0, "PXSCALE": 1.0})).writeto(calibrated)
    fits.PrimaryHDU(np.ones((40, 40), dtype=np.float32)).writeto(plain)
    win = MaskFitsApp([str(calibrated), str(plain)], settings=Settings())
    win.show()
    assert not win.sb_cuts_label.isHidden()
    win.next_image()
    assert win.sb_cuts_label.isHidden()
