"""Coverage for the cube spectrum panel in the main window: shown under the
image (not the sidebar) only for cubes, following the cursor's pixel."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402

N, NY, NX = 6, 20, 30


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _cube_path(tmp_path, name="cube.fits"):
    data = np.arange(N * NY * NX, dtype=np.float32).reshape(N, NY, NX)
    header = fits.Header({"CTYPE3": "Wave", "CRVAL3": 3470.0, "CDELT3": 2.0, "CRPIX3": 1.0,
                          "CUNIT3": "Angstrom", "BUNIT": "erg/s/cm2/Angstrom"})
    path = tmp_path / name
    fits.PrimaryHDU(data, header=header).writeto(path)
    return str(path), data


def _image_path(tmp_path):
    path = tmp_path / "flat.fits"
    fits.PrimaryHDU(np.ones((NY, NX), dtype=np.float32)).writeto(path)
    return str(path)


def _open(paths, qapp):
    win = MaskFitsApp(paths, settings=Settings())
    win.resize(1200, 900)
    win.show()
    qapp.processEvents()
    return win


def _hover(win, ix, iy):
    win._on_motion(*win.img_to_canvas(ix + 0.5, iy + 0.5))


def test_hidden_for_a_plain_image(qapp, tmp_path):
    win = _open([_image_path(tmp_path)], qapp)
    assert win.spectrum_view.isHidden()


def test_shown_under_the_image_only_for_a_cube(qapp, tmp_path):
    path, _ = _cube_path(tmp_path)
    win = _open([path], qapp)
    view = win.spectrum_view
    assert not view.isHidden()
    # Same column as the image: starts right of the sidebar, as wide as the canvas.
    assert view.mapTo(win, view.rect().topLeft()).x() == win.canvas.mapTo(win, win.canvas.rect().topLeft()).x()
    assert view.width() == win.canvas.width()
    assert view.mapTo(win, view.rect().topLeft()).x() > win.sidebar_container.mapTo(win, win.sidebar_container.rect().topRight()).x()
    assert view.mapTo(win, view.rect().topLeft()).y() > win.canvas.mapTo(win, win.canvas.rect().bottomLeft()).y()


def test_axis_and_units_come_from_the_header(qapp, tmp_path):
    path, _ = _cube_path(tmp_path)
    win = _open([path], qapp)
    view = win.spectrum_view
    assert np.allclose(view.axis.values, 3470 + 2 * np.arange(N))
    assert view.x_title() == "λ [Å]"
    assert view.y_title() == "Flux [erg/s/cm2/Å]"


def test_spectrum_follows_the_cursor_pixel(qapp, tmp_path):
    path, data = _cube_path(tmp_path)
    win = _open([path], qapp)
    _hover(win, 7, 3)
    assert np.allclose(win.spectrum_view.spectrum, data[:, 3, 7])
    _hover(win, 12, 15)
    assert np.allclose(win.spectrum_view.spectrum, data[:, 15, 12])


def test_binned_spectrum_is_the_block_mean(qapp, tmp_path):
    path, data = _cube_path(tmp_path)
    win = _open([path], qapp)
    win.bin_factor_entry.setText("2")
    win._toggle_binning()
    _hover(win, 3, 4)  # binned pixel (3, 4) = original x 6-7, y 8-9
    assert np.allclose(win.spectrum_view.spectrum, data[:, 8:10, 6:8].mean(axis=(1, 2)))


def test_marker_follows_the_slice(qapp, tmp_path):
    path, _ = _cube_path(tmp_path)
    win = _open([path], qapp)
    assert win.spectrum_view.marker == 0
    win.switch_slice(4)
    assert win.spectrum_view.marker == 4


def test_hidden_again_after_moving_on_to_a_plain_image(qapp, tmp_path):
    path, _ = _cube_path(tmp_path)
    win = _open([path, _image_path(tmp_path)], qapp)
    assert not win.spectrum_view.isHidden()
    win.index = 1
    win.load_current(reset_view=True)
    assert win.spectrum_view.isHidden()


def test_cube_with_a_sky_wcs_shows_ra_and_dec(qapp, tmp_path):
    """Regression: a cube's WCS has a spectral third axis, so the RA/DEC
    readout (which passed only x and y) silently stayed empty."""
    data = np.ones((N, NY, NX), dtype=np.float32)
    header = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 49.7, "CRVAL2": 41.5,
                          "CRPIX1": 1.0, "CRPIX2": 1.0, "CDELT1": -1e-4, "CDELT2": 1e-4,
                          "CTYPE3": "Wave", "CRVAL3": 3470.0, "CDELT3": 2.0, "CRPIX3": 1.0})
    path = tmp_path / "wcs_cube.fits"
    fits.PrimaryHDU(data, header=header).writeto(path)
    win = _open([str(path)], qapp)
    _hover(win, 5, 5)
    assert win.readout["RA"].text().startswith("3:18:")
    assert win.readout["DEC"].text().startswith("+41:30:")
