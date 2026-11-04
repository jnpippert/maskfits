"""Regression: switching cube slice or FITS extension while binned showed
the new full-resolution data through a view still in binned coordinates
(zoomed in by the bin factor). Binning and smoothing now carry over to the
new slice/extension, with the same view."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402

SIZE = 80


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _data(seed):
    return np.random.default_rng(seed).normal(100.0, 5.0, (SIZE, SIZE)).astype(np.float32)


@pytest.fixture
def cube_path(tmp_path):
    path = tmp_path / "cube.fits"
    fits.PrimaryHDU(np.stack([_data(i) for i in range(4)])).writeto(path)
    return str(path)


@pytest.fixture
def multi_ext_path(tmp_path):
    path = tmp_path / "multi.fits"
    fits.HDUList([fits.PrimaryHDU(_data(0)), fits.ImageHDU(_data(1)), fits.ImageHDU(_data(2))]).writeto(path)
    return str(path)


def _open(path, qapp, **settings):
    win = MaskFitsApp([path], settings=Settings(**settings))
    win.resize(1200, 900)
    win.show()
    qapp.processEvents()
    return win


def _process(win, *, bin_factor=None, sigma=None):
    if sigma is not None:
        win.smooth_sigma_entry.setText(str(sigma))
        win._toggle_smoothing()
    if bin_factor is not None:
        win.bin_factor_entry.setText(str(bin_factor))
        win._toggle_binning()


def _view(win):
    return (win.image.data.shape, round(win.fit_zoom, 6), win.zoom_mult,
            round(win.view_cx, 6), round(win.view_cy, 6))


def _switch(win, kind):
    if kind == "slice":
        win.switch_slice(2)
    else:
        win.switch_extension(win.entry.available_extensions[1])


@pytest.mark.parametrize("kind", ["slice", "extension"])
def test_binning_carries_over_with_the_same_view(qapp, cube_path, multi_ext_path, kind):
    win = _open(cube_path if kind == "slice" else multi_ext_path, qapp)
    win.zoom_mult = 2.0
    _process(win, bin_factor=4)
    win.view_cx, win.view_cy = 7.5, 12.0
    before = _view(win)

    _switch(win, kind)

    assert win.entry.is_binned and win.entry.bin_factor == 4
    assert win.image.data.shape == (SIZE // 4, SIZE // 4)
    assert _view(win) == before
    assert win.bin_button.isChecked()


@pytest.mark.parametrize("kind", ["slice", "extension"])
def test_smoothing_carries_over(qapp, cube_path, multi_ext_path, kind):
    win = _open(cube_path if kind == "slice" else multi_ext_path, qapp)
    _process(win, sigma=1.5)
    before = _view(win)

    _switch(win, kind)

    assert win.entry.is_smoothed and win.entry.smooth_sigma == 1.5
    assert _view(win) == before
    assert win.smooth_button.isChecked()


@pytest.mark.parametrize("kind", ["slice", "extension"])
def test_binning_and_smoothing_together(qapp, cube_path, multi_ext_path, kind):
    win = _open(cube_path if kind == "slice" else multi_ext_path, qapp)
    _process(win, sigma=2.0, bin_factor=2)

    _switch(win, kind)

    assert win.entry.is_smoothed and win.entry.smooth_sigma == 2.0
    assert win.entry.is_binned and win.entry.bin_factor == 2


@pytest.mark.parametrize("kind", ["slice", "extension"])
def test_nothing_applied_stays_unprocessed(qapp, cube_path, multi_ext_path, kind):
    win = _open(cube_path if kind == "slice" else multi_ext_path, qapp)
    before = _view(win)
    _switch(win, kind)
    assert not win.entry.is_binned and not win.entry.is_smoothed
    assert _view(win) == before


@pytest.mark.parametrize("kind", ["slice", "extension"])
def test_mask_painted_while_binned_survives_the_switch(qapp, cube_path, multi_ext_path, kind):
    win = _open(cube_path if kind == "slice" else multi_ext_path, qapp)
    _process(win, bin_factor=4)
    win.image.mask[2, 3] = True  # one binned pixel = a 4x4 full-resolution block
    win._mark_mask_dirty()

    _switch(win, kind)

    assert win.image.mask[2, 3]
    assert win.image.mask.sum() == 1
    win._toggle_binning()
    assert win.image.mask[8:12, 12:16].all() and win.image.mask.sum() == 16


def test_extension_switch_keeps_the_current_factor_over_the_settings_default(qapp, multi_ext_path):
    win = _open(multi_ext_path, qapp, bin_enabled=True, bin_factor=4)
    assert win.entry.bin_factor == 4
    win.bin_factor_entry.setText("2")
    win._apply_bin_entry()
    assert win.entry.bin_factor == 2

    win.switch_extension(win.entry.available_extensions[1])

    assert win.entry.is_binned and win.entry.bin_factor == 2


def test_status_shows_the_switch_not_the_rebinning(qapp, cube_path, multi_ext_path):
    win = _open(cube_path, qapp)
    _process(win, bin_factor=2)
    win.switch_slice(1)
    assert win.status.text().startswith("slice 2/")
    win = _open(multi_ext_path, qapp)
    _process(win, bin_factor=2)
    win.switch_extension(win.entry.available_extensions[1])
    assert win.status.text().startswith("loaded ")
