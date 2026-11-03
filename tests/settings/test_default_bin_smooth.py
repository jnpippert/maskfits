"""Coverage for the Settings defaults for smoothing sigma and bin factor:
the toolbar boxes always start at those values (whether or not the
"default ..." tick boxes are on), the tick boxes decide whether smoothing /
binning is actually applied, and saving Settings applies a change to the
image that's already open."""

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


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.random.default_rng(0).normal(100, 5, (60, 60)).astype(np.float32)).writeto(path)
    return str(path)


def test_toolbar_shows_the_defaults_even_when_they_are_off(qapp, image_path):
    win = MaskFitsApp([image_path], settings=Settings(smooth_sigma=1.5, bin_factor=5))
    assert win.smooth_sigma_entry.text() == "1.5"
    assert win.bin_factor_entry.text() == "5"
    assert not win.entry.is_smoothed
    assert not win.entry.is_binned


def test_ticked_defaults_are_applied_on_startup(qapp, image_path):
    win = MaskFitsApp([image_path], settings=Settings(smooth_enabled=True, smooth_sigma=1.5,
                                                     bin_enabled=True, bin_factor=5))
    assert win.entry.is_smoothed and win.entry.smooth_sigma == 1.5
    assert win.entry.is_binned and win.entry.bin_factor == 5
    assert win.smooth_button.isChecked() and win.bin_button.isChecked()


def test_toggling_uses_the_default_value(qapp, image_path):
    win = MaskFitsApp([image_path], settings=Settings(smooth_sigma=1.5, bin_factor=5))
    win._toggle_smoothing()
    win._toggle_binning()
    assert win.entry.smooth_sigma == 1.5
    assert win.entry.bin_factor == 5


def test_saving_a_new_default_value_updates_the_toolbar(qapp, image_path):
    win = MaskFitsApp([image_path], settings=Settings())
    win._apply_settings_live(Settings(smooth_sigma=4.0, bin_factor=6))
    assert win.smooth_sigma_entry.text() == "4"
    assert win.bin_factor_entry.text() == "6"
    assert not win.entry.is_smoothed and not win.entry.is_binned


def test_saving_with_the_boxes_ticked_applies_to_the_open_image(qapp, image_path):
    win = MaskFitsApp([image_path], settings=Settings())
    win._apply_settings_live(Settings(smooth_enabled=True, smooth_sigma=2.5, bin_enabled=True, bin_factor=3))
    assert win.entry.is_smoothed and win.entry.smooth_sigma == 2.5
    assert win.entry.is_binned and win.entry.bin_factor == 3


def test_saving_a_new_value_while_active_reapplies_it(qapp, image_path):
    win = MaskFitsApp([image_path], settings=Settings(smooth_enabled=True, smooth_sigma=1.5,
                                                     bin_enabled=True, bin_factor=2))
    win._apply_settings_live(Settings(smooth_enabled=True, smooth_sigma=3.0, bin_enabled=True, bin_factor=4))
    assert win.entry.smooth_sigma == 3.0
    assert win.entry.bin_factor == 4


def test_saving_unrelated_settings_keeps_a_manual_toolbar_value(qapp, image_path):
    win = MaskFitsApp([image_path], settings=Settings(smooth_sigma=1.5, bin_factor=5))
    win.smooth_sigma_entry.setText("7")
    win.bin_factor_entry.setText("8")
    win._apply_settings_live(Settings(smooth_sigma=1.5, bin_factor=5, blank_color="#3355ff"))
    assert win.smooth_sigma_entry.text() == "7"
    assert win.bin_factor_entry.text() == "8"
