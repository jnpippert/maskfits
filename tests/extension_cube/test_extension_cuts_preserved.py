"""Switching extensions (Up/Down or the extension combo, via
MaskFitsApp.switch_extension) keeps the current cut levels - the colormap
doesn't change just because the extension does, so ensure_loaded()'s usual
"recompute fresh cuts for a newly loaded image" behavior (right for
prev_image/next_image loading a genuinely different file) is undone for this
one path. Only an explicit stretch-preset change (set_stretch) - or leaving/
entering IsoPy - should move the cuts. The two extensions below use very
different data ranges specifically so a recompute (if it happened) would
produce clearly different numbers, making a regression obvious."""

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


def _write_two_ranges_fits(path):
    rng = np.random.default_rng(0)
    primary = fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(32, 32)).astype(np.float32))
    primary.header["OBJECT"] = "EXT0"
    ext1 = fits.ImageHDU(rng.normal(10_000.0, 500.0, size=(32, 32)).astype(np.float32))
    ext1.header["OBJECT"] = "EXT1"
    fits.HDUList([primary, ext1]).writeto(path, overwrite=True)


def test_switch_extension_keeps_cut_levels(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_two_ranges_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(stretch="zscale"))
    win.show()
    qapp.processEvents()

    before_lo, before_hi = win.entry.lowcut, win.entry.highcut

    win.switch_extension(1)

    assert win.entry.ext == 1
    assert win.entry.lowcut == before_lo
    assert win.entry.highcut == before_hi
    # A fresh zscale recompute against ext1's ~10000-centered data would
    # have produced very different numbers - confirms it didn't happen.
    assert win.entry.highcut < 1000.0


def test_switch_extension_keeps_a_manual_cut_edit(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_two_ranges_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.entry.lowcut, win.entry.highcut = 42.0, 84.0

    win.switch_extension(1)

    assert win.entry.lowcut == 42.0
    assert win.entry.highcut == 84.0


def test_extension_hotkeys_keep_cut_levels(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_two_ranges_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.entry.lowcut, win.entry.highcut = 42.0, 84.0

    win.next_extension()
    assert (win.entry.lowcut, win.entry.highcut) == (42.0, 84.0)

    win.prev_extension()
    assert (win.entry.lowcut, win.entry.highcut) == (42.0, 84.0)


def test_cuts_histogram_display_reflects_the_preserved_cuts(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_two_ranges_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.entry.lowcut, win.entry.highcut = 42.0, 84.0
    win._update_cuts_display()

    win.switch_extension(1)

    assert win.cuts_histogram.vmin == 42.0
    assert win.cuts_histogram.vmax == 84.0


def test_next_image_still_recomputes_cuts_for_a_different_file(qapp, tmp_path):
    """Unlike switch_extension, prev_image/next_image load a genuinely
    different file - the existing fresh-recompute behavior must survive."""
    rng = np.random.default_rng(0)
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(32, 32)).astype(np.float32)).writeto(a, overwrite=True)
    fits.PrimaryHDU(rng.normal(10_000.0, 500.0, size=(32, 32)).astype(np.float32)).writeto(b, overwrite=True)

    win = MaskFitsApp([str(a), str(b)], settings=Settings(stretch="zscale"))
    win.show()
    qapp.processEvents()

    win.entries[0].lowcut, win.entries[0].highcut = 1.0, 2.0  # a stale, deliberately-wrong value
    win.next_image()

    assert win.entry.highcut > 1000.0  # recomputed fresh against b's ~10000-centered data


def test_set_stretch_still_recomputes_cuts_normally(qapp, tmp_path):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.random.default_rng(0).normal(100.0, 5.0, size=(32, 32)).astype(np.float32)).writeto(
        path, overwrite=True
    )
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.entry.lowcut, win.entry.highcut = 1.0, 2.0
    win.set_stretch("zscale")

    assert (win.entry.lowcut, win.entry.highcut) != (1.0, 2.0)


def test_switch_extension_preserves_a_manual_isopy_edit(qapp, tmp_path):
    from maskfits.colormaps import ISOPY_NAME

    path = tmp_path / "multi.fits"
    _write_two_ranges_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(colormap=ISOPY_NAME))
    win.show()
    qapp.processEvents()

    win.cuts_histogram._lo_entry.setText("90")
    win.cuts_histogram._on_lo_entry()
    qapp.processEvents()
    assert win.entry.lowcut == 90.0

    win.switch_extension(1)

    assert win.entry.lowcut == 90.0
