"""Up/Down arrow hotkeys for cycling through a FITS file's extensions,
mirroring the Left/Right prev/next-image ones - stepping through
Entry.available_extensions (not raw ext+/-1, since extensions aren't always
contiguous) and keeping the extension combo in the top row in sync via
load_current()'s existing _update_extension_picker() call."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.shortcuts import SHORTCUT_ACTIONS_BY_ID  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_fits(path, size=32):
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)
    fits.PrimaryHDU(data).writeto(path, overwrite=True)


def _write_three_ext_fits(path):
    rng = np.random.default_rng(0)

    def hdu(cls, tag):
        h = cls(rng.normal(100.0, 5.0, size=(32, 32)).astype(np.float32))
        h.header["OBJECT"] = tag
        return h

    fits.HDUList([
        hdu(fits.PrimaryHDU, "EXT0"),
        hdu(fits.ImageHDU, "EXT1"),
        hdu(fits.ImageHDU, "EXT2"),
    ]).writeto(path, overwrite=True)


def test_default_keys_are_up_and_down():
    # Up counts extensions up (0, 1, 2, ...), Down counts them back down.
    assert SHORTCUT_ACTIONS_BY_ID["next_extension"].defaults == ("Up",)
    assert SHORTCUT_ACTIONS_BY_ID["prev_extension"].defaults == ("Down",)


def test_next_extension_steps_forward_and_updates_the_combo(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_three_ext_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.next_extension()

    assert win.entry.ext == 1
    assert win.image.header.get("OBJECT") == "EXT1"
    assert win.ext_combo.itemData(win.ext_combo.currentIndex()) == 1


def test_prev_extension_steps_backward_and_updates_the_combo(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_three_ext_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(), extension=2)
    win.show()
    qapp.processEvents()

    win.prev_extension()

    assert win.entry.ext == 1
    assert win.image.header.get("OBJECT") == "EXT1"
    assert win.ext_combo.itemData(win.ext_combo.currentIndex()) == 1


def test_next_extension_clamps_at_the_last_extension(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_three_ext_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(), extension=2)
    win.show()
    qapp.processEvents()

    win.next_extension()

    assert win.entry.ext == 2


def test_prev_extension_clamps_at_the_first_extension(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_three_ext_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.prev_extension()

    assert win.entry.ext == 0


def test_extension_hotkeys_are_a_no_op_for_a_single_extension_file(qapp, tmp_path):
    path = tmp_path / "single.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    win.next_extension()
    win.prev_extension()

    assert win.entry.ext == 0


def test_extension_hotkeys_are_wired_to_up_and_down_qshortcuts(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_three_ext_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    keys = [sc.key().toString() for sc in win._shortcuts]
    assert "Up" in keys
    assert "Down" in keys
