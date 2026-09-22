"""MaskFitsApp.export_mask() (the "Export Mask" toolbar button and File
menu action) writes to Settings.export_filename_format's filename - a
$FILENAME placeholder the user fills in via Settings, default
"mask_$FILENAME" - instead of the old hardcoded "mask_<stem>.fits". Save
Mask As... (export_mask_as) is the interactive alternative and keeps its own
default name, unaffected by this setting."""

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


def _write_fits(path, size=32):
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)
    fits.PrimaryHDU(data).writeto(path, overwrite=True)


def test_export_mask_uses_the_default_format(qapp, tmp_path):
    path = tmp_path / "myimage.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(export_dir="file_parent"))
    win.show()
    qapp.processEvents()

    win.export_mask()

    assert (tmp_path / "mask_myimage.fits").exists()


def test_export_mask_uses_a_custom_format(qapp, tmp_path):
    path = tmp_path / "myimage.fits"
    _write_fits(path)
    win = MaskFitsApp(
        [str(path)], settings=Settings(export_dir="file_parent", export_filename_format="$FILENAME_masked"),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()

    assert (tmp_path / "myimage_masked.fits").exists()
    assert not (tmp_path / "mask_myimage.fits").exists()


def test_export_mask_adds_the_fits_suffix_when_the_format_omits_it(qapp, tmp_path):
    path = tmp_path / "myimage.fits"
    _write_fits(path)
    win = MaskFitsApp(
        [str(path)], settings=Settings(export_dir="file_parent", export_filename_format="out_$FILENAME"),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()

    assert (tmp_path / "out_myimage.fits").exists()


def test_export_mask_honors_an_explicit_fits_suffix_in_the_format(qapp, tmp_path):
    path = tmp_path / "myimage.fits"
    _write_fits(path)
    win = MaskFitsApp(
        [str(path)], settings=Settings(export_dir="file_parent", export_filename_format="$FILENAME.fits"),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()

    out = tmp_path / "myimage.fits.fits"
    assert not out.exists()
    assert (tmp_path / "myimage.fits").exists()  # the source file itself, untouched


def test_export_mask_respects_the_export_dir_setting(qapp, tmp_path, monkeypatch):
    subdir = tmp_path / "source"
    subdir.mkdir()
    path = subdir / "myimage.fits"
    _write_fits(path)
    monkeypatch.chdir(tmp_path)

    win = MaskFitsApp(
        [str(path)], settings=Settings(export_dir="cwd", export_filename_format="$FILENAME_masked"),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()

    assert (tmp_path / "myimage_masked.fits").exists()
    assert not (subdir / "myimage_masked.fits").exists()


def test_settings_saved_live_changes_the_export_format_without_restart(qapp, tmp_path):
    path = tmp_path / "myimage.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(export_dir="file_parent"))
    win.show()
    qapp.processEvents()

    win._apply_settings_live(Settings(export_dir="file_parent", export_filename_format="live_$FILENAME"))
    win.export_mask()

    assert (tmp_path / "live_myimage.fits").exists()
