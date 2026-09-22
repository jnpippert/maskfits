"""MaskFitsApp.export_mask() (the "Export Mask" toolbar button and File
menu action) writes to a Settings.export_filename_format(_multi/_cube)
filename - which of the three depends on the current file's own type (see
gui._file_kind): a plain single-extension image, a multi-extension FITS
(export_filename_format_multi, with $EXT for the current extension number),
or a cube (export_filename_format_cube, with $SLICE for the current slice
number). Save Mask As... (export_mask_as) is the interactive alternative and
keeps its own default name, unaffected by any of this."""

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


def _write_multi(path, size=32):
    rng = np.random.default_rng(0)
    primary = fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32))
    ext1 = fits.ImageHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32))
    fits.HDUList([primary, ext1]).writeto(path, overwrite=True)


def _write_cube(path, n_slices=3, size=32):
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, size=(n_slices, size, size)).astype(np.float32)
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


# ----------------------------------------------------------------- multi-ext


def test_export_mask_uses_the_multi_format_for_a_multi_extension_file(qapp, tmp_path):
    path = tmp_path / "myimage.fits"
    _write_multi(path)
    win = MaskFitsApp(
        [str(path)],
        settings=Settings(export_dir="file_parent", export_filename_format_multi="mask_$FILENAME_ext$EXT"),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()
    assert (tmp_path / "mask_myimage_ext0.fits").exists()

    win.switch_extension(1)
    win.export_mask()
    assert (tmp_path / "mask_myimage_ext1.fits").exists()


def test_export_mask_ignores_the_multi_format_for_a_single_extension_file(qapp, tmp_path):
    """A single-extension file must use export_filename_format, not
    export_filename_format_multi, even if the latter is customized."""
    path = tmp_path / "myimage.fits"
    _write_fits(path)
    win = MaskFitsApp(
        [str(path)],
        settings=Settings(
            export_dir="file_parent",
            export_filename_format="mask_$FILENAME",
            export_filename_format_multi="SHOULD_NOT_BE_USED_$FILENAME_ext$EXT",
        ),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()

    assert (tmp_path / "mask_myimage.fits").exists()
    assert not list(tmp_path.glob("SHOULD_NOT_BE_USED*"))


# --------------------------------------------------------------------- cube


def test_export_mask_uses_the_cube_format_for_a_cube(qapp, tmp_path):
    path = tmp_path / "mycube.fits"
    _write_cube(path, n_slices=3)
    win = MaskFitsApp(
        [str(path)],
        settings=Settings(export_dir="file_parent", export_filename_format_cube="mask_$FILENAME_slice$SLICE"),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()
    assert (tmp_path / "mask_mycube_slice0.fits").exists()

    win.switch_slice(2)
    win.export_mask()
    assert (tmp_path / "mask_mycube_slice2.fits").exists()


def test_export_mask_ignores_the_cube_format_for_a_single_extension_file(qapp, tmp_path):
    path = tmp_path / "myimage.fits"
    _write_fits(path)
    win = MaskFitsApp(
        [str(path)],
        settings=Settings(
            export_dir="file_parent",
            export_filename_format="mask_$FILENAME",
            export_filename_format_cube="SHOULD_NOT_BE_USED_$FILENAME_slice$SLICE",
        ),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()

    assert (tmp_path / "mask_myimage.fits").exists()
    assert not list(tmp_path.glob("SHOULD_NOT_BE_USED*"))


def test_export_mask_defaults_produce_distinct_filenames_across_extensions(qapp, tmp_path):
    """The built-in default multi-ext format already includes $EXT, so two
    extensions never collide on the same output filename out of the box."""
    path = tmp_path / "myimage.fits"
    _write_multi(path)
    win = MaskFitsApp([str(path)], settings=Settings(export_dir="file_parent"))
    win.show()
    qapp.processEvents()

    win.export_mask()
    win.switch_extension(1)
    win.export_mask()

    written = sorted(p.name for p in tmp_path.glob("*.fits") if p.name != "myimage.fits")
    assert len(written) == 2
    assert len(set(written)) == 2


def test_export_mask_defaults_produce_distinct_filenames_across_slices(qapp, tmp_path):
    path = tmp_path / "mycube.fits"
    _write_cube(path, n_slices=3)
    win = MaskFitsApp([str(path)], settings=Settings(export_dir="file_parent"))
    win.show()
    qapp.processEvents()

    win.export_mask()
    win.switch_slice(1)
    win.export_mask()

    written = sorted(p.name for p in tmp_path.glob("*.fits") if p.name != "mycube.fits")
    assert len(written) == 2
    assert len(set(written)) == 2


# ------------------------------------------------- $EXT/$SLICE are optional


def test_export_mask_multi_format_works_without_ext(qapp, tmp_path):
    """$EXT is optional in export_filename_format_multi - a format that
    omits it is still valid, it just means every extension's export lands
    on the same filename (the user's own choice)."""
    path = tmp_path / "myimage.fits"
    _write_multi(path)
    win = MaskFitsApp(
        [str(path)],
        settings=Settings(export_dir="file_parent", export_filename_format_multi="mask_$FILENAME"),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()

    assert (tmp_path / "mask_myimage.fits").exists()


def test_export_mask_cube_format_works_without_slice(qapp, tmp_path):
    path = tmp_path / "mycube.fits"
    _write_cube(path, n_slices=3)
    win = MaskFitsApp(
        [str(path)],
        settings=Settings(export_dir="file_parent", export_filename_format_cube="mask_$FILENAME"),
    )
    win.show()
    qapp.processEvents()

    win.export_mask()

    assert (tmp_path / "mask_mycube.fits").exists()
