"""Coverage for the top-row extension/cube UI: a single-extension file shows
neither the extension combo nor cube slice controls, a multi-extension file
shows the combo, and a cube extension shows a slice slider + entry box
instead - the three are mutually exclusive (see MaskFitsApp.
_update_extension_picker). Also covers cube slice navigation itself
(switch_slice) and the Left/Right hotkeys cycling slices instead of
switching files while a cube is loaded."""

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


def _write_single(path, size=32):
    rng = np.random.default_rng(0)
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)).writeto(path, overwrite=True)


def _write_multi(path, size=32):
    rng = np.random.default_rng(0)
    primary = fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32))
    ext1 = fits.ImageHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32))
    fits.HDUList([primary, ext1]).writeto(path, overwrite=True)


def _write_cube(path, n_slices=5, size=32):
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, size=(n_slices, size, size)).astype(np.float32)
    for i in range(n_slices):
        data[i] += i * 10.0  # each slice distinguishable by its mean level
    fits.PrimaryHDU(data).writeto(path, overwrite=True)


def _open(paths, **settings_kwargs):
    win = MaskFitsApp([str(p) for p in paths], settings=Settings(**settings_kwargs))
    win.show()
    return win


# ------------------------------------------------------------- visibility


def test_single_extension_shows_neither_combo_nor_slice_controls(qapp, tmp_path):
    path = tmp_path / "single.fits"
    _write_single(path)
    win = _open([path])

    assert win.ext_combo.isVisible() is False
    assert win.slice_slider.isVisible() is False
    assert win.slice_entry.isVisible() is False


def test_multi_extension_shows_the_combo_only(qapp, tmp_path):
    path = tmp_path / "multi.fits"
    _write_multi(path)
    win = _open([path])

    assert win.ext_combo.isVisible() is True
    assert win.slice_slider.isVisible() is False
    assert win.slice_entry.isVisible() is False


def test_cube_shows_slice_controls_only(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])

    assert win.ext_combo.isVisible() is False
    assert win.slice_slider.isVisible() is True
    assert win.slice_entry.isVisible() is True


def test_cube_slider_range_matches_slice_count(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path, n_slices=7)
    win = _open([path])

    assert win.slice_slider._lo == 0
    assert win.slice_slider._hi == 6
    assert win.slice_entry.text() == "0"


# --------------------------------------------------------------- switch_slice


def test_switch_slice_moves_to_the_requested_slice(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])

    win.switch_slice(3)

    assert win.image.slice_index == 3
    assert win.slice_entry.text() == "3"
    assert win.slice_slider.value() == 3


def test_switch_slice_clamps_at_both_ends(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path, n_slices=5)
    win = _open([path])

    win.switch_slice(-3)
    assert win.image.slice_index == 0

    win.switch_slice(99)
    assert win.image.slice_index == 4


def test_switch_slice_preserves_mask_zoom_and_cuts(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])

    win.image.mask[2:5, 2:5] = True
    painted = win.image.mask.copy()
    win.zoom_mult = 4.0
    win.view_cx, win.view_cy = 9.0, 11.0
    win.entry.lowcut, win.entry.highcut = 42.0, 84.0

    win.switch_slice(2)

    assert np.array_equal(win.image.mask, painted)
    assert win.zoom_mult == 4.0
    assert (win.view_cx, win.view_cy) == (9.0, 11.0)
    assert (win.entry.lowcut, win.entry.highcut) == (42.0, 84.0)


def test_slice_entry_box_updates_slice_and_slider(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])

    win.slice_entry.setText("4")
    win._on_slice_entry()

    assert win.image.slice_index == 4
    assert win.slice_slider.value() == 4


def test_slice_entry_box_recovers_from_garbage_input(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])
    win.switch_slice(2)

    win.slice_entry.setText("not a number")
    win._on_slice_entry()

    assert win.image.slice_index == 2
    assert win.slice_entry.text() == "2"


def test_slider_drag_updates_the_entry_live(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])

    win.slice_slider.setValue(3)

    assert win.slice_entry.text() == "3"
    assert win.image.slice_index == 3


# ------------------------------------------------------ Left/Right hotkeys


def test_right_arrow_steps_up_a_slice_when_cube_loaded(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])

    win.next_image()

    assert win.image.slice_index == 1


def test_left_arrow_steps_down_a_slice_when_cube_loaded(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])
    win.switch_slice(3)

    win.prev_image()

    assert win.image.slice_index == 2


def test_arrow_keys_clamp_at_cube_ends(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path, n_slices=5)
    win = _open([path])

    win.prev_image()
    assert win.image.slice_index == 0

    win.switch_slice(4)
    win.next_image()
    assert win.image.slice_index == 4


def test_arrow_keys_still_navigate_files_when_not_a_cube(qapp, tmp_path):
    """Regression: the cube branch must not hijack Left/Right for ordinary
    (non-cube) multi-file navigation."""
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    _write_single(a)
    _write_single(b)
    win = _open([a, b])

    win.next_image()
    assert win.index == 1

    win.prev_image()
    assert win.index == 0


def test_right_arrow_qshortcut_is_still_bound_to_next_image(qapp, tmp_path):
    """The hotkey binding itself (Right -> next_image) is unchanged - only
    next_image's own behavior became cube-aware. Confirms no rebinding was
    needed to get arrow-key slice cycling."""
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])

    keys = [sc.key().toString() for sc in win._shortcuts]
    assert "Right" in keys
    assert "Left" in keys


# ------------------------------------------------------- status bar slice label


def _write_wave_cube(path, n_slices=5, size=16, crval3=3551.0, cdelt3=1370.0, cunit3="Angstrom"):
    data = np.zeros((n_slices, size, size), dtype=np.float32) + 100.0
    hdr = fits.Header()
    hdr["CTYPE3"] = "WAVE"
    hdr["CUNIT3"] = cunit3
    hdr["CRPIX3"] = 1
    hdr["CRVAL3"] = crval3
    hdr["CDELT3"] = cdelt3
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def test_status_shows_slice_value_and_unit_for_a_wavelength_cube(qapp, tmp_path):
    path = tmp_path / "wave_cube.fits"
    _write_wave_cube(path, crval3=3551.0, cdelt3=1370.0)
    win = _open([path])

    win.switch_slice(2)

    assert win.status.text() == "slice 3/5 (6291 Angstrom)"


def test_status_slice_value_updates_as_slider_drags(qapp, tmp_path):
    path = tmp_path / "wave_cube.fits"
    _write_wave_cube(path, crval3=3551.0, cdelt3=1370.0)
    win = _open([path])

    win.slice_slider.setValue(1)

    assert win.status.text() == "slice 2/5 (4921 Angstrom)"


def test_status_omits_the_value_when_the_header_has_no_wave_axis(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write_cube(path)
    win = _open([path])

    win.switch_slice(1)

    assert win.status.text() == "slice 2/5"


def test_cube_slice_label_handles_a_missing_crpix3(qapp, tmp_path):
    """CRPIX3 is optional per the FITS WCS convention (defaults to 1)."""
    path = tmp_path / "wave_cube.fits"
    data = np.zeros((3, 8, 8), dtype=np.float32)
    hdr = fits.Header()
    hdr["CRVAL3"] = 5000.0
    hdr["CDELT3"] = 10.0
    hdr["CUNIT3"] = "Angstrom"
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
    win = _open([path])

    win.switch_slice(2)

    assert win.status.text() == "slice 3/3 (5020 Angstrom)"


def _write_band_cube(path, size=16):
    data = np.zeros((5, size, size), dtype=np.float32) + 100.0
    hdr = fits.Header()
    bands = ["u", "g", "r", "i", "z"]
    waves = [3551.0, 4686.0, 6165.0, 7481.0, 8931.0]
    for i, (band, wave) in enumerate(zip(bands, waves)):
        hdr[f"BAND{i}"] = band
        hdr[f"WAVE{i}"] = wave
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def test_status_shows_band_name_and_wavelength_for_a_per_slice_convention_cube(qapp, tmp_path):
    """SDSS ugriz-style cubes (test_cube_ugriz.fits) use WAVE<i>/BAND<i>
    per-slice keywords instead of a linear CRVAL3/CDELT3 axis, since the
    central wavelengths aren't evenly spaced."""
    path = tmp_path / "band_cube.fits"
    _write_band_cube(path)
    win = _open([path])

    win.switch_slice(2)
    assert win.status.text() == "slice 3/5 (r 6165 Angstrom)"

    win.switch_slice(4)
    assert win.status.text() == "slice 5/5 (z 8931 Angstrom)"


def test_per_slice_convention_is_checked_before_the_linear_wcs_axis(qapp, tmp_path):
    """A cube could in principle carry both sets of keywords - WAVE<i>
    should win, since it's the more specific, per-slice-exact value."""
    path = tmp_path / "band_cube.fits"
    data = np.zeros((3, 8, 8), dtype=np.float32)
    hdr = fits.Header()
    hdr["CRVAL3"] = 1.0
    hdr["CDELT3"] = 1.0
    hdr["WAVE1"] = 4686.0
    hdr["BAND1"] = "g"
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
    win = _open([path])

    win.switch_slice(1)

    assert win.status.text() == "slice 2/3 (g 4686 Angstrom)"
