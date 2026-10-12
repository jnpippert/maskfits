"""Coverage for File -> Save as DS9 Region...: the options window
(RegionExportWindow) and MaskFitsApp's wiring/writing. The polygon and
file-format logic itself is covered in tests/logic/test_regions.py."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtGui import QAction  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.region_export_window import RegionExportWindow  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def no_popups(monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warning", a[2])))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(("critical", a[2])))
    return shown


def _wcs_header():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [10.0, 10.0]
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    return wcs.to_header()


def _window(tmp_path, shape=(20, 30), with_wcs=False, name="img.fits"):
    path = tmp_path / name
    fits.PrimaryHDU(np.zeros(shape, dtype=np.float32), header=_wcs_header() if with_wcs else None).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    return win


def _polygon_lines(path):
    return [line for line in open(path).read().splitlines() if "polygon(" in line]


def _corners(line):
    """A polygon line's vertices as a set - which corner it starts at is
    an implementation detail."""
    numbers = [float(v) for v in line[line.index("(") + 1:-1].split(",")]
    return set(zip(numbers[0::2], numbers[1::2]))


# ------------------------------------------------------------------ window


def test_file_menu_has_the_action(qapp, tmp_path):
    win = _window(tmp_path)
    assert "Save as DS9 Region..." in [a.text() for a in win.findChildren(QAction)]


def test_default_path_is_the_working_directory(qapp, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    win = _window(tmp_path, name="galaxy.fits")
    win.export_region()
    assert win._region_window.path() == os.path.join(os.getcwd(), "mask_galaxy.reg")


def test_opening_twice_reuses_the_window(qapp, tmp_path):
    win = _window(tmp_path)
    win.export_region()
    first = win._region_window
    win.export_region()
    assert win._region_window is first


def test_no_image_warns_instead_of_opening(qapp, no_popups):
    win = MaskFitsApp([], settings=Settings())
    win.export_region()
    assert win._region_window is None
    assert no_popups and no_popups[0][0] == "warning"


def test_wcs_option_is_disabled_without_a_wcs(qapp):
    dialog = RegionExportWindow("/tmp/x.reg", has_wcs=False)
    assert not dialog.coords_control._buttons["wcs"].isEnabled()
    assert dialog.coords() == "image"
    with_wcs = RegionExportWindow("/tmp/x.reg", has_wcs=True)
    assert with_wcs.coords_control._buttons["wcs"].isEnabled()


def test_color_choices_and_custom_hex(qapp):
    dialog = RegionExportWindow("/tmp/x.reg", has_wcs=False)
    assert dialog.color() == "green"
    dialog.color_combo.setCurrentText("magenta")
    assert dialog.color() == "magenta"
    assert dialog.custom_color.isHidden()

    dialog.color_combo.setCurrentText("custom")
    assert not dialog.custom_color.isHidden()
    dialog.custom_color.setValue("#ff8800")
    assert dialog.color() == "#ff8800"


def test_browse_button_fills_in_the_chosen_path(qapp, monkeypatch):
    dialog = RegionExportWindow("/tmp/x.reg", has_wcs=False)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("/somewhere/else.reg", ""))
    dialog.browse_button.click()
    assert dialog.path() == "/somewhere/else.reg"


def test_cancelled_browse_keeps_the_path(qapp, monkeypatch):
    dialog = RegionExportWindow("/tmp/x.reg", has_wcs=False)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    dialog.browse_button.click()
    assert dialog.path() == "/tmp/x.reg"


def test_save_emits_the_chosen_options(qapp):
    dialog = RegionExportWindow("/tmp/x.reg", has_wcs=True)
    dialog.color_combo.setCurrentText("red")
    dialog.coords_control._buttons["wcs"].click()
    received = []
    dialog.export_requested.connect(lambda *args: received.append(args))
    dialog._save()
    assert received == [("/tmp/x.reg", "red", "wcs")]


def test_save_with_an_empty_path_does_nothing(qapp):
    dialog = RegionExportWindow("", has_wcs=False)
    received = []
    dialog.export_requested.connect(lambda *args: received.append(args))
    dialog._save()
    assert received == []


# ----------------------------------------------------------------- writing


def test_saving_writes_one_polygon_per_group_and_closes_the_window(qapp, tmp_path):
    win = _window(tmp_path)
    win.image.mask[2:5, 3:8] = True
    win.image.mask[10:12, 20:25] = True
    out = tmp_path / "out.reg"
    win.export_region()
    win._region_window.path_entry.setText(str(out))
    win._region_window.color_combo.setCurrentText("cyan")

    win._region_window._save()

    lines = _polygon_lines(out)
    assert len(lines) == 2
    assert lines[0].startswith("polygon(")
    assert _corners(lines[0]) == {(3.5, 2.5), (8.5, 2.5), (8.5, 5.5), (3.5, 5.5)}
    assert "color=cyan" in out.read_text()
    assert "text=" not in out.read_text()
    assert win._region_window is None
    assert win.status.text() == f"exported 2 regions to {out}"
    assert win.status.property("state") == "success"


def test_empty_mask_warns_and_writes_nothing(qapp, tmp_path, no_popups):
    win = _window(tmp_path)
    out = tmp_path / "out.reg"
    win._write_region(str(out), "green", "image")
    assert not out.exists()
    assert no_popups and "empty" in no_popups[0][1]


def test_unwritable_path_reports_an_error(qapp, tmp_path, no_popups):
    win = _window(tmp_path)
    win.image.mask[2, 3] = True
    win._write_region(str(tmp_path / "no_such_dir" / "out.reg"), "green", "image")
    assert no_popups and no_popups[0][0] == "critical"


def test_portrait_image_is_written_in_the_files_own_orientation(qapp, tmp_path):
    win = _window(tmp_path, shape=(30, 20))  # portrait: transposed in memory
    assert win.image.rotated
    # In-memory (row 3, col 7) is the file's (row 7, col 3) = FITS pixel (4, 8).
    win.image.mask[3, 7] = True
    out = tmp_path / "out.reg"
    win._write_region(str(out), "green", "image")
    (line,) = _polygon_lines(out)
    assert _corners(line) == {(3.5, 7.5), (4.5, 7.5), (4.5, 8.5), (3.5, 8.5)}


def test_wcs_export_writes_fk5_sky_coordinates(qapp, tmp_path):
    win = _window(tmp_path, with_wcs=True)
    win.image.mask[9, 9] = True  # FITS pixel (10, 10) = the reference pixel
    out = tmp_path / "out.reg"
    win._write_region(str(out), "green", "wcs")
    text = out.read_text()
    assert text.splitlines()[2] == "fk5"
    numbers = [float(v) for v in _polygon_lines(out)[0][len("polygon("):-1].split(",")]
    # 1e-4 deg: the header has no RADESYS (so ICRS), and ICRS -> FK5 is a
    # real ~0.02" shift.
    assert np.mean(numbers[0::2]) == pytest.approx(150.0, abs=1e-4)
    assert np.mean(numbers[1::2]) == pytest.approx(2.0, abs=1e-4)
