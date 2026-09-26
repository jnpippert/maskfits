"""Coverage for two hotkeys: Tab (toggle_tool, ellipse <-> satellite line
tool) and Ctrl+S (quick_export, same no-dialog write as the Export Mask
toolbar button/File menu action - see gui.MaskFitsApp.export_mask)."""

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
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)).writeto(path, overwrite=True)


def test_tab_hotkey_is_bound_to_toggle_tool():
    assert SHORTCUT_ACTIONS_BY_ID["toggle_tool"].defaults == ("Tab",)


def test_ctrl_s_hotkey_is_bound_to_quick_export():
    assert SHORTCUT_ACTIONS_BY_ID["quick_export"].defaults == ("Ctrl+S",)


def test_toggle_tool_flips_between_ellipse_and_line(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(mode="ellipse"))
    win.show()

    assert win.tool == "ellipse"
    win._toggle_tool()
    assert win.tool == "line"
    win._toggle_tool()
    assert win.tool == "ellipse"


def test_toggle_tool_starting_from_line(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(mode="line"))
    win.show()

    assert win.tool == "line"
    win._toggle_tool()
    assert win.tool == "ellipse"


def test_tab_and_ctrl_s_qshortcuts_are_wired_up(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    keys = [sc.key().toString() for sc in win._shortcuts]
    assert "Tab" in keys
    assert "Ctrl+S" in keys


def test_quick_export_writes_the_default_mask_filename(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    win.export_mask()

    out_path = tmp_path / "mask_img.fits"
    assert out_path.exists()
