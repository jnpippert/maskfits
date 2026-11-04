"""Coverage for Reset Shape: the tool-panel button and its shortcut
(default Backspace) put the active tool's shape back to its starting
values."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import (  # noqa: E402
    DEFAULT_ANGLE,
    DEFAULT_ELLIPTICITY,
    DEFAULT_RADIUS,
    DEFAULT_THICKNESS,
    MaskFitsApp,
)
from maskfits.settings import Settings  # noqa: E402
from maskfits.shortcuts import SHORTCUT_ACTIONS_BY_ID, find_conflict  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, tmp_path):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.zeros((80, 80), dtype=np.float32)).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings(mode="ellipse"))
    win.show()
    qapp.processEvents()
    return win


def _shortcut(win):
    return next(sc for sc in win._shortcuts if sc.key().toString() == "Backspace")


def test_default_key_is_backspace(qapp):
    assert SHORTCUT_ACTIONS_BY_ID["reset_shape"].defaults == ("Backspace",)
    assert SHORTCUT_ACTIONS_BY_ID["reset_shape"].label == "Reset Shape"
    assert find_conflict("reset_shape", "Backspace", {}) is None


def test_starting_values():
    assert (DEFAULT_RADIUS, DEFAULT_ELLIPTICITY, DEFAULT_ANGLE, DEFAULT_THICKNESS) == (40.0, 0, 0, 15)


def test_resets_the_ellipse(win):
    win.radius, win.ellipticity, win.angle = 120.0, 50, 30
    _shortcut(win).activated.emit()
    assert (win.radius, win.ellipticity, win.angle) == (DEFAULT_RADIUS, DEFAULT_ELLIPTICITY, DEFAULT_ANGLE)
    assert float(win._radius_slider.value()) == DEFAULT_RADIUS


def test_resets_the_satellite_thickness_but_keeps_its_style(win):
    win.set_tool("line")
    win.thickness = 60
    win.line_style = "arrow"
    win._sync_tool_option_widgets()
    win.reset_shape_button.click()
    assert win.thickness == DEFAULT_THICKNESS
    assert win.line_style == "arrow"
    assert int(win._thickness_slider.value()) == DEFAULT_THICKNESS


def test_only_the_active_tool_is_reset(win):
    win.thickness = 60
    win.radius = 120.0
    win.reset_shape()  # ellipse active
    assert win.radius == DEFAULT_RADIUS
    assert win.thickness == 60


def test_button_sits_next_to_reset_mask_in_the_toolbar(win):
    from maskfits.widgets import RoundButton

    button = win.reset_shape_button
    assert button.text() == "Reset Shape"
    siblings = [w.text() for w in button.parentWidget().findChildren(RoundButton)]
    assert siblings[siblings.index("Reset Mask") + 1] == "Reset Shape"
    assert not win.sidebar_container.isAncestorOf(button)
    win.set_tool("line")  # the toolbar button isn't rebuilt per tool
    assert win.reset_shape_button is button and button.isVisibleTo(win)


def test_binned_sizes_match_a_fresh_shape(win):
    """Binning divides shape sizes by the factor (same area of sky) - a
    reset while binned must land on the same, not the raw default."""
    win.bin_factor_entry.setText("4")
    win._toggle_binning()
    win.radius = 3.0
    win.reset_shape()
    assert win.radius == DEFAULT_RADIUS / 4
    win.set_tool("line")
    win.thickness = 1
    win.reset_shape()
    assert win.thickness == round(DEFAULT_THICKNESS / 4)
    win._toggle_binning()
    assert win.thickness == round(DEFAULT_THICKNESS / 4) * 4


def test_backspace_while_typing_edits_the_text_instead(win, qapp):
    win.radius = 120.0
    entry = win.smooth_sigma_entry
    entry.setText("2.5")
    entry.setFocus()
    entry.end(False)
    qapp.processEvents()
    try:
        QTest.keyClick(entry, Qt.Key.Key_Backspace)
        assert entry.text() == "2."
        assert win.radius == 120.0
    finally:
        # A focused text box mutes every guarded shortcut app-wide - don't
        # leave it focused for later tests in this process.
        entry.clearFocus()
        win.close()
