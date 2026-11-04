"""Clicking somewhere non-interactive in the main window (e.g. the cube
spectrum panel) takes focus away from a text box typed into before - which
otherwise kept every hotkey muted."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    fits.PrimaryHDU(np.random.default_rng(0).normal(100, 5, (4, 40, 40)).astype(np.float32)).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.resize(1200, 900)
    win.show()
    win.activateWindow()
    qapp.processEvents()
    yield win
    focused = QApplication.focusWidget()
    if focused is not None:
        focused.clearFocus()  # never leave a text box focused for later tests
    win.close()


def _type_into(entry, text, qapp):
    entry.setFocus()
    entry.setText(text)
    qapp.processEvents()
    assert QApplication.focusWidget() is entry


@pytest.mark.parametrize("target", ["spectrum_view", "magnifier", "sb_cuts_label"])
def test_click_on_an_inert_area_ends_editing(win, qapp, target):
    _type_into(win.smooth_sigma_entry, "2.5", qapp)
    QTest.mouseClick(getattr(win, target), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(5, 5))
    assert not isinstance(QApplication.focusWidget(), QLineEdit)


def test_hotkeys_work_again_after_clicking_the_spectrum(win, qapp):
    _type_into(win.smooth_sigma_entry, "2.5", qapp)
    win.zoom_mult = 3.0
    space = next(sc for sc in win._shortcuts if sc.key().toString() == "Space")
    space.activated.emit()
    assert win.zoom_mult == 3.0  # muted while the box has focus
    QTest.mouseClick(win.spectrum_view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(5, 5))
    space.activated.emit()
    assert win.zoom_mult == 1.0


def test_leaving_the_box_applies_the_typed_value(win, qapp):
    win.smooth_sigma_entry.setFocus()
    qapp.processEvents()
    win.smooth_sigma_entry.setText("1.75")
    QTest.mouseClick(win.spectrum_view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(5, 5))
    assert win._read_sigma() == 1.75
