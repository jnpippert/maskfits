"""Regression coverage for a real bug: clicking a HexColorPicker's swatch to
open QColorDialog, then clicking OK, raised the main maskfits window and
left the actual parent (SettingsWindow/ThemeEditorWindow) behind it. Two
distinct causes, both covered here:

1. The dialog was parented to the HexColorPicker itself (a non-top-level
   child widget) instead of its enclosing top-level window, leaving Qt/Cocoa
   unsure which window opened it.
2. The native macOS color picker is backed by NSColorPanel, a single
   OS-level shared panel Qt reuses across calls rather than creating fresh
   each time - its internal window-activation bookkeeping fell out of sync
   after repeated open/close cycles in one session (fine at first, then the
   main window started stealing focus back on OK after a few uses).
   DontUseNativeDialog avoids the shared panel entirely.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QColorDialog, QDialog, QWidget  # noqa: E402

from maskfits.widgets import HexColorPicker  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _fake_get_color(captured):
    def fake(initial, parent, title, options=None):
        captured["parent"] = parent
        captured["options"] = options
        from PySide6.QtGui import QColor

        return QColor()  # invalid - as if the user cancelled

    return fake


def test_color_dialog_parent_is_the_top_level_window(qapp, monkeypatch):
    dialog = QDialog()
    picker = HexColorPicker("#851212", parent=dialog)
    dialog.show()

    captured = {}
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(_fake_get_color(captured)))
    picker._pick()

    assert captured["parent"] is dialog
    assert not isinstance(captured["parent"], HexColorPicker)


def test_color_dialog_parent_is_not_a_plain_child_widget(qapp, monkeypatch):
    """Same check the other way round - a HexColorPicker nested a level
    deeper (inside a plain QWidget row, as in SettingsWindow/
    ThemeEditorWindow) must still resolve to the actual top-level window,
    not that intermediate row widget."""
    window = QDialog()
    row = QWidget(window)
    picker = HexColorPicker("#851212", parent=row)
    window.show()

    captured = {}
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(_fake_get_color(captured)))
    picker._pick()

    assert captured["parent"] is window


def test_color_dialog_avoids_the_shared_native_panel(qapp, monkeypatch):
    captured = {}
    dialog = QDialog()
    picker = HexColorPicker("#851212", parent=dialog)

    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(_fake_get_color(captured)))
    picker._pick()

    assert captured["options"] == QColorDialog.ColorDialogOption.DontUseNativeDialog
