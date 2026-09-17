"""Regression coverage for a real bug: clicking a HexColorPicker's swatch to
open QColorDialog, then clicking OK, raised the main maskfits window and
left the actual parent (SettingsWindow/ThemeEditorWindow) behind it - caused
by passing the HexColorPicker itself (a non-top-level child widget) as the
color dialog's parent instead of its enclosing top-level window, leaving
Qt/Cocoa with the wrong idea of which window opened it.
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


def test_color_dialog_parent_is_the_top_level_window(qapp, monkeypatch):
    dialog = QDialog()
    picker = HexColorPicker("#851212", parent=dialog)
    dialog.show()

    captured = {}

    def fake_get_color(initial, parent, title):
        captured["parent"] = parent
        from PySide6.QtGui import QColor

        return QColor()  # invalid - as if the user cancelled

    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(fake_get_color))
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

    def fake_get_color(initial, parent, title):
        captured["parent"] = parent
        from PySide6.QtGui import QColor

        return QColor()

    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(fake_get_color))
    picker._pick()

    assert captured["parent"] is window
