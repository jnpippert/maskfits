"""Regression coverage for a real bug: pressing Escape closed whichever of
the app's popup dialogs had focus, via QDialog's own default "Escape ==
reject()" behavior. This was especially broken for the keyboard-shortcut
editor: ShortcutCapture treats a quick Escape tap as binding "Esc" itself
and a ~1s hold as cancelling the capture - but QDialog's default handling
could close the whole ShortcutsWindow (and, since Escape wasn't consumed,
the parent SettingsWindow too) as an unrelated side effect of either one.

Each of SettingsWindow/ThemeEditorWindow/ShortcutsWindow/HotkeysWindow now
overrides keyPressEvent to swallow Escape entirely - closing them always
requires an explicit action (Cancel/Save/Close/Delete).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.hotkeys_window import HotkeysWindow  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.settings_window import SettingsWindow  # noqa: E402
from maskfits.shortcuts_window import ShortcutsWindow  # noqa: E402
from maskfits.theme_editor_window import ThemeEditorWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _send_escape(app, widget):
    app.sendEvent(widget, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))


def test_settings_window_survives_escape(qapp):
    dlg = SettingsWindow(Settings())
    dlg.show()
    _send_escape(qapp, dlg)
    assert dlg.isVisible() is True


def test_theme_editor_window_survives_escape(qapp):
    dlg = ThemeEditorWindow()
    dlg.show()
    _send_escape(qapp, dlg)
    assert dlg.isVisible() is True


def test_hotkeys_window_survives_escape(qapp):
    dlg = HotkeysWindow({})
    dlg.show()
    _send_escape(qapp, dlg)
    assert dlg.isVisible() is True


def test_shortcuts_window_survives_escape(qapp):
    dlg = ShortcutsWindow({})
    dlg.show()
    _send_escape(qapp, dlg)
    assert dlg.isVisible() is True


def test_shortcuts_window_survives_escape_sent_to_a_listening_capture(qapp):
    """The exact reported scenario: a ShortcutCapture has focus and is
    listening (mid hold-to-cancel gesture) when Escape arrives."""
    dlg = ShortcutsWindow({})
    dlg.show()
    cap = dlg._captures["reset_mask"][0]
    cap._start_listening()
    _send_escape(qapp, cap)  # a quick tap - starts the hold timer
    assert dlg.isVisible() is True

    cap._on_escape_held()  # simulate the ~1s hold actually completing
    assert dlg.isVisible() is True
    assert cap._listening is False


def test_settings_window_survives_escape_while_shortcuts_child_is_open(qapp):
    """The other half of the report: holding Escape in the child
    ShortcutsWindow must not also close the parent SettingsWindow."""
    settings_dlg = SettingsWindow(Settings())
    settings_dlg.show()
    settings_dlg._edit_shortcuts()
    shortcuts_dlg = settings_dlg._shortcuts_window
    shortcuts_dlg.show()

    cap = shortcuts_dlg._captures["reset_mask"][0]
    cap._start_listening()
    _send_escape(qapp, cap)
    cap._on_escape_held()

    assert shortcuts_dlg.isVisible() is True
    assert settings_dlg.isVisible() is True
