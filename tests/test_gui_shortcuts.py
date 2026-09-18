"""Coverage for the Help -> Hotkeys popup and Settings -> Edit Keyboard
Shortcuts... rebinding, end to end through MaskFitsApp - not just the
individual dialogs/widgets (see test_shortcuts_window.py, test_hotkeys_
window.py, test_shortcut_capture.py) but the full wiring: rebinding in
ShortcutsWindow -> SettingsWindow's own Save -> MaskFitsApp rebuilding its
actual QShortcut objects live, no restart needed.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.hotkeys_window import HotkeysWindow  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.shortcuts import SHORTCUT_ACTIONS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _shortcut_key_strings(win):
    return sorted(sc.key().toString() for sc in win._shortcuts)


def _rebind(cap, key):
    cap._start_listening()
    cap.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


def test_help_menu_has_hotkeys_not_the_old_inline_list(qapp):
    win = MaskFitsApp([], settings=Settings())
    help_menu = None
    for action in win.menuBar().actions():
        if action.text() == "Help":
            help_menu = action.menu()
    assert help_menu is not None
    labels = [a.text() for a in help_menu.actions()]
    assert "Hotkeys" in labels
    assert not any("Left-click" in label for label in labels)


def test_build_shortcuts_creates_one_qshortcut_per_default_key(qapp):
    win = MaskFitsApp([], settings=Settings())
    total_defaults = sum(len(a.defaults) for a in SHORTCUT_ACTIONS)
    assert len(win._shortcuts) == total_defaults
    assert "R" in _shortcut_key_strings(win)


def test_build_shortcuts_honors_settings_overrides_at_startup(qapp):
    win = MaskFitsApp([], settings=Settings(shortcuts={"reset_mask": ["X"]}))
    keys = _shortcut_key_strings(win)
    assert "X" in keys
    assert "R" not in keys


def test_rebuild_shortcuts_replaces_old_qshortcut_objects(qapp):
    win = MaskFitsApp([], settings=Settings())
    assert "R" in _shortcut_key_strings(win)

    win.settings.shortcuts = {"reset_mask": ["X"]}
    win._rebuild_shortcuts()

    keys = _shortcut_key_strings(win)
    assert "X" in keys
    assert "R" not in keys


def test_show_hotkeys_reuses_a_single_instance(qapp):
    win = MaskFitsApp([], settings=Settings())
    win._show_hotkeys()
    first = win._hotkeys_window
    assert isinstance(first, HotkeysWindow)
    win._show_hotkeys()
    assert win._hotkeys_window is first


def test_full_rebind_chain_updates_live_shortcuts(qapp):
    """The exact chain a user drives by hand: Help/Settings -> Edit Keyboard
    Shortcuts... -> click a capture -> press a new key -> Save (sub-editor)
    -> Save (Settings) -> the running window's real QShortcut objects
    reflect the change immediately."""
    win = MaskFitsApp([], settings=Settings())
    win.show()

    win._open_settings()
    settings_dlg = win._settings_window
    settings_dlg._edit_shortcuts()
    shortcuts_dlg = settings_dlg._shortcuts_window

    cap = shortcuts_dlg._captures["reset_mask"][0]
    assert cap.value() == "R"
    _rebind(cap, Qt.Key.Key_X)
    assert cap.value() == "X"

    shortcuts_dlg._save()
    assert settings_dlg.shortcuts == {"reset_mask": ["X"]}

    settings_dlg._save()
    assert win.settings.shortcuts == {"reset_mask": ["X"]}

    keys = _shortcut_key_strings(win)
    assert "X" in keys
    assert "R" not in keys

    # And the new binding actually fires the right handler.
    triggered = []
    win.reset_mask = lambda: triggered.append(True)
    win._rebuild_shortcuts()
    x_shortcut = next(sc for sc in win._shortcuts if sc.key().toString() == "X")
    x_shortcut.activated.emit()
    assert triggered == [True]
