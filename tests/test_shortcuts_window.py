import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from maskfits.shortcuts import SHORTCUT_ACTIONS_BY_ID  # noqa: E402
from maskfits.shortcuts_window import ShortcutsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _rebind(cap, key):
    cap._start_listening()
    cap.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


def test_builds_a_capture_per_default_key(qapp):
    win = ShortcutsWindow({})
    # "undo" has two defaults (Ctrl+Z, U); "clear_mask" has one (R).
    assert len(win._captures["undo"]) == 2
    assert len(win._captures["clear_mask"]) == 1
    assert win._captures["undo"][0].value() == "Ctrl+Z"
    assert win._captures["undo"][1].value() == "U"
    assert win._captures["clear_mask"][0].value() == "R"


def test_existing_overrides_are_shown(qapp):
    win = ShortcutsWindow({"clear_mask": ["X"]})
    assert win._captures["clear_mask"][0].value() == "X"


def test_rebinding_updates_overrides_and_save_emits_cleaned_dict(qapp, monkeypatch):
    win = ShortcutsWindow({})
    _rebind(win._captures["clear_mask"][0], Qt.Key.Key_X)
    assert win.overrides["clear_mask"] == ["X"]

    received = []
    win.shortcuts_saved.connect(received.append)
    monkeypatch.setattr(win, "accept", lambda: None)
    win._save()

    assert received == [{"clear_mask": ["X"]}]


def test_save_drops_overrides_that_match_defaults(qapp, monkeypatch):
    """Rebinding back to the same key as the default shouldn't be persisted
    as an override - keeps the stored set to genuine rebindings only (see
    shortcuts.py's own reasoning for why that matters)."""
    win = ShortcutsWindow({"clear_mask": ["X"]})
    _rebind(win._captures["clear_mask"][0], Qt.Key.Key_R)  # back to the default
    assert win.overrides["clear_mask"] == ["R"]

    received = []
    win.shortcuts_saved.connect(received.append)
    monkeypatch.setattr(win, "accept", lambda: None)
    win._save()

    assert received == [{}]


def test_reset_all_clears_overrides_and_resets_displayed_values(qapp):
    win = ShortcutsWindow({"clear_mask": ["X"], "undo": ["Q"]})
    win._reset_all()
    assert win.overrides == {}
    assert win._captures["clear_mask"][0].value() == "R"
    assert win._captures["undo"][0].value() == "Ctrl+Z"
    assert win._captures["undo"][1].value() == "U"


def test_conflict_warning_shown_for_a_duplicate_key(qapp):
    win = ShortcutsWindow({})
    _rebind(win._captures["undo"][0], Qt.Key.Key_R)  # R is clear_mask's default
    assert SHORTCUT_ACTIONS_BY_ID["clear_mask"].label in win._conflict_label.text()


def test_no_conflict_warning_for_a_unique_key(qapp):
    win = ShortcutsWindow({})
    _rebind(win._captures["clear_mask"][0], Qt.Key.Key_F9)
    assert win._conflict_label.text() == ""


def test_cancel_does_not_emit_saved(qapp, monkeypatch):
    win = ShortcutsWindow({})
    _rebind(win._captures["clear_mask"][0], Qt.Key.Key_X)
    received = []
    win.shortcuts_saved.connect(received.append)
    monkeypatch.setattr(win, "reject", lambda: None)
    win.reject()
    assert received == []


def test_is_non_modal(qapp):
    win = ShortcutsWindow({})
    assert win.isModal() is False
