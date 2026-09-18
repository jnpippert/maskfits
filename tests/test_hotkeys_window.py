import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from maskfits.hotkeys_window import MOUSE_ENTRIES, HotkeysWindow  # noqa: E402
from maskfits.shortcuts import SHORTCUT_ACTIONS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _label_texts(win):
    return [lbl.text() for lbl in win.findChildren(QLabel)]


def test_shows_every_mouse_entry(qapp):
    win = HotkeysWindow({})
    texts = _label_texts(win)
    for keys, desc in MOUSE_ENTRIES:
        assert keys in texts
        assert desc in texts


def test_shows_every_keyboard_action_label(qapp):
    win = HotkeysWindow({})
    texts = _label_texts(win)
    for action in SHORTCUT_ACTIONS:
        assert action.label in texts


def test_shows_default_keys_when_no_overrides(qapp):
    win = HotkeysWindow({})
    texts = _label_texts(win)
    assert "Ctrl+Z / U" in texts  # undo's two defaults joined


def test_shows_rebound_key_instead_of_default(qapp):
    from maskfits.shortcuts import effective_keys

    overrides = {"clear_mask": ["X"]}
    win = HotkeysWindow(overrides)
    texts = _label_texts(win)
    # The window must render whatever effective_keys() itself resolves to
    # (the live source of truth) rather than any hardcoded default text.
    assert " / ".join(effective_keys("clear_mask", overrides)) in texts
    assert "X" in texts
    assert "R" not in texts  # clear_mask's now-overridden default


def test_is_non_modal(qapp):
    win = HotkeysWindow({})
    assert win.isModal() is False
