"""Closing a sub-window of Settings (theme editor - including after deleting
a theme -, mask colors, keyboard shortcuts) hands focus back to Settings
instead of letting the main window come to the front."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import maskfits.custom_themes as ct  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.settings_window import SettingsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def theme_store(tmp_path, monkeypatch):
    ini = str(tmp_path / "themes.ini")
    monkeypatch.setattr(ct, "_store", lambda: QSettings(ini, QSettings.Format.IniFormat))
    ct.save_custom_theme("tmpdel", {"mode": "dark", "icon": "", **{f: "#334455" for f in ct.CUSTOM_THEME_FIELDS}})


def _record_activation(win, monkeypatch):
    calls = []
    monkeypatch.setattr(win, "activateWindow", lambda: calls.append("activate"))
    return calls


def test_focus_returns_after_deleting_a_theme(qapp, theme_store, monkeypatch):
    win = SettingsWindow(Settings())
    win.show()
    win._reload_theme_combo(select="tmpdel")
    win._edit_theme()
    editor = win._theme_editor
    assert editor is not None
    calls = _record_activation(win, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    editor._delete()
    assert win._theme_editor is None
    assert calls == ["activate"]
    assert "tmpdel" not in ct.list_custom_themes()
    win.close()


@pytest.mark.parametrize("opener, attr", [
    ("_edit_mask_colors", "_mask_colors_window"),
    ("_edit_shortcuts", "_shortcuts_window"),
    ("_new_theme", "_theme_editor"),
])
def test_focus_returns_after_closing_a_sub_window(qapp, theme_store, monkeypatch, opener, attr):
    win = SettingsWindow(Settings())
    win.show()
    getattr(win, opener)()
    child = getattr(win, attr)
    calls = _record_activation(win, monkeypatch)
    child.reject()
    assert getattr(win, attr) is None
    assert calls == ["activate"]
    win.close()


def test_no_focus_grab_when_settings_is_hidden(qapp, theme_store, monkeypatch):
    win = SettingsWindow(Settings())
    win.show()
    win._edit_shortcuts()
    child = win._shortcuts_window
    win.hide()
    calls = _record_activation(win, monkeypatch)
    child.reject()
    assert calls == []
