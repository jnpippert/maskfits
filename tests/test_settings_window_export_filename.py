"""Coverage for the Settings dialog's "quick export filename" field - what
Settings.export_filename_format uses to name the file the "Export Mask"
toolbar button/File menu action writes (see gui.export_mask and
maskfits.settings.format_mask_filename)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from maskfits.settings import Settings, format_mask_filename  # noqa: E402
from maskfits.settings_window import SettingsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_defaults_to_the_settings_value(qapp):
    win = SettingsWindow(Settings(export_filename_format="custom_$FILENAME"))
    assert win._export_filename_entry.text() == "custom_$FILENAME"


def test_typing_updates_the_hint_with_a_worked_example(qapp):
    win = SettingsWindow(Settings())
    win._export_filename_entry.setText("$FILENAME_masked")
    assert win._export_filename_hint.text() == f"e.g. {format_mask_filename('$FILENAME_masked', 'myimage')}"


def test_hint_warns_when_the_placeholder_is_missing(qapp):
    win = SettingsWindow(Settings())
    win._export_filename_entry.setText("no_placeholder")
    assert win._export_filename_hint.text() == "must include $FILENAME"


def test_reset_restores_the_default(qapp):
    win = SettingsWindow(Settings(export_filename_format="custom_$FILENAME"))
    win._reset_export_filename_format()
    assert win._export_filename_entry.text() == "mask_$FILENAME"


def test_save_persists_a_valid_custom_format(qapp, monkeypatch):
    saved = []
    monkeypatch.setattr("maskfits.settings_window.save_settings", lambda s, store=None: saved.append(s))

    win = SettingsWindow(Settings())
    win._export_filename_entry.setText("custom_$FILENAME")
    received = []
    win.settings_saved.connect(received.append)
    win._save()

    assert saved and saved[0].export_filename_format == "custom_$FILENAME"
    assert received and received[0].export_filename_format == "custom_$FILENAME"


def test_save_is_refused_without_the_placeholder(qapp, monkeypatch):
    saved = []
    monkeypatch.setattr("maskfits.settings_window.save_settings", lambda s, store=None: saved.append(s))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    win = SettingsWindow(Settings())
    win._export_filename_entry.setText("no_placeholder")
    received = []
    win.settings_saved.connect(received.append)
    win._save()

    assert saved == []
    assert received == []
    assert win._saved is False


def test_save_is_refused_when_emptied(qapp, monkeypatch):
    saved = []
    monkeypatch.setattr("maskfits.settings_window.save_settings", lambda s, store=None: saved.append(s))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    win = SettingsWindow(Settings())
    win._export_filename_entry.setText("")
    received = []
    win.settings_saved.connect(received.append)
    win._save()

    assert saved == []
    assert received == []
