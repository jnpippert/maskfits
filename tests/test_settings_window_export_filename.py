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
    win._reset_export_filename_format("")
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


# ------------------------------------------------------ multi-ext/cube rows


def test_multi_and_cube_rows_default_to_the_settings_values(qapp):
    win = SettingsWindow(Settings(
        export_filename_format_multi="custom_$FILENAME_$EXT",
        export_filename_format_cube="custom_$FILENAME_$SLICE",
    ))
    assert win._export_filename_entries["_multi"].text() == "custom_$FILENAME_$EXT"
    assert win._export_filename_entries["_cube"].text() == "custom_$FILENAME_$SLICE"


def test_multi_row_hint_shows_an_ext_worked_example(qapp):
    win = SettingsWindow(Settings())
    entry = win._export_filename_entries["_multi"]
    entry.setText("mask_$FILENAME_ext$EXT")
    assert win._export_filename_hints["_multi"].text() == "e.g. mask_myimage_ext1.fits"


def test_cube_row_hint_shows_a_slice_worked_example(qapp):
    win = SettingsWindow(Settings())
    entry = win._export_filename_entries["_cube"]
    entry.setText("mask_$FILENAME_slice$SLICE")
    assert win._export_filename_hints["_cube"].text() == "e.g. mask_myimage_slice2.fits"


def test_multi_row_warns_when_ext_placeholder_missing(qapp):
    win = SettingsWindow(Settings())
    win._export_filename_entries["_multi"].setText("mask_$FILENAME")
    assert win._export_filename_hints["_multi"].text() == "must include $FILENAME and $EXT"


def test_cube_row_warns_when_slice_placeholder_missing(qapp):
    win = SettingsWindow(Settings())
    win._export_filename_entries["_cube"].setText("mask_$FILENAME")
    assert win._export_filename_hints["_cube"].text() == "must include $FILENAME and $SLICE"


def test_multi_row_reset_restores_the_default(qapp):
    win = SettingsWindow(Settings(export_filename_format_multi="custom_$FILENAME_$EXT"))
    win._reset_export_filename_format("_multi")
    assert win._export_filename_entries["_multi"].text() == "mask_$FILENAME_ext$EXT"


def test_cube_row_reset_restores_the_default(qapp):
    win = SettingsWindow(Settings(export_filename_format_cube="custom_$FILENAME_$SLICE"))
    win._reset_export_filename_format("_cube")
    assert win._export_filename_entries["_cube"].text() == "mask_$FILENAME_slice$SLICE"


def test_save_persists_all_three_formats(qapp, monkeypatch):
    saved = []
    monkeypatch.setattr("maskfits.settings_window.save_settings", lambda s, store=None: saved.append(s))

    win = SettingsWindow(Settings())
    win._export_filename_entries["_multi"].setText("m_$FILENAME_$EXT")
    win._export_filename_entries["_cube"].setText("c_$FILENAME_$SLICE")
    win._save()

    assert saved[0].export_filename_format_multi == "m_$FILENAME_$EXT"
    assert saved[0].export_filename_format_cube == "c_$FILENAME_$SLICE"


def test_save_is_refused_when_multi_format_missing_ext(qapp, monkeypatch):
    saved = []
    monkeypatch.setattr("maskfits.settings_window.save_settings", lambda s, store=None: saved.append(s))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    win = SettingsWindow(Settings())
    win._export_filename_entries["_multi"].setText("mask_$FILENAME")
    win._save()

    assert saved == []


def test_save_is_refused_when_cube_format_missing_slice(qapp, monkeypatch):
    saved = []
    monkeypatch.setattr("maskfits.settings_window.save_settings", lambda s, store=None: saved.append(s))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    win = SettingsWindow(Settings())
    win._export_filename_entries["_cube"].setText("mask_$FILENAME")
    win._save()

    assert saved == []
