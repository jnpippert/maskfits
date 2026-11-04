"""Coverage for the Settings window's tabs (View / Style / File / Tools)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from maskfits.settings import Settings  # noqa: E402
from maskfits.settings_window import SETTINGS_TABS, SettingsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp):
    window = SettingsWindow(Settings())
    window.show()
    qapp.processEvents()
    return window


def _labels(page):
    return {label.text().rstrip(":") for label in page.findChildren(QLabel) if label.text().endswith(":")}


def test_at_most_five_tabs(dialog):
    assert SETTINGS_TABS == ("view", "style", "file", "tools")
    assert len(SETTINGS_TABS) <= 5
    assert [b.text() for b in dialog.tab_control._buttons.values()] == ["View", "Style", "File", "Tools"]


@pytest.mark.parametrize("tab, expected", [
    ("view", {"Default Colormap", "Default Cut Levels", "Default Scale", "Default Zoom",
              "Default Bin Factor", "Default Smooth Sigma"}),
    ("style", {"Theme", "Accent Color", "Mask Colors", "Zero/NaN Pixel Color", "Center Cross Color"}),
    ("file", {"Export Directory", "Quick Export Filename", "Quick Export Filename (multi-ext)",
              "Quick Export Filename (cube)"}),
    ("tools", {"Default Mode", "Keyboard Shortcuts"}),
])
def test_each_setting_is_on_its_tab(dialog, tab, expected):
    assert _labels(dialog.tab_pages[tab]) == expected


def test_view_tab_is_shown_first(dialog):
    assert dialog.pages.currentWidget() is dialog.tab_pages["view"]


def test_clicking_a_tab_shows_its_page_without_resizing(dialog, qapp):
    size = dialog.size()
    for key in SETTINGS_TABS:
        dialog.tab_control._buttons[key].click()
        qapp.processEvents()
        assert dialog.pages.currentWidget() is dialog.tab_pages[key]
        assert dialog.size() == size


def test_settings_on_every_tab_are_saved(dialog):
    dialog._blank_picker.setValue("#3355ff")  # Style tab
    dialog._export_filename_entries[""].setText("out_$FILENAME")  # File tab
    dialog._on_zoom_changed(2.0)  # View tab
    dialog._on_mode_changed("line")  # Tools tab
    settings = dialog._current_settings()
    assert settings.blank_color == "#3355ff"
    assert settings.export_filename_format == "out_$FILENAME"
    assert settings.zoom == 2.0
    assert settings.mode == "line"


def test_window_is_much_shorter_than_one_long_page(dialog):
    assert dialog.height() < 520
