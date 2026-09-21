"""The toolbar button at the top left cycles through every theme - built-in
dark and light, then each custom theme - instead of only flipping between
dark and light, and shows the current theme's icon: an emoji picked in the
theme editor, or the maskfits logo by default."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.custom_themes import (  # noqa: E402
    BUILTIN_THEME_ICONS,
    next_theme_key,
    save_custom_theme,
    theme_cycle_keys,
    theme_icon,
)
from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.theme import theme_manager  # noqa: E402
from maskfits.theme_editor_window import ThemeEditorWindow  # noqa: E402
from maskfits.widgets import THEME_EMOJI_CHOICES, ThemeIconPicker  # noqa: E402

COLORS = {
    "mode": "dark", "app_bg": "#101010", "panel_bg": "#202020", "panel_border": "#303030",
    "text": "#eeeeee", "text_dim": "#999999", "accent": "#00aa66", "danger": "#cc3333",
    "green": "#33cc66", "warning": "#ccaa33", "blue": "#3366cc", "track": "#404040",
    "button_bg": "#282828", "canvas_bg": "#000000",
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_cycle_is_dark_light_then_custom_themes_alphabetically(store):
    save_custom_theme("Zebra", COLORS, store)
    save_custom_theme("Forest", COLORS, store)
    assert theme_cycle_keys(store) == ["dark", "light", "Forest", "Zebra"]


def test_next_theme_key_wraps_around(store):
    save_custom_theme("Forest", COLORS, store)
    assert next_theme_key("dark", store) == "light"
    assert next_theme_key("light", store) == "Forest"
    assert next_theme_key("Forest", store) == "dark"


def test_next_theme_key_starts_over_for_a_deleted_theme(store):
    assert next_theme_key("Gone", store) == "dark"


def test_without_custom_themes_it_still_toggles_dark_and_light(store):
    assert next_theme_key("dark", store) == "light"
    assert next_theme_key("light", store) == "dark"


def test_theme_icon_defaults(store):
    save_custom_theme("Plain", COLORS, store)
    save_custom_theme("Forest", {**COLORS, "icon": "\U0001F332"}, store)
    assert theme_icon("dark", store) == BUILTIN_THEME_ICONS["dark"]
    assert theme_icon("light", store) == BUILTIN_THEME_ICONS["light"]
    assert theme_icon("Plain", store) == ""  # "" means the maskfits logo
    assert theme_icon("Forest", store) == "\U0001F332"


def test_theme_icon_ignores_a_malformed_stored_value(store):
    save_custom_theme("Odd", {**COLORS, "icon": 5}, store)
    assert theme_icon("Odd", store) == ""


def test_clicking_the_toggle_cycles_the_running_app(qapp, monkeypatch):
    from maskfits import custom_themes

    themes = {"Forest": {**COLORS, "icon": "\U0001F332"}}
    monkeypatch.setattr(custom_themes, "list_custom_themes", lambda store=None: themes)
    monkeypatch.setattr("maskfits.gui.get_custom_theme", lambda name, store=None: themes.get(name))

    win = MaskFitsApp([], settings=Settings(theme="dark"))
    assert win.theme_toggle.text() == BUILTIN_THEME_ICONS["dark"]

    win.theme_toggle.click()
    assert win._theme_key == "light"
    assert win.light_mode is True
    assert win.theme_toggle.text() == BUILTIN_THEME_ICONS["light"]

    win.theme_toggle.click()
    assert win._theme_key == "Forest"
    assert win.theme_toggle.text() == "\U0001F332"
    assert theme_manager().custom is not None

    win.theme_toggle.click()
    assert win._theme_key == "dark"
    assert theme_manager().custom is None
    assert win.theme_toggle.text() == BUILTIN_THEME_ICONS["dark"]


def test_a_custom_theme_without_an_icon_shows_the_logo(qapp, monkeypatch):
    themes = {"Plain": dict(COLORS)}
    monkeypatch.setattr("maskfits.gui.get_custom_theme", lambda name, store=None: themes.get(name))
    monkeypatch.setattr("maskfits.custom_themes.list_custom_themes", lambda store=None: themes)

    win = MaskFitsApp([], settings=Settings(theme="Plain"))
    assert win.theme_toggle.text() == ""
    assert not win.theme_toggle.icon().isNull()


def test_system_theme_counts_as_its_resolved_mode_when_cycling(qapp):
    win = MaskFitsApp([], settings=Settings(theme="system"))
    was_light = win.light_mode
    win.theme_toggle.click()
    assert win._theme_key == ("dark" if was_light else "light")


def test_saving_settings_refreshes_the_toggle_icon(qapp):
    win = MaskFitsApp([], settings=Settings(theme="dark"))
    win._apply_settings_live(Settings(theme="light"))
    assert win._theme_key == "light"
    assert win.theme_toggle.text() == BUILTIN_THEME_ICONS["light"]


def test_the_editor_saves_the_picked_icon(qapp, store, monkeypatch):
    saved = {}
    monkeypatch.setattr("maskfits.theme_editor_window.save_custom_theme",
                        lambda name, colors, s=None: saved.update({name: colors}))
    monkeypatch.setattr("maskfits.theme_editor_window.list_custom_themes", lambda s=None: {})

    editor = ThemeEditorWindow()
    editor._name_entry.setText("Forest")
    editor._icon_picker._on_picked("\U0001F332")
    editor._save()

    assert saved["Forest"]["icon"] == "\U0001F332"


def test_the_editor_defaults_to_the_logo_icon(qapp):
    editor = ThemeEditorWindow()
    assert editor.icon == ""
    assert editor._icon_picker.value() == ""


def test_icon_picker_emits_and_shows_the_choice(qapp):
    picker = ThemeIconPicker("")
    received = []
    picker.iconChanged.connect(received.append)
    picker._on_picked(THEME_EMOJI_CHOICES[0])
    assert received == [THEME_EMOJI_CHOICES[0]]
    assert picker.text() == THEME_EMOJI_CHOICES[0]
    assert picker.value() == THEME_EMOJI_CHOICES[0]

    picker._on_picked("")
    assert picker.text() == ""
    assert not picker.icon().isNull()


def test_emoji_choices_are_unique():
    assert len(THEME_EMOJI_CHOICES) == len(set(THEME_EMOJI_CHOICES))
