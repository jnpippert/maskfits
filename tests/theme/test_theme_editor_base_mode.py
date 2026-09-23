"""Regression coverage for a real bug: the theme editor's Base Mode control
(Dark/Light) changed self.mode but never touched any color field, so
clicking it appeared to do nothing - build_custom_theme only uses `mode` to
derive accent/button hover shades, not the base palette itself. Switching
Base Mode now resets every color field to that built-in theme's defaults, so
it actually works as a starting point to customize from."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.custom_themes import CUSTOM_THEME_FIELDS  # noqa: E402
from maskfits.theme import DARK, LIGHT  # noqa: E402
from maskfits.theme_editor_window import ThemeEditorWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_switching_to_light_resets_every_field_to_lights_defaults(qapp):
    editor = ThemeEditorWindow()  # starts on the "dark" base
    editor._on_mode_changed("light")

    for field in CUSTOM_THEME_FIELDS:
        expected = getattr(LIGHT, field)
        assert editor.colors[field] == expected
        assert editor._pickers[field].value() == expected


def test_switching_back_to_dark_resets_every_field_to_darks_defaults(qapp):
    editor = ThemeEditorWindow()
    editor._on_mode_changed("light")
    editor._on_color_changed("app_bg", "#123456")  # a manual tweak
    editor._on_mode_changed("dark")

    for field in CUSTOM_THEME_FIELDS:
        expected = getattr(DARK, field)
        assert editor.colors[field] == expected
        assert editor._pickers[field].value() == expected


def test_mode_switch_updates_the_live_preview(qapp):
    from maskfits.theme import theme_manager

    editor = ThemeEditorWindow()
    editor._on_mode_changed("light")

    custom = theme_manager().custom
    assert custom is not None
    assert custom.app_bg == LIGHT.app_bg
    assert custom.text == LIGHT.text
