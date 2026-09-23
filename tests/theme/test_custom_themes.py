import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402

from maskfits.custom_themes import (  # noqa: E402
    CUSTOM_THEME_FIELDS,
    build_custom_theme,
    delete_custom_theme,
    get_custom_theme,
    list_custom_themes,
    save_custom_theme,
)
from maskfits.theme import DARK  # noqa: E402

SAMPLE_COLORS = {"mode": "dark", **{f: getattr(DARK, f) for f in CUSTOM_THEME_FIELDS}}


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_no_custom_themes_initially(store):
    assert list_custom_themes(store) == {}
    assert get_custom_theme("anything", store) is None


def test_save_then_get_round_trips(store):
    save_custom_theme("Ocean", SAMPLE_COLORS, store)
    assert get_custom_theme("Ocean", store) == SAMPLE_COLORS
    assert "Ocean" in list_custom_themes(store)


def test_save_overwrites_same_name(store):
    save_custom_theme("Ocean", SAMPLE_COLORS, store)
    edited = {**SAMPLE_COLORS, "accent": "#123456"}
    save_custom_theme("Ocean", edited, store)
    assert get_custom_theme("Ocean", store)["accent"] == "#123456"
    assert len(list_custom_themes(store)) == 1


def test_delete_removes_it(store):
    save_custom_theme("Ocean", SAMPLE_COLORS, store)
    delete_custom_theme("Ocean", store)
    assert get_custom_theme("Ocean", store) is None


def test_delete_of_unknown_name_does_not_raise(store):
    delete_custom_theme("NeverExisted", store)  # should just no-op


@pytest.mark.parametrize("reserved", ["system", "dark", "light"])
def test_save_rejects_reserved_names(store, reserved):
    with pytest.raises(ValueError):
        save_custom_theme(reserved, SAMPLE_COLORS, store)


def test_save_rejects_empty_name(store):
    with pytest.raises(ValueError):
        save_custom_theme("   ", SAMPLE_COLORS, store)


def test_build_custom_theme_uses_given_base_colors():
    theme = build_custom_theme(SAMPLE_COLORS)
    assert theme.mode == "dark"
    assert theme.app_bg == DARK.app_bg
    assert theme.accent == DARK.accent


def test_build_custom_theme_derives_hover_active_and_text():
    colors = {**SAMPLE_COLORS, "accent": "#00aabb"}
    theme = build_custom_theme(colors)
    assert theme.accent == "#00aabb"
    # hover/active are derived, not copied verbatim from the base accent.
    assert theme.accent_hover != theme.accent
    assert theme.accent_active != theme.accent
    assert theme.accent_text in ("#000000", "#ffffff")


def test_build_custom_theme_light_mode_darkens_button_hover():
    colors = {**SAMPLE_COLORS, "mode": "light", "button_bg": "#e0e0e0"}
    theme = build_custom_theme(colors)
    assert theme.mode == "light"
    # light mode should darken for hover, not lighten (matches the built-in
    # LIGHT theme's own accent_hover/accent_active relationship).
    from maskfits.theme import hex_to_rgb

    base_luma = sum(hex_to_rgb(colors["button_bg"]))
    hover_luma = sum(hex_to_rgb(theme.button_hover))
    assert hover_luma < base_luma
