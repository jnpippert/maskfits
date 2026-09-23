import pytest

PySide6 = pytest.importorskip("PySide6")

from maskfits.theme import (  # noqa: E402
    DARK,
    LIGHT,
    apply_accent,
    contrasting_text_color,
    darken,
    derive_accent_shades,
    hex_to_rgb,
    lighten,
)


def test_lighten_moves_toward_white():
    color = "#404040"
    lighter = lighten(color, 0.5)
    assert sum(hex_to_rgb(lighter)) > sum(hex_to_rgb(color))


def test_darken_moves_toward_black():
    color = "#c0c0c0"
    darker = darken(color, 0.5)
    assert sum(hex_to_rgb(darker)) < sum(hex_to_rgb(color))


def test_lighten_clamps_at_white():
    assert lighten("#ffffff", 0.9) == "#ffffff"


def test_darken_clamps_at_black():
    assert darken("#000000", 0.9) == "#000000"


def test_contrasting_text_color_picks_black_for_light_fill():
    assert contrasting_text_color("#ffff00") == "#000000"


def test_contrasting_text_color_picks_white_for_dark_fill():
    assert contrasting_text_color("#111111") == "#ffffff"


def test_derive_accent_shades_dark_mode_brightens_hover_darkens_active():
    accent = "#851212"
    hover, active = derive_accent_shades(accent, "dark")
    assert sum(hex_to_rgb(hover)) > sum(hex_to_rgb(accent))
    assert sum(hex_to_rgb(active)) < sum(hex_to_rgb(accent))


def test_derive_accent_shades_light_mode_darkens_both():
    accent = "#851212"
    hover, active = derive_accent_shades(accent, "light")
    assert sum(hex_to_rgb(hover)) < sum(hex_to_rgb(accent))
    assert sum(hex_to_rgb(active)) < sum(hex_to_rgb(hover))


def test_apply_accent_none_returns_theme_unchanged():
    assert apply_accent(DARK, None) is DARK
    assert apply_accent(LIGHT, "") is LIGHT


def test_apply_accent_overrides_accent_and_derives_variants():
    themed = apply_accent(DARK, "#00aabb")
    assert themed.accent == "#00aabb"
    assert themed.accent_hover != DARK.accent_hover
    assert themed.accent_active != DARK.accent_active
    # Only the accent-related fields should change - everything else stays
    # identical to the base theme.
    assert themed.app_bg == DARK.app_bg
    assert themed.panel_bg == DARK.panel_bg
    assert themed.green == DARK.green


def test_apply_accent_picks_readable_text_for_a_light_accent():
    themed = apply_accent(DARK, "#ffff00")
    assert themed.accent_text == "#000000"
