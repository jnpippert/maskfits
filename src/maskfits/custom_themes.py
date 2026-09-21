"""User-created custom themes (Settings -> theme -> "New theme..."), each a
full palette the user picks by hand for a handful of base color roles, with
hover/pressed/contrasting-text variants derived automatically the same way
theme.apply_accent already derives them for a plain accent-color override on
a built-in theme. Persisted as one JSON blob in QSettings (name -> color
dict), which keeps save/rename/delete simple dict operations instead of
juggling QSettings' native nested groups.

Deliberately does not import maskfits.settings (which imports this module,
for Settings.theme validation) - the ORG_NAME/APP_NAME literals are small
enough to duplicate here rather than risk a circular import.
"""

from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import QSettings

from maskfits.theme import Theme, contrasting_text_color, darken, derive_accent_shades, lighten

ORG_NAME = "maskfits"
APP_NAME = "maskfits"

# The base palette a user picks by hand for a custom theme - hover/pressed/
# contrasting-text variants for accent/danger/button_bg are derived, not
# stored (see build_custom_theme).
CUSTOM_THEME_FIELDS = [
    "app_bg", "panel_bg", "panel_border", "text", "text_dim",
    "accent", "danger", "green", "warning", "blue",
    "track", "button_bg", "canvas_bg",
]

RESERVED_NAMES = {"system", "dark", "light"}

# Toolbar-button icon per built-in theme. A custom theme stores its own under
# an "icon" key alongside its colors (not part of CUSTOM_THEME_FIELDS, which
# is colors only): an emoji, or "" for the default - the maskfits logo.
BUILTIN_THEME_ICONS = {"dark": "\U0001F319", "light": "\u2600\ufe0f"}


def _store() -> QSettings:
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, ORG_NAME, APP_NAME)


def list_custom_themes(store: Optional[QSettings] = None) -> dict[str, dict]:
    s = store if store is not None else _store()
    raw = s.value("custom_themes_json", "", type=str)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def get_custom_theme(name: str, store: Optional[QSettings] = None) -> Optional[dict]:
    return list_custom_themes(store).get(name)


def save_custom_theme(name: str, colors: dict, store: Optional[QSettings] = None) -> None:
    """`colors` must have a "mode" ("dark"/"light") key plus every field in
    CUSTOM_THEME_FIELDS. Overwrites any existing custom theme of the same
    name (used for editing, not just creating)."""
    name = name.strip()
    if not name:
        raise ValueError("A theme name is required.")
    if name in RESERVED_NAMES:
        raise ValueError(f"{name!r} is a built-in theme name and can't be used for a custom theme.")
    s = store if store is not None else _store()
    themes = list_custom_themes(s)
    themes[name] = colors
    s.setValue("custom_themes_json", json.dumps(themes))
    s.sync()


def delete_custom_theme(name: str, store: Optional[QSettings] = None) -> None:
    s = store if store is not None else _store()
    themes = list_custom_themes(s)
    themes.pop(name, None)
    s.setValue("custom_themes_json", json.dumps(themes))
    s.sync()


def build_custom_theme(colors: dict) -> Theme:
    """Builds a full Theme from a custom-theme color dict (see
    CUSTOM_THEME_FIELDS) - hover/pressed/contrasting-text variants for
    accent/danger/button_bg are derived automatically, the same way
    theme.apply_accent already derives them for a plain accent-color
    override on a built-in theme, so the user only ever hand-picks one hex
    value per role."""
    mode = colors.get("mode", "dark")
    accent = colors["accent"]
    accent_hover, accent_active = derive_accent_shades(accent, mode)
    danger = colors["danger"]
    danger_hover, _danger_active = derive_accent_shades(danger, mode)
    button_bg = colors["button_bg"]
    button_hover = darken(button_bg, 0.08) if mode == "light" else lighten(button_bg, 0.12)
    return Theme(
        mode=mode,
        app_bg=colors["app_bg"], panel_bg=colors["panel_bg"], panel_border=colors["panel_border"],
        text=colors["text"], text_dim=colors["text_dim"],
        accent=accent, accent_hover=accent_hover, accent_active=accent_active,
        accent_text=contrasting_text_color(accent),
        danger=danger, danger_hover=danger_hover,
        green=colors["green"], warning=colors["warning"], blue=colors["blue"],
        track=colors["track"], button_bg=button_bg, button_hover=button_hover,
        canvas_bg=colors["canvas_bg"],
    )


def theme_cycle_keys(store: Optional[QSettings] = None) -> list[str]:
    """The themes the toolbar button cycles through, in order: built-in dark
    and light, then every custom theme alphabetically (the same order the
    Settings dialog's theme dropdown lists them). "system" isn't included -
    it isn't a theme of its own, just "follow the OS" between dark and light."""
    return ["dark", "light", *sorted(list_custom_themes(store).keys())]


def next_theme_key(current: str, store: Optional[QSettings] = None) -> str:
    """The theme after `current` in the cycle, wrapping around. A `current`
    that isn't in the cycle (e.g. a custom theme deleted since) starts over
    from the first theme."""
    keys = theme_cycle_keys(store)
    index = keys.index(current) if current in keys else -1
    return keys[(index + 1) % len(keys)]


def theme_icon(key: str, store: Optional[QSettings] = None) -> str:
    """The toolbar-button icon for a theme key: an emoji, or "" meaning the
    maskfits logo (the default for a custom theme that never picked one)."""
    if key in BUILTIN_THEME_ICONS:
        return BUILTIN_THEME_ICONS[key]
    icon = (get_custom_theme(key, store) or {}).get("icon", "")
    return icon if isinstance(icon, str) else ""
