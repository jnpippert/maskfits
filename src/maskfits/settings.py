"""Persisted user-default settings, edited via Help -> Settings and loaded at
startup by run_gui(). CLI flags (-z/-m) still override the persisted default
for that one session only - see cli.py/gui.run_gui.

Backed by QSettings in a plain INI file (QSettings.Format.IniFormat) rather
than the platform-native registry/plist, so the file is easy to inspect by
hand and easy to point at a throwaway path in tests (pass an explicit `store`
to load_settings/save_settings instead of the real user-wide one).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import QSettings

from maskfits.colormaps import COLORMAP_NAMES
from maskfits.custom_themes import list_custom_themes
from maskfits.imagedata import PERCENTILE_PRESETS, STRETCH_NAMES
from maskfits.shortcuts import sanitize_overrides

ORG_NAME = "maskfits"
APP_NAME = "maskfits"

THEME_CHOICES = ["system", "dark", "light"]
MODE_CHOICES = ["ellipse", "line"]
EXPORT_DIR_CHOICES = ["file_parent", "cwd"]
# Cut-levels algorithm (the app's own "Scale" menu: Min/Max, ZScale, then a
# percentile preset per PERCENTILE_PRESETS) - distinct from `scale`, which is
# the linear/log/asinh stretch *function* applied on top of those cuts.
STRETCH_CHOICES = ["minmax", "zscale"] + [f"pct{p}" for p in PERCENTILE_PRESETS]


@dataclass
class Settings:
    theme: str = "system"
    mode: str = "ellipse"
    colormap: str = "Grayscale"
    scale: str = "linear"
    stretch: str = "zscale"
    bin_enabled: bool = False
    bin_factor: int = 3
    smooth_enabled: bool = False
    smooth_sigma: float = 3.0
    export_dir: str = "file_parent"
    zoom: float = 1.0
    # Hex string ("#rrggbb") or None to use the built-in crimson accent.
    accent_color: Optional[str] = None
    # Only actions the user has actually rebound, as {action_id: [key, ...]}
    # - see maskfits.shortcuts. Anything not present here still uses that
    # action's own default key(s).
    shortcuts: dict[str, list[str]] = field(default_factory=dict)


def _backing_store() -> QSettings:
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, ORG_NAME, APP_NAME)


def load_settings(store: Optional[QSettings] = None) -> Settings:
    s = store if store is not None else _backing_store()
    defaults = Settings()

    theme = s.value("theme", defaults.theme, type=str)
    mode = s.value("mode", defaults.mode, type=str)
    colormap = s.value("colormap", defaults.colormap, type=str)
    scale = s.value("scale", defaults.scale, type=str)
    stretch = s.value("stretch", defaults.stretch, type=str)
    export_dir = s.value("export_dir", defaults.export_dir, type=str)
    accent = s.value("accent_color", "", type=str)
    # A custom theme's name is a valid `theme` value too, alongside the
    # three built-ins - checked here (rather than a static THEME_CHOICES
    # list) so a persisted custom-theme selection survives a reload; a name
    # that no longer exists (theme deleted since) falls back to "system"
    # the same as any other invalid value.
    valid_themes = THEME_CHOICES + list(list_custom_themes(s).keys())

    return Settings(
        theme=theme if theme in valid_themes else defaults.theme,
        mode=mode if mode in MODE_CHOICES else defaults.mode,
        colormap=colormap if colormap in COLORMAP_NAMES else defaults.colormap,
        scale=scale if scale in STRETCH_NAMES else defaults.scale,
        stretch=stretch if stretch in STRETCH_CHOICES else defaults.stretch,
        bin_enabled=s.value("bin_enabled", defaults.bin_enabled, type=bool),
        bin_factor=s.value("bin_factor", defaults.bin_factor, type=int),
        smooth_enabled=s.value("smooth_enabled", defaults.smooth_enabled, type=bool),
        smooth_sigma=s.value("smooth_sigma", defaults.smooth_sigma, type=float),
        export_dir=export_dir if export_dir in EXPORT_DIR_CHOICES else defaults.export_dir,
        zoom=s.value("zoom", defaults.zoom, type=float),
        accent_color=accent or None,
        shortcuts=_load_shortcuts(s),
    )


def _load_shortcuts(s: QSettings) -> dict[str, list[str]]:
    raw = s.value("shortcuts_json", "", type=str)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return sanitize_overrides(parsed)


def save_settings(settings: Settings, store: Optional[QSettings] = None) -> None:
    s = store if store is not None else _backing_store()
    s.setValue("theme", settings.theme)
    s.setValue("mode", settings.mode)
    s.setValue("colormap", settings.colormap)
    s.setValue("scale", settings.scale)
    s.setValue("stretch", settings.stretch)
    s.setValue("bin_enabled", settings.bin_enabled)
    s.setValue("bin_factor", settings.bin_factor)
    s.setValue("smooth_enabled", settings.smooth_enabled)
    s.setValue("smooth_sigma", settings.smooth_sigma)
    s.setValue("export_dir", settings.export_dir)
    s.setValue("zoom", settings.zoom)
    s.setValue("accent_color", settings.accent_color or "")
    s.setValue("shortcuts_json", json.dumps(settings.shortcuts))
    s.sync()
