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
    # The quick-export ("Export Mask" toolbar button / File menu) output
    # filename - one of three formats, chosen by the current file's own
    # type (see MaskFitsApp.export_mask): a plain single-extension image,
    # a multi-extension FITS, or a cube. $FILENAME (the source file's own
    # stem - see format_mask_filename) is required in all three.
    # $EXT (the extension number) and $SLICE (the slice number) are each
    # optional - the user may leave them out and accept that exporting
    # from different extensions/slices then overwrites the same file -
    # but $EXT only makes sense (and is only allowed) in
    # export_filename_format_multi, and $SLICE only in
    # export_filename_format_cube; neither is allowed in the plain
    # single-extension export_filename_format, and each is also barred
    # from the other's format (see is_valid_export_filename_format's
    # `forbidden`). A ".fits"/".fit"/".fts" suffix is optional in all
    # three - added automatically if missing.
    export_filename_format: str = "mask_$FILENAME"
    export_filename_format_multi: str = "mask_$FILENAME"
    export_filename_format_cube: str = "mask_$FILENAME"
    zoom: float = 1.0
    # Hex string ("#rrggbb") or None to use the built-in crimson accent.
    accent_color: Optional[str] = None
    # Only actions the user has actually rebound, as {action_id: [key, ...]}
    # - see maskfits.shortcuts. Anything not present here still uses that
    # action's own default key(s).
    shortcuts: dict[str, list[str]] = field(default_factory=dict)


def is_valid_export_filename_format(fmt: str, *, forbidden: tuple[str, ...] = ()) -> bool:
    """A quick-export filename format must be non-empty and actually use
    the $FILENAME placeholder - otherwise every exported mask would
    overwrite the same file. `forbidden` names any placeholders that must
    NOT be present (e.g. "$EXT" is meaningless - and barred - in the
    single-extension and cube formats; "$SLICE" is barred in the single-
    extension and multi-extension ones) since each only resolves to
    anything for its own file type - see MaskFitsApp.export_mask - and
    would otherwise pass through the output filename unresolved. $EXT
    itself is optional (not required) in the multi-extension format, and
    $SLICE likewise in the cube one."""
    if not fmt or "$FILENAME" not in fmt:
        return False
    return not any(p in fmt for p in forbidden)


def format_mask_filename(fmt: str, stem: str, *, ext: Optional[int] = None,
                          slice_index: Optional[int] = None) -> str:
    """Resolves a Settings.export_filename_format(_multi/_cube) string into
    an actual output filename: $FILENAME replaced with the source file's
    own stem (see MaskFitsApp._mask_stem), $EXT/$SLICE (when given) with
    the current extension/slice number, and a ".fits" suffix added unless
    the result already ends with a recognized FITS extension - so both
    "mask_$FILENAME" (the default) and "$FILENAME_out.fits" work as typed."""
    name = fmt.replace("$FILENAME", stem)
    if ext is not None:
        name = name.replace("$EXT", str(ext))
    if slice_index is not None:
        name = name.replace("$SLICE", str(slice_index))
    if not name.lower().endswith((".fits", ".fit", ".fts")):
        name += ".fits"
    return name


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
    export_filename_format = s.value("export_filename_format", defaults.export_filename_format, type=str)
    export_filename_format_multi = s.value(
        "export_filename_format_multi", defaults.export_filename_format_multi, type=str
    )
    export_filename_format_cube = s.value(
        "export_filename_format_cube", defaults.export_filename_format_cube, type=str
    )
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
        export_filename_format=(
            export_filename_format
            if is_valid_export_filename_format(export_filename_format, forbidden=("$EXT", "$SLICE"))
            else defaults.export_filename_format
        ),
        export_filename_format_multi=(
            export_filename_format_multi
            if is_valid_export_filename_format(export_filename_format_multi, forbidden=("$SLICE",))
            else defaults.export_filename_format_multi
        ),
        export_filename_format_cube=(
            export_filename_format_cube
            if is_valid_export_filename_format(export_filename_format_cube, forbidden=("$EXT",))
            else defaults.export_filename_format_cube
        ),
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
    s.setValue("export_filename_format", settings.export_filename_format)
    s.setValue("export_filename_format_multi", settings.export_filename_format_multi)
    s.setValue("export_filename_format_cube", settings.export_filename_format_cube)
    s.setValue("zoom", settings.zoom)
    s.setValue("accent_color", settings.accent_color or "")
    s.setValue("shortcuts_json", json.dumps(settings.shortcuts))
    s.sync()
