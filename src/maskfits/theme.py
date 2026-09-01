"""Color palette and fonts shared across the maskfits GUI.

Two palettes live here - DARK (default) and LIGHT. The module-level names
below (APP_BG, PANEL_BG, ...) always resolve to whichever one is active.
gui.py/widgets.py/cuts_dialog.py/colormaps.py all import these names directly
(`from maskfits.theme import PANEL_BG`), which copies the value into each of
those modules' own namespaces at import time - so switching themes can't just
mutate this module's globals, it has to reach into those already-imported
modules and patch the same names there too. See set_theme().
"""

import sys

DARK = dict(
    APP_BG="#141415",
    PANEL_BG="#1d1d1f",
    PANEL_BORDER="#2f2f32",
    TEXT="#eae7e2",
    TEXT_DIM="#96938d",
    ACCENT="#851212",
    ACCENT_HOVER="#a3201f",
    ACCENT_ACTIVE="#5c0d0d",
    DANGER="#c1554a",
    DANGER_HOVER="#d16e63",
    GREEN="#22c55e",
    TRACK="#3a3a3d",
    BUTTON_BG="#28282b",
    BUTTON_HOVER="#333336",
)

LIGHT = dict(
    APP_BG="#eeeeec",
    PANEL_BG="#ffffff",
    PANEL_BORDER="#d8d8d5",
    TEXT="#1c1c1e",
    TEXT_DIM="#68686c",
    # Same crimson identity as dark mode, but hover/active move darker rather
    # than lighter - on a white panel, a solid fill gets more contrast (and
    # visible hover feedback) by darkening, not brightening.
    ACCENT="#851212",
    ACCENT_HOVER="#6b0e0e",
    ACCENT_ACTIVE="#4a0a0a",
    DANGER="#b2453b",
    DANGER_HOVER="#c1554a",
    GREEN="#178a43",
    TRACK="#d3d3d0",
    BUTTON_BG="#e7e7e4",
    BUTTON_HOVER="#dadad7",
)

# The image viewport stays dark in both themes (like most image editors keep
# their canvas neutral regardless of chrome theme) - it's not part of either
# palette above.
CANVAS_BG = "#0a0a0b"

# Crimson is dark enough in both themes that text on top of an accent- (or
# danger-) filled surface always needs to stay white, regardless of which
# theme's TEXT color is otherwise in effect - so this is a fixed constant,
# not part of either palette.
ACCENT_TEXT = "#ffffff"

FONT_FAMILY = "Helvetica"
FONT = (FONT_FAMILY, 11)
FONT_SMALL = (FONT_FAMILY, 10)
FONT_LABEL = (FONT_FAMILY, 10)

MODE = "dark"
globals().update(DARK)

# Every module that does `from maskfits.theme import <color name>` needs its
# copy patched too when the theme changes.
_THEMED_MODULES = ("maskfits.gui", "maskfits.widgets", "maskfits.cuts_dialog", "maskfits.colormaps")


def set_theme(mode: str) -> None:
    """Switch the active palette ("dark" or "light") and propagate the new
    color values into every module that already imported the old ones.

    This only updates the plain module-level names - it doesn't touch any
    widget already on screen (those baked in whatever color was active when
    they were built). Pairing this with a full widget-tree rebuild is what
    actually re-themes the app; see MaskFitsApp._rebuild_ui.
    """
    global MODE
    palette = LIGHT if mode == "light" else DARK
    MODE = "light" if mode == "light" else "dark"
    globals().update(palette)
    for modname in _THEMED_MODULES:
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        for name, value in palette.items():
            if hasattr(mod, name):
                setattr(mod, name, value)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
