"""Color palette, fonts, and live theming for the maskfits Qt GUI.

Two `Theme` instances live here - DARK (default) and LIGHT. Unlike the old
Tkinter app (which baked colors into each widget's canvas draw calls at
construction time, so a theme switch had to destroy and rebuild the whole
widget tree), Qt widgets restyle live: `theme_manager().set_mode(...)` swaps
the active `Theme`, re-applies a QSS stylesheet + QPalette to the whole
QApplication, and emits `theme_changed` so any custom-painted widget (which
reads `current_theme()` fresh inside its own paintEvent, never caching colors
at construction) can repaint itself.
"""

from __future__ import annotations

import colorsys
import sys
from dataclasses import dataclass, replace
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QPalette


@dataclass(frozen=True)
class Theme:
    mode: str
    app_bg: str
    panel_bg: str
    panel_border: str
    text: str
    text_dim: str
    accent: str
    accent_hover: str
    accent_active: str
    # Text color for solid accent fills (buttons, menu selection, ...) -
    # white for the default crimson accent in both themes, but a *custom*
    # accent (see apply_accent) may be light enough that white text on it
    # would be illegible, so this is a per-Theme field rather than the fixed
    # constant it used to be.
    accent_text: str
    danger: str
    danger_hover: str
    green: str
    warning: str
    blue: str
    track: str
    button_bg: str
    button_hover: str
    canvas_bg: str


FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else "Helvetica"
FONT_SIZE = 11
FONT_SIZE_SMALL = 10

# Accent hover/active/text below are derived the same way a user's own
# custom accent override is (see derive_accent_shades/contrasting_text_color
# further down) - dark mode brightens on hover and darkens further on press,
# light mode darkens for both; #adb18d is light enough that black text reads
# better on it than white.
DARK = Theme(
    mode="dark",
    app_bg="#141415", panel_bg="#1d1d1f", panel_border="#2f2f32",
    text="#eae7e2", text_dim="#96938d",
    accent="#adb18d", accent_hover="#bcbfa2", accent_active="#8e9365", accent_text="#000000",
    danger="#c1554a", danger_hover="#d16e63",
    green="#22c55e", warning="#e8a33d", blue="#3b82f6",
    track="#3a3a3d", button_bg="#28282b", button_hover="#333336", canvas_bg="#0a0a0b",
)

LIGHT = Theme(
    mode="light",
    app_bg="#eeeeec", panel_bg="#ffffff", panel_border="#d8d8d5",
    text="#1c1c1e", text_dim="#68686c",
    accent="#adb18d", accent_hover="#9da176", accent_active="#80845a", accent_text="#000000",
    danger="#b2453b", danger_hover="#c1554a",
    green="#178a43", warning="#c9781f", blue="#2563eb",
    track="#d3d3d0", button_bg="#e7e7e4", button_hover="#dadad7", canvas_bg="#ffffff",
)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def detect_os_light_mode() -> bool:
    """Best-effort read of the OS-wide light/dark preference, used to pick the
    app's initial theme (Settings.theme == "system") so it opens matching the
    desktop instead of always defaulting to dark. Falls back to dark (returns
    False) wherever this can't be determined - an unrecognized platform, or
    the lookup failing for any reason (missing registry key, sandboxed
    `defaults`, etc.)."""
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return bool(value)
        if sys.platform == "darwin":
            import subprocess

            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=1,
            )
            return result.returncode != 0
    except Exception:
        pass
    return False


def _clamp01(x: float) -> float:
    return max(0.0, min(x, 1.0))


def lighten(color: str, amount: float) -> str:
    """Moves `color` toward white by `amount` (0-1) in HSL lightness."""
    r, g, b = hex_to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    r2, g2, b2 = colorsys.hls_to_rgb(h, _clamp01(l + amount * (1 - l)), s)
    return f"#{round(r2 * 255):02x}{round(g2 * 255):02x}{round(b2 * 255):02x}"


def darken(color: str, amount: float) -> str:
    """Moves `color` toward black by `amount` (0-1) in HSL lightness."""
    r, g, b = hex_to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    r2, g2, b2 = colorsys.hls_to_rgb(h, _clamp01(l * (1 - amount)), s)
    return f"#{round(r2 * 255):02x}{round(g2 * 255):02x}{round(b2 * 255):02x}"


def contrasting_text_color(color: str) -> str:
    """Picks black or white, whichever reads better on a solid `color` fill
    (simple relative-luminance threshold - good enough for picking a legible
    label color, not full WCAG contrast-ratio math)."""
    r, g, b = hex_to_rgb(color)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luma > 150 else "#ffffff"


def derive_accent_shades(color: str, mode: str) -> tuple[str, str]:
    """(hover, active) shades for a custom accent color, matching the
    relationship the built-in crimson accent has to its own hover/active in
    each theme: dark mode brightens on hover and darkens further on press;
    light mode darkens for both (brightening a solid fill on a white panel
    loses contrast instead of gaining it)."""
    if mode == "light":
        return darken(color, 0.12), darken(color, 0.30)
    return lighten(color, 0.18), darken(color, 0.22)


def apply_accent(theme: Theme, accent: Optional[str]) -> Theme:
    """Returns `theme` with its accent/accent_hover/accent_active/accent_text
    replaced by a user-chosen accent color, or `theme` unchanged if `accent`
    is None (no override - use the built-in crimson)."""
    if not accent:
        return theme
    hover, active = derive_accent_shades(accent, theme.mode)
    return replace(theme, accent=accent, accent_hover=hover, accent_active=active,
                    accent_text=contrasting_text_color(accent))


class ThemeManager(QObject):
    """App-wide singleton owning the currently active Theme.

    Access the active theme via `current_theme()`; switch it via
    `theme_manager().set_mode(...)` (built-in dark/light, optionally with
    `set_accent(...)` layered on top) or `set_custom(...)` (a fully
    user-defined Theme - see maskfits.custom_themes). Either re-applies the
    stylesheet/palette to the running QApplication and emits theme_changed
    for anything that needs to react (custom-painted widgets, the Windows
    dark-titlebar hook). Deliberately knows nothing about *how* a custom
    Theme is built or persisted (see maskfits.custom_themes) - it just holds
    whichever Theme instance it's given.
    """

    theme_changed = Signal(Theme)

    def __init__(self) -> None:
        super().__init__()
        self._mode = "dark"
        self._accent: Optional[str] = None
        self._custom: Optional[Theme] = None
        self._theme = DARK

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def mode(self) -> str:
        """The last built-in mode selected via set_mode - kept even while a
        custom theme (set_custom) is active, so switching back to a built-in
        (or reverting a preview) doesn't need the caller to remember it."""
        return self._mode

    @property
    def accent(self) -> Optional[str]:
        return self._accent

    @property
    def custom(self) -> Optional[Theme]:
        return self._custom

    def set_mode(self, mode: str) -> None:
        """Selects a built-in theme ('dark'/'light') and clears any custom
        theme (set_custom) override - deliberately NOT short-circuited when
        the mode is unchanged, since this is also how the very first theme
        gets applied at startup (there's no prior "different" state to
        compare against then), and re-applying identical QSS is cheap."""
        self._mode = mode
        self._custom = None
        self._apply()

    def set_accent(self, accent: Optional[str]) -> None:
        """Overrides the current built-in theme's accent color (see
        apply_accent); None reverts to the built-in crimson. No-op while a
        custom theme is active - it already bakes its own accent in, see
        set_custom - but is remembered and reapplied as soon as set_mode
        switches back to a built-in theme."""
        self._accent = accent
        self._apply()

    def set_custom(self, theme: Theme) -> None:
        """Activates a fully custom Theme (see
        custom_themes.build_custom_theme), overriding set_mode/set_accent
        until the next set_mode call."""
        self._custom = theme
        self._apply()

    def _apply(self) -> None:
        if self._custom is not None:
            theme = self._custom
        else:
            base = LIGHT if self._mode == "light" else DARK
            theme = apply_accent(base, self._accent)
        self._theme = theme
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_qss(theme))
            app.setPalette(build_palette(theme))
        self.theme_changed.emit(theme)


_manager: Optional[ThemeManager] = None


def theme_manager() -> ThemeManager:
    global _manager
    if _manager is None:
        _manager = ThemeManager()
    return _manager


def current_theme() -> Theme:
    return theme_manager().theme


def build_palette(theme: Theme) -> QPalette:
    """A QPalette matching the Theme, for native chrome that reads palette
    roles rather than QSS (dialogs, some menu internals, disabled-state
    fallbacks)."""
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(theme.app_bg))
    p.setColor(QPalette.ColorRole.WindowText, QColor(theme.text))
    p.setColor(QPalette.ColorRole.Base, QColor(theme.panel_bg))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(theme.button_bg))
    p.setColor(QPalette.ColorRole.Text, QColor(theme.text))
    p.setColor(QPalette.ColorRole.Button, QColor(theme.button_bg))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(theme.text))
    p.setColor(QPalette.ColorRole.Highlight, QColor(theme.accent))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(theme.accent_text))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme.panel_bg))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(theme.text))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.text_dim))
    disabled_text = QColor(theme.text_dim)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
    return p


def build_qss(theme: Theme) -> str:
    """The app-wide stylesheet, applied via QApplication.setStyleSheet().

    Unlike Tkinter's tk.Menu (confirmed this session to fully ignore bg/fg
    config on Windows, for both the menu bar strip and its dropdown popups),
    Qt's QMenuBar/QMenu fully respect QSS - the entire reason for this port.
    """
    t = theme
    return f"""
    /* Transparent by default - only the actual top-level windows get a
    solid fill (below), plus whatever specific widget classes declare their
    own background further down (RoundedPanel, QPushButton, QLineEdit, ...).
    The old version instead gave EVERY QWidget an opaque app_bg background,
    which meant every plain QWidget used purely for layout (a toolbar
    "chunk" wrapper, a row container, tool_options_container, ...) silently
    painted its own app_bg rectangle over whatever panel (panel_bg) it
    actually sat inside of - a recurring visible-rectangle bug fixed one
    widget at a time before this rule was inverted to fix the whole class
    of it at once. */
    QWidget {{
        background-color: transparent;
        color: {t.text};
        font-family: "{FONT_FAMILY}";
        font-size: {FONT_SIZE}pt;
        selection-background-color: {t.accent};
        selection-color: {t.accent_text};
    }}

    QMainWindow, QDialog {{
        background-color: {t.app_bg};
    }}

    QToolTip {{
        background-color: {t.panel_bg};
        color: {t.text};
        border: 1px solid {t.panel_border};
        padding: 4px 6px;
    }}

    /* -------------------------------------------------------------- panels */

    RoundedPanel, .RoundedPanel {{
        background-color: {t.panel_bg};
        border: 1px solid {t.panel_border};
        border-radius: 14px;
    }}

    /* Plain QWidget containers nested inside a RoundedPanel (the toolbar's
    FlowLayout and its two rows, its per-control "chunk" wrappers) must stay
    transparent - otherwise the global QWidget background rule below paints
    each of them its own app_bg rectangle, visibly darker than the panel_bg
    panel they sit inside of. */
    #flowLayout, #flowRow, #toolbarChunk {{
        background-color: transparent;
    }}

    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: transparent;
    }}
    QScrollBar:vertical {{
        background-color: transparent;
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {t.button_bg};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {t.button_hover};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background-color: transparent;
    }}

    /* ------------------------------------------------------------- labels */

    QLabel {{
        background-color: transparent;
    }}
    QLabel[dim="true"] {{
        color: {t.text_dim};
    }}
    /* Overrides [dim="true"] below it (same specificity, later wins) - the
    status label stays dim="true" always and toggles this on top of it for a
    quick visual "this succeeded" cue (e.g. after a mask export) without the
    user having to read the message. */
    QLabel[state="success"] {{
        color: {t.green};
    }}

    /* ------------------------------------------------------------ buttons */

    QPushButton {{
        background-color: {t.button_bg};
        color: {t.text};
        border: none;
        border-radius: 10px;
        padding: 5px 11px;
    }}
    QPushButton:hover {{
        background-color: {t.button_hover};
    }}
    QPushButton:disabled {{
        color: {t.text_dim};
    }}
    QPushButton:checked {{
        background-color: {t.accent};
        color: {t.accent_text};
    }}
    QPushButton:checked:hover {{
        background-color: {t.accent_hover};
    }}
    QPushButton[accent="true"] {{
        background-color: {t.accent};
        color: {t.accent_text};
    }}
    QPushButton[accent="true"]:hover {{
        background-color: {t.accent_hover};
    }}
    QPushButton[danger="true"] {{
        background-color: {t.danger};
        color: {t.accent_text};
    }}
    QPushButton[danger="true"]:hover {{
        background-color: {t.danger_hover};
    }}
    QPushButton[flat="true"] {{
        background-color: transparent;
        padding: 2px;
    }}
    QPushButton[flat="true"]:hover {{
        background-color: {t.button_hover};
    }}

    /* ------------------------------------------------------------- inputs */

    QLineEdit {{
        background-color: {t.button_bg};
        color: {t.text};
        border: 1px solid {t.panel_border};
        border-radius: 6px;
        padding: 3px 6px;
        selection-background-color: {t.accent};
        selection-color: {t.accent_text};
    }}
    QLineEdit:focus {{
        border: 1px solid {t.accent};
    }}
    QLineEdit:disabled {{
        color: {t.text_dim};
    }}

    QComboBox {{
        background-color: {t.button_bg};
        color: {t.text};
        border: 1px solid {t.panel_border};
        border-radius: 6px;
        padding: 3px 8px;
    }}
    QComboBox:disabled {{
        color: {t.text_dim};
    }}
    QComboBox QAbstractItemView {{
        background-color: {t.panel_bg};
        color: {t.text};
        border: 1px solid {t.panel_border};
        selection-background-color: {t.accent};
        selection-color: {t.accent_text};
    }}

    QCheckBox {{
        background-color: transparent;
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border-radius: 4px;
        border: 1px solid {t.panel_border};
        background: {t.button_bg};
    }}
    QCheckBox::indicator:checked {{
        background: {t.accent};
        border: 1px solid {t.accent};
    }}

    /* RoundSlider is fully custom-painted (see widgets.py) - no QSS needed
    here at all; QSlider's own native complex-control rendering couldn't be
    fully suppressed via stylesheet, which is exactly why it isn't used. */

    /* --------------------------------------------------------- menu bar */

    QMenuBar {{
        background-color: {t.panel_bg};
        color: {t.text};
        border-bottom: 1px solid {t.panel_border};
        padding: 2px;
    }}
    QMenuBar::item {{
        background-color: transparent;
        padding: 4px 10px;
        border-radius: 6px;
    }}
    QMenuBar::item:selected {{
        background-color: {t.button_hover};
    }}
    QMenuBar::item:pressed {{
        background-color: {t.accent};
        color: {t.accent_text};
    }}

    QMenu {{
        background-color: {t.panel_bg};
        color: {t.text};
        border: 1px solid {t.panel_border};
        padding: 4px;
    }}
    QMenu::item {{
        padding: 5px 24px 5px 12px;
        border-radius: 6px;
    }}
    QMenu::item:selected {{
        background-color: {t.accent};
        color: {t.accent_text};
    }}
    QMenu::item:disabled {{
        color: {t.text_dim};
    }}
    QMenu::separator {{
        height: 1px;
        background: {t.panel_border};
        margin: 4px 6px;
    }}
    QMenu::indicator {{
        width: 13px;
        height: 13px;
    }}

    /* ------------------------------------------------------------ dialog */

    QMessageBox, QFileDialog {{
        background-color: {t.app_bg};
    }}
    """
