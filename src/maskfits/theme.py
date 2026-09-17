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

import sys
from dataclasses import dataclass
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
    danger: str
    danger_hover: str
    green: str
    warning: str
    blue: str
    track: str
    button_bg: str
    button_hover: str
    canvas_bg: str


# Crimson (accent/danger) fills are dark enough in both themes that text on
# top of them always needs to stay white, regardless of which theme's `text`
# color is otherwise in effect - so this is a fixed constant, not a Theme field.
ACCENT_TEXT = "#ffffff"

FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else "Helvetica"
FONT_SIZE = 11
FONT_SIZE_SMALL = 10

DARK = Theme(
    mode="dark",
    app_bg="#141415", panel_bg="#1d1d1f", panel_border="#2f2f32",
    text="#eae7e2", text_dim="#96938d",
    accent="#851212", accent_hover="#a3201f", accent_active="#5c0d0d",
    danger="#c1554a", danger_hover="#d16e63",
    green="#22c55e", warning="#e8a33d", blue="#3b82f6",
    track="#3a3a3d", button_bg="#28282b", button_hover="#333336", canvas_bg="#0a0a0b",
)

LIGHT = Theme(
    mode="light",
    app_bg="#eeeeec", panel_bg="#ffffff", panel_border="#d8d8d5",
    text="#1c1c1e", text_dim="#68686c",
    # Same crimson identity as dark mode, but hover/active move darker rather
    # than lighter - on a white panel, a solid fill gets more contrast (and
    # visible hover feedback) by darkening, not brightening.
    accent="#851212", accent_hover="#6b0e0e", accent_active="#4a0a0a",
    danger="#b2453b", danger_hover="#c1554a",
    green="#178a43", warning="#c9781f", blue="#2563eb",
    track="#d3d3d0", button_bg="#e7e7e4", button_hover="#dadad7", canvas_bg="#ffffff",
)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


class ThemeManager(QObject):
    """App-wide singleton owning the currently active Theme.

    Access the active theme via `current_theme()`; switch it via
    `theme_manager().set_mode(...)`, which re-applies the stylesheet/palette
    to the running QApplication and emits `theme_changed` for anything that
    needs to react (custom-painted widgets, the Windows dark-titlebar hook).
    """

    theme_changed = Signal(Theme)

    def __init__(self) -> None:
        super().__init__()
        self._theme = DARK

    @property
    def theme(self) -> Theme:
        return self._theme

    def set_mode(self, mode: str) -> None:
        """Applies the theme's QSS/QPalette to the running QApplication and
        emits theme_changed - deliberately NOT short-circuited when the mode
        is unchanged, since this is also how the very first theme gets
        applied at startup (there's no prior "different" state to compare
        against then), and re-applying identical QSS is cheap."""
        theme = LIGHT if mode == "light" else DARK
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
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(ACCENT_TEXT))
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
        selection-color: {ACCENT_TEXT};
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
        color: {ACCENT_TEXT};
    }}
    QPushButton:checked:hover {{
        background-color: {t.accent_hover};
    }}
    QPushButton[accent="true"] {{
        background-color: {t.accent};
        color: {ACCENT_TEXT};
    }}
    QPushButton[accent="true"]:hover {{
        background-color: {t.accent_hover};
    }}
    QPushButton[danger="true"] {{
        background-color: {t.danger};
        color: {ACCENT_TEXT};
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
        selection-color: {ACCENT_TEXT};
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
        selection-color: {ACCENT_TEXT};
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
        color: {ACCENT_TEXT};
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
        color: {ACCENT_TEXT};
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
