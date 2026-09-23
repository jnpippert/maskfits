"""Qt GUI: view FITS images and paint circular/elliptical or line (satellite trail) masks."""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional

import numpy as np
from astropy.io import fits
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QColor,
    QIcon,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

from maskfits.automask_window import AutoMaskWindow
from maskfits.binning import bin_func, bin_mask, unbin_mask
from maskfits.colormaps import (
    COLORMAP_LUTS,
    COLORMAP_NAMES,
    ISOPY_NAME,
    auto_mask_tint_for,
    build_isopy_lut,
    mask_tint_for,
)
from maskfits.custom_themes import build_custom_theme, get_custom_theme, next_theme_key, theme_icon
from maskfits.cuts_histogram import CutsHistogram
from maskfits.hotkeys_window import HotkeysWindow
from maskfits.imagedata import (
    PERCENTILE_PRESETS,
    STRETCH_NAMES,
    STRETCHES,
    FitsImage,
    gaussian_smooth,
    isopy_cuts_and_stops,
    list_image_extensions,
    load_fits_image,
    minmax_cuts,
    percentile_cuts,
    safe_span,
    zscale_cuts,
)
from maskfits.layouts import FlowLayout
from maskfits.masking import (
    ellipse_mask,
    ellipse_polygon_points,
    extend_line_to_borders,
    extend_ray_to_border,
    line_mask,
)
from maskfits.settings import Settings, format_mask_filename, load_settings
from maskfits.settings_window import SettingsWindow
from maskfits.shortcuts import SHORTCUT_ACTIONS, effective_keys
from maskfits.theme import current_theme, detect_os_light_mode, theme_manager
from maskfits.update_check import UpdateCheckWorker, check_for_updates, get_clone_url
from maskfits.update_dialog import show_update_dialog
from maskfits.widgets import RoundButton, RoundedPanel, RoundSlider, SegmentedControl, SidebarToggle, ThemeToggle

MAG_SIZE = 31
MAG_BLOCK = 6
PAN_W = PAN_H = MAG_SIZE * MAG_BLOCK
MAX_SHAPE_SIZE = 500
RADIUS_MIN = 0.5  # small enough to mask a single pixel
SIDEBAR_W = 340  # fixed - see MaskFitsApp._toggle_sidebar's docstring for why
SIDEBAR_TOGGLE_W = 10
ZOOM_STEP = 1.25
ZOOM_MULT_MIN = 0.5
ZOOM_MULT_MAX = 100.0

LINE_STYLES = [
    ("segment", "Segment"),
    ("arrow", "Arrow"),
    ("line", "Line"),
]

SCALE_OPTIONS = [(name, name.capitalize()) for name in STRETCH_NAMES]

ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
ICON_ICO_PATH = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")


def _set_windows_app_id() -> None:
    """Give the process its own Application User Model ID on Windows.

    Without this, Windows treats the process as plain python.exe/pythonw.exe
    for taskbar purposes. Must run before QApplication is constructed.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("maskfits.app")
    except Exception:
        pass


def _app_icon() -> QIcon:
    path = ICON_ICO_PATH if (sys.platform == "win32" and os.path.exists(ICON_ICO_PATH)) else ICON_PATH
    return QIcon(path) if os.path.exists(path) else QIcon()


def _set_windows_titlebar_dark(widget: QWidget, dark: bool) -> None:
    """Match the native window chrome (title bar, drawn by the Windows
    compositor) to the app's current theme - a plain Qt window keeps a
    plain white title bar on Windows 10/11 regardless of the OS dark-mode
    setting unless this DWM attribute is set explicitly.

    A Qt top-level widget's winId() IS the real HWND directly - no need for
    the GetParent() indirection a Tkinter root window needed.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = int(widget.winId())
        value = ctypes.c_int(1 if dark else 0)
        for attr in (20, 19):
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))
            if result == 0:
                break
    except Exception:
        pass


class Entry:
    """A single loaded (or not-yet-loaded) FITS file slot."""

    def __init__(self, path: Optional[str] = None, ext: int = 0):
        self.path = path
        self.image: Optional[FitsImage] = None
        # Which HDU to load - defaults to 0 (or the CLI's -x/--extension,
        # for whichever files were passed on the command line), but
        # ensure_loaded falls back to the first HDU that actually has 2D
        # image data if this one doesn't (e.g. an empty primary HDU with the
        # real data in extension 1+, or a requested extension this
        # particular file doesn't have at all).
        self.ext: int = ext
        self.available_extensions: list[tuple[int, str]] = []
        self.lowcut = 0.0
        self.highcut = 1.0
        # Smoothing and binning are independent and composable: original_data
        # is the one pristine array (captured the first time either is turned
        # on), and up to three more arrays cache each combination actually
        # used - smoothed-only, binned-only, and smoothed-then-binned - keyed
        # on the sigma/factor they were built at.
        self.original_data: Optional[np.ndarray] = None

        self.is_smoothed = False
        self.smooth_sigma: Optional[float] = None
        self.smoothed_cache: Optional[np.ndarray] = None
        self._smoothed_cache_sigma: Optional[float] = None

        self.is_binned = False
        self.bin_factor: Optional[int] = None
        self.binned_cache: Optional[np.ndarray] = None
        self._binned_cache_factor: Optional[int] = None

        self.smoothed_binned_cache: Optional[np.ndarray] = None
        self._smoothed_binned_cache_sigma: Optional[float] = None
        self._smoothed_binned_cache_factor: Optional[int] = None

        self.mask_backup: Optional[np.ndarray] = None
        self.mask_dirty: bool = False
        self._binned_mask_cache: Optional[np.ndarray] = None
        self._binned_mask_cache_factor: Optional[int] = None

        self._isopy_cache_image: Optional[FitsImage] = None
        self._isopy_vmin: float = 0.0
        self._isopy_vmax: float = 1.0
        self._isopy_lut: np.ndarray = np.zeros((256, 3), dtype=np.uint8)

        # The user's own lowcut/highcut, remembered from just before IsoPy's
        # fixed cuts last overwrote them - IsoPy's cuts come from its own
        # surface-brightness formula, not user editing (see
        # isopy_cuts_and_lut), so switching to another colormap should bring
        # the user's previous choice back rather than leaving IsoPy's values
        # in place. None whenever this entry isn't currently "inside" an
        # IsoPy view (i.e. nothing pending to restore).
        self._pre_isopy_lowcut: Optional[float] = None
        self._pre_isopy_highcut: Optional[float] = None

    def isopy_cuts_and_lut(self) -> tuple[float, float, np.ndarray]:
        if self.image is not None and self._isopy_cache_image is not self.image:
            vmin, vmax, stops = isopy_cuts_and_stops(self.image.header, self.image.wcs)
            self._isopy_vmin, self._isopy_vmax = vmin, vmax
            self._isopy_lut = build_isopy_lut(stops)
            self._isopy_cache_image = self.image
        return self._isopy_vmin, self._isopy_vmax, self._isopy_lut

    def ensure_loaded(self, stretch: str) -> None:
        if self.image is not None or self.path is None:
            return
        if not self.available_extensions:
            self.available_extensions = list_image_extensions(self.path)
        valid_exts = [i for i, _ in self.available_extensions]
        if valid_exts and self.ext not in valid_exts:
            self.ext = valid_exts[0]
        self.image = load_fits_image(self.path, ext=self.ext)
        self.apply_stretch(stretch)

    def apply_stretch(self, stretch: str) -> None:
        if self.image is None:
            return
        if stretch == "zscale":
            self.lowcut, self.highcut = zscale_cuts(self.image.data)
        elif stretch.startswith("pct"):
            percent = float(stretch[3:])
            self.lowcut, self.highcut = percentile_cuts(self.image.data, percent)
        else:
            self.lowcut, self.highcut = minmax_cuts(self.image.data)


class ImageCanvas(QWidget):
    """The main zoomable/pannable image view - a thin event-routing shell;
    all actual state and math lives on MaskFitsApp (see img_to_canvas/
    canvas_to_img/render/_on_button/etc.), mirroring how the old Tkinter
    version kept everything on the app object and the canvas was just a
    dumb drawing surface."""

    def __init__(self, app: "MaskFitsApp", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app = app
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        theme_manager().theme_changed.connect(lambda _t: self.update())

    def resizeEvent(self, event) -> None:  # noqa: N802
        self.app.canvas_w, self.app.canvas_h = self.width(), self.height()
        if self.app.image is not None:
            self.app.fit_zoom = self.app._compute_fit_zoom()
        self.app.render()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(current_theme().canvas_bg))
        if self.app._base_pixmap is not None:
            painter.drawPixmap(self.app._base_pos, self.app._base_pixmap)
        self.app._paint_overlays(painter)
        # _paint_overlays may have left a translucent fill brush set (for the
        # ellipse hover preview) - without clearing it, drawRect() below
        # would fill this ENTIRE border rectangle with that brush, not just
        # stroke its outline, tinting the whole canvas the moment the mouse
        # first hovers over it.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor(current_theme().panel_border))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.setFocus()
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.app._on_pan_start(pos.x(), pos.y())
            else:
                self.app._on_button(pos.x(), pos.y(), erase=False)
        elif event.button() == Qt.MouseButton.RightButton:
            self.app._on_button(pos.x(), pos.y(), erase=True)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.app._on_middle_click()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        self.app._on_motion(pos.x(), pos.y())
        buttons = event.buttons()
        if buttons & Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.app._on_pan_drag(pos.x(), pos.y())
            else:
                self.app._on_drag(pos.x(), pos.y(), erase=False)
        elif buttons & Qt.MouseButton.RightButton:
            self.app._on_drag(pos.x(), pos.y(), erase=True)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.app._pan_drag = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        pos = event.position()
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        self.app._zoom_at(pos.x(), pos.y(), factor)


class MagnifierWidget(QWidget):
    """Small fixed-size panel showing a pixel-block-zoomed crop around the
    cursor, plus x/y/RA/DEC/value readouts live above/below it in the
    sidebar (see MaskFitsApp._build_sidebar)."""

    def __init__(self, app: "MaskFitsApp", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app = app
        self.setFixedSize(PAN_W, PAN_H)
        theme_manager().theme_changed.connect(lambda _t: self.update())

    def paintEvent(self, event) -> None:  # noqa: N802
        theme = current_theme()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.canvas_bg))
        app = self.app
        image = app.image
        if image is None or app._cursor_img_pos is None:
            painter.setPen(QColor(theme.panel_border))
            step = 10
            for x in range(0, self.width(), step):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), step):
                painter.drawLine(0, y, self.width(), y)
            return

        ny, nx = image.data.shape
        half = MAG_SIZE // 2
        cx_i = int(round(app._cursor_img_pos[0]))
        cy_i = int(round(app._cursor_img_pos[1]))
        x0, y0 = cx_i - half, cy_i - half

        crop = np.full((MAG_SIZE, MAG_SIZE), np.nan, dtype=np.float64)
        mask_crop = np.zeros((MAG_SIZE, MAG_SIZE), dtype=bool)
        sx0, sx1 = max(x0, 0), min(x0 + MAG_SIZE, nx)
        sy0, sy1 = max(y0, 0), min(y0 + MAG_SIZE, ny)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image.data[sy0:sy1, sx0:sx1]
            mask_crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image.mask[sy0:sy1, sx0:sx1]

        entry = app.entry
        span = safe_span(entry.lowcut, entry.highcut)
        norm = np.clip((crop - entry.lowcut) / span, 0, 1)
        norm = np.where(np.isnan(crop), 0.12, norm)
        rgb = app._scale_and_color(norm)
        app._tint_masked(rgb, mask_crop)
        rgb = np.ascontiguousarray(rgb[::-1])  # same bottom-up flip as render()

        square = min(self.width(), self.height())
        block = max(square // MAG_SIZE, 1)
        disp = block * MAG_SIZE
        qimg = QImage(rgb.data, MAG_SIZE, MAG_SIZE, 3 * MAG_SIZE, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            disp, disp, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
        ox, oy = (self.width() - disp) // 2, (self.height() - disp) // 2
        painter.drawPixmap(ox, oy, pixmap)

        cxp, cyp = ox + half * block, oy + half * block
        painter.setPen(QPen(QColor(theme.green), 2))
        painter.drawRect(cxp, cyp, block, block)


MODE_FLAGS = {"s": "line", "e": "ellipse"}


class MaskFitsApp(QMainWindow):
    def _size_to_screen(self, preferred_w: int, preferred_h: int) -> None:
        """Open windowed at `preferred_w`x`preferred_h`, but never larger
        than the screen's available space (e.g. a Full HD 1920x1080 display
        can't fit a window sized for a MacBook Pro's much taller default) -
        shrunk to fit with a margin and centered, rather than just clamped
        to the raw screen size which would touch the edges/taskbar."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(preferred_w, preferred_h)
            return
        available = screen.availableGeometry()
        margin = 60
        floor_w, floor_h = 640, 480
        width = min(preferred_w, max(available.width() - margin, floor_w))
        height = min(preferred_h, max(available.height() - margin, floor_h))
        # Without an explicit minimum, QMainWindow refuses to shrink below
        # whatever its layout naturally needs (sidebar + toolbar + panels),
        # which can exceed a laptop's screen width. That's harmless on its
        # own, but a window manager that later tries to maximize/resize the
        # window to the real screen size and gets a bigger buffer back than
        # it asked for can treat it as a protocol violation (observed as a
        # Wayland "xdg_surface buffer does not match configured maximized
        # state" fatal error under WSLg) - so cap the minimum explicitly to
        # whatever we're willing to shrink to, rather than leave it to the
        # layout's own (possibly larger) size hint.
        self.setMinimumSize(floor_w, floor_h)
        self.resize(width, height)
        x = available.x() + (available.width() - width) // 2
        y = available.y() + (available.height() - height) // 2
        self.move(x, y)

    def __init__(self, paths: list[str], settings: Optional[Settings] = None, *,
                 extension: Optional[int] = None):
        super().__init__()
        self.setWindowTitle("maskfits")
        self._size_to_screen(1400, 980)
        self.setWindowIcon(_app_icon())

        self.settings = settings if settings is not None else load_settings()

        self._apply_theme_setting(self.settings.theme, self.settings.accent_color)
        theme_manager().theme_changed.connect(self._on_theme_changed)

        # -x/--extension (CLI-only, like vmin/vmax - not a Settings field,
        # since which HDU to open is a per-invocation detail, not a
        # persistent preference). A file that doesn't have this extension
        # falls back to its own first valid one instead of failing - see
        # Entry.ensure_loaded - so this is a safe no-op for a single-
        # extension file.
        start_ext = extension if extension is not None else 0
        self.entries: list[Entry] = [Entry(p, ext=start_ext) for p in paths] or [Entry(None)]
        self.index = 0

        self.stretch = self.settings.stretch
        self.scale_function = self.settings.scale
        self.colormap = self.settings.colormap
        self.invert_colormap = False
        self.mask_alpha = 100
        self.show_center_cross = False
        self.tool = self.settings.mode
        self.ellipticity = 0
        self.angle = 0
        self.radius = 40.0
        self.thickness = 15
        self.line_style = "segment"
        # Session defaults from Settings (Help -> Settings): applied to each
        # entry the first time it's displayed with no binning/smoothing of
        # its own yet - see load_current(). None means "off by default".
        self.export_dir_mode = self.settings.export_dir
        self._default_bin_factor = self.settings.bin_factor if self.settings.bin_enabled else None
        self._default_smooth_sigma = self.settings.smooth_sigma if self.settings.smooth_enabled else None
        # Whichever of these the active tool currently has built (see
        # _rebuild_tool_options) - lets _sync_tool_option_widgets() update a
        # slider's displayed value in place after a programmatic change,
        # without tearing down/rebuilding the whole tool-options panel just
        # for that (see its own docstring for why that rebuild is unsafe to
        # use as a generic refresh).
        self._radius_slider: Optional[RoundSlider] = None
        self._ellipticity_slider: Optional[RoundSlider] = None
        self._angle_slider: Optional[RoundSlider] = None
        self._thickness_slider: Optional[RoundSlider] = None
        self._line_style_control: Optional[SegmentedControl] = None

        self.fit_zoom = 1.0
        # Reset to 1.0 by reset_zoom() during load_current() below, then set
        # to the real Settings-driven default afterward - see the end of
        # this __init__.
        self.zoom_mult = 1.0
        self.view_cx = 0.0
        self.view_cy = 0.0
        self.canvas_w = 1
        self.canvas_h = 1
        self._cursor_img_pos: Optional[tuple[float, float]] = None
        self._cursor_canvas_pos: Optional[tuple[float, float]] = None

        self._pan_drag: Optional[tuple[float, float, float, float]] = None
        self._line_anchor: Optional[tuple[float, float]] = None
        self._line_anchor_erase = False
        self._undo: Optional[tuple[int, np.ndarray]] = None
        self._redo: Optional[tuple[int, np.ndarray]] = None
        self.sidebar_collapsed = False

        self._base_pixmap: Optional[QPixmap] = None
        self._base_pos = QPointF(0, 0)

        # Auto Mask: a pending preview overlay (boolean array matching
        # _auto_mask_entry.image.data's current shape) shown on top of the
        # image + manual mask, but not written into the real mask until the
        # user confirms in AutoMaskWindow.
        self._auto_mask_window: Optional[AutoMaskWindow] = None
        self._settings_window: Optional[SettingsWindow] = None
        self._hotkeys_window: Optional[HotkeysWindow] = None
        self._update_check_worker: Optional[UpdateCheckWorker] = None
        self._auto_mask_entry: Optional[Entry] = None
        self._auto_mask_preview: Optional[np.ndarray] = None

        self._build_ui()
        self._build_shortcuts()
        self._rebuild_tool_options()
        self.load_current(reset_view=True)
        # load_current(reset_view=True) just ran reset_zoom(), which always
        # sets zoom_mult back to 1.0 - so the Settings-driven default zoom
        # (like a CLI -z override in run_gui) has to be (re-)applied after
        # it, not before.
        self.zoom_mult = max(min(self.settings.zoom, ZOOM_MULT_MAX), ZOOM_MULT_MIN)
        self._update_zoom_label()
        self.render()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        _set_windows_titlebar_dark(self, dark=not self.light_mode)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._auto_mask_window is not None:
            self._auto_mask_window.close()
        if self._settings_window is not None:
            self._settings_window.close()
        if self._hotkeys_window is not None:
            self._hotkeys_window.close()
        super().closeEvent(event)

    # ---------------------------------------------------------------- theme

    def _apply_theme_setting(self, theme_key: str, accent: Optional[str]) -> None:
        """Resolves a Settings.theme value ("system"/"dark"/"light", or a
        custom theme's name - see maskfits.custom_themes) to an actual Theme
        and applies it. Used at startup and whenever Settings are saved with
        a different theme (see _apply_settings_live). Sets self.light_mode
        BEFORE touching theme_manager() - its theme_changed signal fires
        synchronously and _on_theme_changed reads self.light_mode, so it
        must already be correct by the time that happens."""
        self._theme_key = theme_key
        custom_colors = None if theme_key in ("system", "dark", "light") else get_custom_theme(theme_key)
        if custom_colors is not None:
            theme_obj = build_custom_theme(custom_colors)
            self.light_mode = theme_obj.mode == "light"
            theme_manager().set_custom(theme_obj)
        else:
            self.light_mode = detect_os_light_mode() if theme_key == "system" else (theme_key == "light")
            theme_manager().set_mode("light" if self.light_mode else "dark")
            theme_manager().set_accent(accent)

    def _cycle_theme(self) -> None:
        """Toolbar button: switches to the next theme - dark, light, then
        every custom theme - and wraps around. "system" (follow the OS) isn't
        a step in the cycle, so it counts as whichever of dark/light it
        currently resolves to."""
        current = self._theme_key
        if current == "system":
            current = "light" if self.light_mode else "dark"
        self._apply_theme_setting(next_theme_key(current), self.settings.accent_color)

    def _refresh_theme_toggle(self) -> None:
        if not hasattr(self, "theme_toggle"):
            return
        key = self._theme_key
        if key == "system":
            key = "light" if self.light_mode else "dark"
        self.theme_toggle.set_icon_spec(theme_icon(key))
        name = key.capitalize() if key in ("dark", "light") else key
        self.theme_toggle.setToolTip(f"Theme: {name} (Click For The Next One)")

    def _preview_theme_icon(self, spec: str) -> None:
        """Live icon preview from an open Settings/theme-editor window -
        e.g. cycling the theme combo, or editing a theme's icon, before
        either is saved. Bypasses _refresh_theme_toggle's self._theme_key
        lookup entirely (that only updates on an actual commit - see
        _apply_settings_live/_on_theme_renamed), which is exactly why this
        preview used to go stale while Settings was open: the colors
        updated live via theme_manager()'s own signal, but the icon lookup
        never saw anything happen until Save."""
        if hasattr(self, "theme_toggle"):
            self.theme_toggle.set_icon_spec(spec)

    def _on_theme_changed(self, theme) -> None:
        self.light_mode = theme.mode == "light"
        _set_windows_titlebar_dark(self, dark=not self.light_mode)
        self._refresh_theme_toggle()
        self.render()

    # ---------------------------------------------------------------- menu

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open...", self.open_files)
        file_menu.addAction("Export Mask", self.export_mask)
        file_menu.addAction("Save Mask As...", self.export_mask_as)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        mode_menu = menubar.addMenu("Mode")
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        self._mode_actions: dict[str, QAction] = {}
        for value, label in (("ellipse", "Ellipse Mask"), ("line", "Line Mask (satellite trail)")):
            action = QAction(label, self, checkable=True)
            action.setChecked(value == self.tool)
            action.triggered.connect(lambda _checked=False, v=value: self.set_tool(v))
            mode_group.addAction(action)
            mode_menu.addAction(action)
            self._mode_actions[value] = action

        scale_menu = menubar.addMenu("Scale")
        scale_menu.addAction("Min/Max", lambda: self.set_stretch("minmax"))
        scale_menu.addAction("ZScale", lambda: self.set_stretch("zscale"))
        scale_menu.addSeparator()
        for percent in PERCENTILE_PRESETS:
            scale_menu.addAction(f"{percent}%", lambda p=percent: self.set_stretch(f"pct{p}"))
        scale_menu.addSeparator()
        scale_group = QActionGroup(self)
        scale_group.setExclusive(True)
        self._scale_actions: dict[str, QAction] = {}
        for value, label in SCALE_OPTIONS:
            action = QAction(label, self, checkable=True)
            action.setChecked(value == self.scale_function)
            action.triggered.connect(lambda _checked=False, v=value: self.set_scale_function(v))
            scale_group.addAction(action)
            scale_menu.addAction(action)
            self._scale_actions[value] = action
        scale_menu.addSeparator()
        scale_menu.addAction("Reset", self.reset_scale)

        color_menu = menubar.addMenu("Color")
        color_group = QActionGroup(self)
        color_group.setExclusive(True)
        self._color_actions: dict[str, QAction] = {}
        for name in COLORMAP_NAMES:
            action = QAction(name, self, checkable=True)
            action.setChecked(name == self.colormap)
            action.triggered.connect(lambda _checked=False, v=name: self.set_colormap(v))
            color_group.addAction(action)
            color_menu.addAction(action)
            self._color_actions[name] = action
        color_menu.addSeparator()
        self._invert_action = QAction("Invert Colormap", self, checkable=True)
        self._invert_action.setChecked(self.invert_colormap)
        self._invert_action.triggered.connect(self._on_invert_action)
        color_menu.addAction(self._invert_action)

        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About", self._show_help)
        settings_action = help_menu.addAction("Settings...", self._open_settings)
        update_action = help_menu.addAction("Check New Repo Version...", self._check_repo_version)
        hotkeys_action = help_menu.addAction("Hotkeys", self._show_hotkeys)
        # On macOS, Qt auto-detects action text like "About"/"Settings..."
        # and silently relocates it out of whatever menu it was added to
        # into the application's own menu (top-left, next to the app name)
        # instead of leaving it under Help. NoRole pins all four to
        # stay exactly where they're placed, on every platform.
        about_action.setMenuRole(QAction.MenuRole.NoRole)
        settings_action.setMenuRole(QAction.MenuRole.NoRole)
        update_action.setMenuRole(QAction.MenuRole.NoRole)
        hotkeys_action.setMenuRole(QAction.MenuRole.NoRole)
        if sys.platform == "darwin":
            # ...but macOS users do expect Settings in the native app-menu
            # spot too (Cmd+,) - a second action, explicitly given
            # PreferencesRole, gets pulled out of Help into that spot by Qt,
            # so it's reachable from both places rather than picking one.
            mac_settings_action = QAction("Settings...", self)
            mac_settings_action.triggered.connect(self._open_settings)
            mac_settings_action.setMenuRole(QAction.MenuRole.PreferencesRole)
            mac_settings_action.setShortcut(QKeySequence("Ctrl+,"))
            help_menu.addAction(mac_settings_action)

    def _open_settings(self) -> None:
        if self._settings_window is not None:
            self._settings_window.raise_()
            self._settings_window.activateWindow()
            return
        dialog = SettingsWindow(self.settings, self)
        dialog.settings_saved.connect(self._apply_settings_live)
        dialog.theme_renamed.connect(self._on_theme_renamed)
        dialog.theme_icon_previewed.connect(self._preview_theme_icon)
        dialog.finished.connect(self._clear_settings_window)
        self._settings_window = dialog
        dialog.show()

    def _clear_settings_window(self, _result=None) -> None:
        self._settings_window = None

    def _on_theme_renamed(self, old: str, new: str) -> None:
        """A custom theme was renamed in the theme editor (which already
        moved its stored data and any persisted Settings selection) - keep
        this window's own idea of the current theme pointing at it."""
        if self._theme_key == old:
            self._theme_key = new
            self._refresh_theme_toggle()
        if self.settings.theme == old:
            self.settings.theme = new

    def _apply_settings_live(self, settings: Settings) -> None:
        """Settings were just saved (persisted to disk by the dialog itself)
        - also apply the live-appliable ones to the currently running
        session, so Save visibly does something instead of only affecting
        the next launch. The theme itself was already live-previewed by the
        dialog as the user edited it; Save just means "keep what's showing"."""
        self.settings = settings
        self._theme_key = settings.theme
        self._refresh_theme_toggle()
        self.export_dir_mode = settings.export_dir
        self._default_bin_factor = settings.bin_factor if settings.bin_enabled else None
        self._default_smooth_sigma = settings.smooth_sigma if settings.smooth_enabled else None
        self.set_colormap(settings.colormap)
        self.set_stretch(settings.stretch)
        self.set_scale_function(settings.scale)
        self.set_tool(settings.mode)
        self.zoom_mult = max(min(settings.zoom, ZOOM_MULT_MAX), ZOOM_MULT_MIN)
        self._update_zoom_label()
        self._rebuild_shortcuts()
        self.render()

    def _show_hotkeys(self) -> None:
        if self._hotkeys_window is not None:
            self._hotkeys_window.raise_()
            self._hotkeys_window.activateWindow()
            return
        dialog = HotkeysWindow(self.settings.shortcuts, self)
        dialog.finished.connect(self._clear_hotkeys_window)
        self._hotkeys_window = dialog
        dialog.show()

    def _clear_hotkeys_window(self, _result=None) -> None:
        self._hotkeys_window = None

    def _check_repo_version(self) -> None:
        """Compares the local git clone against its GitHub remote (git
        fetch + rev-list - see update_check.check_for_updates) and reports
        the result - for contributors/devs working from a checkout, not the
        general "is my install up to date" question (see
        start_background_update_check, which checks PyPI instead). Only
        meaningful for an actual git clone; anything else (offline, no git,
        a plain `pip install maskfits` with no .git) just gets a clear
        message instead of crashing."""
        self.setCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            available, message = check_for_updates()
        finally:
            self.unsetCursor()
        title = "maskfits - New Commits Available" if available else "maskfits"
        clone_url = get_clone_url()
        command = f"git clone {clone_url}" if clone_url else None
        show_update_dialog(self, title, message, available=available, command=command)

    def start_background_update_check(self) -> None:
        """Silently checks PyPI for a newer release once on startup (see
        run_gui) and only pops a dialog for an actual major-version bump -
        never for "up to date", offline, or any other non-update outcome,
        so a normal launch stays silent. Works the same way regardless of
        how maskfits was installed (pip or a git clone), unlike the manual
        "Check New Repo Version..." action above, which is git-specific.
        Runs check_for_major_pip_update() (a blocking network call) on a
        daemon thread so a slow or absent network connection can never
        delay startup or freeze the window; UpdateCheckWorker.finished is a
        Qt signal, safe to emit from any thread - Qt auto-queues its
        delivery back onto this (GUI) thread."""
        worker = UpdateCheckWorker()
        worker.finished.connect(self._on_background_update_check)
        self._update_check_worker = worker  # keep alive until it finishes
        threading.Thread(target=worker.run, daemon=True).start()

    def _on_background_update_check(self, available: bool, message: str) -> None:
        self._update_check_worker = None
        if available:
            show_update_dialog(self, "maskfits - Update Available", message, available=True,
                                command="pip install --upgrade maskfits")

    def _show_help(self) -> None:
        # The full hotkey list lives in Help -> Hotkeys instead (a live view
        # against the current Settings-driven bindings, since keyboard
        # shortcuts are rebindable - a hardcoded copy of them here would go
        # stale the moment the user rebinds anything).
        QMessageBox.information(
            self, "maskfits",
            "Ellipse mode: stamp shapes sized by the radius, ellipticity, and angle sliders\n"
            "(ellipticity 0 is a circle)\n\n"
            "Satellite mode styles:\n"
            "  Segment - click a start point, click an end point\n"
            "  Arrow   - click start, click a second point; the trail extends\n"
            "            past it to the image border\n"
            "  Line    - click two points; the trail extends to both borders\n\n"
            "See Help > Hotkeys for the full list of mouse and keyboard controls.",
        )

    # -------------------------------------------------------------- layout

    def _build_ui(self) -> None:
        self._build_menu()

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(6)

        toolbar_panel = RoundedPanel(central, mode="hug")
        root_layout.addWidget(toolbar_panel)
        self._build_toolbar(toolbar_panel.inner)

        body_widget = QWidget(central)
        root_layout.addWidget(body_widget, 1)
        body = QHBoxLayout(body_widget)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.sidebar_container = QWidget(body_widget)
        self.sidebar_container.setFixedWidth(SIDEBAR_W)
        sidebar_layout = QVBoxLayout(self.sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_panel = RoundedPanel(self.sidebar_container, mode="fill", scrollable=True)
        sidebar_layout.addWidget(sidebar_panel)
        body.addWidget(self.sidebar_container)
        self._build_sidebar(sidebar_panel.inner)

        self.sidebar_toggle = SidebarToggle(body_widget, width=SIDEBAR_TOGGLE_W)
        self.sidebar_toggle.clicked.connect(self._toggle_sidebar)
        body.addWidget(self.sidebar_toggle)

        self.canvas = ImageCanvas(self, body_widget)
        body.addWidget(self.canvas, 1)

        status_panel = RoundedPanel(central, mode="hug")
        root_layout.addWidget(status_panel)
        status_layout = QHBoxLayout(status_panel.inner)
        status_layout.setContentsMargins(14, 8, 14, 8)
        self.status = QLabel("new file")
        self.status.setProperty("dim", True)
        self.status.setProperty("state", "")
        status_layout.addWidget(self.status, 1)
        copyright_label = QLabel("© Jan-Niklas Pippert 2026")
        copyright_label.setProperty("dim", True)
        status_layout.addWidget(copyright_label)

    def _toggle_sidebar(self) -> None:
        """The strip between the sidebar and canvas is a show/hide toggle,
        not a drag-resize handle (see SIDEBAR_W's own comment) - collapsing
        just hides the sidebar entirely rather than shrinking it, so
        nothing in it (buttons, the histogram, the cut-level boxes) is ever
        at risk of being clipped."""
        self.sidebar_collapsed = not self.sidebar_collapsed
        self.sidebar_container.setVisible(not self.sidebar_collapsed)
        self.sidebar_toggle.set_collapsed(self.sidebar_collapsed)

    def _chunk(self, *widgets: QWidget) -> QWidget:
        box = QWidget()
        box.setObjectName("toolbarChunk")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for w in widgets:
            layout.addWidget(w)
        return box

    def _build_toolbar(self, parent: QWidget) -> None:
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(0, 0, 0, 0)
        layout = FlowLayout(parent, h_spacing=10, v_spacing=6)
        layout.setContentsMargins(14, 8, 14, 8)
        outer.addWidget(layout)

        self.theme_toggle = ThemeToggle()
        self.theme_toggle.clicked.connect(self._cycle_theme)
        layout.addWidget(self.theme_toggle)
        self._refresh_theme_toggle()

        self.zoom_label = QLabel("1")
        self.zoom_label.setFixedWidth(36)
        reset_zoom_btn = RoundButton("Reset Zoom")
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        layout.addWidget(self._chunk(self.zoom_label, reset_zoom_btn))

        self.smooth_button = RoundButton("Smooth", checkable=True)
        self.smooth_button.clicked.connect(self._toggle_smoothing)
        self.smooth_sigma_entry = QLineEdit("2")
        self.smooth_sigma_entry.setFixedWidth(44)
        self.smooth_sigma_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.smooth_sigma_entry.editingFinished.connect(self._apply_sigma_entry)
        layout.addWidget(self._chunk(self.smooth_button, self.smooth_sigma_entry))

        self.bin_button = RoundButton("Bin", checkable=True)
        self.bin_button.clicked.connect(self._toggle_binning)
        self.bin_factor_entry = QLineEdit("4")
        self.bin_factor_entry.setFixedWidth(44)
        self.bin_factor_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bin_factor_entry.editingFinished.connect(self._apply_bin_entry)
        layout.addWidget(self._chunk(self.bin_button, self.bin_factor_entry))

        self.scale_control = SegmentedControl(SCALE_OPTIONS, self.scale_function)
        self.scale_control.valueChanged.connect(self.set_scale_function)
        layout.addWidget(self.scale_control)

        prev_btn = RoundButton("<-")
        prev_btn.clicked.connect(self.prev_image)
        self.counter_label = QLabel("1/1")
        self.counter_label.setFixedWidth(50)
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        next_btn = RoundButton("->")
        next_btn.clicked.connect(self.next_image)
        layout.addWidget(self._chunk(prev_btn, self.counter_label, next_btn))

        self.filename_label = QLabel("noname")
        self.ext_combo = QComboBox()
        self.ext_combo.currentIndexChanged.connect(self._on_ext_combo_changed)
        # Cube slice controls - mutually exclusive with ext_combo (see
        # _update_extension_picker): a single-extension file shows neither,
        # a multi-extension one shows ext_combo, a cube extension shows
        # these instead, since a cube's "extensions" are its wavelength/
        # slice axis, not separate HDUs.
        self.slice_slider = RoundSlider(0, 1, 0, integer=True)
        self.slice_slider.setFixedWidth(90)
        self.slice_slider.valueChanged.connect(self._on_slice_slider_changed)
        self.slice_entry = QLineEdit("0")
        self.slice_entry.setFixedWidth(44)
        self.slice_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slice_entry.editingFinished.connect(self._on_slice_entry)
        self.slice_slider.hide()
        self.slice_entry.hide()
        layout.addWidget(self._chunk(self.filename_label, self.ext_combo,
                                      self.slice_slider, self.slice_entry))

        auto_mask_btn = RoundButton("Auto Mask")
        auto_mask_btn.clicked.connect(self.open_auto_mask)
        reset_btn = RoundButton("Reset Mask")
        reset_btn.clicked.connect(self.reset_mask)
        layout.addWidget(self._chunk(auto_mask_btn, reset_btn))

        export_btn = RoundButton("Export Mask", accent=True)
        export_btn.clicked.connect(self.export_mask)
        kill_btn = RoundButton("Kill", danger=True)
        kill_btn.clicked.connect(self.kill_current)
        layout.add_right_widget(self._chunk(export_btn, kill_btn))

    def _build_sidebar(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.magnifier = MagnifierWidget(self)
        mag_row = QHBoxLayout()
        mag_row.addStretch(1)
        mag_row.addWidget(self.magnifier)
        mag_row.addStretch(1)
        layout.addLayout(mag_row)

        readout_layout = QVBoxLayout()
        readout_layout.setSpacing(2)
        self.readout: dict[str, QLabel] = {}
        for left_key, right_key in (("x", "RA"), ("y", "DEC")):
            row = QHBoxLayout()
            row.addWidget(self._dim_label(f"{left_key}:"))
            left_lbl = QLabel("")
            # Fixed width + center alignment, so "x"/"y"'s varying digit
            # count doesn't shift "RA:"/"DEC:" sideways between the two rows
            # - they stay anchored under each other instead of drifting.
            left_lbl.setFixedWidth(36)
            left_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(left_lbl)
            row.addSpacing(10)
            row.addWidget(self._dim_label(f"{right_key}:"))
            right_lbl = QLabel("")
            row.addWidget(right_lbl)
            row.addStretch(1)
            self.readout[left_key] = left_lbl
            self.readout[right_key] = right_lbl
            readout_layout.addLayout(row)
        value_row = QHBoxLayout()
        value_row.addWidget(self._dim_label("value:"))
        value_lbl = QLabel("")
        value_row.addWidget(value_lbl)
        value_row.addStretch(1)
        self.readout["value"] = value_lbl
        readout_layout.addLayout(value_row)
        layout.addLayout(readout_layout)

        self._divider(layout)

        # Live pixel-value histogram with draggable lowcut/highcut lines,
        # embedded directly in the sidebar instead of a separate popup.
        self.cuts_histogram = CutsHistogram(np.array([0.0, 1.0]), 0.0, 1.0, bins=40)
        self.cuts_histogram.cuts_applied.connect(self._on_histogram_apply)
        layout.addWidget(self.cuts_histogram)

        alpha_label = QLabel(f"Mask opacity: {self.mask_alpha}%")
        alpha_label.setProperty("dim", True)
        self._alpha_label = alpha_label
        layout.addWidget(alpha_label)
        self.alpha_slider = RoundSlider(0, 100, self.mask_alpha, integer=True)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        self.alpha_slider.sliderReleased.connect(self.render)
        layout.addWidget(self.alpha_slider)

        self._divider(layout)

        self.mode_control = SegmentedControl([("ellipse", "Ellipse"), ("line", "Satellite")], self.tool)
        self.mode_control.valueChanged.connect(self.set_tool)
        layout.addWidget(self.mode_control)

        self.tool_options_container = QWidget()
        self.tool_options_layout = QVBoxLayout(self.tool_options_container)
        self.tool_options_layout.setContentsMargins(0, 4, 0, 0)
        self.tool_options_layout.setSpacing(4)
        layout.addWidget(self.tool_options_container)
        layout.addStretch(1)

    @staticmethod
    def _dim_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("dim", True)
        return lbl

    @staticmethod
    def _divider(layout: QVBoxLayout) -> None:
        line = QWidget()
        line.setFixedHeight(1)
        line.setAutoFillBackground(True)
        pal = line.palette()
        pal.setColor(line.backgroundRole(), QColor(current_theme().panel_border))
        line.setPalette(pal)
        layout.addWidget(line)

    @staticmethod
    def _fmt_slider_value(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    def _build_slider_row(self, layout: QVBoxLayout, name: str, value: float, lo: float, hi: float, *,
                           suffix: str = "", integer: bool = False, on_change=None) -> RoundSlider:
        row = QHBoxLayout()
        unit_hint = f" ({suffix.strip()})" if suffix.strip() else ""
        row.addWidget(self._dim_label(f"{name}{unit_hint}:"))
        row.addStretch(1)
        entry = QLineEdit(self._fmt_slider_value(value))
        entry.setFixedWidth(60)
        entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(entry)
        layout.addLayout(row)

        slider = RoundSlider(lo, hi, value, integer=integer)
        layout.addWidget(slider)

        def apply_entry() -> None:
            try:
                v = float(entry.text())
            except ValueError:
                entry.setText(self._fmt_slider_value(slider.value()))
                return
            v = max(lo, min(v, hi))
            slider.setValue(v)
            entry.setText(self._fmt_slider_value(slider.value()))
            if on_change is not None:
                on_change(slider.value())

        def sync_entry(v: float) -> None:
            entry.setText(self._fmt_slider_value(v))
            if on_change is not None:
                on_change(v)

        entry.editingFinished.connect(apply_entry)
        slider.valueChanged.connect(sync_entry)
        return slider

    def _rebuild_tool_options(self) -> None:
        """Tears down and rebuilds the tool-options panel for whichever tool
        is now active - ONLY call this for an actual tool switch (ellipse
        <-> satellite). It unconditionally cancels a pending satellite-mode
        start point, appropriate when the tool itself is changing but not
        when only a slider's value needs refreshing after a programmatic
        change - see _sync_tool_option_widgets(), which updates the existing
        sliders in place instead, specifically to avoid that side effect.
        """
        while self.tool_options_layout.count():
            item = self.tool_options_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # takeAt() only detaches it from the layout - it stays
                # visible, at its last computed geometry, until deleteLater()
                # actually runs (a real Qt event, not guaranteed to happen
                # before the next paint). Left implicit, a switch back and
                # forth between ellipse/satellite briefly overlapped the old
                # tool's labels with the new one's - hide() makes it
                # invisible immediately regardless of when the deferred
                # delete itself gets processed.
                widget.hide()
                widget.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())
        self._cancel_pending_line()
        self._radius_slider = self._ellipticity_slider = self._angle_slider = self._thickness_slider = None
        self._line_style_control = None

        if self.tool == "ellipse":
            self._radius_slider = self._build_slider_row(
                self.tool_options_layout, "Radius", self.radius, RADIUS_MIN, MAX_SHAPE_SIZE,
                suffix=" px", on_change=self._on_radius_changed)
            self._ellipticity_slider = self._build_slider_row(
                self.tool_options_layout, "Ellipticity", self.ellipticity, 0, 90,
                suffix="%", integer=True, on_change=self._on_ellipticity_changed)
            self._angle_slider = self._build_slider_row(
                self.tool_options_layout, "Angle", self.angle, -180, 180,
                suffix="°", integer=True, on_change=self._on_angle_changed)
        else:
            self._thickness_slider = self._build_slider_row(
                self.tool_options_layout, "Thickness", self.thickness, 1, MAX_SHAPE_SIZE,
                suffix=" px", integer=True, on_change=self._on_thickness_changed)
            style_box = QWidget()
            style_layout = QVBoxLayout(style_box)
            style_layout.setContentsMargins(0, 8, 0, 4)
            style_layout.addWidget(self._dim_label("Style"))
            self._line_style_control = SegmentedControl(LINE_STYLES, self.line_style)
            self._line_style_control.valueChanged.connect(self._on_line_style_changed)
            style_layout.addWidget(self._line_style_control)
            self.tool_options_layout.addWidget(style_box)

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

    def _on_radius_changed(self, v: float) -> None:
        self.radius = v
        self._refresh_active_preview()

    def _on_ellipticity_changed(self, v: float) -> None:
        self.ellipticity = v
        self._refresh_active_preview()

    def _on_angle_changed(self, v: float) -> None:
        self.angle = v
        self._refresh_active_preview()

    def _on_thickness_changed(self, v: float) -> None:
        self.thickness = v
        self._refresh_active_preview()

    def _on_line_style_changed(self, v: str) -> None:
        self.line_style = v
        self._refresh_active_preview()

    def _on_alpha_changed(self, v: float) -> None:
        self.mask_alpha = int(v)
        self._alpha_label.setText(f"Mask Opacity: {self.mask_alpha}%")

    def set_tool(self, value: str) -> None:
        if value == self.tool:
            return
        self.tool = value
        if hasattr(self, "mode_control"):
            self.mode_control.set_value(value)
        if hasattr(self, "_mode_actions"):
            self._mode_actions[value].setChecked(True)
        self._rebuild_tool_options()
        self._update_shape_preview_state()

    # --------------------------------------------------------- image state

    @property
    def entry(self) -> Entry:
        return self.entries[self.index]

    @property
    def image(self) -> Optional[FitsImage]:
        return self.entry.image

    @property
    def zoom(self) -> float:
        """Effective image-to-canvas pixel scale: fit-to-window baseline times the user multiplier."""
        return self.fit_zoom * self.zoom_mult

    def _apply_session_defaults(self, entry: Entry) -> None:
        """Applies the Settings-driven default bin factor / smooth sigma
        (Help -> Settings) to a freshly-loaded entry - only when it isn't
        already binned/smoothed, so this never fights a value the user set
        explicitly. Runs every time load_current() lands on an entry, which
        - given _release_mask() already resets is_binned/is_smoothed when
        navigating away from an entry - means "every time you're newly
        looking at this entry's data", not just the very first time."""
        if entry.image is None:
            return
        if self._default_bin_factor is not None and not entry.is_binned:
            self.bin_factor_entry.setText(str(self._default_bin_factor))
            self._toggle_binning()
        if self._default_smooth_sigma is not None and not entry.is_smoothed:
            self.smooth_sigma_entry.setText(self._fmt(self._default_smooth_sigma))
            self._toggle_smoothing()

    def load_current(self, reset_view: bool = False) -> None:
        entry = self.entry
        try:
            entry.ensure_loaded(self.stretch)
        except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
            QMessageBox.critical(self, "maskfits", f"Could not load {entry.path}:\n{exc}")
            entry.path = None

        if entry.image is not None and reset_view:
            self.reset_zoom()

        self._apply_session_defaults(entry)

        self.filename_label.setText(os.path.basename(entry.path) if entry.path else "noname")
        self._update_extension_picker()
        self.counter_label.setText(f"{self.index + 1}/{len(self.entries)}")
        self._sync_isopy_cuts()
        self._update_cuts_display()
        self._update_smooth_button()
        self._update_bin_button()
        self._set_status(f"loaded {entry.path}" if entry.path else "new file")

        self.render()

    @staticmethod
    def _file_kind(entry: Entry) -> str:
        """Classifies the currently active extension as "single", "multi",
        or "cube" - "single"/"multi"/"cube" drive both which top-row
        control _update_extension_picker shows and which
        Settings.export_filename_format(_multi/_cube) export_mask() uses,
        so the two stay consistent by construction."""
        image = entry.image
        if image is not None and image.cube is not None:
            return "cube"
        if len(entry.available_extensions) > 1:
            return "multi"
        return "single"

    def _update_extension_picker(self) -> None:
        """Shows exactly one of the extension combo / cube slice controls,
        never both - a single-extension file shows neither (nothing to
        pick), a multi-extension file shows the combo (unchanged), and a
        cube extension shows the slice slider+entry instead, since a
        cube's "extensions" are its wavelength/slice axis, not separate
        HDUs to combo between."""
        entry = self.entry
        image = entry.image
        kind = self._file_kind(entry)
        is_cube = kind == "cube"

        self.ext_combo.blockSignals(True)
        self.ext_combo.clear()
        extensions = entry.available_extensions
        for idx, label in extensions:
            self.ext_combo.addItem(label, idx)
        current_index = next((i for i, (idx, _) in enumerate(extensions) if idx == entry.ext), -1)
        if current_index >= 0:
            self.ext_combo.setCurrentIndex(current_index)
        self.ext_combo.setVisible(not is_cube and len(extensions) > 1)
        self.ext_combo.blockSignals(False)

        self.slice_slider.setVisible(is_cube)
        self.slice_entry.setVisible(is_cube)
        if is_cube:
            n_slices = image.cube.shape[0]
            self.slice_slider.set_range(0, max(n_slices - 1, 0))
            self._sync_slice_controls()

    def _sync_slice_controls(self) -> None:
        """Reflects entry.image.slice_index onto the slider/entry - called
        from every path that can change it (the slider itself, the entry
        box, the Left/Right hotkeys) so all three always agree, no matter
        which one triggered the change."""
        image = self.entry.image
        if image is None or image.cube is None:
            return
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(image.slice_index)
        self.slice_slider.blockSignals(False)
        self.slice_entry.setText(str(image.slice_index))

    def _on_ext_combo_changed(self, index: int) -> None:
        if index < 0:
            return
        ext = self.ext_combo.itemData(index)
        if ext is not None:
            self.switch_extension(ext)

    def _on_slice_slider_changed(self, value: float) -> None:
        self.switch_slice(int(value))

    def _on_slice_entry(self) -> None:
        image = self.entry.image
        if image is None or image.cube is None:
            return
        try:
            index = int(self.slice_entry.text())
        except ValueError:
            index = image.slice_index
        self.switch_slice(index)
        # Covers the no-op case (invalid text, or a value switch_slice()
        # clamped/ignored) where it returns early without resyncing - the
        # entry box must still snap back to the actual current value.
        self._sync_slice_controls()

    def _set_status(self, text: str, *, success: bool = False) -> None:
        """Set the bottom status bar text, optionally tinted green (theme's
        `green` token) as a quick "this succeeded" visual cue - e.g. after a
        mask export - so the user doesn't have to read the message to know
        it worked. Always routes through here (never self.status.setText
        directly) so the green tint never lingers on an unrelated later
        message."""
        self.status.setText(text)
        state = "success" if success else ""
        if self.status.property("state") != state:
            self.status.setProperty("state", state)
            style = self.status.style()
            style.unpolish(self.status)
            style.polish(self.status)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:.4g}"

    def _update_cuts_display(self) -> None:
        if self.image is not None:
            self.cuts_histogram.set_data(self.image.data, self.entry.lowcut, self.entry.highcut)

    def _sync_isopy_cuts(self) -> None:
        """Applies (or un-applies) IsoPy's fixed starting cuts to the
        current entry.

        Called both when the colormap itself changes and when navigating
        between images while IsoPy stays active (each entry has its own
        header/WCS-derived IsoPy cuts). Only seeds lowcut/highcut from
        IsoPy's own formula the FIRST time an entry enters IsoPy (guarded by
        _pre_isopy_lowcut being None, the same flag that remembers what to
        restore) - after that, the user's own histogram edits (the cuts
        histogram stays fully interactive in IsoPy too, unlike before) are
        left alone across navigation/re-syncs, rather than being silently
        reset back to the formula's values on every call. Leaving IsoPy
        restores what the user had before entering it, not IsoPy's values -
        and the next time IsoPy is entered, its formula is re-seeded fresh.
        """
        entry = self.entry
        is_isopy = self.colormap == ISOPY_NAME
        if is_isopy and self.image is not None:
            if entry._pre_isopy_lowcut is None:
                entry._pre_isopy_lowcut, entry._pre_isopy_highcut = entry.lowcut, entry.highcut
                vmin, vmax, _ = entry.isopy_cuts_and_lut()
                entry.lowcut, entry.highcut = vmin, vmax
        elif entry._pre_isopy_lowcut is not None:
            entry.lowcut, entry.highcut = entry._pre_isopy_lowcut, entry._pre_isopy_highcut
            entry._pre_isopy_lowcut = None
            entry._pre_isopy_highcut = None

    def set_colormap(self, name: str) -> None:
        if name == self.colormap:
            return
        self.colormap = name
        if hasattr(self, "_color_actions"):
            self._color_actions[name].setChecked(True)
        self._sync_isopy_cuts()
        self._update_cuts_display()
        self.render()

    def _on_invert_action(self, checked: bool) -> None:
        self.invert_colormap = checked
        self.render()

    def _on_histogram_apply(self, lowcut: float, highcut: float) -> None:
        self.entry.lowcut = lowcut
        self.entry.highcut = highcut
        self.render()

    def set_stretch(self, stretch: str) -> None:
        self.stretch = stretch
        self.entry.apply_stretch(stretch)
        self._update_cuts_display()
        self.render()

    def reset_scale(self) -> None:
        self.set_scale_function("linear")

    def set_scale_function(self, value: str) -> None:
        if value == self.scale_function:
            return
        self.scale_function = value
        if hasattr(self, "scale_control"):
            self.scale_control.set_value(value)
        if hasattr(self, "_scale_actions"):
            self._scale_actions[value].setChecked(True)
        self.render()

    def switch_extension(self, ext: int) -> None:
        """Different extensions of the same file are often just different
        representations of the same pixel grid (e.g. a science frame and its
        weight/variance map) - so if the extension being switched to has the
        same shape, the mask painted so far carries over instead of being
        discarded, letting the user mask against whichever representation
        shows a feature most clearly. A different shape can't share a mask
        at all (nothing to line pixels up with), so that's treated like
        opening a new file - the mask (and undo history, via
        _release_mask()) is dropped entirely."""
        entry = self.entry
        if entry.path is None or ext == entry.ext:
            return
        carry_mask = carry_shape = carry_rotated = None
        if entry.image is not None:
            full_res_mask = self._unbin_mask_cached(entry) if entry.is_binned else entry.image.mask
            carry_mask = full_res_mask.copy()
            carry_shape = carry_mask.shape
            carry_rotated = entry.image.rotated
        # The colormap doesn't change here - only the extension does - so
        # the cut levels the user is currently looking at shouldn't either.
        # ensure_loaded() (via load_current below) always recomputes fresh
        # cuts for a newly loaded image (right, for prev_image/next_image
        # loading a genuinely different file) - saved here and restored
        # after, so that recompute is effectively undone for an extension
        # switch specifically. Only an explicit stretch-preset change
        # (set_stretch) or leaving/entering IsoPy should ever move the cuts.
        saved_lowcut, saved_highcut = entry.lowcut, entry.highcut
        self._release_mask()
        entry.ext = ext
        entry.image = None
        # reset_view=False, unlike prev_image/next_image switching to a
        # different file entirely - an extension is still the same file, so
        # the user's current zoom/pan is worth keeping (it's exactly the
        # framing they were just looking at, e.g. to compare the same
        # region across a science frame and its weight map).
        self.load_current(reset_view=False)
        new_image = entry.image
        if (carry_mask is not None and new_image is not None
                and new_image.data.shape == carry_shape and new_image.rotated == carry_rotated):
            new_image.mask = carry_mask
        if new_image is not None:
            entry.lowcut, entry.highcut = saved_lowcut, saved_highcut
            self._update_cuts_display()
            self.render()

    def switch_slice(self, index: int) -> None:
        """Cube slice navigation (the slice slider/entry, cube-only) -
        unlike switch_extension, every slice of a cube always shares the
        same shape by definition, so the mask, zoom/pan, and cut levels
        always carry over unconditionally - no "different shape, start
        fresh" case to handle. No file re-read either: a cube's full data
        is already in memory (see imagedata.load_fits_image), so this is
        just a re-index, not a reload."""
        entry = self.entry
        image = entry.image
        if image is None or image.cube is None:
            return
        index = max(0, min(index, image.cube.shape[0] - 1))
        if index == image.slice_index:
            return
        carry_mask = (self._unbin_mask_cached(entry) if entry.is_binned else image.mask).copy()
        self._release_mask()
        image.slice_index = index
        image.data = image.cube[index]
        image.mask = carry_mask
        self._sync_slice_controls()
        self._update_cuts_display()
        suffix = self._cube_slice_label(image)
        self._set_status(f"slice {index + 1}/{image.cube.shape[0]}{f' ({suffix})' if suffix else ''}")
        self.render()

    @staticmethod
    def _cube_slice_label(image: FitsImage) -> str:
        """The current slice's position along the cube's third axis (e.g.
        "r 6165 Angstrom" for an SDSS-band cube, "4686 Angstrom" for a
        linear wavelength cube) - empty string if the header doesn't
        describe that axis at all, since not every cube extension will.

        Two conventions, checked in order:
        - Per-slice WAVE<i>/BAND<i> keywords (e.g. WAVE0/BAND0, WAVE1/
          BAND1, ...) - for a handful of discrete, unevenly-spaced bands
          (SDSS ugriz's central wavelengths aren't evenly spaced, so a
          single linear axis can't describe them). No CUNIT3 keyword
          accompanies this convention in practice, so it defaults to
          Angstrom - the only unit this app's own cube generators use it
          for; not a general assumption for arbitrary WAVE<i> data.
        - Standard FITS linear WCS axis (CRVAL3/CDELT3/CRPIX3/CUNIT3):
          pixel i (0-based) is CRVAL3 + (i - (CRPIX3-1))*CDELT3.
        """
        header = image.header
        index = image.slice_index

        wave_key = f"WAVE{index}"
        if wave_key in header:
            try:
                value = float(header[wave_key])
            except (TypeError, ValueError):
                return ""
            band = str(header.get(f"BAND{index}", "")).strip()
            unit = str(header.get("CUNIT3", "Angstrom")).strip()
            text = f"{value:.6g} {unit}".strip()
            return f"{band} {text}".strip() if band else text

        if "CRVAL3" not in header or "CDELT3" not in header:
            return ""
        try:
            crval3 = float(header["CRVAL3"])
            cdelt3 = float(header["CDELT3"])
            crpix3 = float(header.get("CRPIX3", 1))
        except (TypeError, ValueError):
            return ""
        value = crval3 + (index - (crpix3 - 1)) * cdelt3
        unit = str(header.get("CUNIT3", "")).strip()
        return f"{value:.6g} {unit}".strip()

    def prev_extension(self) -> None:
        """Down-arrow hotkey - steps to the previous entry in
        available_extensions (not just entry.ext - 1, since extensions
        aren't always contiguous - e.g. only 0 and 2 have image data).
        load_current() (via switch_extension) refreshes the extension combo
        in the top row to match."""
        exts = [i for i, _ in self.entry.available_extensions]
        if self.entry.ext not in exts:
            return
        idx = exts.index(self.entry.ext)
        if idx > 0:
            self.switch_extension(exts[idx - 1])

    def next_extension(self) -> None:
        """Up-arrow hotkey - see prev_extension."""
        exts = [i for i, _ in self.entry.available_extensions]
        if self.entry.ext not in exts:
            return
        idx = exts.index(self.entry.ext)
        if idx < len(exts) - 1:
            self.switch_extension(exts[idx + 1])

    # ------------------------------------------------------------- navigation

    def _release_mask(self) -> None:
        entry = self.entry
        image = entry.image
        if image is not None:
            if entry.original_data is not None:
                image.data = entry.original_data
            image.mask = np.zeros(image.data.shape, dtype=bool)
        entry.original_data = None
        entry.is_smoothed = False
        entry.smooth_sigma = None
        entry.smoothed_cache = None
        entry._smoothed_cache_sigma = None
        entry.is_binned = False
        entry.bin_factor = None
        entry.binned_cache = None
        entry._binned_cache_factor = None
        entry.smoothed_binned_cache = None
        entry._smoothed_binned_cache_sigma = None
        entry._smoothed_binned_cache_factor = None
        entry.mask_backup = None
        entry.mask_dirty = False
        entry._binned_mask_cache = None
        entry._binned_mask_cache_factor = None
        self._undo = None
        self._redo = None

    def open_files(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(
            self, "Open FITS files", "",
            "FITS files (*.fits *.fit *.fts *.fits.gz);;All files (*.*)",
        )
        if not paths:
            return
        self._release_mask()
        if len(self.entries) == 1 and self.entries[0].path is None:
            self.entries = []
        start = len(self.entries)
        self.entries.extend(Entry(p) for p in paths)
        self.index = start
        self.load_current(reset_view=True)

    def prev_image(self) -> None:
        """Left-arrow hotkey - while a cube is loaded, this instead steps
        one slice down (right is up, left is down - see switch_slice),
        since Left/Right are the natural "step through" direction and a
        cube's slices are what there is to step through, not other loaded
        files."""
        image = self.image
        if image is not None and image.cube is not None:
            self.switch_slice(image.slice_index - 1)
            return
        if self.index > 0:
            self._release_mask()
            self.index -= 1
            self.load_current(reset_view=True)

    def next_image(self) -> None:
        """Right-arrow hotkey - see prev_image."""
        image = self.image
        if image is not None and image.cube is not None:
            self.switch_slice(image.slice_index + 1)
            return
        if self.index < len(self.entries) - 1:
            self._release_mask()
            self.index += 1
            self.load_current(reset_view=True)

    def kill_current(self) -> None:
        if len(self.entries) == 1:
            self.entries = [Entry(None)]
            self.index = 0
        else:
            del self.entries[self.index]
            self.index = min(self.index, len(self.entries) - 1)
        self.load_current(reset_view=True)

    def reset_mask(self) -> None:
        if self.image is None:
            return
        self._push_undo()
        self.image.mask[:] = False
        self._mark_mask_dirty()
        self.render()
        self._set_status("mask cleared")

    # ------------------------------------------------------------- smoothing

    def _ensure_original(self, entry: Entry) -> None:
        if entry.original_data is None:
            entry.original_data = entry.image.data

    def _smoothed_cache_for(self, entry: Entry, sigma: float) -> np.ndarray:
        if entry.smoothed_cache is None or entry._smoothed_cache_sigma != sigma:
            entry.smoothed_cache = gaussian_smooth(entry.original_data, sigma)
            entry._smoothed_cache_sigma = sigma
        return entry.smoothed_cache

    def _binned_cache_for(self, entry: Entry, factor: int) -> np.ndarray:
        if entry.binned_cache is None or entry._binned_cache_factor != factor:
            entry.binned_cache = bin_func(entry.original_data, factor)
            entry._binned_cache_factor = factor
        return entry.binned_cache

    def _smoothed_binned_cache_for(self, entry: Entry, sigma: float, factor: int) -> np.ndarray:
        if (entry.smoothed_binned_cache is None
                or entry._smoothed_binned_cache_sigma != sigma
                or entry._smoothed_binned_cache_factor != factor):
            base = self._smoothed_cache_for(entry, sigma)
            entry.smoothed_binned_cache = bin_func(base, factor)
            entry._smoothed_binned_cache_sigma = sigma
            entry._smoothed_binned_cache_factor = factor
        return entry.smoothed_binned_cache

    def _current_display_data(self, entry: Entry) -> np.ndarray:
        if entry.is_smoothed and entry.is_binned:
            return self._smoothed_binned_cache_for(entry, entry.smooth_sigma, entry.bin_factor)
        if entry.is_smoothed:
            return self._smoothed_cache_for(entry, entry.smooth_sigma)
        if entry.is_binned:
            return self._binned_cache_for(entry, entry.bin_factor)
        return entry.original_data

    def _mark_mask_dirty(self, entry: Optional[Entry] = None) -> None:
        (entry if entry is not None else self.entry).mask_dirty = True

    def _unbin_mask_cached(self, entry: Entry) -> np.ndarray:
        if not entry.mask_dirty:
            return entry.mask_backup
        return unbin_mask(entry.image.mask, entry.bin_factor, entry.mask_backup)

    def _bin_mask_cached(self, entry: Entry, full_res_mask: np.ndarray, factor: int) -> np.ndarray:
        if (not entry.mask_dirty and entry._binned_mask_cache is not None
                and entry._binned_mask_cache_factor == factor):
            return entry._binned_mask_cache
        computed = bin_mask(full_res_mask, factor)
        entry._binned_mask_cache = computed
        entry._binned_mask_cache_factor = factor
        return computed

    def _read_sigma(self) -> float:
        try:
            sigma = float(self.smooth_sigma_entry.text())
        except ValueError:
            entry = self.entry
            sigma = entry.smooth_sigma if entry.smooth_sigma is not None else 3.0
        sigma = max(sigma, 0.0)
        self.smooth_sigma_entry.setText(self._fmt(sigma))
        return sigma

    def _update_smooth_button(self) -> None:
        smoothed = self.image is not None and self.entry.is_smoothed
        self.smooth_button.setText("Unsmooth" if smoothed else "Smooth")
        self.smooth_button.setChecked(smoothed)

    def _toggle_smoothing(self) -> None:
        entry = self.entry
        if entry.image is None:
            self._update_smooth_button()
            return
        if entry.is_smoothed:
            entry.is_smoothed = False
            entry.image.data = self._current_display_data(entry)
            self._update_smooth_button()
            self._update_cuts_display()
            self.render()
            self._set_status("smoothing removed")
            return
        sigma = self._read_sigma()
        if sigma <= 0:
            self._update_smooth_button()
            return
        self._ensure_original(entry)
        entry.smooth_sigma = sigma
        entry.is_smoothed = True
        entry.image.data = self._current_display_data(entry)
        self._update_smooth_button()
        self._update_cuts_display()
        self.render()
        self._set_status(f"smoothed (sigma={self._fmt(sigma)})")

    def _apply_sigma_entry(self) -> None:
        entry = self.entry
        sigma = self._read_sigma()
        if entry.image is None or not entry.is_smoothed:
            return
        if sigma <= 0:
            entry.is_smoothed = False
            entry.image.data = self._current_display_data(entry)
            self._update_smooth_button()
            self._update_cuts_display()
            self.render()
            self._set_status("smoothing removed")
            return
        entry.smooth_sigma = sigma
        entry.image.data = self._current_display_data(entry)
        self._update_cuts_display()
        self.render()
        self._set_status(f"smoothed (sigma={self._fmt(sigma)})")

    # -------------------------------------------------------------- binning

    def _read_bin_factor(self) -> int:
        try:
            factor = int(round(float(self.bin_factor_entry.text())))
        except ValueError:
            entry = self.entry
            factor = entry.bin_factor if entry.bin_factor is not None else 3
        factor = max(factor, 1)
        if self.image is not None:
            factor = min(factor, max(min(self.image.data.shape), 1))
        self.bin_factor_entry.setText(str(factor))
        return factor

    def _update_bin_button(self) -> None:
        binned = self.image is not None and self.entry.is_binned
        self.bin_button.setText("Unbin" if binned else "Bin")
        self.bin_button.setChecked(binned)

    def _toggle_binning(self) -> None:
        entry = self.entry
        if entry.image is None:
            self._update_bin_button()
            return
        if entry.is_binned:
            old_factor = entry.bin_factor
            entry.image.mask = self._unbin_mask_cached(entry)
            entry.mask_backup = None
            entry.mask_dirty = False
            entry.is_binned = False
            entry.image.data = self._current_display_data(entry)
            self._undo = None
            self._redo = None
            self._update_bin_button()
            self._update_cuts_display()
            self._rescale_view_for_bin_change(old_factor)
            self._set_status("binning removed")
            return
        factor = self._read_bin_factor()
        if factor <= 1:
            self._update_bin_button()
            return
        self._ensure_original(entry)
        entry.mask_backup = entry.image.mask
        entry.bin_factor = factor
        entry.is_binned = True
        entry.image.data = self._current_display_data(entry)
        entry.image.mask = self._bin_mask_cached(entry, entry.mask_backup, factor)
        entry.mask_dirty = False
        self._undo = None
        self._redo = None
        self._update_bin_button()
        self._update_cuts_display()
        self._rescale_view_for_bin_change(1.0 / factor)
        self._set_status(f"binned {factor}x{factor}")

    def _apply_bin_entry(self) -> None:
        entry = self.entry
        factor = self._read_bin_factor()
        if entry.image is None or not entry.is_binned:
            return
        old_factor = entry.bin_factor
        if factor <= 1:
            entry.image.mask = self._unbin_mask_cached(entry)
            entry.mask_backup = None
            entry.mask_dirty = False
            entry.is_binned = False
            entry.image.data = self._current_display_data(entry)
            self._undo = None
            self._redo = None
            self._update_bin_button()
            self._update_cuts_display()
            self._rescale_view_for_bin_change(old_factor)
            self._set_status("binning removed")
            return
        full_res_mask = self._unbin_mask_cached(entry)
        entry.mask_backup = full_res_mask
        entry.bin_factor = factor
        entry.image.data = self._current_display_data(entry)
        entry.image.mask = self._bin_mask_cached(entry, full_res_mask, factor)
        entry.mask_dirty = False
        self._undo = None
        self._redo = None
        self._update_cuts_display()
        self._rescale_view_for_bin_change(old_factor / factor)
        self._set_status(f"binned {factor}x{factor}")

    def _rescale_view_for_bin_change(self, k: float) -> None:
        self.view_cx *= k
        self.view_cy *= k
        if self.image is not None:
            self.fit_zoom = self._compute_fit_zoom()
        self.radius = max(min(self.radius * k, MAX_SHAPE_SIZE), RADIUS_MIN)
        self.thickness = max(min(int(round(self.thickness * k)), MAX_SHAPE_SIZE), 1)
        self._sync_tool_option_widgets()
        self._update_zoom_label()
        self.render()

    def _sync_tool_option_widgets(self) -> None:
        """Refreshes the tool-option sliders/entries after a programmatic
        value change (a bin-factor rescale, the E/W hotkey, or the 1/2/3/4
        hotkeys) that didn't go through their own widgets - updates
        whichever controls are currently built in place, rather than
        routing through _rebuild_tool_options(). That rebuild unconditionally
        cancels a pending satellite-mode start point (appropriate for an
        actual tool switch, its real purpose) - reusing it here for a
        same-tool value refresh would wipe that start point and its preview
        even though the tool itself never changed."""
        if self.tool == "ellipse":
            if self._radius_slider is not None:
                self._radius_slider.setValue(self.radius)
            if self._ellipticity_slider is not None:
                self._ellipticity_slider.setValue(self.ellipticity)
            if self._angle_slider is not None:
                self._angle_slider.setValue(self.angle)
        else:
            if self._thickness_slider is not None:
                self._thickness_slider.setValue(self.thickness)
            if self._line_style_control is not None:
                self._line_style_control.set_value(self.line_style)

    def _full_res_mask(self, entry: Entry) -> np.ndarray:
        if entry.is_binned:
            return self._unbin_mask_cached(entry)
        return entry.image.mask

    def _adjust_shape_size(self, direction: int) -> None:
        step = max(1, round(5 / self.zoom_mult))
        if self.tool == "ellipse":
            self.radius = max(RADIUS_MIN, min(self.radius + direction * step, MAX_SHAPE_SIZE))
        elif self.tool == "line":
            self.thickness = max(1, min(self.thickness + direction * step, MAX_SHAPE_SIZE))
        else:
            return
        self._sync_tool_option_widgets()
        self._refresh_active_preview()

    def _hotkey_digit(self, n: int) -> None:
        if self.tool == "ellipse":
            if n == 1:
                self._adjust_ellipticity(-1)
            elif n == 2:
                self._adjust_ellipticity(1)
            elif n == 3:
                self._adjust_angle(-1)
            elif n == 4:
                self._adjust_angle(1)
        elif self.tool == "line" and 1 <= n <= len(LINE_STYLES):
            self.line_style = LINE_STYLES[n - 1][0]
            self._sync_tool_option_widgets()
            self._refresh_active_preview()

    def _adjust_ellipticity(self, direction: int) -> None:
        if self.tool != "ellipse":
            return
        self.ellipticity = max(0, min(self.ellipticity + direction * 2, 90))
        self._sync_tool_option_widgets()
        self._refresh_active_preview()

    def _adjust_angle(self, direction: int) -> None:
        if self.tool != "ellipse":
            return
        self.angle = max(-180, min(self.angle + direction * 2, 180))
        self._sync_tool_option_widgets()
        self._refresh_active_preview()

    # -------------------------------------------------------------- export

    @staticmethod
    def _mask_stem(path: str) -> str:
        stem = os.path.basename(path)
        if stem.endswith(".fits.gz"):
            return stem[: -len(".fits.gz")]
        return os.path.splitext(stem)[0]

    def _build_mask_hdu(self, entry: Entry) -> "fits.PrimaryHDU":
        header = entry.image.header.copy()
        header["OBJECT"] = "MASK"
        full_res_mask = self._full_res_mask(entry)
        mask = full_res_mask.T if entry.image.rotated else full_res_mask
        exported = (~mask).astype("uint8")
        return fits.PrimaryHDU(data=exported, header=header)

    def _export_dir(self, entry: Entry) -> str:
        """The folder export_mask/export_mask_as default to, per the
        "export directory" Setting: the file's own folder (default), or the
        directory maskfits was launched from."""
        if self.export_dir_mode == "cwd":
            return os.getcwd()
        return os.path.dirname(entry.path)

    def export_mask(self) -> None:
        """The "Export Mask" toolbar button and File menu action - always
        writes to a Settings.export_filename_format(_multi/_cube)
        filename, in Settings.export_dir's folder, with no dialog - which
        of the three formats depends on the current file's own type (see
        _file_kind): a plain single-extension image uses
        export_filename_format (default "mask_$FILENAME"), a multi-
        extension FITS uses export_filename_format_multi ($EXT also
        available, the current extension number), and a cube uses
        export_filename_format_cube ($SLICE also available, the current
        slice number). Save Mask As... (export_mask_as) is the
        interactive, pick-your-own-name/location alternative and isn't
        affected by any of this."""
        entry = self.entry
        if entry.image is None or entry.path is None:
            QMessageBox.warning(self, "maskfits", "No image loaded to export a mask for.")
            return
        stem = self._mask_stem(entry.path)
        kind = self._file_kind(entry)
        if kind == "cube":
            filename = format_mask_filename(
                self.settings.export_filename_format_cube, stem, slice_index=entry.image.slice_index
            )
        elif kind == "multi":
            filename = format_mask_filename(self.settings.export_filename_format_multi, stem, ext=entry.ext)
        else:
            filename = format_mask_filename(self.settings.export_filename_format, stem)
        out_path = os.path.join(self._export_dir(entry), filename)
        self._build_mask_hdu(entry).writeto(out_path, overwrite=True)
        self._set_status(f"exported mask to {out_path}", success=True)

    def export_mask_as(self) -> None:
        entry = self.entry
        if entry.image is None or entry.path is None:
            QMessageBox.warning(self, "maskfits", "No image loaded to export a mask for.")
            return
        default_name = f"mask_{self._mask_stem(entry.path)}.fits"
        out_path, _filter = QFileDialog.getSaveFileName(
            self, "Save Mask As", os.path.join(self._export_dir(entry), default_name),
            "FITS files (*.fits *.fit *.fts);;All files (*.*)",
        )
        if not out_path:
            return
        self._build_mask_hdu(entry).writeto(out_path, overwrite=True)
        self._set_status(f"exported mask to {out_path}", success=True)

    # ---------------------------------------------------------- auto mask

    def open_auto_mask(self) -> None:
        if self.image is None:
            return
        if self._auto_mask_window is not None:
            self._auto_mask_window.raise_()
            self._auto_mask_window.activateWindow()
            return
        self._auto_mask_window = AutoMaskWindow(self, self.entry)
        self._auto_mask_window.show()

    def set_auto_mask_preview(self, entry: Entry, preview: np.ndarray) -> None:
        self._auto_mask_entry = entry
        self._auto_mask_preview = preview
        self.render()

    def confirm_auto_mask(self, entry: Entry, preview: np.ndarray) -> None:
        if entry.image is not None and preview.shape == entry.image.mask.shape:
            if entry is self.entry:
                self._push_undo()
            entry.image.mask = entry.image.mask | preview
            self._mark_mask_dirty(entry)
            self._set_status(f"auto mask applied: {int(preview.sum()):,} px")
        else:
            self._set_status("auto mask discarded: image changed while the window was open")
        self._clear_auto_mask_preview()

    def discard_auto_mask(self, _entry: Entry) -> None:
        self._set_status("auto mask discarded")
        self._clear_auto_mask_preview()

    def _clear_auto_mask_preview(self) -> None:
        self._auto_mask_window = None
        self._auto_mask_entry = None
        self._auto_mask_preview = None
        self.render()

    # --------------------------------------------------------------- undo

    def _push_undo(self) -> None:
        if self.image is None:
            return
        self._undo = (self.index, self.image.mask.copy())
        self._redo = None  # a fresh edit invalidates any pending redo

    def undo(self) -> None:
        if self._undo is None:
            return
        idx, mask = self._undo
        if idx < len(self.entries) and self.entries[idx].image is not None:
            current = self.entries[idx].image.mask.copy()
            self.entries[idx].image.mask = mask
            self._mark_mask_dirty(self.entries[idx])
            self._redo = (idx, current)
            self._undo = None
            if idx == self.index:
                self.render()

    def redo(self) -> None:
        if self._redo is None:
            return
        idx, mask = self._redo
        if idx < len(self.entries) and self.entries[idx].image is not None:
            current = self.entries[idx].image.mask.copy()
            self.entries[idx].image.mask = mask
            self._mark_mask_dirty(self.entries[idx])
            self._undo = (idx, current)
            self._redo = None
            if idx == self.index:
                self.render()

    def _on_middle_click(self) -> None:
        if self.tool != "ellipse":
            self._on_cancel_or_redo()

    def _on_cancel_or_redo(self) -> None:
        if self.tool == "line" and self._line_anchor is not None:
            self._cancel_pending_line()
            return
        self.redo()

    def _cycle_colormap(self) -> None:
        idx = COLORMAP_NAMES.index(self.colormap)
        self.set_colormap(COLORMAP_NAMES[(idx + 1) % len(COLORMAP_NAMES)])

    # ------------------------------------------------------- coordinate math

    def img_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        # y is inverted relative to array/canvas indexing: FITS convention
        # (row 0 at the bottom of the image) while canvas y increases
        # downward - so increasing iy must map to a DEcreasing canvas y.
        cx = self.canvas_w / 2 + (ix - self.view_cx) * self.zoom
        cy = self.canvas_h / 2 - (iy - self.view_cy) * self.zoom
        return cx, cy

    def canvas_to_img(self, cx: float, cy: float) -> tuple[float, float]:
        ix = self.view_cx + (cx - self.canvas_w / 2) / self.zoom
        iy = self.view_cy - (cy - self.canvas_h / 2) / self.zoom
        return ix, iy

    # ------------------------------------------------------------ rendering

    def _compute_fit_zoom(self) -> float:
        if self.image is None or self.canvas_w <= 1 or self.canvas_h <= 1:
            return 1.0
        ny, nx = self.image.data.shape
        return min(self.canvas_w / nx, self.canvas_h / ny) or 1.0

    def _update_zoom_label(self) -> None:
        text = f"{self.zoom_mult:.2f}".rstrip("0").rstrip(".")
        self.zoom_label.setText(text or "1")

    def render(self) -> None:
        if hasattr(self, "magnifier"):
            self.magnifier.update()
        self._base_pixmap = None
        if not hasattr(self, "canvas"):
            return
        image = self.image
        if image is None or self.canvas_w <= 1:
            self.canvas.update()
            return

        data = image.data
        ny, nx = data.shape
        ix0, iy0 = self.canvas_to_img(0, 0)
        ix1, iy1 = self.canvas_to_img(self.canvas_w, self.canvas_h)
        x0 = max(int(np.floor(min(ix0, ix1))), 0)
        x1 = min(int(np.ceil(max(ix0, ix1))), nx)
        y0 = max(int(np.floor(min(iy0, iy1))), 0)
        y1 = min(int(np.ceil(max(iy0, iy1))), ny)
        if x1 <= x0 or y1 <= y0:
            self.canvas.update()
            return

        entry = self.entry
        crop_h, crop_w = y1 - y0, x1 - x0
        disp_w = max(int(round(crop_w * self.zoom)), 1)
        disp_h = max(int(round(crop_h * self.zoom)), 1)

        # When zoomed out, downsample toward display resolution BEFORE the
        # per-pixel stretch/colormap work, instead of computing it at full
        # source resolution and throwing most of it away in the final resize.
        step_x = max(crop_w // max(disp_w, 1), 1)
        step_y = max(crop_h // max(disp_h, 1), 1)
        crop = data[y0:y1:step_y, x0:x1:step_x]
        mask_crop = image.mask[y0:y1:step_y, x0:x1:step_x]

        span = safe_span(entry.lowcut, entry.highcut)
        norm = np.clip((crop - entry.lowcut) / span, 0, 1)
        norm = np.nan_to_num(norm, nan=0.0)
        rgb = self._scale_and_color(norm)

        self._tint_masked(rgb, mask_crop)

        if (self._auto_mask_preview is not None and self._auto_mask_entry is entry
                and self._auto_mask_preview.shape == image.mask.shape):
            preview_crop = self._auto_mask_preview[y0:y1:step_y, x0:x1:step_x]
            self._tint_preview(rgb, preview_crop)

        # crop's row 0 is array row y0 (the smallest iy in view), but with
        # y0 = bottom_row and y1 = top_row - so flip vertically before
        # handing it to Qt, which always draws its own row 0 at the top.
        rgb = np.ascontiguousarray(rgb[::-1])

        # rgb's actual shape is the STRIDED/downsampled crop (data[y0:y1:
        # step_y, x0:x1:step_x] above), which is smaller than crop_h/crop_w
        # (the full-resolution source span) whenever step_x/step_y > 1 - true
        # for any real-sized image displayed below 1:1 zoom, i.e. almost
        # always. QImage takes width/height/bytesPerLine as the CALLER's
        # claim about its buffer, with no way to cross-check them against
        # rgb's actual size - passing crop_w/crop_h there told Qt the buffer
        # was far bigger than it really is, so it read past the end of
        # rgb's memory (a native out-of-bounds read - not a Python
        # exception, a segfault). Use the array's own real dimensions.
        sampled_h, sampled_w = rgb.shape[0], rgb.shape[1]
        qimg = QImage(rgb.data, sampled_w, sampled_h, 3 * sampled_w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        resample = (Qt.TransformationMode.FastTransformation if self.zoom >= 1
                    else Qt.TransformationMode.SmoothTransformation)
        pixmap = pixmap.scaled(disp_w, disp_h, Qt.AspectRatioMode.IgnoreAspectRatio, resample)

        # anchor point below places the (now-flipped) top-left corner at this
        # canvas point - that corner corresponds to image coordinate (x0, y1),
        # the crop's top edge under the inverted y-axis, not (x0, y0).
        cx0, cy0 = self.img_to_canvas(x0, y1)
        self._base_pixmap = pixmap
        self._base_pos = QPointF(cx0, cy0)
        self.canvas.update()

    def _active_lut(self) -> np.ndarray:
        name = self.colormap
        if name == ISOPY_NAME:
            _, _, lut = self.entry.isopy_cuts_and_lut()
        else:
            lut = COLORMAP_LUTS[name]
        return lut[::-1] if self.invert_colormap else lut

    def _scale_and_color(self, norm: np.ndarray) -> np.ndarray:
        stretch = STRETCHES[self.scale_function]
        stretched = np.clip(np.asarray(stretch(norm)), 0.0, 1.0)
        gray = (stretched * 255).astype(np.uint8)
        return self._active_lut()[gray]

    def _mask_tint(self) -> tuple[int, int, int]:
        return mask_tint_for(self.colormap, self._active_lut(), current_theme().accent)

    def _auto_mask_tint(self) -> tuple[int, int, int]:
        return auto_mask_tint_for(self.colormap, self._active_lut(), current_theme().blue)

    def _tint_masked(self, rgb: np.ndarray, mask_crop: np.ndarray) -> None:
        if not mask_crop.any():
            return
        self._blend_tint(rgb, mask_crop, self._mask_tint(), self.mask_alpha / 100.0)

    def _tint_preview(self, rgb: np.ndarray, preview_crop: np.ndarray) -> None:
        if not preview_crop.any():
            return
        self._blend_tint(rgb, preview_crop, self._auto_mask_tint(), 0.55)

    @staticmethod
    def _blend_tint(rgb: np.ndarray, region: np.ndarray, tint: tuple[int, int, int], alpha: float) -> None:
        for ch, tint_v in enumerate(tint):
            channel = rgb[..., ch].astype(np.float32)
            blended = channel * (1 - alpha) + tint_v * alpha
            rgb[..., ch] = np.where(region, blended, channel).astype(np.uint8)

    # ------------------------------------------------------------ overlays

    def _paint_overlays(self, painter: QPainter) -> None:
        """The center-cross marker (toggled independent of the cursor) plus
        hover previews for the active tool - the latter drawn as vector
        shapes directly (rotated-ellipse outline via the exact same
        geometry as the real mask stamp, thick flat-capped line for the
        satellite trail), unlike the old Tkinter version which had to
        pre-render a small RGBA raster because Tk canvas ovals can't be
        rotated."""
        if self.image is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.show_center_cross:
            self._paint_center_cross(painter)
        if self._cursor_canvas_pos is None:
            return
        cx, cy = self._cursor_canvas_pos
        if self.tool == "ellipse":
            self._paint_ellipse_preview(painter, cx, cy)
        elif self._line_anchor is not None:
            self._paint_line_preview(painter, cx, cy)

    def _paint_center_cross(self, painter: QPainter) -> None:
        """A small crosshair marking the image's exact center (nx/2, ny/2
        in image pixel coordinates, the same point reset_zoom() re-centers
        the view on) - toggled by the K hotkey, to help judge how well a
        source is centered in the frame. Colored via _center_cross_color()
        - the inverse of whatever's actually displayed there right now
        (colormap AND mask tint included) - rather than a fixed color, so
        it can never blend into a same-colored mask covering the center
        the way a flat tint (even the mask's own color) eventually would."""
        ny, nx = self.image.data.shape
        cx, cy = self.img_to_canvas(nx / 2.0, ny / 2.0)
        half = 9.0
        pen = QPen(QColor(*self._center_cross_color()), 1.5)
        painter.setPen(pen)
        painter.drawLine(QPointF(cx - half, cy), QPointF(cx + half, cy))
        painter.drawLine(QPointF(cx, cy - half), QPointF(cx, cy + half))

    def _center_cross_color(self) -> tuple[int, int, int]:
        """The exact inverse (255-r, 255-g, 255-b) of the average displayed
        color (colormap-mapped, then mask-tinted the same way render() does)
        over the central 15x15 image pixels - guarantees contrast against
        whatever's actually there, mask or no mask, any colormap."""
        image = self.image
        entry = self.entry
        ny, nx = image.data.shape
        half = 7  # 2*7+1 = 15
        cy_i, cx_i = ny // 2, nx // 2
        y0, y1 = max(cy_i - half, 0), min(cy_i + half + 1, ny)
        x0, x1 = max(cx_i - half, 0), min(cx_i + half + 1, nx)
        crop = image.data[y0:y1, x0:x1]
        mask_crop = image.mask[y0:y1, x0:x1]
        span = safe_span(entry.lowcut, entry.highcut)
        norm = np.nan_to_num(np.clip((crop - entry.lowcut) / span, 0, 1), nan=0.0)
        rgb = self._scale_and_color(norm)
        self._tint_masked(rgb, mask_crop)
        avg = rgb.reshape(-1, 3).mean(axis=0)
        return (int(255 - avg[0]), int(255 - avg[1]), int(255 - avg[2]))

    def _toggle_center_cross(self) -> None:
        self.show_center_cross = not self.show_center_cross
        self._refresh_active_preview()

    def _paint_ellipse_preview(self, painter: QPainter, cx: float, cy: float) -> None:
        a, b, angle = self._current_round_params()
        disp_a, disp_b = a * self.zoom, b * self.zoom
        pts = ellipse_polygon_points(cx, cy, disp_a, disp_b, angle)
        poly = QPolygonF([QPointF(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)])
        tint = self._mask_tint()
        fill = QColor(*tint)
        fill.setAlpha(round(0.25 * 255))
        painter.setBrush(fill)
        painter.setPen(QPen(QColor(*tint), 1))
        painter.drawPolygon(poly)

    def _paint_line_preview(self, painter: QPainter, cx: float, cy: float) -> None:
        x0, y0 = self._line_anchor
        x1, y1 = self.canvas_to_img(cx, cy)
        ex0, ey0, ex1, ey1 = self._extend_for_style(x0, y0, x1, y1)
        sx0, sy0 = self.img_to_canvas(ex0, ey0)
        sx1, sy1 = self.img_to_canvas(ex1, ey1)
        disp_width = max(self.thickness * self.zoom, 1.0)
        tint = self._mask_tint()
        color = QColor(*tint)
        color.setAlpha(round(0.5 * 255))
        pen = QPen(color, disp_width)
        # line_mask's actual mask math clips the perpendicular band strictly
        # to the segment's own length (a true flat-cut rectangle, not a
        # rounded capsule) - FlatCap matches that exactly.
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(sx0, sy0), QPointF(sx1, sy1))

    def _extend_for_style(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        if self.image is None:
            return x0, y0, x1, y1
        shape = self.image.data.shape
        if self.line_style == "arrow":
            return extend_ray_to_border(shape, x0, y0, x1, y1)
        if self.line_style == "line":
            return extend_line_to_borders(shape, x0, y0, x1, y1)
        return x0, y0, x1, y1

    def _current_round_params(self) -> tuple[float, float, float]:
        r = self.radius
        b = r * (1 - self.ellipticity / 100.0)
        return r, b, self.angle

    def _refresh_active_preview(self) -> None:
        if hasattr(self, "canvas"):
            self.canvas.update()

    def _update_shape_preview_state(self) -> None:
        self._refresh_active_preview()

    def _cancel_pending_line(self) -> None:
        self._line_anchor = None
        self._refresh_active_preview()

    # ---------------------------------------------------------------- input

    def _on_motion(self, cx: float, cy: float) -> None:
        self._cursor_canvas_pos = (cx, cy)
        ix, iy = self.canvas_to_img(cx, cy)
        self._cursor_img_pos = (ix, iy)
        self.magnifier.update()
        ix_i, iy_i = int(round(ix)), int(round(iy))
        self.readout["x"].setText(str(ix_i))
        self.readout["y"].setText(str(iy_i))

        image = self.image
        if image is not None and 0 <= iy_i < image.data.shape[0] and 0 <= ix_i < image.data.shape[1]:
            self.readout["value"].setText(self._fmt(float(image.data[iy_i, ix_i])))
        else:
            self.readout["value"].setText("")

        if image is not None and image.wcs is not None:
            try:
                wcs_ix, wcs_iy = (iy, ix) if image.rotated else (ix, iy)
                if self.entry.is_binned:
                    factor = self.entry.bin_factor
                    wcs_ix, wcs_iy = wcs_ix * factor, wcs_iy * factor
                sky = image.wcs.pixel_to_world(wcs_ix, wcs_iy)
                self.readout["RA"].setText(sky.ra.to_string(unit="hourangle", sep=":", precision=2))
                self.readout["DEC"].setText(sky.dec.to_string(sep=":", precision=1, alwayssign=True))
            except Exception:
                self.readout["RA"].setText("")
                self.readout["DEC"].setText("")
        else:
            self.readout["RA"].setText("")
            self.readout["DEC"].setText("")

        self._refresh_active_preview()

    def _on_button(self, cx: float, cy: float, erase: bool) -> None:
        if self.image is None:
            return
        if self.tool == "ellipse":
            self._push_undo()
            self._stamp_round(cx, cy, erase)
        elif self.tool == "line":
            if self._line_anchor is None:
                self._push_undo()
                self._line_anchor = self.canvas_to_img(cx, cy)
                self._line_anchor_erase = erase
            else:
                self._finalize_click_line(cx, cy)

    def _on_drag(self, cx: float, cy: float, erase: bool) -> None:
        if self.tool == "ellipse" and self.image is not None:
            self._stamp_round(cx, cy, erase)

    def _finalize_click_line(self, cx: float, cy: float) -> None:
        if self.image is None or self._line_anchor is None:
            return
        x0, y0 = self._line_anchor
        x1, y1 = self.canvas_to_img(cx, cy)
        ex0, ey0, ex1, ey1 = self._extend_for_style(x0, y0, x1, y1)
        stamp = line_mask(self.image.data.shape, ex0, ey0, ex1, ey1, self.thickness)
        self.image.mask = (self.image.mask & ~stamp) if self._line_anchor_erase else (self.image.mask | stamp)
        self._mark_mask_dirty()
        self._line_anchor = None
        self.render()

    def _stamp_round(self, cx: float, cy: float, erase: bool) -> None:
        image = self.image
        if image is None:
            return
        ix, iy = self.canvas_to_img(cx, cy)
        a, b, angle = self._current_round_params()
        stamp = ellipse_mask(image.data.shape, ix, iy, a, b, angle)
        image.mask = (image.mask & ~stamp) if erase else (image.mask | stamp)
        self._mark_mask_dirty()
        self.render()

    def _on_pan_start(self, cx: float, cy: float) -> None:
        self._pan_drag = (cx, cy, self.view_cx, self.view_cy)

    def _on_pan_drag(self, cx: float, cy: float) -> None:
        if self._pan_drag is None:
            return
        sx, sy, ocx, ocy = self._pan_drag
        self.view_cx = ocx - (cx - sx) / self.zoom
        self.view_cy = ocy + (cy - sy) / self.zoom
        self.render()

    def reset_zoom(self) -> None:
        self.zoom_mult = 1.0
        if self.image is not None:
            ny, nx = self.image.data.shape
            self.view_cx, self.view_cy = nx / 2, ny / 2
            self.fit_zoom = self._compute_fit_zoom()
        self._update_zoom_label()
        self.render()

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        if self.image is None:
            return
        new_mult = max(min(self.zoom_mult * factor, ZOOM_MULT_MAX), ZOOM_MULT_MIN)
        if new_mult == self.zoom_mult:
            return
        ix, iy = self.canvas_to_img(cx, cy)
        self.zoom_mult = new_mult
        new_cx, new_cy = self.canvas_to_img(cx, cy)
        self.view_cx += ix - new_cx
        self.view_cy += iy - new_cy
        self._update_zoom_label()
        self.render()
        self._refresh_active_preview()

    # ----------------------------------------------------------- shortcuts

    def _guarded(self, func):
        """Wraps a hotkey action so it's a no-op while a text entry has
        focus - QShortcut fires regardless of focus by default, but a plain
        letter hotkey (s, r, e, w, c, i, b, u, y, 1-4, ...) firing while
        the user is typing a number into a sigma/bin/cut/slider entry would
        both insert nothing useful there AND trigger the hotkey unexpectedly,
        the same failure mode the old Tkinter version's global-click-to-
        defocus existed to avoid."""

        def wrapped() -> None:
            if isinstance(QApplication.focusWidget(), QLineEdit):
                return
            func()

        return wrapped

    def _build_shortcuts(self) -> None:
        # (handler, guarded) per action id (see maskfits.shortcuts) - the
        # metadata (label, default keys) lives there, independent of self,
        # so both this and the Settings shortcut editor/Hotkeys popup can
        # use it without needing a live MaskFitsApp instance.
        handlers: dict[str, tuple[object, bool]] = {
            "undo": (self.undo, True),
            "redo": (self.redo, True),
            "prev_image": (self.prev_image, True),
            "next_image": (self.next_image, True),
            "reset_mask": (self.reset_mask, True),
            "grow_shape": (lambda: self._adjust_shape_size(1), True),
            "shrink_shape": (lambda: self._adjust_shape_size(-1), True),
            "cycle_colormap": (self._cycle_colormap, True),
            "invert_colormap": (lambda: self._invert_action.trigger(), True),
            "toggle_smooth": (self._toggle_smoothing, True),
            "toggle_bin": (self._toggle_binning, True),
            "reset_zoom": (self.reset_zoom, True),
            "toggle_center_cross": (self._toggle_center_cross, True),
            "prev_extension": (self.prev_extension, True),
            "next_extension": (self.next_extension, True),
            "digit_1": (lambda: self._hotkey_digit(1), True),
            "digit_2": (lambda: self._hotkey_digit(2), True),
            "digit_3": (lambda: self._hotkey_digit(3), True),
            "digit_4": (lambda: self._hotkey_digit(4), True),
            "cancel_line": (self._cancel_pending_line, False),
        }
        self._shortcuts: list[QShortcut] = []
        for action in SHORTCUT_ACTIONS:
            func, guarded = handlers[action.id]
            for seq in effective_keys(action.id, self.settings.shortcuts):
                sc = QShortcut(QKeySequence(seq), self)
                sc.activated.connect(self._guarded(func) if guarded else func)
                self._shortcuts.append(sc)

    def _rebuild_shortcuts(self) -> None:
        """Tears down and recreates every QShortcut from the current
        self.settings.shortcuts - called after Settings are saved so a
        rebinding takes effect immediately, no restart needed."""
        for sc in self._shortcuts:
            sc.deleteLater()
        self._build_shortcuts()


def run_gui(paths: list[str], zoom: Optional[float] = None, mode: Optional[str] = None, *,
            vmin: Optional[float] = None, vmax: Optional[float] = None, scale: Optional[str] = None,
            cuts: Optional[str] = None, colormap: Optional[str] = None, binning: Optional[int] = None,
            smooth: Optional[int] = None, extension: Optional[int] = None) -> int:
    _set_windows_app_id()
    if QApplication.instance() is None:
        # Must be set before the QApplication exists. Without this, a
        # fractional host display scale (e.g. Windows' 150% on a laptop,
        # relayed through WSLg's Wayland compositor) can get rounded to the
        # nearest integer factor and then re-applied on top of the
        # compositor's own scaling, making the window appear far larger
        # than the screen - PassThrough uses the exact reported factor
        # instead of rounding it.
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    app = QApplication.instance() or QApplication(sys.argv)
    # The native per-platform style (macOS in particular) draws QSlider's
    # groove/sub-page/add-page/handle itself as a "complex control" and
    # ignores most of the QSS geometry rules for them (border-radius,
    # background) regardless of what's in build_qss() - Fusion is the one
    # bundled Qt style that actually honors that stylesheet, which is what
    # makes the slider track render as a rounded pill and the handle as a
    # true circle instead of native's plain rectangle/native-shaped thumb.
    fusion_style = QStyleFactory.create("Fusion")
    if fusion_style is not None:
        app.setStyle(fusion_style)
    app.setWindowIcon(_app_icon())

    # Every CLI flag but -z/-m used to be patched onto the window after
    # construction, which only worked by accident for zoom/mode (their
    # startup application happens to run late enough in __init__ to survive
    # being overwritten again afterward) - colormap/scale/stretch/bin/smooth
    # are all applied INSIDE __init__ instead, so overriding them that late
    # would just be undone or silently ignored. Merging every CLI override
    # into one Settings object up front, before constructing MaskFitsApp,
    # lets all of them flow through the same Settings-driven startup path -
    # this never touches the persisted config (save_settings is never
    # called here), so it's session-only, same as -z/-m already were.
    settings = load_settings()
    if mode is not None:
        settings.mode = MODE_FLAGS[mode]
    if zoom is not None:
        settings.zoom = zoom
    if scale is not None:
        settings.scale = scale
    if cuts is not None:
        settings.stretch = cuts
    if colormap is not None:
        settings.colormap = colormap
    if binning is not None:
        settings.bin_enabled = True
        settings.bin_factor = binning
    if smooth is not None:
        settings.smooth_enabled = True
        settings.smooth_sigma = float(smooth)

    window = MaskFitsApp(paths, settings=settings, extension=extension)
    # --cuts (self.stretch) is correctly applied once during construction
    # for a non-IsoPy colormap - but when IsoPy is the STARTING colormap,
    # _sync_isopy_cuts() runs right after in that same initial load and
    # unconditionally overwrites lowcut/highcut with its own fixed formula
    # (the seed it applies the first time an entry enters IsoPy), silently
    # discarding whatever --cuts just computed. Re-applying --cuts here,
    # after construction, makes it win regardless of colormap - the same
    # "explicit CLI override always wins" fix vmin/vmax already needed for
    # the same reason (see their own comment below).
    if cuts is not None:
        window.entry.apply_stretch(cuts)
    # vmin/vmax have no Settings field (they're per-entry, not a session
    # default) - applied directly to the just-loaded entry instead, after
    # whatever --cuts/the stretch algorithm already computed, so they
    # correctly override just the one(s) actually given.
    if vmin is not None:
        window.entry.lowcut = vmin
    if vmax is not None:
        window.entry.highcut = vmax
    if cuts is not None or vmin is not None or vmax is not None:
        window._update_cuts_display()
        window.render()
    window.show()
    window.raise_()
    window.activateWindow()
    window.start_background_update_check()
    return app.exec()
