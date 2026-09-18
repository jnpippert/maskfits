"""Help -> Settings: a dialog for editing the persisted user defaults (see
maskfits.settings) that run_gui() loads at startup - theme, default tool/
colormap/cut-levels/scale, default binning/smoothing, export directory,
zoom, and accent color.

Theme and accent color preview live as the user edits them (both are already
global, instantly-appliable state via theme_manager()); every other field
only takes effect on Save, applied both to the persisted QSettings store and
(via the settings_saved signal) to the currently running session. Cancel (or
closing without Save) reverts the live theme/accent preview back to whatever
was active when the dialog opened - see _revert_preview.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from maskfits.colormaps import COLORMAP_NAMES
from maskfits.custom_themes import build_custom_theme, get_custom_theme, list_custom_themes
from maskfits.imagedata import PERCENTILE_PRESETS, STRETCH_NAMES
from maskfits.settings import (
    EXPORT_DIR_CHOICES,
    MODE_CHOICES,
    STRETCH_CHOICES,
    Settings,
    save_settings,
)
from maskfits.shortcuts_window import ShortcutsWindow
from maskfits.theme import detect_os_light_mode, theme_manager
from maskfits.theme_editor_window import ThemeEditorWindow
from maskfits.widgets import HexColorPicker, RoundButton, RoundSlider, SegmentedControl

MODE_OPTIONS = [("ellipse", "Ellipse"), ("line", "Satellite")]
SCALE_OPTIONS = [(name, name.capitalize()) for name in STRETCH_NAMES]
# Cut-levels algorithm (the app's own "Scale" menu: Min/Max, ZScale, then a
# percentile preset) - distinct from SCALE_OPTIONS above, which is the
# linear/log/asinh stretch *function* applied on top of whichever cuts these
# pick.
STRETCH_OPTIONS = [("minmax", "Min/Max"), ("zscale", "ZScale")] + [
    (f"pct{p}", f"{p}%") for p in PERCENTILE_PRESETS
]
EXPORT_DIR_OPTIONS = [("file_parent", "File's folder"), ("cwd", "Working directory")]
BUILTIN_THEME_LABELS = [("system", "System"), ("dark", "Dark"), ("light", "Light")]

BIN_FACTOR_MIN, BIN_FACTOR_MAX = 2, 20
SMOOTH_SIGMA_MIN, SMOOTH_SIGMA_MAX = 0.1, 10.0
ZOOM_MIN, ZOOM_MAX = 0.5, 20.0


class SettingsWindow(QDialog):
    settings_saved = Signal(Settings)

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        # Non-modal, same as AutoMaskWindow - a modal dialog that itself
        # opens another modal dialog (ThemeEditorWindow, and QMessageBox
        # from within it) nests Cocoa "modal sessions" on macOS, which
        # throws "modalSession has been exited prematurely" once the inner
        # one closes before the outer one. Shown via .show(), not .exec().
        self.setModal(False)

        # Snapshot so Cancel/close-without-Save can revert the live theme/
        # accent preview - see _revert_preview.
        self._snapshot_mode = theme_manager().mode
        self._snapshot_accent = theme_manager().accent
        self._snapshot_custom = theme_manager().custom
        self._saved = False
        self._theme_editor: Optional[ThemeEditorWindow] = None
        self._shortcuts_window: Optional[ShortcutsWindow] = None

        self.theme = settings.theme
        self.mode = settings.mode
        self.colormap = settings.colormap
        self.scale = settings.scale
        self.stretch = settings.stretch
        self.bin_enabled = settings.bin_enabled
        self.bin_factor = settings.bin_factor
        self.smooth_enabled = settings.smooth_enabled
        self.smooth_sigma = settings.smooth_sigma
        self.export_dir = settings.export_dir
        self.zoom = settings.zoom
        self.accent_color = settings.accent_color
        self.shortcuts: dict[str, list[str]] = dict(settings.shortcuts)

        self._build()
        self._center_on_parent()

    def _center_on_parent(self) -> None:
        parent = self.parentWidget()
        self.adjustSize()
        if parent is None:
            return
        geo = parent.frameGeometry()
        x = max(geo.x() + (geo.width() - self.width()) // 2, 0)
        y = max(geo.y() + (geo.height() - self.height()) // 2, 0)
        self.move(x, y)

    # ---------------------------------------------------------------- build

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(0)

        title = QLabel("Settings")
        layout.addWidget(title)
        desc = QLabel("Defaults loaded the next time maskfits is opened from the command line.\n"
                       "A -z/-m flag at launch still overrides the zoom/mode default for that session only.")
        desc.setWordWrap(True)
        desc.setProperty("dim", True)
        desc.setFixedWidth(420)
        layout.addSpacing(2)
        layout.addWidget(desc)

        self._theme_row(layout)
        self._segment_row(layout, "default mode", MODE_OPTIONS, self.mode, self._on_mode_changed)
        self._combo_row(layout, "default colormap", COLORMAP_NAMES, self.colormap, self._on_colormap_changed)
        self._labeled_combo_row(layout, "default cut levels", STRETCH_OPTIONS, self.stretch,
                                 self._on_stretch_changed)
        self._segment_row(layout, "default scale", SCALE_OPTIONS, self.scale, self._on_scale_changed)
        self._segment_row(layout, "export directory", EXPORT_DIR_OPTIONS, self.export_dir,
                           self._on_export_dir_changed)
        self._slider_row(layout, "default zoom", self.zoom, ZOOM_MIN, ZOOM_MAX, self._on_zoom_changed)
        self._slider_row(layout, "default bin factor", self.bin_factor, BIN_FACTOR_MIN, BIN_FACTOR_MAX,
                          self._on_bin_factor_changed, integer=True, enabled_checkbox=True,
                          checked=self.bin_enabled, on_toggled=self._on_bin_enabled_changed)
        self._slider_row(layout, "default smooth sigma", self.smooth_sigma, SMOOTH_SIGMA_MIN, SMOOTH_SIGMA_MAX,
                          self._on_smooth_sigma_changed, enabled_checkbox=True, checked=self.smooth_enabled,
                          on_toggled=self._on_smooth_enabled_changed)
        self._accent_row(layout)
        self._shortcuts_row(layout)

        layout.addSpacing(10)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        save_btn = RoundButton("Save", accent=True)
        save_btn.clicked.connect(self._save)
        cancel_btn = RoundButton("Cancel")
        cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _row_label(self, layout: QVBoxLayout, label: str) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setProperty("dim", True)
        row.addWidget(lbl)
        row.addStretch(1)
        layout.addSpacing(10)
        return row

    def _segment_row(self, layout: QVBoxLayout, label: str, options: list[tuple[str, str]], value: str,
                      on_change) -> None:
        row = self._row_label(layout, label)
        control = SegmentedControl(options, value)
        control.valueChanged.connect(on_change)
        row.addWidget(control)
        layout.addLayout(row)

    def _combo_row(self, layout: QVBoxLayout, label: str, options: list[str], value: str, on_change) -> None:
        row = self._row_label(layout, label)
        combo = QComboBox()
        combo.addItems(options)
        if value in options:
            combo.setCurrentIndex(options.index(value))
        combo.currentTextChanged.connect(on_change)
        row.addWidget(combo)
        layout.addLayout(row)

    def _labeled_combo_row(self, layout: QVBoxLayout, label: str, options: list[tuple[str, str]], value: str,
                            on_change) -> None:
        """Like _combo_row, but the combo's displayed text (e.g. "99.5%")
        differs from the value it represents (e.g. "pct99.5") - see
        STRETCH_OPTIONS."""
        row = self._row_label(layout, label)
        combo = QComboBox()
        for key, display in options:
            combo.addItem(display, key)
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda i: on_change(combo.itemData(i)))
        row.addWidget(combo)
        layout.addLayout(row)

    def _slider_row(self, layout: QVBoxLayout, label: str, value: float, lo: float, hi: float, on_change, *,
                     integer: bool = False, enabled_checkbox: bool = False, checked: bool = True,
                     on_toggled=None) -> None:
        row = QHBoxLayout()
        checkbox: Optional[QCheckBox] = None
        if enabled_checkbox:
            checkbox = QCheckBox()
            checkbox.setChecked(checked)
            checkbox.toggled.connect(on_toggled)
            row.addWidget(checkbox)
        lbl = QLabel(f"{label}:")
        lbl.setProperty("dim", True)
        row.addWidget(lbl)
        row.addStretch(1)
        entry = QLineEdit(f"{value:.3g}")
        entry.setFixedWidth(56)
        row.addWidget(entry)
        layout.addSpacing(10)
        layout.addLayout(row)

        slider = RoundSlider(lo, hi, value, integer=integer)
        slider.setFixedWidth(420)
        layout.addSpacing(0)
        layout.addWidget(slider)

        def apply_entry() -> None:
            try:
                v = float(entry.text())
            except ValueError:
                entry.setText(f"{slider.value():.3g}")
                return
            v = max(lo, min(v, hi))
            slider.setValue(v)
            entry.setText(f"{slider.value():.3g}")
            on_change(slider.value())

        def sync_entry(v: float) -> None:
            entry.setText(f"{v:.3g}")
            on_change(v)

        entry.editingFinished.connect(apply_entry)
        slider.valueChanged.connect(sync_entry)

    # ---------------------------------------------------------------- theme

    def _theme_row(self, layout: QVBoxLayout) -> None:
        row = self._row_label(layout, "theme")
        self._theme_combo = QComboBox()
        self._theme_combo.currentIndexChanged.connect(self._on_theme_combo_changed)
        row.addWidget(self._theme_combo)

        self._new_theme_btn = RoundButton("New Theme...")
        self._new_theme_btn.clicked.connect(self._new_theme)
        row.addWidget(self._new_theme_btn)

        self._edit_theme_btn = RoundButton("Edit")
        self._edit_theme_btn.clicked.connect(self._edit_theme)
        row.addWidget(self._edit_theme_btn)

        layout.addLayout(row)
        self._reload_theme_combo(select=self.theme)

    def _reload_theme_combo(self, *, select: str) -> None:
        combo = self._theme_combo
        combo.blockSignals(True)
        combo.clear()
        for key, label in BUILTIN_THEME_LABELS:
            combo.addItem(label, key)
        custom_names = sorted(list_custom_themes().keys())
        for name in custom_names:
            combo.addItem(name, name)
        index = combo.findData(select)
        if index < 0:
            index = 0  # the selected custom theme was deleted elsewhere - fall back to "system"
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        # setCurrentIndex() above is signal-blocked (rebuilding the combo's
        # items must not fire currentIndexChanged for every addItem), so the
        # live preview has to be (re-)applied explicitly here - notably for
        # the "deleted the currently-active custom theme" case, where the
        # combo needs to fall back to "system" for real, not just visually.
        self._apply_theme_choice(combo.itemData(index))

    def _sync_theme_dependent_ui(self) -> None:
        # Called once from _theme_row(), before _accent_row() has built the
        # widget this also toggles - harmless no-op then, since _accent_row
        # calls this again itself once that widget exists.
        if not hasattr(self, "_accent_row_widget"):
            return
        is_custom = self.theme not in ("system", "dark", "light")
        self._edit_theme_btn.setEnabled(is_custom)
        # A custom theme already bakes its own accent in as one of its
        # fields - the separate accent-color override only makes sense on
        # top of a built-in theme.
        self._accent_row_widget.setEnabled(not is_custom)

    def _apply_theme_choice(self, key: str) -> None:
        self.theme = key
        if key in ("system", "dark", "light"):
            mode = ("light" if detect_os_light_mode() else "dark") if key == "system" else key
            theme_manager().set_mode(mode)
            theme_manager().set_accent(self.accent_color)
        else:
            custom = get_custom_theme(key)
            if custom is not None:
                theme_manager().set_custom(build_custom_theme(custom))
        self._sync_theme_dependent_ui()

    def _on_theme_combo_changed(self, index: int) -> None:
        key = self._theme_combo.itemData(index)
        if key is None:
            return
        self._apply_theme_choice(key)

    def _new_theme(self) -> None:
        if self._theme_editor is not None:
            self._theme_editor.raise_()
            self._theme_editor.activateWindow()
            return
        editor = ThemeEditorWindow(self)
        editor.theme_saved.connect(self._on_theme_editor_saved)
        editor.finished.connect(self._clear_theme_editor)
        self._theme_editor = editor
        editor.show()

    def _edit_theme(self) -> None:
        if self.theme in ("system", "dark", "light"):
            return
        if self._theme_editor is not None:
            self._theme_editor.raise_()
            self._theme_editor.activateWindow()
            return
        editor = ThemeEditorWindow(self, edit_name=self.theme)
        editor.theme_saved.connect(self._on_theme_editor_saved)
        editor.theme_deleted.connect(self._on_theme_editor_deleted)
        editor.finished.connect(self._clear_theme_editor)
        self._theme_editor = editor
        editor.show()

    def _clear_theme_editor(self, _result=None) -> None:
        self._theme_editor = None

    def _on_theme_editor_saved(self, name: str) -> None:
        self._reload_theme_combo(select=name)

    def _on_theme_editor_deleted(self, name: str) -> None:
        self._reload_theme_combo(select="system")

    # --------------------------------------------------------------- accent

    def _accent_row(self, layout: QVBoxLayout) -> None:
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("accent color:")
        lbl.setProperty("dim", True)
        row.addWidget(lbl)
        row.addStretch(1)

        self._accent_picker = HexColorPicker(self.accent_color, allow_empty=True, placeholder="default")
        self._accent_picker.colorChanged.connect(self._on_accent_changed)
        row.addWidget(self._accent_picker)

        reset_btn = RoundButton("Reset")
        reset_btn.clicked.connect(lambda: self._accent_picker.setValue(None))
        row.addWidget(reset_btn)

        layout.addSpacing(10)
        layout.addWidget(row_widget)
        self._accent_row_widget = row_widget
        self._sync_theme_dependent_ui()

    def _on_accent_changed(self, value: str) -> None:
        self.accent_color = value or None
        theme_manager().set_accent(self.accent_color)

    # ----------------------------------------------------------- shortcuts

    def _shortcuts_row(self, layout: QVBoxLayout) -> None:
        row = self._row_label(layout, "keyboard shortcuts")
        edit_btn = RoundButton("Edit Keyboard Shortcuts...")
        edit_btn.clicked.connect(self._edit_shortcuts)
        row.addWidget(edit_btn)
        layout.addLayout(row)

    def _edit_shortcuts(self) -> None:
        if self._shortcuts_window is not None:
            self._shortcuts_window.raise_()
            self._shortcuts_window.activateWindow()
            return
        editor = ShortcutsWindow(self.shortcuts, self)
        editor.shortcuts_saved.connect(self._on_shortcuts_saved)
        editor.finished.connect(self._clear_shortcuts_window)
        self._shortcuts_window = editor
        editor.show()

    def _clear_shortcuts_window(self, _result=None) -> None:
        self._shortcuts_window = None

    def _on_shortcuts_saved(self, overrides: dict) -> None:
        self.shortcuts = overrides

    # ------------------------------------------------------------- setters

    def _on_mode_changed(self, value: str) -> None:
        self.mode = value

    def _on_colormap_changed(self, value: str) -> None:
        self.colormap = value

    def _on_scale_changed(self, value: str) -> None:
        self.scale = value

    def _on_stretch_changed(self, value: str) -> None:
        self.stretch = value

    def _on_export_dir_changed(self, value: str) -> None:
        self.export_dir = value

    def _on_zoom_changed(self, value: float) -> None:
        self.zoom = value

    def _on_bin_factor_changed(self, value: float) -> None:
        self.bin_factor = int(value)

    def _on_bin_enabled_changed(self, checked: bool) -> None:
        self.bin_enabled = checked

    def _on_smooth_sigma_changed(self, value: float) -> None:
        self.smooth_sigma = value

    def _on_smooth_enabled_changed(self, checked: bool) -> None:
        self.smooth_enabled = checked

    # ---------------------------------------------------------------- done

    def _current_settings(self) -> Settings:
        return Settings(
            theme=self.theme,
            mode=self.mode if self.mode in MODE_CHOICES else "ellipse",
            colormap=self.colormap if self.colormap in COLORMAP_NAMES else "Grayscale",
            scale=self.scale if self.scale in STRETCH_NAMES else "linear",
            stretch=self.stretch if self.stretch in STRETCH_CHOICES else "zscale",
            bin_enabled=self.bin_enabled,
            bin_factor=self.bin_factor,
            smooth_enabled=self.smooth_enabled,
            smooth_sigma=self.smooth_sigma,
            export_dir=self.export_dir if self.export_dir in EXPORT_DIR_CHOICES else "file_parent",
            zoom=self.zoom,
            accent_color=self.accent_color,
            shortcuts=self.shortcuts,
        )

    def _save(self) -> None:
        settings = self._current_settings()
        save_settings(settings)
        self._saved = True
        self.settings_saved.emit(settings)
        self.accept()

    def _cancel(self) -> None:
        self._revert_preview()
        self.reject()

    def _revert_preview(self) -> None:
        if self._snapshot_custom is not None:
            theme_manager().set_custom(self._snapshot_custom)
        else:
            theme_manager().set_mode(self._snapshot_mode)
            theme_manager().set_accent(self._snapshot_accent)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._theme_editor is not None:
            self._theme_editor.close()
        if self._shortcuts_window is not None:
            self._shortcuts_window.close()
        if not self._saved:
            self._revert_preview()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            # QDialog's default Escape-closes behavior is a poor fit here -
            # Cancel/close is one click away either way, and it was closing
            # this window as a side effect of holding Escape to cancel a
            # shortcut capture in the child ShortcutsWindow (Escape bubbles
            # up when nothing below consumes it). Use Cancel to close.
            return
        super().keyPressEvent(event)
