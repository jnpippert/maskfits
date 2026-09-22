"""A modal editor for a single custom theme (Settings -> theme -> "New
theme..." / "Edit") - a name, a base dark/light mode (only used to derive
sensible hover/pressed shades and OS-chrome integration, not itself a stored
color), and a HexColorPicker per base color role in
custom_themes.CUSTOM_THEME_FIELDS.

Every edit previews live via theme_manager().set_custom() so the user sees
the actual running app change as they pick colors, the same live-preview
pattern the accent-color field already uses; Cancel (or closing without
Save) reverts to whatever theme was active before the editor opened. Built-in
dark/light themes never reach here - only maskfits.settings_window opens
this, and only for "New theme..." or editing an existing *custom* theme.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QVBoxLayout, QWidget

from maskfits.custom_themes import (
    CUSTOM_THEME_FIELDS,
    RESERVED_NAMES,
    build_custom_theme,
    delete_custom_theme,
    get_custom_theme,
    list_custom_themes,
    rename_custom_theme,
    save_custom_theme,
)
from maskfits.theme import DARK, LIGHT, theme_manager
from maskfits.widgets import HexColorPicker, RoundButton, SegmentedControl, ThemeIconPicker

FIELD_LABELS = {
    "app_bg": "App Background",
    "panel_bg": "Panel Background",
    "panel_border": "Panel Border",
    "text": "Text",
    "text_dim": "Dim Text",
    "accent": "Accent",
    "danger": "Danger",
    "green": "Success / Green",
    "warning": "Warning",
    "blue": "Info / Blue",
    "track": "Slider Track",
    "button_bg": "Button Background",
    "canvas_bg": "Image Canvas Background",
}


class ThemeEditorWindow(QDialog):
    theme_saved = Signal(str)
    theme_deleted = Signal(str)
    theme_renamed = Signal(str, str)  # old name, new name - emitted before theme_saved

    def __init__(self, parent: Optional[QWidget] = None, *, edit_name: Optional[str] = None):
        super().__init__(parent)
        self._editing = edit_name is not None
        self.setWindowTitle("Edit Theme" if self._editing else "New Theme")
        # Non-modal, same as AutoMaskWindow/SettingsWindow - shown via
        # .show(), not .exec(), so it never nests inside SettingsWindow's
        # own (also non-modal) Cocoa window session. See SettingsWindow's
        # __init__ docstring comment for the modal-nesting bug this avoids.
        self.setModal(False)

        # So Cancel/close-without-Save can put back exactly what was active
        # before this editor started previewing changes live.
        self._snapshot_mode = theme_manager().mode
        self._snapshot_accent = theme_manager().accent
        self._snapshot_custom = theme_manager().custom
        self._saved = False

        if self._editing:
            stored = get_custom_theme(edit_name) or {}
            self.name = edit_name
            self.mode = stored.get("mode", "dark")
            icon = stored.get("icon", "")
            self.icon = icon if isinstance(icon, str) else ""
            base = DARK if self.mode != "light" else LIGHT
            self.colors = {f: stored.get(f, getattr(base, f)) for f in CUSTOM_THEME_FIELDS}
        else:
            self.name = ""
            self.mode = "dark"
            self.icon = ""
            self.colors = {f: getattr(DARK, f) for f in CUSTOM_THEME_FIELDS}

        self._pickers: dict[str, HexColorPicker] = {}
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

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(0)

        title = QLabel("Edit Theme" if self._editing else "New Theme")
        layout.addWidget(title)
        desc = QLabel("Each edit previews live in the main window - Save to keep it, Cancel to revert.")
        desc.setWordWrap(True)
        desc.setProperty("dim", True)
        desc.setFixedWidth(360)
        layout.addSpacing(2)
        layout.addWidget(desc)

        name_row = QHBoxLayout()
        name_lbl = QLabel("Name:")
        name_lbl.setProperty("dim", True)
        name_row.addWidget(name_lbl)
        name_row.addStretch(1)
        self._name_entry = QLineEdit(self.name)
        self._name_entry.setFixedWidth(180)
        name_row.addWidget(self._name_entry)
        layout.addSpacing(10)
        layout.addLayout(name_row)

        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Base Mode:")
        mode_lbl.setProperty("dim", True)
        mode_row.addWidget(mode_lbl)
        mode_row.addStretch(1)
        mode_control = SegmentedControl([("dark", "Dark"), ("light", "Light")], self.mode)
        mode_control.valueChanged.connect(self._on_mode_changed)
        mode_row.addWidget(mode_control)
        layout.addSpacing(10)
        layout.addLayout(mode_row)

        icon_row = QHBoxLayout()
        icon_lbl = QLabel("Icon:")
        icon_lbl.setProperty("dim", True)
        icon_lbl.setToolTip("Shown on the toolbar button that cycles through themes.")
        icon_row.addWidget(icon_lbl)
        icon_row.addStretch(1)
        self._icon_picker = ThemeIconPicker(self.icon)
        self._icon_picker.iconChanged.connect(self._on_icon_changed)
        icon_row.addWidget(self._icon_picker)
        layout.addSpacing(10)
        layout.addLayout(icon_row)

        for field in CUSTOM_THEME_FIELDS:
            row = QHBoxLayout()
            lbl = QLabel(f"{FIELD_LABELS.get(field, field)}:")
            lbl.setProperty("dim", True)
            row.addWidget(lbl)
            row.addStretch(1)
            picker = HexColorPicker(self.colors[field])
            picker.colorChanged.connect(lambda value, f=field: self._on_color_changed(f, value))
            self._pickers[field] = picker
            row.addWidget(picker)
            layout.addSpacing(8)
            layout.addLayout(row)

        layout.addSpacing(10)
        btn_row = QHBoxLayout()
        if self._editing:
            delete_btn = RoundButton("Delete", danger=True)
            delete_btn.clicked.connect(self._delete)
            btn_row.addWidget(delete_btn)
        btn_row.addStretch(1)
        cancel_btn = RoundButton("Cancel")
        cancel_btn.clicked.connect(self._cancel)
        save_btn = RoundButton("Save", accent=True)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self._preview()

    # --------------------------------------------------------------- edits

    def _on_mode_changed(self, value: str) -> None:
        self.mode = value
        self._preview()

    def _on_icon_changed(self, spec: str) -> None:
        self.icon = spec

    def _on_color_changed(self, field: str, value: str) -> None:
        if not value:
            return
        self.colors[field] = value
        self._preview()

    def _preview(self) -> None:
        theme_manager().set_custom(build_custom_theme({"mode": self.mode, **self.colors}))

    # ---------------------------------------------------------------- done

    def _save(self) -> None:
        name = self._name_entry.text().strip()
        if not name:
            QMessageBox.warning(self, "maskfits", "A theme name is required.")
            return
        if name in RESERVED_NAMES:
            QMessageBox.warning(self, "maskfits", f"{name!r} is a built-in theme name - pick another.")
            return
        if name != self.name and name in list_custom_themes():
            QMessageBox.warning(self, "maskfits", f"A theme named {name!r} already exists.")
            return
        renamed = self._editing and name != self.name
        if renamed:
            rename_custom_theme(self.name, name)
        save_custom_theme(name, {"mode": self.mode, "icon": self.icon, **self.colors})
        self._saved = True
        if renamed:
            self.theme_renamed.emit(self.name, name)
        self.theme_saved.emit(name)
        self.accept()

    def _delete(self) -> None:
        confirm = QMessageBox.question(self, "maskfits", f"Delete theme {self.name!r}?")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        delete_custom_theme(self.name)
        self._saved = True  # closeEvent shouldn't re-preview a theme that no longer exists
        self._revert_preview()
        self.theme_deleted.emit(self.name)
        self.reject()

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
        if not self._saved:
            self._revert_preview()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            # QDialog's default Escape-closes behavior isn't wanted here -
            # Cancel/Save are one click away either way. Use Cancel to close.
            return
        super().keyPressEvent(event)
