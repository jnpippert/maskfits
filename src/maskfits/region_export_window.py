"""File -> Save as DS9 Region...: the options window (region color,
coordinate system, output path). It only collects options - Save emits
export_requested and MaskFitsApp does the actual writing (see
MaskFitsApp._write_region and maskfits.regions).
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from maskfits.regions import DS9_COLORS
from maskfits.widgets import HexColorPicker, RoundButton, SegmentedControl

CUSTOM_COLOR = "custom"
PATH_WIDTH = 380


class RegionExportWindow(QDialog):
    # (path, color, coords) - color is a DS9 color name or "#rrggbb",
    # coords is "image" or "wcs".
    export_requested = Signal(str, str, str)

    def __init__(self, default_path: str, has_wcs: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Save as DS9 Region")
        # Non-modal, like the other option windows - a modal dialog that
        # opens a second (file) dialog hits a Cocoa modal-nesting bug.
        self.setModal(False)
        self._has_wcs = has_wcs
        self._build(default_path)
        self._center_on_parent()

    def _center_on_parent(self) -> None:
        self.adjustSize()
        parent = self.parentWidget()
        if parent is None:
            return
        geo = parent.frameGeometry()
        self.move(max(geo.x() + (geo.width() - self.width()) // 2, 0),
                  max(geo.y() + (geo.height() - self.height()) // 2, 0))

    def _build(self, default_path: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(0)

        desc = QLabel(
            "Writes the current mask as a DS9 region file: one polygon per connected "
            "group of masked pixels, without labels. Holes are written as excluded polygons."
        )
        desc.setWordWrap(True)
        desc.setProperty("dim", True)
        desc.setFixedWidth(PATH_WIDTH + 80)
        layout.addWidget(desc)

        row = self._row(layout, "color")
        self.color_combo = QComboBox()
        self.color_combo.addItems(DS9_COLORS + [CUSTOM_COLOR])
        self.color_combo.currentTextChanged.connect(self._on_color_choice)
        row.addWidget(self.color_combo)
        self.custom_color = HexColorPicker("#00ff00")
        self.custom_color.hide()
        row.addWidget(self.custom_color)

        row = self._row(layout, "coordinates")
        self.coords_control = SegmentedControl([("image", "Image (x, y)"), ("wcs", "WCS (fk5)")], "image")
        row.addWidget(self.coords_control)
        if not self._has_wcs:
            wcs_button = self.coords_control._buttons["wcs"]
            wcs_button.setEnabled(False)
            wcs_button.setToolTip("This image has no WCS")

        row = self._row(layout, "file", stretch=False)
        self.path_entry = QLineEdit(default_path)
        self.path_entry.setMinimumWidth(PATH_WIDTH)
        row.addWidget(self.path_entry, 1)
        browse_btn = RoundButton("\U0001F4C1")  # folder icon
        browse_btn.setToolTip("Browse...")
        browse_btn.clicked.connect(self._browse)
        row.addWidget(browse_btn)
        self.browse_button = browse_btn

        layout.addSpacing(14)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = RoundButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        save_btn = RoundButton("Save", accent=True)
        save_btn.clicked.connect(self._save)
        buttons.addWidget(save_btn)
        layout.addLayout(buttons)

    def _row(self, layout: QVBoxLayout, label: str, *, stretch: bool = True) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setProperty("dim", True)
        row.addWidget(lbl)
        if stretch:
            row.addStretch(1)
        layout.addSpacing(10)
        layout.addLayout(row)
        return row

    def _on_color_choice(self, choice: str) -> None:
        self.custom_color.setVisible(choice == CUSTOM_COLOR)

    def color(self) -> str:
        choice = self.color_combo.currentText()
        return (self.custom_color.value() or "green") if choice == CUSTOM_COLOR else choice

    def coords(self) -> str:
        return self.coords_control.value()

    def path(self) -> str:
        return os.path.expanduser(self.path_entry.text().strip())

    def _browse(self) -> None:
        chosen, _filter = QFileDialog.getSaveFileName(
            self, "Save as DS9 Region", self.path() or os.getcwd(),
            "DS9 region files (*.reg);;All files (*.*)",
        )
        if chosen:
            self.path_entry.setText(chosen)

    def _save(self) -> None:
        if self.path():
            self.export_requested.emit(self.path(), self.color(), self.coords())
