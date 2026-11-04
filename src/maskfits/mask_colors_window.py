"""Settings -> "Edit Mask Colors...": a custom mask overlay color per
colormap. Each row shows the colormap as a gradient with the mask color as a
bar right below it, so the contrast between the two is visible while
picking. An empty field means "auto" - the colormap's automatic color (see
colormaps.mask_tint_for).

No live preview, like the shortcuts editor - changes take effect when this
window and then Settings are saved.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from astropy.io import fits
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from maskfits.colormaps import COLORMAP_LUTS, COLORMAP_NAMES, ISOPY_NAME, build_isopy_lut, mask_tint_for
from maskfits.imagedata import isopy_cuts_and_stops
from maskfits.settings import sanitize_mask_colors
from maskfits.theme import current_theme, hex_to_rgb, on_theme_changed, rgb_to_hex
from maskfits.widgets import HexColorPicker, RoundButton

PREVIEW_W = 220
GRADIENT_H = 16
MASK_BAR_H = 10


def colormap_lut(name: str) -> np.ndarray:
    """The LUT drawn for a colormap here. IsoPy's real stops depend on each
    image's zero point and pixel scale, so it's shown with the defaults."""
    if name == ISOPY_NAME:
        _vmin, _vmax, stops = isopy_cuts_and_stops(fits.Header(), None)
        return build_isopy_lut(stops)
    return COLORMAP_LUTS[name]


def auto_mask_color(name: str) -> tuple[int, int, int]:
    return mask_tint_for(name, colormap_lut(name), current_theme().accent)


class ColormapMaskPreview(QWidget):
    """The colormap as a horizontal gradient, the mask color as a bar below."""

    def __init__(self, lut: np.ndarray, mask_rgb: tuple[int, int, int], parent: Optional[QWidget] = None):
        super().__init__(parent)
        strip = np.ascontiguousarray(np.repeat(lut[np.newaxis, :, :], 2, axis=0))
        self._gradient = QImage(strip.data, strip.shape[1], strip.shape[0], 3 * strip.shape[1],
                                QImage.Format.Format_RGB888).copy()
        self.mask_rgb = mask_rgb
        self.setFixedSize(PREVIEW_W, GRADIENT_H + MASK_BAR_H)

    def set_mask_rgb(self, rgb: tuple[int, int, int]) -> None:
        self.mask_rgb = rgb
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.drawImage(QRect(0, 0, self.width(), GRADIENT_H), self._gradient)
        painter.fillRect(0, GRADIENT_H, self.width(), MASK_BAR_H, QColor(*self.mask_rgb))
        painter.setPen(QColor(current_theme().panel_border))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)


class MaskColorsWindow(QDialog):
    mask_colors_saved = Signal(dict)

    def __init__(self, mask_colors: dict[str, str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Mask Colors")
        # Non-modal, same as the other Settings sub-windows (Cocoa
        # modal-nesting bug - see SettingsWindow).
        self.setModal(False)
        self.mask_colors: dict[str, str] = dict(mask_colors)  # working copy
        self.pickers: dict[str, HexColorPicker] = {}
        self.previews: dict[str, ColormapMaskPreview] = {}
        self._build()
        on_theme_changed(self, self._refresh_auto_colors)
        self._center_on_parent()

    def _center_on_parent(self) -> None:
        parent = self.parentWidget()
        self.adjustSize()
        if parent is None:
            return
        geo = parent.frameGeometry()
        self.move(max(geo.x() + (geo.width() - self.width()) // 2, 0),
                  max(geo.y() + (geo.height() - self.height()) // 2, 0))

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(0)

        layout.addWidget(QLabel("Mask Colors"))
        desc = QLabel("The mask overlay color for each colormap, shown below the colormap itself. "
                      "Leave a field empty to use the automatic color.")
        desc.setWordWrap(True)
        desc.setProperty("dim", True)
        desc.setFixedWidth(PREVIEW_W + 260)
        layout.addSpacing(2)
        layout.addWidget(desc)

        for name in COLORMAP_NAMES:
            layout.addSpacing(12)
            row = QHBoxLayout()
            column = QVBoxLayout()
            column.setSpacing(4)
            label = QLabel(name)
            label.setProperty("dim", True)
            column.addWidget(label)
            preview = ColormapMaskPreview(colormap_lut(name), self.effective_color(name))
            column.addWidget(preview)
            row.addLayout(column)
            row.addStretch(1)

            picker = HexColorPicker(self.mask_colors.get(name), allow_empty=True, placeholder="Auto")
            picker.set_empty_color(rgb_to_hex(auto_mask_color(name)))
            picker.colorChanged.connect(lambda value, n=name: self._on_color_changed(n, value))
            row.addWidget(picker, 0, Qt.AlignmentFlag.AlignBottom)
            reset_btn = RoundButton("Reset")
            reset_btn.clicked.connect(lambda _checked=False, p=picker: p.setValue(None))
            row.addWidget(reset_btn, 0, Qt.AlignmentFlag.AlignBottom)
            layout.addLayout(row)
            self.pickers[name] = picker
            self.previews[name] = preview

        layout.addSpacing(16)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = RoundButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        save_btn = RoundButton("Save", accent=True)
        save_btn.clicked.connect(self._save)
        buttons.addWidget(save_btn)
        layout.addLayout(buttons)

    def effective_color(self, name: str) -> tuple[int, int, int]:
        custom = self.mask_colors.get(name)
        return hex_to_rgb(custom) if custom else auto_mask_color(name)

    def _on_color_changed(self, name: str, value: str) -> None:
        if value:
            self.mask_colors[name] = value
        else:
            self.mask_colors.pop(name, None)
        self.previews[name].set_mask_rgb(self.effective_color(name))

    def _refresh_auto_colors(self) -> None:
        # Grayscale's automatic color is the theme accent.
        for name, preview in self.previews.items():
            preview.set_mask_rgb(self.effective_color(name))
            self.pickers[name].set_empty_color(rgb_to_hex(auto_mask_color(name)))

    def _save(self) -> None:
        self.mask_colors_saved.emit(sanitize_mask_colors(self.mask_colors))
        self.accept()
