"""The 1D spectrum panel shown under the image for data cubes: the cube's
values along its third axis at the cursor's pixel, against the spectral
axis from the header (see imagedata.cube_spectral_axis), with a marker at
the slice currently displayed.

Custom-painted rather than matplotlib - the same reason as the other plots
in this app: no heavy plotting dependency, and it follows the live theme.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from maskfits.imagedata import SpectralAxis
from maskfits.theme import current_theme, on_theme_changed

SPECTRUM_H = 200
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 86, 14, 12, 36
TARGET_TICKS = 6
# Up to this many slices, each one is also drawn as a dot - with only a few
# bands (e.g. ugriz) a bare line would hide where the samples actually are.
DOT_MAX_SLICES = 40


def nice_ticks(lo: float, hi: float, target: int = TARGET_TICKS) -> list[float]:
    """Round tick positions (steps of 1, 2 or 5 x 10^n) covering [lo, hi]."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo] if math.isfinite(lo) else []
    raw = (hi - lo) / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    # Whole multiples of the step, rounded to 12 significant digits so
    # float noise (e.g. -3 * 1e-18 = -3.0000000000000002e-18) can't push a
    # tick just outside the range.
    first, last = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    return [float(f"{k * step:.12g}") for k in range(first, last + 1)]


# Axis symbol per spectral quantity (see imagedata.SpectralAxis.quantity).
QUANTITY_SYMBOLS = {"wavelength": "\u03bb", "frequency": "\u03bd", "velocity": "v", "slice": "slice"}
ANGSTROM_SPELLINGS = {"angstrom", "angstroms", "aa", "a", "\u00e5"}


def display_unit(unit: str) -> str:
    """Angstrom as the \u00c5 symbol - on its own or inside a compound unit
    like "erg/s/cm2/Angstrom"."""
    unit = unit.strip()
    if unit.lower() in ANGSTROM_SPELLINGS:
        return "\u00c5"
    return re.sub(r"angstroms?", "\u00c5", unit, flags=re.IGNORECASE)


def axis_title(name: str, unit: str) -> str:
    unit = display_unit(unit)
    return f"{name} [{unit}]" if unit else name


class SpectrumView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(SPECTRUM_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.axis: Optional[SpectralAxis] = None
        self.flux_unit = ""
        self.spectrum: Optional[np.ndarray] = None
        self.marker: Optional[int] = None
        on_theme_changed(self, self.update)

    # ------------------------------------------------------------- state

    def set_axis(self, axis: SpectralAxis, flux_unit: str) -> None:
        self.axis = axis
        self.flux_unit = flux_unit
        self.update()

    def set_spectrum(self, values: Optional[np.ndarray]) -> None:
        self.spectrum = None if values is None else np.asarray(values, dtype=float)
        self.update()

    def set_marker(self, index: Optional[int]) -> None:
        self.marker = index
        self.update()

    def x_title(self) -> str:
        if self.axis is None:
            return ""
        quantity = self.axis.quantity
        return axis_title(QUANTITY_SYMBOLS.get(quantity, quantity), self.axis.unit)

    def y_title(self) -> str:
        return axis_title("Flux", self.flux_unit)

    def y_range(self) -> Optional[tuple[float, float]]:
        if self.spectrum is None:
            return None
        finite = self.spectrum[np.isfinite(self.spectrum)]
        if finite.size == 0:
            return None
        lo, hi = float(finite.min()), float(finite.max())
        if hi == lo:
            pad = abs(lo) * 0.1 or 1.0
        else:
            pad = (hi - lo) * 0.06
        return lo - pad, hi + pad

    # ------------------------------------------------------------- paint

    def plot_rect(self) -> QRectF:
        return QRectF(MARGIN_LEFT, MARGIN_TOP, max(self.width() - MARGIN_LEFT - MARGIN_RIGHT, 1),
                      max(self.height() - MARGIN_TOP - MARGIN_BOTTOM, 1))

    def paintEvent(self, event) -> None:  # noqa: N802
        theme = current_theme()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.canvas_bg))
        painter.setPen(QColor(theme.panel_border))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        if self.axis is None or self.axis.values.size == 0:
            return

        rect = self.plot_rect()
        dim = QColor(theme.text_dim)
        painter.setPen(dim)
        painter.drawText(QRectF(rect.left(), self.height() - 18, rect.width(), 16),
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, self.x_title())
        # Y title rotated along the left edge, centered on the plot.
        painter.save()
        painter.translate(12, rect.center().y())
        painter.rotate(-90)
        font = painter.font()
        while painter.fontMetrics().horizontalAdvance(self.y_title()) > rect.height() and font.pointSizeF() > 7:
            font.setPointSizeF(font.pointSizeF() - 0.5)  # shrink a long unit to fit the axis
            painter.setFont(font)
        painter.drawText(QRectF(-rect.height() / 2, -8, rect.height(), 16), Qt.AlignmentFlag.AlignCenter,
                         self.y_title())
        painter.restore()

        xs = self.axis.values
        x_lo, x_hi = float(np.nanmin(xs)), float(np.nanmax(xs))
        if x_hi == x_lo:
            x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
        if xs.size <= DOT_MAX_SLICES:
            # Room for the first/last dot, which would otherwise be cut in half.
            pad = (x_hi - x_lo) * 0.03
            x_lo, x_hi = x_lo - pad, x_hi + pad

        def px(x: float) -> float:
            return rect.left() + (x - x_lo) / (x_hi - x_lo) * rect.width()

        grid = QColor(theme.panel_border)
        for tick in nice_ticks(x_lo, x_hi):
            x = px(tick)
            painter.setPen(grid)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.setPen(dim)
            painter.drawText(QRectF(x - 40, rect.bottom() + 2, 80, 14), Qt.AlignmentFlag.AlignHCenter,
                             f"{tick:.6g}")

        y_range = self.y_range()
        if y_range is not None:
            y_lo, y_hi = y_range

            def py(y: float) -> float:
                return rect.bottom() - (y - y_lo) / (y_hi - y_lo) * rect.height()

            for tick in nice_ticks(y_lo, y_hi, target=4):
                y = py(tick)
                painter.setPen(grid)
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
                painter.setPen(dim)
                painter.drawText(QRectF(0, y - 7, MARGIN_LEFT - 6, 14),
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{tick:.4g}")

        painter.setPen(grid)
        painter.drawRect(rect)

        if self.marker is not None and 0 <= self.marker < xs.size:
            painter.setPen(QPen(QColor(theme.warning), 1.5))
            x = px(float(xs[self.marker]))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

        if y_range is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(rect)
        points = [(px(float(x)), py(float(y))) if np.isfinite(y) else None for x, y in zip(xs, self.spectrum)]
        path = QPainterPath()
        pen_down = False
        for point in points:
            if point is None:
                pen_down = False  # NaN: break the line instead of bridging the gap
                continue
            if pen_down:
                path.lineTo(*point)
            else:
                path.moveTo(*point)
                pen_down = True
        accent = QColor(theme.accent)
        painter.setPen(QPen(accent, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        if xs.size <= DOT_MAX_SLICES:
            painter.setBrush(accent)
            for point in points:
                if point is not None:
                    painter.drawEllipse(QPointF(*point), 3, 3)
