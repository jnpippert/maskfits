"""Embeddable pixel-value histogram with draggable lower/upper cut lines.

Used inline in the sidebar (see gui.MaskFitsApp._build_sidebar) so manual cut
levels can be set directly against a live histogram, without a separate
popup window.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from maskfits.imagedata import safe_span
from maskfits.theme import current_theme, theme_manager
from maskfits.widgets import RoundButton

SAMPLE_TARGET = 500_000

# How far past the true pixel min/max the draggable/typeable cut range
# extends, in multiples of the sample's standard deviation. Hard-stopping
# exactly at data_min/data_max is too restrictive - pushing a cut past a
# saturated star core or a cosmic-ray hit, or below the noise floor, is a
# normal thing to want - so the allowed range gets generous headroom instead
# of an unmovable wall right at the observed extremes.
SIGMA_MARGIN = 10.0


class _HistogramCanvas(QWidget):
    """The actual drawable/draggable histogram area - kept separate from the
    entry-box/toggle row so paintEvent only has to think about the plot."""

    def __init__(self, owner: "CutsHistogram", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._owner = owner
        self.setFixedHeight(150)
        self.setMouseTracking(True)
        self._drag: Optional[str] = None  # "lo" | "hi" | None
        self._drag_range: Optional[tuple[float, float]] = None
        # Cached bar heights (log1p-scaled counts) + the range they were
        # computed for - re-binning the full sample is the expensive part,
        # so it's only redone when the display range actually changes
        # (full_rebin), not on every pixel of a drag.
        self._bar_heights: Optional[np.ndarray] = None
        self._max_count = 0
        self._bars_range: Optional[tuple[float, float]] = None

    # -------------------------------------------------------------- layout

    def _pad(self) -> float:
        return max(round(self.height() * 0.11), 10)

    def _pad_left(self) -> float:
        return max(round(self.width() * 0.15), 30)

    def _handle_h(self) -> float:
        return max(round(self.height() * 0.045), 6)

    def _value_to_x(self, value: float, lo: float, hi: float) -> float:
        frac = (value - lo) / (hi - lo) if hi > lo else 0.0
        frac = min(max(frac, 0.0), 1.0)
        return self._pad_left() + frac * (self.width() - self._pad_left() - self._pad())

    def _x_to_value(self, x: float, lo: float, hi: float) -> float:
        span = self.width() - self._pad_left() - self._pad()
        frac = (x - self._pad_left()) / max(span, 1)
        frac = min(max(frac, 0.0), 1.0)
        return lo + frac * (hi - lo)

    # ------------------------------------------------------------- drawing

    def ensure_bars(self, disp_lo: float, disp_hi: float, force: bool = False) -> None:
        if not force and self._bars_range == (disp_lo, disp_hi) and self._bar_heights is not None:
            return
        owner = self._owner
        counts, _edges = np.histogram(owner.sample, bins=owner.bins, range=(disp_lo, disp_hi))
        self._max_count = int(counts.max()) if counts.size else 0
        heights = np.log1p(counts.astype(np.float64))
        self._bar_heights = heights
        self._bars_range = (disp_lo, disp_hi)

    @staticmethod
    def _fmt_tick(t: int) -> str:
        if t <= 0:
            return "0"
        return f"1E+{round(np.log10(t))}"

    def paintEvent(self, _event) -> None:  # noqa: N802
        theme = current_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.panel_bg))
        painter.setPen(QPen(QColor(theme.panel_border), 1))
        painter.drawRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))

        owner = self._owner
        disp_lo, disp_hi = (self._drag_range if self._drag_range is not None
                             else owner.compute_disp_range())
        self.ensure_bars(disp_lo, disp_hi)

        pad = self._pad()
        pad_left = self._pad_left()
        w, h = self.width(), self.height()
        plot_h = h - 2 * pad
        bar_w = (w - pad_left - pad) / max(owner.bins, 1)
        heights = self._bar_heights if self._bar_heights is not None else np.zeros(owner.bins)
        max_h = heights.max() if heights.size and heights.max() > 0 else 1.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.text_dim))
        for i, hgt in enumerate(heights):
            x0 = pad_left + i * bar_w
            y1 = h - pad
            y0 = y1 - (hgt / max_h) * plot_h
            painter.drawRect(QRectF(x0, y0, bar_w, y1 - y0))

        # y axis (log-scaled ticks matching the log1p bar-height scale)
        painter.setPen(QPen(QColor(theme.text_dim), 1))
        ticks = [0]
        v = 1
        while v <= self._max_count:
            ticks.append(v)
            v *= 10
        if len(ticks) == 1:
            ticks.append(max(self._max_count, 1))
        plot_bottom = h - pad
        for t in ticks:
            frac = (np.log1p(t) / max_h) if max_h > 0 else 0.0
            y = plot_bottom - frac * plot_h
            painter.drawLine(int(pad_left - 5), int(y), int(pad_left), int(y))
            painter.drawText(QRectF(0, y - 7, pad_left - 8, 14),
                              Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._fmt_tick(t))

        # overlay: selection band + lowcut/highcut lines + drag handles.
        # The band is drawn on top of the bars (not behind them), so it
        # needs real alpha transparency to still show the bars underneath -
        # a fully opaque brush here would just blot out the histogram
        # within the selected range.
        x_lo = self._value_to_x(owner.vmin, disp_lo, disp_hi)
        x_hi = self._value_to_x(owner.vmax, disp_lo, disp_hi)
        band_color = QColor(theme.button_hover)
        band_color.setAlpha(110)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(band_color)
        painter.drawRect(QRectF(x_lo, pad, x_hi - x_lo, h - 2 * pad))

        lo_color = QColor(theme.danger)
        hi_color = QColor(theme.green)
        painter.setPen(QPen(lo_color, 2))
        painter.drawLine(int(x_lo), int(pad), int(x_lo), int(h - pad))
        painter.setPen(QPen(hi_color, 2))
        painter.drawLine(int(x_hi), int(pad), int(x_hi), int(h - pad))

        handle_h = self._handle_h()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(lo_color)
        painter.drawRect(QRectF(x_lo - 5, pad - handle_h, 10, handle_h))
        painter.setBrush(hi_color)
        painter.drawRect(QRectF(x_hi - 5, pad - handle_h, 10, handle_h))

    # ---------------------------------------------------------------- input

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        owner = self._owner
        disp_lo, disp_hi = owner.compute_disp_range()
        x_lo = self._value_to_x(owner.vmin, disp_lo, disp_hi)
        x_hi = self._value_to_x(owner.vmax, disp_lo, disp_hi)
        x = event.position().x()
        if abs(x - x_lo) <= 8:
            self._drag = "lo"
        elif abs(x - x_hi) <= 8:
            self._drag = "hi"
        else:
            self._drag = None
            return
        self._drag_range = (disp_lo, disp_hi)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is None or self._drag_range is None:
            return
        owner = self._owner
        value = self._x_to_value(event.position().x(), *self._drag_range)
        # No from_entry here (unlike _on_lo_entry/_on_hi_entry) - that
        # param exists to stop set_cuts() from overwriting the text box the
        # user is actively typing into, which doesn't apply to a histogram
        # drag. Passing "lo"/"hi" here (the bug) suppressed updating the
        # very entry box whose handle was being dragged, leaving it stale.
        if self._drag == "lo":
            value = min(value, owner.vmax)
            owner.set_cuts(value, owner.vmax)
        else:
            value = max(value, owner.vmin)
            owner.set_cuts(owner.vmin, value)
        self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is None:
            return
        self._drag = None
        self._drag_range = None
        self.ensure_bars(*self._owner.compute_disp_range(), force=True)
        self.update()
        self._owner.cuts_applied.emit(self._owner.vmin, self._owner.vmax)


class CutsHistogram(QWidget):
    """A live pixel-value histogram with draggable vmin/vmax lines and numeric
    entry boxes. Emits `cutsApplied(lowcut, highcut)` as the user drags or
    types - continuously while dragging is deliberately NOT re-triggered
    (see _HistogramCanvas.ensure_bars/mouseMoveEvent): re-binning on every
    mouse-motion pixel is what makes dragging feel sluggish, so a full
    rebin (and the resulting cuts-applied signal) only fires on release or
    on a typed entry.
    """

    cuts_applied = Signal(float, float)

    def __init__(self, data: np.ndarray, lowcut: float, highcut: float, *, bins: int = 40,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.bins = bins
        self.vmin = lowcut
        self.vmax = highcut
        self.show_full_range = False
        self.sample = np.array([0.0, 1.0])
        self.data_min = 0.0
        self.data_max = 1.0
        self.range_min = 0.0
        self.range_max = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.canvas = _HistogramCanvas(self)
        layout.addWidget(self.canvas)

        entries = QHBoxLayout()
        entries.setSpacing(6)
        lo_label = QLabel("lowcut:")
        lo_label.setProperty("dim", True)
        entries.addWidget(lo_label)
        self._lo_entry = QLineEdit(self._fmt(lowcut))
        self._lo_entry.setFixedWidth(72)
        self._lo_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lo_entry.editingFinished.connect(self._on_lo_entry)
        entries.addWidget(self._lo_entry)

        hi_label = QLabel("highcut:")
        hi_label.setProperty("dim", True)
        entries.addWidget(hi_label)
        self._hi_entry = QLineEdit(self._fmt(highcut))
        self._hi_entry.setFixedWidth(72)
        self._hi_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hi_entry.editingFinished.connect(self._on_hi_entry)
        entries.addWidget(self._hi_entry)

        self._range_btn = RoundButton("Full", checkable=True)
        self._range_btn.toggled.connect(self._on_toggle_range_view)
        entries.addWidget(self._range_btn)
        entries.addStretch(1)
        layout.addLayout(entries)

        theme_manager().theme_changed.connect(lambda _t: self.canvas.update())
        self._set_sample(data)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:.6g}"

    # ------------------------------------------------------------- geometry

    def resize_width(self, width: int) -> None:
        """Re-fit to a new width - e.g. the sidebar panel it's embedded in
        was resized. Callers should only call this once a resize settles
        (see gui.MaskFitsApp._on_sidebar_grip_release), for the same reason
        drags elsewhere in this file only trigger a full rebin on release."""
        self.canvas.ensure_bars(*self.compute_disp_range(), force=True)
        self.canvas.update()

    # ------------------------------------------------------------- data

    def set_data(self, data: np.ndarray, lowcut: float, highcut: float) -> None:
        """Resample from a (possibly new) data array and sync the cut values.

        Call whenever the displayed image or its cuts change from outside -
        switching images, changing the stretch/cut preset, or toggling
        smoothing/binning (which changes the array the histogram should
        actually reflect)."""
        self._set_sample(data)
        self.vmin, self.vmax = lowcut, highcut
        self._lo_entry.setText(self._fmt(lowcut))
        self._hi_entry.setText(self._fmt(highcut))
        self.canvas.ensure_bars(*self.compute_disp_range(), force=True)
        self.canvas.update()

    def _set_sample(self, data: np.ndarray) -> None:
        flat = data.ravel()
        stride = max(1, flat.size // SAMPLE_TARGET)
        sample = flat[::stride]
        sample = sample[np.isfinite(sample) & (sample != 0)]
        if sample.size == 0:
            sample = np.array([0.0, 1.0])
        self.sample = sample
        self.data_min = float(sample.min())
        self.data_max = float(sample.max())
        if self.data_max <= self.data_min:
            self.data_max = self.data_min + 1.0
        sigma = float(np.std(sample))
        margin = SIGMA_MARGIN * sigma if sigma > 0 else (self.data_max - self.data_min)
        self.range_min = self.data_min - margin
        self.range_max = self.data_max + margin

    def compute_disp_range(self) -> tuple[float, float]:
        if self.show_full_range:
            return self.data_min, self.data_max
        span = safe_span(self.vmin, self.vmax)
        margin = span * 0.5
        lo = max(self.vmin - margin, self.range_min)
        hi = min(self.vmax + margin, self.range_max)
        if hi <= lo:
            lo, hi = self.range_min, self.range_max
        return lo, hi

    def _on_toggle_range_view(self, checked: bool) -> None:
        self.show_full_range = checked
        self.canvas.ensure_bars(*self.compute_disp_range(), force=True)
        self.canvas.update()

    def set_cuts(self, lowcut: float, highcut: float, from_entry: Optional[str] = None) -> None:
        self.vmin, self.vmax = lowcut, highcut
        if from_entry != "lo":
            self._lo_entry.setText(self._fmt(lowcut))
        if from_entry != "hi":
            self._hi_entry.setText(self._fmt(highcut))

    def _on_lo_entry(self) -> None:
        try:
            value = float(self._lo_entry.text())
        except ValueError:
            self._lo_entry.setText(self._fmt(self.vmin))
            return
        value = max(value, self.range_min)
        value = min(value, self.vmax)
        self.set_cuts(value, self.vmax, from_entry="lo")
        self.canvas.ensure_bars(*self.compute_disp_range(), force=True)
        self.canvas.update()
        self.cuts_applied.emit(self.vmin, self.vmax)

    def _on_hi_entry(self) -> None:
        try:
            value = float(self._hi_entry.text())
        except ValueError:
            self._hi_entry.setText(self._fmt(self.vmax))
            return
        value = min(value, self.range_max)
        value = max(value, self.vmin)
        self.set_cuts(self.vmin, value, from_entry="hi")
        self.canvas.ensure_bars(*self.compute_disp_range(), force=True)
        self.canvas.update()
        self.cuts_applied.emit(self.vmin, self.vmax)
