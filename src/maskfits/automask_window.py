"""A non-modal window for interactively building a threshold-based auto mask.

Computes an iterative sigma-clipped background level (+ error) from the
current entry's displayed data (whatever resolution/smoothing is currently
active - same as manual painting), then flags pixels above bg + kappa *
bg_err as a translucent preview overlay drawn on the main canvas by the app.
Nothing touches the real mask until the user clicks Confirm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QCloseEvent, QColor, QKeyEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from maskfits.automask import (
    BG_METHODS,
    auto_mask_preview,
    background_stats,
    expand_mask,
    filter_by_group_size,
    filter_by_shape,
    fit_gaussian_to_histogram,
    sigma_clip_mask,
    valid_pixels,
)
from maskfits.theme import DARK, LIGHT, current_theme, on_theme_changed
from maskfits.widgets import RangeSlider, RoundButton, RoundSlider, SegmentedControl

if TYPE_CHECKING:
    from maskfits.gui import Entry, MaskFitsApp

GROUP_SIZE_MIN, GROUP_SIZE_MAX = 1, 500  # px
# Starting values of the window's sliders.
DEFAULT_CLEANUP_KAPPA = 2.0
DEFAULT_CLEANUP_ITERATIONS = 10
DEFAULT_MASK_KAPPA = 5.0
DEFAULT_EXPAND_PX = 2
HIST_W = 420
HIST_H = 140
HIST_BINS = 60


class _AutoMaskHistCanvas(QWidget):
    """Histogram of only the pixels that survived clean-up clipping, with the
    Gaussian-fit curve and background/threshold marker lines overlaid."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(HIST_W, HIST_H)
        self._vals: Optional[np.ndarray] = None
        self._bg = 0.0
        self._threshold = 0.0
        on_theme_changed(self, self.update)

    def set_data(self, vals: np.ndarray, bg: float, threshold: float) -> None:
        self._vals = vals
        self._bg = bg
        self._threshold = threshold
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        theme = current_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.panel_bg))
        painter.setPen(QPen(QColor(theme.panel_border), 1))
        painter.drawRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))

        vals = self._vals
        if vals is None or vals.size == 0:
            return

        mean = float(vals.mean())
        std = float(vals.std()) or 1.0
        lo, hi = mean - 8 * std, mean + 8 * std
        if hi <= lo:
            hi = lo + 1.0

        pad = 10
        pad_left = 10
        w, h = HIST_W, HIST_H
        plot_w = w - pad_left - pad
        plot_h = h - 2 * pad
        y_bottom = h - pad

        counts, _edges = np.histogram(vals, bins=HIST_BINS, range=(lo, hi))
        heights = np.log1p(counts.astype(np.float64))
        max_h = heights.max() or 1.0
        bar_w = plot_w / HIST_BINS

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.text_dim))
        for i in range(HIST_BINS):
            if counts[i] == 0:
                continue
            x0 = pad_left + i * bar_w
            bar_h = (heights[i] / max_h) * plot_h
            painter.drawRect(QRectF(x0, y_bottom - bar_h, bar_w, bar_h))

        def x_of(v: float) -> float:
            frac = (v - lo) / (hi - lo) if hi > lo else 0.0
            frac = min(max(frac, 0.0), 1.0)
            return pad_left + frac * plot_w

        fit = fit_gaussian_to_histogram(vals, HIST_BINS, (lo, hi))
        if fit is not None:
            amplitude, mu, sigma = fit
            xs = np.linspace(lo, hi, 150)
            predicted = np.clip(amplitude * np.exp(-0.5 * ((xs - mu) / sigma) ** 2), 0, None)
            fit_heights = np.log1p(predicted)
            path = QPainterPath()
            for i, (x, fh) in enumerate(zip(xs, fit_heights)):
                px = pad_left + ((x - lo) / (hi - lo)) * plot_w
                py = y_bottom - (fh / max_h) * plot_h
                if i == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            pen = QPen(QColor(theme.blue), 2)
            pen.setDashPattern([5, 3])
            painter.setPen(pen)
            painter.drawPath(path)

        bg_x = x_of(self._bg)
        painter.setPen(QPen(QColor(theme.green), 2))
        painter.drawLine(int(bg_x), pad, int(bg_x), h - pad)
        th_x = x_of(self._threshold)
        painter.setPen(QPen(QColor(theme.warning), 2))
        painter.drawLine(int(th_x), pad, int(th_x), h - pad)


class AutoMaskWindow(QDialog):
    def __init__(self, app: "MaskFitsApp", entry: "Entry"):
        super().__init__(app)
        self.app = app
        self.entry = entry
        # A snapshot reference to whatever's currently displayed (binned/
        # smoothed or not) - matches how manual painting already treats
        # "the current image state" as the thing to operate on. Pixels the
        # user already masked manually (or with a previously-confirmed auto
        # mask) are excluded from background stats AND from being (re-)
        # flagged here - see _apply()'s `valid` computation.
        self.data = entry.image.data
        self.existing_mask = entry.image.mask.copy()
        self.setWindowTitle("Auto Mask")
        self.setModal(False)

        self.bg_method = "median"
        self.cleanup_kappa = DEFAULT_CLEANUP_KAPPA
        self.cleanup_iterations = DEFAULT_CLEANUP_ITERATIONS
        self.kappa = DEFAULT_MASK_KAPPA
        # Off by default: keep every flagged region, whatever its size.
        self.group_size_enabled = False
        self.min_group_size = 1
        self.max_group_size = 50
        self.expand_px = DEFAULT_EXPAND_PX
        # None, "lines" or "circles" - see automask.filter_by_shape.
        self.shape_filter: Optional[str] = None

        self._preview: Optional[np.ndarray] = None
        self._resolved = False

        self._build()
        self._apply()
        on_theme_changed(self, self._style_action_buttons)

        self._center_on_parent()

    # ---------------------------------------------------------------- build

    def _center_on_parent(self) -> None:
        parent = self.app
        self.adjustSize()
        geo = parent.frameGeometry()
        x = max(geo.x() + (geo.width() - self.width()) // 2, 0)
        y = max(geo.y() + (geo.height() - self.height()) // 2, 0)
        self.move(x, y)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(0)
        # The word-wrapped labels report a smaller minimum height than they
        # need at their width, so a freely sized dialog came out too short
        # and the layout squeezed the fixed-height histogram row - the
        # canvas then overflowed into the legend below it. Sizing the
        # dialog to exactly its layout's needs avoids that.
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        title = QLabel("Auto Mask")
        layout.addWidget(title)
        desc = QLabel(
            "Flags pixels above background + κ × background error as a preview "
            "overlay, updated live as the sliders below change - nothing is written "
            "to the real mask until Confirm."
        )
        desc.setWordWrap(True)
        desc.setProperty("dim", True)
        desc.setFixedWidth(HIST_W)
        layout.addSpacing(2)
        layout.addWidget(desc)

        self._method_row(layout, "Background", [(m, m.capitalize()) for m in BG_METHODS],
                          self.bg_method, self._on_bg_method_changed)
        self._slider_row(layout, "Clean-Up κ  (background isolation)", self.cleanup_kappa, 0.5, 10.0,
                          on_change=self._on_cleanup_kappa_changed)
        self._slider_row(layout, "Clean-Up Iterations", self.cleanup_iterations, 1, 20, integer=True,
                          on_change=self._on_cleanup_iterations_changed)

        self.hist_canvas = _AutoMaskHistCanvas(self)
        hist_row = QHBoxLayout()
        hist_row.addWidget(self.hist_canvas)
        hist_row.addStretch(1)
        layout.addSpacing(10)
        layout.addLayout(hist_row)

        legend = QHBoxLayout()
        legend.setSpacing(0)
        theme = current_theme()
        self._legend_swatch(legend, theme.blue, "Gaussian Fit")
        self._legend_swatch(legend, theme.green, "Background")
        self._legend_swatch(legend, theme.warning, "Mask Threshold")
        legend.addStretch(1)
        layout.addLayout(legend)

        # The threshold/expand knobs live below the histogram, separate from
        # the clean-up block above it - they shape the FINAL flagged mask,
        # not the background estimate the histogram is showing.
        self._slider_row(layout, "Mask κ  (threshold above background)", self.kappa, 0.5, 20.0,
                          on_change=self._on_kappa_changed)
        self._group_size_row(layout)
        self._shape_row(layout)
        self._slider_row(layout, "Expand  (pad each flagged region, px)", self.expand_px, 0, 20, integer=True,
                          on_change=self._on_expand_changed)

        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        layout.addSpacing(8)
        layout.addWidget(self.stats_label)

        layout.addSpacing(4)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        confirm_btn = RoundButton("Confirm")
        confirm_btn.clicked.connect(self._confirm)
        discard_btn = RoundButton("Discard", danger=True)
        discard_btn.clicked.connect(self._discard)
        btn_row.addWidget(discard_btn)
        btn_row.addWidget(confirm_btn)
        layout.addLayout(btn_row)
        self._confirm_btn = confirm_btn
        self._discard_btn = discard_btn
        self._style_action_buttons()

    def _style_action_buttons(self) -> None:
        """Confirm/Discard always use the built-in dark/light theme's green
        and red (whichever mode is active), never a custom theme's - they
        must stay recognizable as accept/reject whatever the palette."""
        base = LIGHT if current_theme().mode == "light" else DARK
        self._confirm_btn.setStyleSheet(
            f"QPushButton {{ background-color: {base.green}; color: white; }}"
            f"QPushButton:hover {{ background-color: {base.green}; }}"
        )
        self._discard_btn.setStyleSheet(
            f"QPushButton {{ background-color: {base.danger}; color: white; }}"
            f"QPushButton:hover {{ background-color: {base.danger_hover}; }}"
        )

    @staticmethod
    def _legend_swatch(layout: QHBoxLayout, color: str, label: str) -> None:
        box = QHBoxLayout()
        box.setSpacing(4)
        swatch = QLabel()
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
        box.addWidget(swatch)
        text = QLabel(label)
        text.setProperty("dim", True)
        box.addWidget(text)
        wrapper = QWidget()
        wrapper.setLayout(box)
        layout.addWidget(wrapper)
        layout.setSpacing(12)

    def _method_row(self, layout: QVBoxLayout, label: str, options: list[tuple[str, str]], value: str,
                     on_change) -> None:
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setProperty("dim", True)
        row.addWidget(lbl)
        row.addStretch(1)
        control = SegmentedControl(options, value)
        control.valueChanged.connect(on_change)
        row.addWidget(control)
        layout.addSpacing(10)
        layout.addLayout(row)

    def _group_size_row(self, layout: QVBoxLayout) -> None:
        """Optional size window for flagged regions: a checkbox (the limits
        only apply while checked - see _apply()), min and max entries, and a
        two-handle slider for both."""
        row = QHBoxLayout()
        self.group_size_checkbox = QCheckBox()
        self.group_size_checkbox.setChecked(self.group_size_enabled)
        self.group_size_checkbox.toggled.connect(self._on_group_size_enabled_changed)
        row.addWidget(self.group_size_checkbox)
        lbl = QLabel("Group Size  (px, keeps flagged regions within this range):")
        lbl.setProperty("dim", True)
        row.addWidget(lbl)
        row.addStretch(1)
        entries = []
        for value in (self.min_group_size, self.max_group_size):
            entry = QLineEdit(str(value))
            entry.setFixedWidth(48)
            entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            entries.append(entry)
        self.min_group_entry, self.max_group_entry = entries
        row.addWidget(self.min_group_entry)
        dash = QLabel("\u2013")
        dash.setProperty("dim", True)
        row.addWidget(dash)
        row.addWidget(self.max_group_entry)
        layout.addSpacing(10)
        layout.addLayout(row)

        self.group_size_slider = RangeSlider(GROUP_SIZE_MIN, GROUP_SIZE_MAX, self.min_group_size,
                                             self.max_group_size, integer=True)
        self.group_size_slider.setFixedWidth(HIST_W)
        layout.addWidget(self.group_size_slider)

        def apply_entries() -> None:
            lower, upper = self.group_size_slider.values()
            try:
                lower = float(self.min_group_entry.text())
                upper = float(self.max_group_entry.text())
            except ValueError:
                pass
            self.group_size_slider.setValues(lower, upper)
            sync_entries(*self.group_size_slider.values())

        def sync_entries(lower: float, upper: float) -> None:
            self.min_group_entry.setText(str(int(lower)))
            self.max_group_entry.setText(str(int(upper)))

        self.min_group_entry.editingFinished.connect(apply_entries)
        self.max_group_entry.editingFinished.connect(apply_entries)
        self.group_size_slider.valuesChanged.connect(sync_entries)
        self.group_size_slider.valuesChanged.connect(self._on_group_size_changed)

    def _shape_row(self, layout: QVBoxLayout) -> None:
        """"Satellites" / "Galaxies & Stars": keep only the elongated
        (trail, spike, plane) or only the round flagged regions. Ticking one
        unticks the other; both unticked keeps every shape."""
        row = QHBoxLayout()
        self.shape_checkboxes: dict[str, QCheckBox] = {}
        tooltips = {
            "lines": "Keep only regions at least 3x longer than wide (satellite trails, spikes, planes).",
            "circles": "Keep only regions rounder than that (stars, galaxies).",
        }
        for value, text in (("lines", "Satellites"), ("circles", "Galaxies && Stars")):
            checkbox = QCheckBox(text)
            checkbox.setToolTip(tooltips[value])
            checkbox.toggled.connect(lambda checked, v=value: self._on_shape_toggled(v, checked))
            if self.shape_checkboxes:
                row.addSpacing(16)
            row.addWidget(checkbox)
            self.shape_checkboxes[value] = checkbox
        row.addStretch(1)
        layout.addSpacing(10)
        layout.addLayout(row)

    def _slider_row(self, layout: QVBoxLayout, label: str, value: float, lo: float, hi: float, *,
                     integer: bool = False, on_change=None) -> None:
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setProperty("dim", True)
        row.addWidget(lbl)
        row.addStretch(1)
        entry = QLineEdit(f"{value:.3g}")
        entry.setFixedWidth(56)
        entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(entry)
        layout.addSpacing(10)
        layout.addLayout(row)

        slider = RoundSlider(lo, hi, value, integer=integer)
        slider.setFixedWidth(HIST_W)
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
            if on_change is not None:
                on_change(slider.value())

        def sync_entry(v: float) -> None:
            entry.setText(f"{v:.3g}")
            if on_change is not None:
                on_change(v)

        entry.editingFinished.connect(apply_entry)
        slider.valueChanged.connect(sync_entry)

    # ------------------------------------------------------- param setters

    def _on_bg_method_changed(self, v: str) -> None:
        self.bg_method = v
        self._apply()

    def _on_cleanup_kappa_changed(self, v: float) -> None:
        self.cleanup_kappa = v
        self._apply()

    def _on_cleanup_iterations_changed(self, v: float) -> None:
        self.cleanup_iterations = int(v)
        self._apply()

    def _on_kappa_changed(self, v: float) -> None:
        self.kappa = v
        self._apply()

    def _on_group_size_changed(self, lower: float, upper: float) -> None:
        self.min_group_size, self.max_group_size = int(lower), int(upper)
        if self.group_size_enabled:
            self._apply()

    def _on_group_size_enabled_changed(self, checked: bool) -> None:
        self.group_size_enabled = checked
        self._apply()

    def _on_shape_toggled(self, value: str, checked: bool) -> None:
        if checked:
            for other, checkbox in self.shape_checkboxes.items():
                if other != value and checkbox.isChecked():
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)
            self.shape_filter = value
        elif self.shape_filter == value:
            self.shape_filter = None
        self._apply()

    def _on_expand_changed(self, v: float) -> None:
        self.expand_px = int(v)
        self._apply()

    # ---------------------------------------------------------------- logic

    def _apply(self) -> None:
        data = self.data
        # Already-masked pixels (manually painted, or a previously-confirmed
        # auto mask) are excluded from background stats and can never be
        # (re-)flagged - as far as this tool is concerned they don't exist
        # in the image.
        valid = valid_pixels(data) & ~self.existing_mask
        kept = sigma_clip_mask(data, valid, self.cleanup_kappa, max_iter=self.cleanup_iterations)
        bg, bg_err = background_stats(data, kept, self.bg_method)
        preview = auto_mask_preview(data, valid, bg, bg_err, self.kappa)
        threshold = bg + self.kappa * bg_err
        # Group-size filtering runs on the RAW flagged regions, before expand
        # pads them - padding first would inflate every group's size and
        # defeat the point of keeping only compact, point-like sources.
        if self.group_size_enabled:
            preview = filter_by_group_size(preview, max_size=self.max_group_size, min_size=self.min_group_size)
        # Shape is judged on the raw flags too - expand would round off a
        # thin streak's ends and width.
        if self.shape_filter is not None:
            preview = filter_by_shape(preview, self.shape_filter)
        # Expansion only pads the FINAL flagged regions - it has no bearing
        # on background isolation, so it's applied after everything the
        # histogram/stats below are about.
        preview = expand_mask(preview, self.expand_px)

        self._preview = preview

        flagged = int(preview.sum())
        total = data.size
        pct = 100.0 * flagged / total if total else 0.0
        self.stats_label.setText(
            f"Background: {bg:.4g}    Error: {bg_err:.4g}    Threshold: {threshold:.4g}\n"
            f"Flagged: {flagged:,} px ({pct:.2f}%)    Background Pixels Used: {int(kept.sum()):,}"
        )
        self.hist_canvas.set_data(data[kept], bg, threshold)
        self.app.set_auto_mask_preview(self.entry, preview)

    def _confirm(self) -> None:
        self._resolved = True
        if self._preview is not None:
            self.app.confirm_auto_mask(self.entry, self._preview)
        else:
            self.app.discard_auto_mask(self.entry)
        self.close()

    def _discard(self) -> None:
        self._resolved = True
        self.app.discard_auto_mask(self.entry)
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._resolved:
            self._resolved = True
            self.app.discard_auto_mask(self.entry)
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._discard()
            return
        super().keyPressEvent(event)
