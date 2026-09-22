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
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
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
    fit_gaussian_to_histogram,
    neural_network_mask,
    sigma_clip_mask,
    valid_pixels,
)
from maskfits.theme import current_theme, theme_manager
from maskfits.widgets import RoundButton, RoundSlider, SegmentedControl

if TYPE_CHECKING:
    from maskfits.gui import Entry, MaskFitsApp

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
        theme_manager().theme_changed.connect(lambda _t: self.update())

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

        self.bg_method = "constant"
        self.error_method = "sigma"
        self.cleanup_kappa = 4.0
        self.cleanup_iterations = 10
        self.kappa = 5.0
        self.max_group_size_enabled = True
        self.max_group_size = 50
        self.expand_px = 5

        self._preview: Optional[np.ndarray] = None
        self.nn_model_path: Optional[str] = None
        self._resolved = False

        self._build()
        self._apply()
        theme_manager().theme_changed.connect(lambda _t: self._style_confirm_button())

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

        self._method_row(layout, "background method", [(m, m.capitalize()) for m in BG_METHODS],
                          self.bg_method, self._on_bg_method_changed)
        self._method_row(layout, "background error", [("sigma", "Sigma"), ("sem", "SEM")],
                          self.error_method, self._on_error_method_changed)
        self._slider_row(layout, "clean-up κ  (background isolation)", self.cleanup_kappa, 0.5, 10.0,
                          on_change=self._on_cleanup_kappa_changed)
        self._slider_row(layout, "clean-up iterations", self.cleanup_iterations, 1, 20, integer=True,
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
        self._legend_swatch(legend, theme.text_dim, "background (kept after clipping)")
        self._legend_swatch(legend, theme.blue, "gaussian fit")
        self._legend_swatch(legend, theme.green, "background")
        self._legend_swatch(legend, theme.warning, "mask threshold")
        legend.addStretch(1)
        layout.addLayout(legend)

        # The threshold/expand knobs live below the histogram, separate from
        # the clean-up block above it - they shape the FINAL flagged mask,
        # not the background estimate the histogram is showing.
        self._slider_row(layout, "mask κ  (threshold above background)", self.kappa, 0.5, 20.0,
                          on_change=self._on_kappa_changed)
        self._slider_row(layout, "max group size  (px, drops larger flagged regions)", self.max_group_size,
                          1, 500, integer=True, enabled_checkbox=True, on_change=self._on_max_group_size_changed)
        self._slider_row(layout, "expand  (pad each flagged region, px)", self.expand_px, 0, 20, integer=True,
                          on_change=self._on_expand_changed)

        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        layout.addSpacing(8)
        layout.addWidget(self.stats_label)

        self._build_nn_section(layout)

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
        self._style_confirm_button()

    def _style_confirm_button(self) -> None:
        theme = current_theme()
        self._confirm_btn.setStyleSheet(
            f"QPushButton {{ background-color: {theme.green}; color: white; }}"
            f"QPushButton:hover {{ background-color: {theme.green}; }}"
        )

    def _build_nn_section(self, layout: QVBoxLayout) -> None:
        """Forward-looking hook, not functional yet: lets a model file be
        picked and wires an "apply" button through to
        automask.neural_network_mask, which currently just raises
        NotImplementedError - so the interface (pick a model, apply it, merge
        its output the same way as the threshold preview) is in place for a
        real model to be dropped into later without any GUI changes."""
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addSpacing(2)
        layout.addWidget(divider)
        layout.addSpacing(8)

        caption = QLabel("Neural network masking (future / experimental):")
        caption.setProperty("dim", True)
        layout.addWidget(caption)

        row = QHBoxLayout()
        self.nn_model_label = QLabel("no model loaded")
        self.nn_model_label.setProperty("dim", True)
        row.addWidget(self.nn_model_label, 1)
        load_btn = RoundButton("Load Model...")
        load_btn.clicked.connect(self._load_nn_model)
        row.addWidget(load_btn)
        layout.addSpacing(4)
        layout.addLayout(row)

        nn_btn_row = QHBoxLayout()
        apply_btn = RoundButton("Apply NN Mask")
        apply_btn.clicked.connect(self._apply_nn)
        nn_btn_row.addWidget(apply_btn)
        nn_btn_row.addStretch(1)
        layout.addSpacing(6)
        layout.addLayout(nn_btn_row)

        self.nn_status_label = QLabel()
        self.nn_status_label.setWordWrap(True)
        self.nn_status_label.setFixedWidth(HIST_W)
        self.nn_status_label.setProperty("dim", True)
        layout.addSpacing(4)
        layout.addWidget(self.nn_status_label)

    def _load_nn_model(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Load a masking model")
        if not path:
            return
        self.nn_model_path = path
        self.nn_model_label.setText(path.rsplit("/", 1)[-1])

    def _apply_nn(self) -> None:
        if not self.nn_model_path:
            self.nn_status_label.setText("Load a model file first.")
            return
        try:
            preview = neural_network_mask(self.data, self.nn_model_path)
        except NotImplementedError as exc:
            self.nn_status_label.setText(str(exc))
            return
        self._preview = preview
        self.app.set_auto_mask_preview(self.entry, preview)
        self.nn_status_label.setText(f"NN mask applied: {int(preview.sum()):,} px flagged.")

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

    def _slider_row(self, layout: QVBoxLayout, label: str, value: float, lo: float, hi: float, *,
                     integer: bool = False, enabled_checkbox: bool = False, on_change=None) -> None:
        row = QHBoxLayout()
        checkbox: Optional[QCheckBox] = None
        if enabled_checkbox:
            # An optional on/off switch for this parameter - the constraint
            # only applies while checked (see _apply()); the slider/entry
            # stay interactive either way, they just have no effect while
            # unchecked.
            checkbox = QCheckBox()
            checkbox.setChecked(self.max_group_size_enabled)
            checkbox.toggled.connect(self._on_max_group_size_enabled_changed)
            row.addWidget(checkbox)
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

    def _on_error_method_changed(self, v: str) -> None:
        self.error_method = v
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

    def _on_max_group_size_changed(self, v: float) -> None:
        self.max_group_size = int(v)
        self._apply()

    def _on_max_group_size_enabled_changed(self, checked: bool) -> None:
        self.max_group_size_enabled = checked
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
        bg, bg_err = background_stats(data, kept, self.error_method)
        preview = auto_mask_preview(data, valid, bg, bg_err, self.kappa)
        threshold = bg + self.kappa * bg_err
        # Group-size filtering runs on the RAW flagged regions, before expand
        # pads them - padding first would inflate every group's size and
        # defeat the point of keeping only compact, point-like sources.
        max_group_size = self.max_group_size if self.max_group_size_enabled else 0
        preview = filter_by_group_size(preview, max_group_size)
        # Expansion only pads the FINAL flagged regions - it has no bearing
        # on background isolation, so it's applied after everything the
        # histogram/stats below are about.
        preview = expand_mask(preview, self.expand_px)

        self._preview = preview

        flagged = int(preview.sum())
        total = data.size
        pct = 100.0 * flagged / total if total else 0.0
        self.stats_label.setText(
            f"background: {bg:.4g}    error: {bg_err:.4g}    threshold: {threshold:.4g}\n"
            f"flagged: {flagged:,} px ({pct:.2f}%)    background pixels used: {int(kept.sum()):,}"
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
