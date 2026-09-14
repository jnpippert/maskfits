"""A Toplevel window for interactively building a threshold-based auto mask.

Computes an iterative sigma-clipped background level (+ error) from the
current entry's displayed data (whatever resolution/smoothing is currently
active - same as manual painting), then flags pixels above bg + kappa *
bg_err as a translucent preview overlay drawn on the main canvas by the app.
Nothing touches the real mask until the user clicks Confirm.
"""

import tkinter as tk
from tkinter import filedialog
from typing import TYPE_CHECKING, Optional

import numpy as np

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
from maskfits.theme import (
    ACCENT_TEXT,
    BLUE,
    BUTTON_BG,
    FONT,
    FONT_SMALL,
    GREEN,
    PANEL_BG,
    PANEL_BORDER,
    TEXT,
    TEXT_DIM,
    WARNING,
)
from maskfits.widgets import RoundButton, RoundSlider, SegmentedControl

if TYPE_CHECKING:
    from maskfits.gui import Entry, MaskFitsApp

HIST_W = 420
HIST_H = 140
HIST_BINS = 60


class AutoMaskWindow(tk.Toplevel):
    def __init__(self, app: "MaskFitsApp", entry: "Entry"):
        super().__init__(app.root)
        self.app = app
        self.entry = entry
        # A snapshot reference to whatever's currently displayed (binned/
        # smoothed or not) - matches how manual painting already treats
        # "the current image state" as the thing to operate on. Pixels the
        # user already masked manually (or with a previously-confirmed auto
        # mask) are excluded from background stats AND from being (re-)
        # flagged here - see _apply()'s `valid` computation - since as far
        # as this tool is concerned, they no longer exist in the image.
        self.data = entry.image.data
        self.existing_mask = entry.image.mask.copy()
        self.title("Auto Mask")
        self.configure(bg=PANEL_BG)
        self.resizable(False, False)
        self.transient(app.root)

        self.bg_method = tk.StringVar(value="constant")
        self.error_method = tk.StringVar(value="sigma")
        self.cleanup_kappa = tk.DoubleVar(value=4.0)
        self.cleanup_iterations = tk.IntVar(value=10)
        self.kappa = tk.DoubleVar(value=5.0)
        self.max_group_size = tk.IntVar(value=50)
        self.expand_px = tk.IntVar(value=5)

        self._preview: Optional[np.ndarray] = None
        self.nn_model_path: Optional[str] = None

        self._build()
        # Live-updates the preview on every parameter change - no separate
        # Apply step. Fires on every tick of a slider drag (same as the
        # main app's other live sliders, e.g. mask_alpha/radius), not just on
        # release - simplest way to make it feel immediate, at the cost of
        # recomputing more often than strictly necessary during a drag.
        for var in (self.bg_method, self.error_method, self.cleanup_kappa,
                    self.cleanup_iterations, self.kappa, self.max_group_size, self.expand_px):
            var.trace_add("write", lambda *_: self._apply())
        self.protocol("WM_DELETE_WINDOW", self._discard)
        self.bind("<Escape>", lambda e: self._discard())
        self._apply()

        self.update_idletasks()
        self._center_on_parent()

    # ---------------------------------------------------------------- build

    def _center_on_parent(self) -> None:
        self.update_idletasks()
        px, py = self.app.root.winfo_rootx(), self.app.root.winfo_rooty()
        pw, ph = self.app.root.winfo_width(), self.app.root.winfo_height()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{max(px + (pw - w) // 2, 0)}+{max(py + (ph - h) // 2, 0)}")

    def _build(self) -> None:
        tk.Label(self, text="Auto Mask", bg=PANEL_BG, fg=TEXT, font=FONT).pack(
            anchor="w", padx=16, pady=(12, 0))
        tk.Label(
            self, bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, justify="left", wraplength=HIST_W,
            text="Flags pixels above background + κ × background error as a preview "
                 "overlay, updated live as the sliders below change - nothing is written "
                 "to the real mask until Confirm.",
        ).pack(anchor="w", padx=16, pady=(2, 0))

        self._method_row("background method", self.bg_method, [(m, m.capitalize()) for m in BG_METHODS])
        self._method_row("background error", self.error_method, [("sigma", "Sigma"), ("sem", "SEM")])
        self._slider_row("clean-up κ  (background isolation)", self.cleanup_kappa, 0.5, 10.0)
        self._slider_row("clean-up iterations", self.cleanup_iterations, 1, 20)

        self.hist_canvas = tk.Canvas(self, width=HIST_W, height=HIST_H, bg=PANEL_BG,
                                      highlightthickness=1, highlightbackground=PANEL_BORDER)
        self.hist_canvas.pack(padx=16, pady=(10, 4))

        legend = tk.Frame(self, bg=PANEL_BG)
        legend.pack(anchor="w", padx=16)
        self._legend_swatch(legend, TEXT_DIM, "background (kept after clipping)")
        self._legend_swatch(legend, BLUE, "gaussian fit")
        self._legend_swatch(legend, GREEN, "background")
        self._legend_swatch(legend, WARNING, "mask threshold")

        # The threshold/expand knobs live below the histogram, separate from
        # the clean-up block above it - they shape the FINAL flagged mask,
        # not the background estimate the histogram is showing.
        self._slider_row("mask κ  (threshold above background)", self.kappa, 0.5, 20.0)
        self._slider_row("max group size  (px, drops larger flagged regions)", self.max_group_size, 1, 500)
        self._slider_row("expand  (pad each flagged region, px)", self.expand_px, 0, 20)

        self.stats_label = tk.Label(self, bg=PANEL_BG, fg=TEXT, font=FONT_SMALL, justify="left", anchor="w")
        self.stats_label.pack(anchor="w", padx=16, pady=(8, 4), fill="x")

        self._build_nn_section()

        btn_row = tk.Frame(self, bg=PANEL_BG)
        btn_row.pack(fill="x", padx=16, pady=(0, 16))
        RoundButton(btn_row, "discard", command=self._discard, outer_bg=PANEL_BG, danger=True).pack(side="right")
        RoundButton(btn_row, "confirm", command=self._confirm, outer_bg=PANEL_BG,
                    bg=GREEN, hover_bg=GREEN, fg=ACCENT_TEXT).pack(side="right", padx=(0, 8))

    def _build_nn_section(self) -> None:
        """Forward-looking hook, not functional yet: lets a model file be
        picked and wires an "apply" button through to
        automask.neural_network_mask, which currently just raises
        NotImplementedError - so the interface (pick a model, apply it,
        merge its output the same way as the threshold preview) is in place
        for a real model to be dropped into later without any GUI changes.
        """
        tk.Frame(self, bg=PANEL_BORDER, height=1).pack(fill="x", padx=16, pady=(2, 10))
        tk.Label(self, text="Neural network masking (future / experimental):",
                 bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, anchor="w").pack(anchor="w", padx=16)

        row = tk.Frame(self, bg=PANEL_BG)
        row.pack(fill="x", padx=16, pady=(4, 0))
        self.nn_model_label = tk.Label(row, text="no model loaded", bg=PANEL_BG, fg=TEXT_DIM,
                                        font=FONT_SMALL, anchor="w")
        self.nn_model_label.pack(side="left", fill="x", expand=True)
        RoundButton(row, "load model...", command=self._load_nn_model, outer_bg=PANEL_BG).pack(side="right")

        nn_btn_row = tk.Frame(self, bg=PANEL_BG)
        nn_btn_row.pack(fill="x", padx=16, pady=(6, 0))
        RoundButton(nn_btn_row, "apply nn mask", command=self._apply_nn, outer_bg=PANEL_BG).pack(anchor="w")

        self.nn_status_label = tk.Label(self, bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL,
                                         justify="left", wraplength=HIST_W, anchor="w")
        self.nn_status_label.pack(anchor="w", padx=16, pady=(4, 0))

    def _load_nn_model(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Load a masking model")
        if not path:
            return
        self.nn_model_path = path
        self.nn_model_label.config(text=path.rsplit("/", 1)[-1])

    def _apply_nn(self) -> None:
        if not self.nn_model_path:
            self.nn_status_label.config(text="Load a model file first.")
            return
        try:
            preview = neural_network_mask(self.data, self.nn_model_path)
        except NotImplementedError as exc:
            self.nn_status_label.config(text=str(exc))
            return
        self._preview = preview
        self.app.set_auto_mask_preview(self.entry, preview)
        self.nn_status_label.config(text=f"NN mask applied: {int(preview.sum()):,} px flagged.")

    def _legend_swatch(self, parent: tk.Widget, color: str, label: str) -> None:
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(side="left", padx=(0, 12), pady=(2, 6))
        sw = tk.Canvas(row, width=10, height=10, bg=PANEL_BG, highlightthickness=0)
        sw.pack(side="left", padx=(0, 4))
        sw.create_rectangle(0, 0, 10, 10, fill=color, outline="")
        tk.Label(row, text=label, bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).pack(side="left")

    def _method_row(self, label: str, var: tk.StringVar, options: list[tuple[str, str]]) -> None:
        row = tk.Frame(self, bg=PANEL_BG)
        row.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(row, text=f"{label}:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, anchor="w").pack(side="left")
        SegmentedControl(row, options, var, outer_bg=PANEL_BG).pack(side="right")

    def _slider_row(self, label: str, var: tk.Variable, lo: float, hi: float) -> None:
        row = tk.Frame(self, bg=PANEL_BG)
        row.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(row, text=f"{label}:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, anchor="w").pack(side="left")

        entry_str = tk.StringVar(value=f"{var.get():.3g}")

        def sync_entry(*_args: object) -> None:
            if entry.winfo_exists():
                entry_str.set(f"{var.get():.3g}")

        trace_id = var.trace_add("write", sync_entry)

        def apply_entry(_event: Optional[tk.Event] = None) -> None:
            try:
                value = float(entry_str.get())
            except ValueError:
                entry_str.set(f"{var.get():.3g}")
                return
            value = max(lo, min(value, hi))
            if isinstance(var, tk.IntVar):
                value = int(round(value))
            var.set(value)
            entry_str.set(f"{var.get():.3g}")

        entry = tk.Entry(row, textvariable=entry_str, width=6, justify="center", bg=BUTTON_BG, fg=TEXT,
                          insertbackground=TEXT, relief="flat", highlightthickness=1,
                          highlightbackground=PANEL_BORDER, highlightcolor=WARNING, font=FONT_SMALL)
        entry.pack(side="right")
        entry.bind("<Return>", apply_entry)
        entry.bind("<FocusOut>", apply_entry)
        entry.bind("<Destroy>", lambda e: var.trace_remove("write", trace_id), add="+")

        RoundSlider(self, var, lo, hi, width=HIST_W, height=20, outer_bg=PANEL_BG).pack(padx=16, pady=(0, 4))

    # ---------------------------------------------------------------- logic

    def _apply(self) -> None:
        data = self.data
        # Already-masked pixels (manually painted, or a previously-confirmed
        # auto mask) are excluded from background stats and can never be
        # (re-)flagged - as far as this tool is concerned they don't exist
        # in the image.
        valid = valid_pixels(data) & ~self.existing_mask
        kept = sigma_clip_mask(data, valid, self.cleanup_kappa.get(), max_iter=self.cleanup_iterations.get())
        bg, bg_err = background_stats(data, kept, self.error_method.get())
        preview = auto_mask_preview(data, valid, bg, bg_err, self.kappa.get())
        threshold = bg + self.kappa.get() * bg_err
        # Group-size filtering runs on the RAW flagged regions, before
        # expand pads them - padding first would inflate every group's size
        # and defeat the point of keeping only compact, point-like sources.
        preview = filter_by_group_size(preview, self.max_group_size.get())
        # Expansion only pads the FINAL flagged regions - it has no bearing
        # on background isolation, so it's applied after everything the
        # histogram/stats below are about.
        preview = expand_mask(preview, self.expand_px.get())

        self._preview = preview

        flagged = int(preview.sum())
        total = data.size
        pct = 100.0 * flagged / total if total else 0.0
        self.stats_label.config(
            text=(f"background: {bg:.4g}    error: {bg_err:.4g}    threshold: {threshold:.4g}\n"
                  f"flagged: {flagged:,} px ({pct:.2f}%)    background pixels used: {int(kept.sum()):,}")
        )
        self._draw_histogram(kept, bg, threshold)
        self.app.set_auto_mask_preview(self.entry, preview)

    def _draw_histogram(self, kept: np.ndarray, bg: float, threshold: float) -> None:
        """Histogram of only the pixels that SURVIVED clean-up clipping (the
        ones actually used for the background/error stats) - not the full
        valid distribution. The x-axis is always the kept pixels' own
        mean +/- 8 sigma - a fixed rule, not adjustable from the window, and
        NOT the same thing as bg/bg_err (which may be a SEM, far too narrow
        a span to plot against) - so a huge kappa can push bg/threshold
        outside this range; their marker lines just clip to the edge then."""
        canvas = self.hist_canvas
        canvas.delete("all")
        vals = self.data[kept]
        if vals.size == 0:
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

        counts, _edges = np.histogram(vals, bins=HIST_BINS, range=(lo, hi))
        heights = np.log1p(counts.astype(np.float64))
        max_h = heights.max() or 1.0
        bar_w = plot_w / HIST_BINS
        y_bottom = h - pad

        for i in range(HIST_BINS):
            if counts[i] == 0:
                continue
            x0 = pad_left + i * bar_w
            x1 = x0 + bar_w
            bar_h = (heights[i] / max_h) * plot_h
            canvas.create_rectangle(x0, y_bottom - bar_h, x1, y_bottom, fill=TEXT_DIM, outline="")

        def x_of(v: float) -> float:
            frac = (v - lo) / (hi - lo) if hi > lo else 0.0
            frac = min(max(frac, 0.0), 1.0)
            return pad_left + frac * plot_w

        fit = fit_gaussian_to_histogram(vals, HIST_BINS, (lo, hi))
        if fit is not None:
            amplitude, mu, sigma = fit
            xs = np.linspace(lo, hi, 150)
            # Predicted counts under the fit, put through the SAME log1p
            # height scale as the bars above, so the curve visually lines up
            # with them instead of living in a different (linear-count) space.
            predicted = np.clip(amplitude * np.exp(-0.5 * ((xs - mu) / sigma) ** 2), 0, None)
            fit_heights = np.log1p(predicted)
            points = []
            for x, fh in zip(xs, fit_heights):
                points.append(pad_left + ((x - lo) / (hi - lo)) * plot_w)
                points.append(y_bottom - (fh / max_h) * plot_h)
            if len(points) >= 4:
                canvas.create_line(*points, fill=BLUE, width=2, dash=(5, 3))

        bg_x = x_of(bg)
        canvas.create_line(bg_x, pad, bg_x, h - pad, fill=GREEN, width=2)
        th_x = x_of(threshold)
        canvas.create_line(th_x, pad, th_x, h - pad, fill=WARNING, width=2)

    def _confirm(self) -> None:
        if self._preview is not None:
            self.app.confirm_auto_mask(self.entry, self._preview)
        else:
            self.app.discard_auto_mask(self.entry)
        self.destroy()

    def _discard(self) -> None:
        self.app.discard_auto_mask(self.entry)
        self.destroy()
