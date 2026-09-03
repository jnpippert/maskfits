"""Embeddable pixel-value histogram with draggable lower/upper cut lines.

Used inline in the sidebar (see MaskFitsApp._build_sidebar) so manual cut
levels can be set directly against a live histogram, without a separate
popup window.
"""

import tkinter as tk
from typing import Callable, Optional

import numpy as np

from maskfits.theme import (
    BUTTON_BG,
    BUTTON_HOVER,
    DANGER,
    FONT_SMALL,
    GREEN,
    PANEL_BG,
    PANEL_BORDER,
    TEXT,
    TEXT_DIM,
)
from maskfits.widgets import RoundButton

SAMPLE_TARGET = 500_000


class CutsHistogram(tk.Frame):
    """A live pixel-value histogram with draggable vmin/vmax lines and numeric
    entry boxes. Changes apply via `on_apply(lowcut, highcut)` as the user
    drags or types. `width`/`height`/`bins` size it to fit wherever it's
    embedded - the padding and handle size scale with width/height too.
    """

    def __init__(self, parent: tk.Widget, data: np.ndarray, lowcut: float, highcut: float,
                 on_apply: Callable[[float, float], None], *,
                 width: int = 480, height: int = 220, bins: int = 80,
                 outer_bg: Optional[str] = None):
        outer_bg = outer_bg if outer_bg is not None else parent["bg"]
        super().__init__(parent, bg=outer_bg)
        self.on_apply = on_apply
        self.canvas_w = width
        self.canvas_h = height
        self.bins = bins
        self.pad = max(round(height * 0.11), 10)
        self.pad_left = max(round(width * 0.15), 30)
        self.handle_h = max(round(height * 0.045), 6)

        self._drag: Optional[str] = None  # "lo" | "hi" | None
        self._drag_range: Optional[tuple[float, float]] = None
        self.show_full_range = False

        # overlay canvas item ids, created once and repositioned afterwards
        self._sel_id: Optional[int] = None
        self._line_lo_id: Optional[int] = None
        self._line_hi_id: Optional[int] = None
        self._handle_lo_id: Optional[int] = None
        self._handle_hi_id: Optional[int] = None

        self.vmin_var = tk.DoubleVar(value=lowcut)
        self.vmax_var = tk.DoubleVar(value=highcut)
        self.lo_str = tk.StringVar(value=self._fmt(lowcut))
        self.hi_str = tk.StringVar(value=self._fmt(highcut))

        self._build(outer_bg)
        self._set_sample(data)
        self._redraw(full_rebin=True)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:.6g}"

    def _build(self, outer_bg: str) -> None:
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg=PANEL_BG,
                                 highlightthickness=1, highlightbackground=PANEL_BORDER)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Lowcut, highcut, and the full-range toggle all in one row - fits
        # comfortably now that the sidebar (and so this widget) is wide
        # enough for all three without anything getting clipped.
        entries = tk.Frame(self, bg=outer_bg)
        entries.pack(fill="x", pady=(8, 0))

        tk.Label(entries, text="lowcut:", bg=outer_bg, fg=TEXT_DIM, font=FONT_SMALL).pack(side="left")
        lo_entry = tk.Entry(entries, textvariable=self.lo_str, width=8, justify="center", bg=BUTTON_BG, fg=TEXT,
                             insertbackground=TEXT, relief="flat", highlightthickness=1,
                             highlightbackground=PANEL_BORDER, highlightcolor=DANGER, font=FONT_SMALL)
        lo_entry.pack(side="left", padx=(6, 14))
        lo_entry.bind("<Return>", self._on_lo_entry)
        lo_entry.bind("<FocusOut>", self._on_lo_entry)

        tk.Label(entries, text="highcut:", bg=outer_bg, fg=TEXT_DIM, font=FONT_SMALL).pack(side="left")
        hi_entry = tk.Entry(entries, textvariable=self.hi_str, width=8, justify="center", bg=BUTTON_BG, fg=TEXT,
                             insertbackground=TEXT, relief="flat", highlightthickness=1,
                             highlightbackground=PANEL_BORDER, highlightcolor=GREEN, font=FONT_SMALL)
        hi_entry.pack(side="left", padx=(6, 8))
        hi_entry.bind("<Return>", self._on_hi_entry)
        hi_entry.bind("<FocusOut>", self._on_hi_entry)

        self._range_btn = RoundButton(
            entries, "full", command=self._toggle_range_view, outer_bg=outer_bg,
            toggle=True, active=self.show_full_range, height=22, radius=7,
        )
        self._range_btn.pack(side="left")

    # ------------------------------------------------------------- data

    def set_data(self, data: np.ndarray, lowcut: float, highcut: float) -> None:
        """Resample from a (possibly new) data array and sync the cut values.

        Call whenever the displayed image or its cuts change from outside -
        switching images, changing the stretch/cut preset, or toggling
        smoothing/binning (which changes the array the histogram should
        actually reflect).
        """
        self._set_sample(data)
        self.vmin_var.set(lowcut)
        self.vmax_var.set(highcut)
        self.lo_str.set(self._fmt(lowcut))
        self.hi_str.set(self._fmt(highcut))
        self._redraw(full_rebin=True)

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

    def _toggle_range_view(self) -> None:
        self.show_full_range = not self.show_full_range
        self._range_btn.set_active(self.show_full_range)
        self._redraw(full_rebin=True)

    # ------------------------------------------------------------- geometry

    def _compute_disp_range(self) -> tuple[float, float]:
        if self.show_full_range:
            return self.data_min, self.data_max
        vmin, vmax = self.vmin_var.get(), self.vmax_var.get()
        span = max(vmax - vmin, 1e-9)
        margin = span * 0.5
        lo = max(vmin - margin, self.data_min)
        hi = min(vmax + margin, self.data_max)
        if hi <= lo:
            lo, hi = self.data_min, self.data_max
        return lo, hi

    def _value_to_x(self, value: float, lo: float, hi: float) -> float:
        frac = (value - lo) / (hi - lo) if hi > lo else 0.0
        frac = min(max(frac, 0.0), 1.0)
        return self.pad_left + frac * (self.canvas_w - self.pad_left - self.pad)

    def _x_to_value(self, x: float, lo: float, hi: float) -> float:
        frac = (x - self.pad_left) / max(self.canvas_w - self.pad_left - self.pad, 1)
        frac = min(max(frac, 0.0), 1.0)
        return lo + frac * (hi - lo)

    # -------------------------------------------------------------- drawing

    def _redraw(self, full_rebin: bool) -> None:
        """Recompute+redraw histogram bars only when the display range actually changed.

        During a drag the range is frozen, so only the cheap overlay (lines/handles/
        shading) needs to move each step - re-binning on every mouse-motion pixel is
        what makes dragging feel sluggish.
        """
        if full_rebin or self._drag_range is None:
            disp_lo, disp_hi = self._compute_disp_range()
            self._draw_bars(disp_lo, disp_hi)
        else:
            disp_lo, disp_hi = self._drag_range
        self._draw_overlay(disp_lo, disp_hi)

    def _draw_bars(self, disp_lo: float, disp_hi: float) -> None:
        self.canvas.delete("bar")
        self.canvas.delete("yaxis")
        counts, _edges = np.histogram(self.sample, bins=self.bins, range=(disp_lo, disp_hi))
        max_count = int(counts.max()) if counts.size else 0
        heights = np.log1p(counts.astype(np.float64))
        max_h = heights.max() or 1.0
        plot_h = self.canvas_h - 2 * self.pad
        bar_w = (self.canvas_w - self.pad_left - self.pad) / self.bins
        for i, h in enumerate(heights):
            x0 = self.pad_left + i * bar_w
            x1 = x0 + bar_w
            y1 = self.canvas_h - self.pad
            y0 = y1 - (h / max_h) * plot_h
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=TEXT_DIM, outline="", tags="bar")
        self._draw_y_axis(max_count, max_h)

    @staticmethod
    def _fmt_tick(t: int) -> str:
        """Every tick is an exact power of ten (see below), so scientific
        notation is just its exponent - 100 becomes "1E+2"."""
        if t <= 0:
            return "0"
        return f"1E+{round(np.log10(t))}"

    def _draw_y_axis(self, max_count: int, max_h: float) -> None:
        """Log-scaled y ticks (0, 1, 10, 100, ...) matching the log1p bar-height scale."""
        ticks = [0]
        v = 1
        while v <= max_count:
            ticks.append(v)
            v *= 10
        if len(ticks) == 1:
            ticks.append(max(max_count, 1))

        plot_bottom = self.canvas_h - self.pad
        plot_h = self.canvas_h - 2 * self.pad
        for t in ticks:
            frac = (np.log1p(t) / max_h) if max_h > 0 else 0.0
            y = plot_bottom - frac * plot_h
            self.canvas.create_line(self.pad_left - 5, y, self.pad_left, y, fill=TEXT_DIM, tags="yaxis")
            self.canvas.create_text(self.pad_left - 8, y, text=self._fmt_tick(t), fill=TEXT_DIM, font=FONT_SMALL,
                                     anchor="e", tags="yaxis")

    def _draw_overlay(self, disp_lo: float, disp_hi: float) -> None:
        x_lo = self._value_to_x(self.vmin_var.get(), disp_lo, disp_hi)
        x_hi = self._value_to_x(self.vmax_var.get(), disp_lo, disp_hi)

        if self._sel_id is None:
            # BUTTON_HOVER is already tuned as "a bit of emphasis over the
            # panel background" in both themes, which is exactly what the
            # selected-range highlight needs - reusing it means this stays
            # correct in light mode too, instead of a fixed dark-only tint.
            self._sel_id = self.canvas.create_rectangle(
                x_lo, self.pad, x_hi, self.canvas_h - self.pad, fill=BUTTON_HOVER, outline="")
            self._line_lo_id = self.canvas.create_line(
                x_lo, self.pad, x_lo, self.canvas_h - self.pad, fill=DANGER, width=2)
            self._line_hi_id = self.canvas.create_line(
                x_hi, self.pad, x_hi, self.canvas_h - self.pad, fill=GREEN, width=2)
            self._handle_lo_id = self.canvas.create_rectangle(
                x_lo - 5, self.pad - self.handle_h, x_lo + 5, self.pad, fill=DANGER, outline="")
            self._handle_hi_id = self.canvas.create_rectangle(
                x_hi - 5, self.pad - self.handle_h, x_hi + 5, self.pad, fill=GREEN, outline="")
        else:
            self.canvas.coords(self._sel_id, x_lo, self.pad, x_hi, self.canvas_h - self.pad)
            self.canvas.coords(self._line_lo_id, x_lo, self.pad, x_lo, self.canvas_h - self.pad)
            self.canvas.coords(self._line_hi_id, x_hi, self.pad, x_hi, self.canvas_h - self.pad)
            self.canvas.coords(self._handle_lo_id, x_lo - 5, self.pad - self.handle_h, x_lo + 5, self.pad)
            self.canvas.coords(self._handle_hi_id, x_hi - 5, self.pad - self.handle_h, x_hi + 5, self.pad)

        self.canvas.tag_lower(self._sel_id)
        self.canvas.tag_raise(self._line_lo_id)
        self.canvas.tag_raise(self._line_hi_id)
        self.canvas.tag_raise(self._handle_lo_id)
        self.canvas.tag_raise(self._handle_hi_id)

    # ---------------------------------------------------------------- input

    def _on_press(self, event: tk.Event) -> None:
        disp_lo, disp_hi = self._compute_disp_range()
        x_lo = self._value_to_x(self.vmin_var.get(), disp_lo, disp_hi)
        x_hi = self._value_to_x(self.vmax_var.get(), disp_lo, disp_hi)
        if abs(event.x - x_lo) <= 8:
            self._drag = "lo"
        elif abs(event.x - x_hi) <= 8:
            self._drag = "hi"
        else:
            self._drag = None
            return
        self._drag_range = (disp_lo, disp_hi)

    def _on_drag(self, event: tk.Event) -> None:
        """Move the histogram lines/entries live, but don't apply to the main
        image yet - that only happens on release, so continuous dragging
        doesn't force a full re-render on every mouse-motion pixel."""
        if self._drag is None or self._drag_range is None:
            return
        value = self._x_to_value(event.x, *self._drag_range)
        if self._drag == "lo":
            value = min(value, self.vmax_var.get())
            self.vmin_var.set(value)
            self.lo_str.set(self._fmt(value))
        else:
            value = max(value, self.vmin_var.get())
            self.vmax_var.set(value)
            self.hi_str.set(self._fmt(value))
        self._redraw(full_rebin=False)

    def _on_release(self, _event: tk.Event) -> None:
        if self._drag is None:
            return
        self._drag = None
        self._drag_range = None
        self._redraw(full_rebin=True)
        self.on_apply(self.vmin_var.get(), self.vmax_var.get())

    def _on_lo_entry(self, _event: tk.Event) -> None:
        try:
            value = float(self.lo_str.get())
        except ValueError:
            self.lo_str.set(self._fmt(self.vmin_var.get()))
            return
        value = min(value, self.vmax_var.get())
        self.vmin_var.set(value)
        self.lo_str.set(self._fmt(value))
        self._redraw(full_rebin=True)
        self.on_apply(self.vmin_var.get(), self.vmax_var.get())

    def _on_hi_entry(self, _event: tk.Event) -> None:
        try:
            value = float(self.hi_str.get())
        except ValueError:
            self.hi_str.set(self._fmt(self.vmax_var.get()))
            return
        value = max(value, self.vmin_var.get())
        self.vmax_var.set(value)
        self.hi_str.set(self._fmt(value))
        self._redraw(full_rebin=True)
        self.on_apply(self.vmin_var.get(), self.vmax_var.get())
