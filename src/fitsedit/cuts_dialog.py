"""Manual lower/upper cut dialog with a live, draggable pixel-value histogram."""

import tkinter as tk
from typing import Callable, Optional

import numpy as np

from fitsedit.theme import (
    ACCENT,
    APP_BG,
    BUTTON_BG,
    DANGER,
    FONT,
    FONT_SMALL,
    GREEN,
    PANEL_BG,
    PANEL_BORDER,
    TEXT,
    TEXT_DIM,
)
from fitsedit.widgets import RoundButton

HIST_BINS = 80
CANVAS_W, CANVAS_H = 520, 240
PAD = 24
PAD_LEFT = 48
HANDLE_H = 8
SAMPLE_TARGET = 500_000
SELECTION_TINT = "#2c2620"
LO_COLOR = DANGER
HI_COLOR = GREEN


class ManualCutsWindow(tk.Toplevel):
    """A non-modal dialog: histogram + draggable vmin/vmax lines + numeric entry boxes.

    Changes apply live via `on_apply(lowcut, highcut)` as the user drags or types.
    """

    def __init__(self, parent: tk.Misc, data: np.ndarray, lowcut: float, highcut: float,
                 on_apply: Callable[[float, float], None]):
        super().__init__(parent)
        self.title("Manual Cuts")
        self.configure(bg=APP_BG)
        self.resizable(False, False)
        self.transient(parent)

        self.on_apply = on_apply
        self._drag: Optional[str] = None  # "lo" | "hi" | None
        self._drag_range: Optional[tuple[float, float]] = None
        self._last_range = (0.0, 1.0)
        self.show_full_range = False

        # overlay canvas item ids, created once and repositioned afterwards
        self._sel_id: Optional[int] = None
        self._line_lo_id: Optional[int] = None
        self._line_hi_id: Optional[int] = None
        self._handle_lo_id: Optional[int] = None
        self._handle_hi_id: Optional[int] = None

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

        self.vmin_var = tk.DoubleVar(value=lowcut)
        self.vmax_var = tk.DoubleVar(value=highcut)
        self.lo_str = tk.StringVar(value=self._fmt(lowcut))
        self.hi_str = tk.StringVar(value=self._fmt(highcut))

        self._build()
        self._redraw(full_rebin=True)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:.6g}"

    def _build(self) -> None:
        frame = tk.Frame(self, bg=APP_BG)
        frame.pack(padx=16, pady=16)

        header = tk.Frame(frame, bg=APP_BG)
        header.pack(fill="x", pady=(0, 10))
        tk.Label(header, text="Manual cut levels", bg=APP_BG, fg=TEXT, font=FONT).pack(side="left")
        self._range_btn = RoundButton(
            header, "show full range", command=self._toggle_range_view, outer_bg=APP_BG, height=26,
        )
        self._range_btn.pack(side="right")

        self.canvas = tk.Canvas(frame, width=CANVAS_W, height=CANVAS_H, bg=PANEL_BG,
                                 highlightthickness=1, highlightbackground=PANEL_BORDER)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        entries = tk.Frame(frame, bg=APP_BG)
        entries.pack(fill="x", pady=(14, 0))

        tk.Label(entries, text="lower cut:", bg=APP_BG, fg=TEXT_DIM, font=FONT_SMALL).grid(row=0, column=0, sticky="w")
        lo_entry = tk.Entry(entries, textvariable=self.lo_str, width=14, bg=BUTTON_BG, fg=TEXT,
                             insertbackground=TEXT, relief="flat", highlightthickness=1,
                             highlightbackground=PANEL_BORDER, highlightcolor=LO_COLOR)
        lo_entry.grid(row=0, column=1, padx=(8, 24))
        lo_entry.bind("<Return>", self._on_lo_entry)
        lo_entry.bind("<FocusOut>", self._on_lo_entry)

        tk.Label(entries, text="upper cut:", bg=APP_BG, fg=TEXT_DIM, font=FONT_SMALL).grid(row=0, column=2, sticky="w")
        hi_entry = tk.Entry(entries, textvariable=self.hi_str, width=14, bg=BUTTON_BG, fg=TEXT,
                             insertbackground=TEXT, relief="flat", highlightthickness=1,
                             highlightbackground=PANEL_BORDER, highlightcolor=HI_COLOR)
        hi_entry.grid(row=0, column=3, padx=(8, 0))
        hi_entry.bind("<Return>", self._on_hi_entry)
        hi_entry.bind("<FocusOut>", self._on_hi_entry)

        RoundButton(frame, "Close", command=self.destroy, outer_bg=APP_BG, accent=True).pack(anchor="e", pady=(14, 0))

    def _toggle_range_view(self) -> None:
        self.show_full_range = not self.show_full_range
        self._range_btn.set_text("zoom to cuts" if self.show_full_range else "show full range")
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
        return PAD_LEFT + frac * (CANVAS_W - PAD_LEFT - PAD)

    def _x_to_value(self, x: float, lo: float, hi: float) -> float:
        frac = (x - PAD_LEFT) / max(CANVAS_W - PAD_LEFT - PAD, 1)
        frac = min(max(frac, 0.0), 1.0)
        return lo + frac * (hi - lo)

    # -------------------------------------------------------------- drawing

    def _redraw(self, full_rebin: bool) -> None:
        """Recompute+redraw histogram bars only when the display range actually changed.

        During a drag the range is frozen, so only the cheap overlay (lines/handles/
        shading) needs to move each step - re-binning ~80 bars on every pixel of mouse
        motion was the source of the sluggish dragging.
        """
        if full_rebin or self._drag_range is None:
            disp_lo, disp_hi = self._compute_disp_range()
            self._last_range = (disp_lo, disp_hi)
            self._draw_bars(disp_lo, disp_hi)
        else:
            disp_lo, disp_hi = self._drag_range
        self._draw_overlay(disp_lo, disp_hi)

    def _draw_bars(self, disp_lo: float, disp_hi: float) -> None:
        self.canvas.delete("bar")
        self.canvas.delete("yaxis")
        counts, _edges = np.histogram(self.sample, bins=HIST_BINS, range=(disp_lo, disp_hi))
        max_count = int(counts.max()) if counts.size else 0
        heights = np.log1p(counts.astype(np.float64))
        max_h = heights.max() or 1.0
        plot_h = CANVAS_H - 2 * PAD
        bar_w = (CANVAS_W - PAD_LEFT - PAD) / HIST_BINS
        for i, h in enumerate(heights):
            x0 = PAD_LEFT + i * bar_w
            x1 = x0 + bar_w
            y1 = CANVAS_H - PAD
            y0 = y1 - (h / max_h) * plot_h
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=TEXT_DIM, outline="", tags="bar")
        self._draw_y_axis(max_count, max_h)

    def _draw_y_axis(self, max_count: int, max_h: float) -> None:
        """Log-scaled y ticks (0, 1, 10, 100, ...) matching the log1p bar-height scale."""
        ticks = [0]
        v = 1
        while v <= max_count:
            ticks.append(v)
            v *= 10
        if len(ticks) == 1:
            ticks.append(max(max_count, 1))

        plot_bottom = CANVAS_H - PAD
        plot_h = CANVAS_H - 2 * PAD
        for t in ticks:
            frac = (np.log1p(t) / max_h) if max_h > 0 else 0.0
            y = plot_bottom - frac * plot_h
            self.canvas.create_line(PAD_LEFT - 5, y, PAD_LEFT, y, fill=TEXT_DIM, tags="yaxis")
            self.canvas.create_text(PAD_LEFT - 8, y, text=str(t), fill=TEXT_DIM, font=FONT_SMALL,
                                     anchor="e", tags="yaxis")
        self.canvas.create_text((PAD_LEFT + CANVAS_W - PAD) / 2, 10, text="pixel count", fill=TEXT_DIM,
                                 font=FONT_SMALL, tags="yaxis")

    def _draw_overlay(self, disp_lo: float, disp_hi: float) -> None:
        x_lo = self._value_to_x(self.vmin_var.get(), disp_lo, disp_hi)
        x_hi = self._value_to_x(self.vmax_var.get(), disp_lo, disp_hi)

        if self._sel_id is None:
            self._sel_id = self.canvas.create_rectangle(x_lo, PAD, x_hi, CANVAS_H - PAD, fill=SELECTION_TINT, outline="")
            self._line_lo_id = self.canvas.create_line(x_lo, PAD, x_lo, CANVAS_H - PAD, fill=LO_COLOR, width=2)
            self._line_hi_id = self.canvas.create_line(x_hi, PAD, x_hi, CANVAS_H - PAD, fill=HI_COLOR, width=2)
            self._handle_lo_id = self.canvas.create_rectangle(
                x_lo - 5, PAD - HANDLE_H, x_lo + 5, PAD, fill=LO_COLOR, outline="")
            self._handle_hi_id = self.canvas.create_rectangle(
                x_hi - 5, PAD - HANDLE_H, x_hi + 5, PAD, fill=HI_COLOR, outline="")
        else:
            self.canvas.coords(self._sel_id, x_lo, PAD, x_hi, CANVAS_H - PAD)
            self.canvas.coords(self._line_lo_id, x_lo, PAD, x_lo, CANVAS_H - PAD)
            self.canvas.coords(self._line_hi_id, x_hi, PAD, x_hi, CANVAS_H - PAD)
            self.canvas.coords(self._handle_lo_id, x_lo - 5, PAD - HANDLE_H, x_lo + 5, PAD)
            self.canvas.coords(self._handle_hi_id, x_hi - 5, PAD - HANDLE_H, x_hi + 5, PAD)

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
        """Move the histogram lines/entries live, but don't touch the main image yet.

        Re-rendering the full main image on every mouse-motion pixel is what made
        dragging feel sluggish - the main canvas only updates once, on release.
        """
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
