"""Small dark-themed, rounded-corner Tkinter widgets used by the fitsedit GUI."""

import tkinter as tk
import tkinter.font as tkfont
from typing import Callable, Optional

from fitsedit.theme import (
    ACCENT,
    ACCENT_HOVER,
    BUTTON_BG,
    BUTTON_HOVER,
    DANGER,
    DANGER_HOVER,
    FONT_SMALL,
    PANEL_BG,
    PANEL_BORDER,
    TEXT,
    TRACK,
)


def rounded_rect_points(x0: float, y0: float, x1: float, y1: float, r: float) -> list[float]:
    """Point list for a rounded rectangle, for use with create_polygon(..., smooth=True)."""
    r = max(min(r, (x1 - x0) / 2, (y1 - y0) / 2), 0)
    return [
        x0 + r, y0,
        x1 - r, y0,
        x1, y0,
        x1, y0 + r,
        x1, y1 - r,
        x1, y1,
        x1 - r, y1,
        x0 + r, y1,
        x0, y1,
        x0, y1 - r,
        x0, y0 + r,
        x0, y0,
    ]


class RoundedPanel(tk.Frame):
    """A card-like panel with rounded corners. Pack/grid children into `.inner`."""

    def __init__(self, parent: tk.Widget, *, radius: int = 14, bg: str = PANEL_BG,
                 outer_bg: str = None, border: str = PANEL_BORDER):
        outer_bg = outer_bg if outer_bg is not None else parent["bg"]
        super().__init__(parent, bg=outer_bg)
        self.radius = radius
        self.bg_color = bg
        self.border_color = border
        self.canvas = tk.Canvas(self, bg=outer_bg, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", self._redraw)

    def _redraw(self, event: tk.Event) -> None:
        w, h = event.width, event.height
        self.canvas.delete("bg")
        if w > 2 and h > 2:
            pts = rounded_rect_points(1, 1, w - 1, h - 1, self.radius)
            self.canvas.create_polygon(pts, smooth=True, fill=self.bg_color, outline=self.border_color, tags="bg")
            self.canvas.tag_lower("bg")
        self.canvas.itemconfig(self._win, width=w, height=h)


class RoundButton(tk.Canvas):
    """A rounded-rectangle button, optionally behaving as a toggle for segmented controls."""

    def __init__(self, parent: tk.Widget, text: str, command: Optional[Callable[[], None]] = None, *,
                 width: Optional[int] = None, height: int = 30, radius: int = 10,
                 outer_bg: Optional[str] = None, bg: str = BUTTON_BG, hover_bg: str = BUTTON_HOVER,
                 fg: str = TEXT, font=FONT_SMALL, accent: bool = False, danger: bool = False,
                 toggle: bool = False, active: bool = False, padx: int = 14):
        outer_bg = outer_bg if outer_bg is not None else parent["bg"]
        self._text = text
        self._command = command
        self._radius = radius
        self._fg = fg
        self._font = font
        self._accent = accent
        self._danger = danger
        self._toggle = toggle
        self._active = active
        self._base_bg = bg
        self._hover_bg = hover_bg
        self._hovering = False
        self._pressed = False

        measured = tkfont.Font(font=font)
        text_w = measured.measure(text)
        text_h = measured.metrics("linespace")
        if width is None:
            width = text_w + padx * 2
        height = max(height, text_h + 10)

        super().__init__(parent, width=width, height=height, bg=outer_bg, highlightthickness=0, cursor="hand2")
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonPress-1>", lambda e: setattr(self, "_pressed", True))
        self.bind("<ButtonRelease-1>", self._on_release)
        self._redraw()

    def _set_hover(self, hovering: bool) -> None:
        self._hovering = hovering
        self._redraw()

    def _on_release(self, event: tk.Event) -> None:
        was_pressed = self._pressed
        self._pressed = False
        inside = 0 <= event.x <= int(self["width"]) and 0 <= event.y <= int(self["height"])
        if was_pressed and inside and self._command is not None:
            self._command()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._redraw()

    def set_text(self, text: str) -> None:
        self._text = text
        self._redraw()

    def _current_fill(self) -> str:
        if self._danger:
            return DANGER_HOVER if self._hovering else DANGER
        if self._accent or (self._toggle and self._active):
            return ACCENT_HOVER if self._hovering else ACCENT
        return self._hover_bg if self._hovering else self._base_bg

    def _redraw(self) -> None:
        self.delete("all")
        w, h = int(self["width"]), int(self["height"])
        pts = rounded_rect_points(1, 1, w - 1, h - 1, self._radius)
        self.create_polygon(pts, smooth=True, fill=self._current_fill(), outline="")
        self.create_text(w / 2, h / 2, text=self._text, fill=self._fg, font=self._font)


class SegmentedControl(tk.Frame):
    """A row of toggle buttons acting as a single-select group bound to a StringVar."""

    def __init__(self, parent: tk.Widget, options: list[tuple[str, str]], variable: tk.StringVar, *,
                 outer_bg: Optional[str] = None, command: Optional[Callable[[str], None]] = None):
        outer_bg = outer_bg if outer_bg is not None else parent["bg"]
        super().__init__(parent, bg=outer_bg)
        self.variable = variable
        self.command = command
        self.buttons: dict[str, RoundButton] = {}
        for value, label in options:
            btn = RoundButton(
                self, label, command=lambda v=value: self._select(v),
                outer_bg=outer_bg, toggle=True, active=(variable.get() == value),
                height=26, radius=7,
            )
            btn.pack(side="left", padx=(0, 4), pady=2)
            self.buttons[value] = btn

    def _select(self, value: str) -> None:
        self.variable.set(value)
        for v, btn in self.buttons.items():
            btn.set_active(v == value)
        if self.command:
            self.command(value)


class RoundSlider(tk.Canvas):
    """A pill-shaped, mouse-draggable slider bound to an IntVar/DoubleVar."""

    def __init__(self, parent: tk.Widget, variable, from_: float, to: float, *,
                 width: int = 200, height: int = 20, outer_bg: Optional[str] = None,
                 track_color: str = TRACK, fill_color: str = ACCENT, thumb_color: str = TEXT,
                 on_change: Optional[Callable[[float], None]] = None):
        outer_bg = outer_bg if outer_bg is not None else parent["bg"]
        super().__init__(parent, width=width, height=height, bg=outer_bg, highlightthickness=0, cursor="hand2")
        self.variable = variable
        self.from_ = from_
        self.to = to
        self.on_change = on_change
        self.track_color = track_color
        self.fill_color = fill_color
        self.thumb_color = thumb_color
        self._pad = 9

        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<ButtonPress-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)
        self._trace_id = variable.trace_add("write", lambda *_: self._redraw())
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._redraw()

    def _on_destroy(self, event: tk.Event) -> None:
        try:
            self.variable.trace_remove("write", self._trace_id)
        except tk.TclError:
            pass

    def _value_to_x(self, value: float, w: int) -> float:
        span = self.to - self.from_
        frac = (value - self.from_) / span if span else 0.0
        frac = min(max(frac, 0.0), 1.0)
        return self._pad + frac * (w - 2 * self._pad)

    def _x_to_value(self, x: float, w: int) -> float:
        frac = (x - self._pad) / max(w - 2 * self._pad, 1)
        frac = min(max(frac, 0.0), 1.0)
        return self.from_ + frac * (self.to - self.from_)

    def _on_drag(self, event: tk.Event) -> None:
        w = self.winfo_width() or int(self["width"])
        value = self._x_to_value(event.x, w)
        if isinstance(self.variable, tk.IntVar):
            value = int(round(value))
        self.variable.set(value)
        if self.on_change:
            self.on_change(value)

    def _redraw(self) -> None:
        self.delete("all")
        w = self.winfo_width() or int(self["width"])
        h = self.winfo_height() or int(self["height"])
        cy = h / 2
        track_h = 6
        pts = rounded_rect_points(self._pad, cy - track_h / 2, w - self._pad, cy + track_h / 2, track_h / 2)
        self.create_polygon(pts, smooth=True, fill=self.track_color, outline="")

        thumb_x = self._value_to_x(self.variable.get(), w)
        if thumb_x > self._pad:
            fpts = rounded_rect_points(self._pad, cy - track_h / 2, thumb_x, cy + track_h / 2, track_h / 2)
            self.create_polygon(fpts, smooth=True, fill=self.fill_color, outline="")

        r = 7
        self.create_oval(thumb_x - r, cy - r, thumb_x + r, cy + r, fill=self.thumb_color, outline=self.fill_color, width=2)
