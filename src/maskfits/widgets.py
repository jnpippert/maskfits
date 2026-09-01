"""Small rounded-corner Tkinter widgets used by the maskfits GUI, themed via maskfits.theme."""

import tkinter as tk
import tkinter.font as tkfont
from typing import Callable, Optional

from maskfits.theme import (
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
    """A card-like panel with rounded corners. Pack/grid children into `.inner`.

    mode="fill" (default): the panel takes whatever size its parent gives it
      (e.g. a sidebar stretched to the window height by its container).
    mode="hug": the panel sizes itself to its content's natural height instead
      (e.g. a toolbar or status bar that should stay compact).
    scrollable: only meaningful with mode="fill" - if the content's natural
      height exceeds the space available, scroll instead of clipping it.
    """

    def __init__(self, parent: tk.Widget, *, radius: int = 14, bg: Optional[str] = None,
                 outer_bg: Optional[str] = None, border: Optional[str] = None,
                 mode: str = "fill", scrollable: bool = False):
        # Resolved from the current theme at call time (not baked in as a
        # default argument) so a freshly-built panel always picks up whichever
        # palette is active, including right after a theme switch.
        bg = bg if bg is not None else PANEL_BG
        border = border if border is not None else PANEL_BORDER
        outer_bg = outer_bg if outer_bg is not None else parent["bg"]
        super().__init__(parent, bg=outer_bg)
        self.radius = radius
        self.bg_color = bg
        self.border_color = border
        self.mode = mode
        self.scrollable = scrollable and mode == "fill"

        self.canvas = tk.Canvas(self, bg=outer_bg, highlightthickness=0, width=1, height=1)
        self.canvas.pack(fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")

        self.bind("<Configure>", self._sync)
        self.inner.bind("<Configure>", self._sync)
        if self.scrollable:
            self.canvas.bind("<Enter>", lambda e: self._bind_wheel())
            self.canvas.bind("<Leave>", lambda e: self._unbind_wheel())

    def _sync(self, _event: Optional[tk.Event] = None) -> None:
        if self.mode == "hug":
            w = self.winfo_width()
            h = self.inner.winfo_reqheight()
            if w <= 1:
                return
            self.canvas.configure(width=w, height=h)
            self.canvas.itemconfig(self._win, width=w, height=h)
            self._draw_bg(w, h)
        else:
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w <= 1 or h <= 1:
                return
            if self.scrollable:
                content_h = max(self.inner.winfo_reqheight(), h)
                self.canvas.itemconfig(self._win, width=w)
                self.canvas.configure(scrollregion=(0, 0, w, content_h))
            else:
                self.canvas.itemconfig(self._win, width=w, height=h)
            self._draw_bg(w, h)

    def _draw_bg(self, w: int, h: int) -> None:
        self.canvas.delete("bg")
        if w > 2 and h > 2:
            pts = rounded_rect_points(1, 1, w - 1, h - 1, self.radius)
            self.canvas.create_polygon(pts, smooth=True, fill=self.bg_color, outline=self.border_color, tags="bg")
            self.canvas.tag_lower("bg")

    def _bind_wheel(self) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(3, "units"))

    def _unbind_wheel(self) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")


class RoundButton(tk.Canvas):
    """A rounded-rectangle button, optionally behaving as a toggle for segmented controls."""

    def __init__(self, parent: tk.Widget, text: str, command: Optional[Callable[[], None]] = None, *,
                 width: Optional[int] = None, height: int = 30, radius: int = 10,
                 outer_bg: Optional[str] = None, bg: Optional[str] = None, hover_bg: Optional[str] = None,
                 fg: Optional[str] = None, font=FONT_SMALL, accent: bool = False, danger: bool = False,
                 toggle: bool = False, active: bool = False, padx: int = 14):
        # Resolved at call time (not a baked-in default) so a button built
        # after a theme switch picks up the newly active palette.
        bg = bg if bg is not None else BUTTON_BG
        hover_bg = hover_bg if hover_bg is not None else BUTTON_HOVER
        fg = fg if fg is not None else TEXT
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
    """A row of toggle buttons acting as a single-select group bound to a StringVar.

    Stays in sync regardless of what changes the variable - a menu radiobutton, a
    CLI startup flag, or a plain `.set()` - not just its own button clicks.
    """

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

        self._trace_id = variable.trace_add("write", self._on_variable_changed)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _on_variable_changed(self, *_args: object) -> None:
        current = self.variable.get()
        for value, btn in self.buttons.items():
            if btn.winfo_exists():
                btn.set_active(value == current)

    def _select(self, value: str) -> None:
        self.variable.set(value)
        if self.command:
            self.command(value)

    def _on_destroy(self, _event: tk.Event) -> None:
        try:
            self.variable.trace_remove("write", self._trace_id)
        except tk.TclError:
            pass


class RoundSlider(tk.Canvas):
    """A pill-shaped, mouse-draggable slider bound to an IntVar/DoubleVar."""

    def __init__(self, parent: tk.Widget, variable, from_: float, to: float, *,
                 width: int = 200, height: int = 20, outer_bg: Optional[str] = None,
                 track_color: Optional[str] = None, fill_color: Optional[str] = None,
                 thumb_color: Optional[str] = None,
                 on_change: Optional[Callable[[float], None]] = None):
        # Resolved at call time (not a baked-in default) so a slider built
        # after a theme switch picks up the newly active palette.
        track_color = track_color if track_color is not None else TRACK
        fill_color = fill_color if fill_color is not None else ACCENT
        thumb_color = thumb_color if thumb_color is not None else TEXT
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


class ThemeToggle(tk.Label):
    """A sun/moon emoji button for switching between light and dark mode -
    plain Unicode glyphs rather than hand-drawn icons, so it renders as the
    platform's native full-color emoji."""

    SUN = "☀️"
    MOON = "\U0001F319"

    def __init__(self, parent: tk.Widget, command: Optional[Callable[[], None]] = None, *,
                 light: bool = False, outer_bg: Optional[str] = None):
        outer_bg = outer_bg if outer_bg is not None else parent["bg"]
        super().__init__(parent, text=self.SUN if light else self.MOON, bg=outer_bg,
                          font=("Apple Color Emoji", 16), cursor="hand2")
        self._command = command
        self.bind("<Button-1>", lambda e: self._command() if self._command else None)
