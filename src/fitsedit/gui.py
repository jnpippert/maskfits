"""Tkinter GUI: view FITS images and paint circular/elliptical or line (satellite trail) masks."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np
from astropy.io import fits
from PIL import Image, ImageTk

from fitsedit.imagedata import FitsImage, load_fits_image, minmax_cuts, zscale_cuts
from fitsedit.masking import (
    ellipse_mask,
    ellipse_polygon_points,
    extend_line_to_borders,
    extend_ray_to_border,
    line_mask,
)
from fitsedit.theme import (
    ACCENT,
    APP_BG,
    CANVAS_BG,
    FONT,
    FONT_SMALL,
    PANEL_BG,
    PANEL_BORDER,
    TEXT,
    TEXT_DIM,
    hex_to_rgb,
)
from fitsedit.widgets import RoundButton, RoundedPanel, RoundSlider, SegmentedControl, rounded_rect_points

PAN_W, PAN_H = 220, 150
PREVIEW_W, PREVIEW_H = 150, 120
SIDEBAR_W = 240
ZOOM_STEP = 1.25
ZOOM_MULT_MIN = 1.0
ZOOM_MULT_MAX = 40.0
MAG_SIZE = 25
MAGNIFIER_GREEN = "#22c55e"
ACCENT_RGB = hex_to_rgb(ACCENT)

LINE_STYLES = [
    ("draw", "Draw"),
    ("segment", "Segment"),
    ("arrow", "Arrow"),
    ("line", "Line"),
]


class Entry:
    """A single loaded (or not-yet-loaded) FITS file slot."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.image: Optional[FitsImage] = None
        self.lowcut = 0.0
        self.highcut = 1.0

    def ensure_loaded(self, stretch: str) -> None:
        if self.image is not None or self.path is None:
            return
        self.image = load_fits_image(self.path)
        self.apply_stretch(stretch)

    def apply_stretch(self, stretch: str) -> None:
        if self.image is None:
            return
        if stretch == "zscale":
            self.lowcut, self.highcut = zscale_cuts(self.image.data)
        else:
            self.lowcut, self.highcut = minmax_cuts(self.image.data)


class FitsEditApp:
    def __init__(self, root: tk.Tk, paths: list[str]):
        self.root = root
        self.root.title("fitsedit")
        self.root.geometry("1400x900")
        self.root.configure(bg=APP_BG)

        self.entries: list[Entry] = [Entry(p) for p in paths] or [Entry(None)]
        self.index = 0

        self.stretch = "zscale"
        self.tool = tk.StringVar(value="ellipse")
        self.axis_a = tk.IntVar(value=40)
        self.axis_b = tk.IntVar(value=40)
        self.angle = tk.IntVar(value=0)
        self.thickness = tk.IntVar(value=15)
        self.line_style = tk.StringVar(value="draw")

        self.fit_zoom = 1.0
        self.zoom_mult = 1.0
        self.view_cx = 0.0
        self.view_cy = 0.0
        self.canvas_w = 1
        self.canvas_h = 1
        self._cursor_img_pos: Optional[tuple[float, float]] = None

        self._pan_drag: Optional[tuple[int, int, float, float]] = None
        self._line_drag_start: Optional[tuple[float, float]] = None
        self._line_drag_erase = False
        self._line_anchor: Optional[tuple[float, float]] = None
        self._line_anchor_erase = False
        self._preview_item: Optional[int] = None
        self._undo: Optional[tuple[int, np.ndarray]] = None

        self._build_menu()
        self._build_layout()

        self.tool.trace_add("write", lambda *_: self._on_tool_changed())
        self.line_style.trace_add("write", lambda *_: self._cancel_pending_line())
        self.root.bind("<Control-z>", self._on_undo)
        self.root.bind("<Escape>", lambda e: self._cancel_pending_line())

        self._rebuild_tool_options()
        self.load_current(reset_view=True)

    # ---------------------------------------------------------------- menu

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", command=self.open_files)
        file_menu.add_command(label="Export Mask", command=self.export_mask)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_radiobutton(label="Circle / Ellipse Mask", variable=self.tool, value="ellipse")
        mode_menu.add_radiobutton(label="Line Mask (satellite trail)", variable=self.tool, value="line")
        menubar.add_cascade(label="Mode", menu=mode_menu)

        cuts_menu = tk.Menu(menubar, tearoff=0)
        cuts_menu.add_command(label="Min/Max", command=lambda: self.set_stretch("minmax"))
        cuts_menu.add_command(label="ZScale", command=lambda: self.set_stretch("zscale"))
        menubar.add_cascade(label="Cuts", menu=cuts_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_help)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _show_help(self) -> None:
        messagebox.showinfo(
            "fitsedit",
            "fitsedit IMAGE1 IMAGE2 ...\n\n"
            "Left-click / drag: paint mask with the current tool\n"
            "Right-click / drag: erase mask\n"
            "Middle-click drag: pan the view\n"
            "Mouse wheel: zoom in (zoom 1 shows the full image; you can only zoom in from there)\n"
            "Ctrl+Z: undo last mask stroke\n"
            "Esc: cancel a pending line click\n\n"
            "Ellipse mode: stamp shapes sized by the major/minor axis and angle sliders\n\n"
            "Satellite mode styles:\n"
            "  Draw    - click and drag a trail freehand\n"
            "  Segment - click a start point, click an end point\n"
            "  Arrow   - click start, click a second point; the trail extends\n"
            "            past it to the image border\n"
            "  Line    - click two points; the trail extends to both borders",
        )

    # -------------------------------------------------------------- layout

    def _build_layout(self) -> None:
        toolbar_container = tk.Frame(self.root, bg=APP_BG)
        toolbar_container.pack(side="top", fill="x", padx=10, pady=(10, 6))
        toolbar_panel = RoundedPanel(toolbar_container, outer_bg=APP_BG, bg=PANEL_BG, radius=14, mode="hug")
        toolbar_panel.pack(fill="x")
        self._build_toolbar(toolbar_panel.inner)

        body = tk.Frame(self.root, bg=APP_BG)
        body.pack(side="top", fill="both", expand=True, padx=10)

        sidebar_container = tk.Frame(body, width=SIDEBAR_W, bg=APP_BG)
        sidebar_container.pack(side="left", fill="y", padx=(0, 6))
        sidebar_container.pack_propagate(False)
        sidebar_panel = RoundedPanel(sidebar_container, outer_bg=APP_BG, bg=PANEL_BG, radius=16, scrollable=True)
        sidebar_panel.pack(fill="both", expand=True)
        self._build_sidebar(sidebar_panel.inner)

        canvas_container = tk.Frame(body, bg=APP_BG)
        canvas_container.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_container, bg=CANVAS_BG, highlightthickness=1, highlightbackground=PANEL_BORDER)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<ButtonPress-1>", lambda e: self._on_button(e, erase=False))
        self.canvas.bind("<B1-Motion>", lambda e: self._on_drag(e, erase=False))
        self.canvas.bind("<ButtonRelease-1>", lambda e: self._on_release(e, erase=False))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._on_button(e, erase=True))
        self.canvas.bind("<B3-Motion>", lambda e: self._on_drag(e, erase=True))
        self.canvas.bind("<ButtonRelease-3>", lambda e: self._on_release(e, erase=True))
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", lambda e: setattr(self, "_pan_drag", None))
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, ZOOM_STEP))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / ZOOM_STEP))

        status_container = tk.Frame(self.root, bg=APP_BG)
        status_container.pack(side="bottom", fill="x", padx=10, pady=(6, 10))
        status_panel = RoundedPanel(status_container, outer_bg=APP_BG, bg=PANEL_BG, radius=10, mode="hug")
        status_panel.pack(fill="x")
        self.status = tk.Label(status_panel.inner, text="new file", bg=PANEL_BG, fg=TEXT_DIM, anchor="w", font=FONT_SMALL)
        self.status.pack(fill="x", padx=14, pady=8)

    def _build_toolbar(self, parent: tk.Frame) -> None:
        parent.configure(bg=PANEL_BG)
        pad = dict(padx=4, pady=10)

        tk.Label(parent, text="Zoom:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).pack(side="left", padx=(14, 4))
        self.zoom_label = tk.Label(parent, text="1", bg=PANEL_BG, fg=TEXT, font=FONT, width=5)
        self.zoom_label.pack(side="left")
        RoundButton(parent, "-", command=self.zoom_out, outer_bg=PANEL_BG, width=32).pack(side="left", **pad)
        RoundButton(parent, "+", command=self.zoom_in, outer_bg=PANEL_BG, width=32).pack(side="left", **pad)
        RoundButton(parent, "reset", command=self.reset_zoom, outer_bg=PANEL_BG).pack(side="left", padx=(0, 16), pady=10)

        RoundButton(parent, "<-", command=self.prev_image, outer_bg=PANEL_BG, width=36).pack(side="left", **pad)
        self.counter_label = tk.Label(parent, text="1/1", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, width=6)
        self.counter_label.pack(side="left")
        RoundButton(parent, "->", command=self.next_image, outer_bg=PANEL_BG, width=36).pack(side="left", padx=(4, 16), pady=10)

        self.filename_label = tk.Label(parent, text="noname", bg=PANEL_BG, fg=TEXT, font=FONT)
        self.filename_label.pack(side="left")

        RoundButton(parent, "kill", command=self.kill_current, outer_bg=PANEL_BG, danger=True).pack(side="right", padx=14, pady=10)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        parent.configure(bg=PANEL_BG)

        self.magnifier_canvas = tk.Canvas(parent, width=PAN_W, height=PAN_H, bg=CANVAS_BG, highlightthickness=0)
        self.magnifier_canvas.pack(padx=10, pady=10)

        info = tk.Frame(parent, bg=PANEL_BG)
        info.pack(fill="x", padx=14, pady=(0, 10))
        self.readout = {}
        for row, key in enumerate(["x", "y", "value", "RA", "DEC"]):
            tk.Label(info, text=f"{key}:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).grid(row=row, column=0, sticky="w")
            lbl = tk.Label(info, text="", bg=PANEL_BG, fg=TEXT, font=FONT_SMALL)
            lbl.grid(row=row, column=1, sticky="w", padx=(8, 0))
            self.readout[key] = lbl

        self._divider(parent)

        cuts = tk.Frame(parent, bg=PANEL_BG)
        cuts.pack(fill="x", padx=14, pady=10)
        tk.Label(cuts, text="lowcut:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).grid(row=0, column=0, sticky="w")
        self.lowcut_label = tk.Label(cuts, text="0", bg=PANEL_BG, fg=TEXT, font=FONT_SMALL)
        self.lowcut_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        tk.Label(cuts, text="highcut:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).grid(row=1, column=0, sticky="w")
        self.highcut_label = tk.Label(cuts, text="1", bg=PANEL_BG, fg=TEXT, font=FONT_SMALL)
        self.highcut_label.grid(row=1, column=1, sticky="w", padx=(8, 0))

        RoundButton(parent, "export mask", command=self.export_mask, outer_bg=PANEL_BG, accent=True,
                    width=SIDEBAR_W - 28).pack(padx=14, pady=(4, 12))

        self._divider(parent)

        mode_frame = tk.Frame(parent, bg=PANEL_BG)
        mode_frame.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(mode_frame, text="mode", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w")
        SegmentedControl(
            mode_frame, [("ellipse", "Ellipse"), ("line", "Satellite")], self.tool, outer_bg=PANEL_BG,
        ).pack(anchor="w", pady=(4, 0))

        self.tool_options_frame = tk.Frame(parent, bg=PANEL_BG)
        self.tool_options_frame.pack(fill="x")

        self.preview_canvas: Optional[tk.Canvas] = None

    def _divider(self, parent: tk.Frame) -> None:
        tk.Frame(parent, bg=PANEL_BORDER, height=1).pack(fill="x", padx=14)

    def _build_slider_row(self, parent: tk.Frame, name: str, var: tk.IntVar, lo: int, hi: int) -> None:
        label = tk.Label(parent, bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, anchor="w")
        label.pack(fill="x", padx=14, pady=(6, 2))

        def update_label(*_args: object) -> None:
            if label.winfo_exists():
                label.config(text=f"{name}: {var.get()}")

        trace_id = var.trace_add("write", update_label)
        label.bind("<Destroy>", lambda e: var.trace_remove("write", trace_id), add="+")
        update_label()

        RoundSlider(parent, var, lo, hi, width=SIDEBAR_W - 40, height=22, outer_bg=PANEL_BG).pack(padx=14, pady=(0, 4))

    def _rebuild_tool_options(self) -> None:
        for child in list(self.tool_options_frame.winfo_children()):
            child.destroy()
        self._cancel_pending_line()

        if self.tool.get() == "ellipse":
            self._build_slider_row(self.tool_options_frame, "major axis", self.axis_a, 1, 300)
            self._build_slider_row(self.tool_options_frame, "minor axis", self.axis_b, 1, 300)
            self._build_slider_row(self.tool_options_frame, "angle", self.angle, 0, 180)
        else:
            self._build_slider_row(self.tool_options_frame, "thickness", self.thickness, 1, 100)
            style_frame = tk.Frame(self.tool_options_frame, bg=PANEL_BG)
            style_frame.pack(fill="x", padx=14, pady=(8, 4))
            tk.Label(style_frame, text="style", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w")
            SegmentedControl(style_frame, LINE_STYLES, self.line_style, outer_bg=PANEL_BG).pack(anchor="w", pady=(4, 0))

        self.preview_canvas = tk.Canvas(self.tool_options_frame, width=PREVIEW_W, height=PREVIEW_H,
                                         bg=CANVAS_BG, highlightthickness=0)
        self.preview_canvas.pack(padx=14, pady=14)

        for var in (self.axis_a, self.axis_b, self.angle, self.thickness):
            var.trace_add("write", lambda *_: self.render_tool_preview())
        self.render_tool_preview()

    def _on_tool_changed(self) -> None:
        self._rebuild_tool_options()
        self._update_shape_preview(-1000, -1000)

    def _cancel_pending_line(self) -> None:
        self._line_anchor = None
        self._line_drag_start = None
        if hasattr(self, "canvas"):
            self.canvas.delete("line_preview")

    # --------------------------------------------------------- image state

    @property
    def entry(self) -> Entry:
        return self.entries[self.index]

    @property
    def image(self) -> Optional[FitsImage]:
        return self.entry.image

    @property
    def zoom(self) -> float:
        """Effective image-to-canvas pixel scale: fit-to-window baseline times the user multiplier."""
        return self.fit_zoom * self.zoom_mult

    def load_current(self, reset_view: bool = False) -> None:
        entry = self.entry
        try:
            entry.ensure_loaded(self.stretch)
        except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
            messagebox.showerror("fitsedit", f"Could not load {entry.path}:\n{exc}")
            entry.path = None

        if entry.image is not None and reset_view:
            self.reset_zoom()

        self.filename_label.config(text=os.path.basename(entry.path) if entry.path else "noname")
        self.counter_label.config(text=f"{self.index + 1}/{len(self.entries)}")
        self.lowcut_label.config(text=self._fmt(entry.lowcut))
        self.highcut_label.config(text=self._fmt(entry.highcut))
        self.status.config(text=f"loaded {entry.path}" if entry.path else "new file")

        self.render()

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:.4g}"

    def set_stretch(self, stretch: str) -> None:
        self.stretch = stretch
        self.entry.apply_stretch(stretch)
        self.lowcut_label.config(text=self._fmt(self.entry.lowcut))
        self.highcut_label.config(text=self._fmt(self.entry.highcut))
        self.render()

    # ------------------------------------------------------------- navigation

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Open FITS files",
            filetypes=[("FITS files", "*.fits *.fit *.fts *.fits.gz"), ("All files", "*.*")],
        )
        if not paths:
            return
        if len(self.entries) == 1 and self.entries[0].path is None:
            self.entries = []
        start = len(self.entries)
        self.entries.extend(Entry(p) for p in paths)
        self.index = start
        self.load_current(reset_view=True)

    def prev_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.load_current(reset_view=True)

    def next_image(self) -> None:
        if self.index < len(self.entries) - 1:
            self.index += 1
            self.load_current(reset_view=True)

    def kill_current(self) -> None:
        if len(self.entries) == 1:
            self.entries = [Entry(None)]
            self.index = 0
        else:
            del self.entries[self.index]
            self.index = min(self.index, len(self.entries) - 1)
        self.load_current(reset_view=True)

    # -------------------------------------------------------------- export

    def export_mask(self) -> None:
        entry = self.entry
        if entry.image is None or entry.path is None:
            messagebox.showwarning("fitsedit", "No image loaded to export a mask for.")
            return
        base, _ = os.path.splitext(entry.path)
        if base.endswith(".fits"):
            base, _ = os.path.splitext(base)
        out_path = f"{base}_mask.fits"
        hdu = fits.PrimaryHDU(data=entry.image.mask.astype("uint8"), header=entry.image.header.copy())
        hdu.writeto(out_path, overwrite=True)
        self.status.config(text=f"exported mask to {out_path}")

    # --------------------------------------------------------------- undo

    def _push_undo(self) -> None:
        if self.image is None:
            return
        self._undo = (self.index, self.image.mask.copy())

    def _on_undo(self, _event: Optional[tk.Event] = None) -> None:
        if self._undo is None:
            return
        idx, mask = self._undo
        if idx < len(self.entries) and self.entries[idx].image is not None:
            self.entries[idx].image.mask = mask
            self._undo = None
            if idx == self.index:
                self.render()

    # ------------------------------------------------------- coordinate math

    def img_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        cx = self.canvas_w / 2 + (ix - self.view_cx) * self.zoom
        cy = self.canvas_h / 2 + (iy - self.view_cy) * self.zoom
        return cx, cy

    def canvas_to_img(self, cx: float, cy: float) -> tuple[float, float]:
        ix = self.view_cx + (cx - self.canvas_w / 2) / self.zoom
        iy = self.view_cy + (cy - self.canvas_h / 2) / self.zoom
        return ix, iy

    # ------------------------------------------------------------ rendering

    def _compute_fit_zoom(self) -> float:
        if self.image is None or self.canvas_w <= 1 or self.canvas_h <= 1:
            return 1.0
        ny, nx = self.image.data.shape
        return min(self.canvas_w / nx, self.canvas_h / ny) or 1.0

    def _update_zoom_label(self) -> None:
        text = f"{self.zoom_mult:.2f}".rstrip("0").rstrip(".")
        self.zoom_label.config(text=text or "1")

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self.canvas_w, self.canvas_h = event.width, event.height
        if self.image is not None:
            self.fit_zoom = self._compute_fit_zoom()
        self.render()

    def render(self) -> None:
        self.render_magnifier()
        self.canvas.delete("img")
        image = self.image
        if image is None or self.canvas_w <= 1:
            return

        data = image.data
        ny, nx = data.shape
        ix0, iy0 = self.canvas_to_img(0, 0)
        ix1, iy1 = self.canvas_to_img(self.canvas_w, self.canvas_h)
        x0 = max(int(np.floor(min(ix0, ix1))), 0)
        x1 = min(int(np.ceil(max(ix0, ix1))), nx)
        y0 = max(int(np.floor(min(iy0, iy1))), 0)
        y1 = min(int(np.ceil(max(iy0, iy1))), ny)
        if x1 <= x0 or y1 <= y0:
            return

        entry = self.entry
        crop = data[y0:y1, x0:x1]
        span = max(entry.highcut - entry.lowcut, 1e-12)
        norm = np.clip((crop - entry.lowcut) / span, 0, 1)
        gray = (norm * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

        mask_crop = image.mask[y0:y1, x0:x1]
        if mask_crop.any():
            alpha = 0.55
            ar, ag, ab = ACCENT_RGB
            for ch, accent_v in enumerate((ar, ag, ab)):
                channel = rgb[..., ch].astype(np.float32)
                blended = channel * (1 - alpha) + accent_v * alpha
                rgb[..., ch] = np.where(mask_crop, blended, channel).astype(np.uint8)

        pil_img = Image.fromarray(rgb, mode="RGB")
        disp_w = max(int(round((x1 - x0) * self.zoom)), 1)
        disp_h = max(int(round((y1 - y0) * self.zoom)), 1)
        resample = Image.NEAREST if self.zoom >= 1 else Image.BOX
        pil_img = pil_img.resize((disp_w, disp_h), resample)

        cx0, cy0 = self.img_to_canvas(x0, y0)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(cx0, cy0, image=self._photo, anchor="nw", tags="img")
        self.canvas.tag_lower("img")

    def render_magnifier(self) -> None:
        canvas = self.magnifier_canvas
        canvas.delete("all")
        w, h = PAN_W, PAN_H
        image = self.image
        if image is None or self._cursor_img_pos is None:
            self._draw_grid(canvas, w, h)
            return

        ny, nx = image.data.shape
        half = MAG_SIZE // 2
        cx_i = int(round(self._cursor_img_pos[0]))
        cy_i = int(round(self._cursor_img_pos[1]))
        x0, y0 = cx_i - half, cy_i - half

        crop = np.full((MAG_SIZE, MAG_SIZE), np.nan, dtype=np.float64)
        mask_crop = np.zeros((MAG_SIZE, MAG_SIZE), dtype=bool)
        sx0, sx1 = max(x0, 0), min(x0 + MAG_SIZE, nx)
        sy0, sy1 = max(y0, 0), min(y0 + MAG_SIZE, ny)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image.data[sy0:sy1, sx0:sx1]
            mask_crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image.mask[sy0:sy1, sx0:sx1]

        entry = self.entry
        span = max(entry.highcut - entry.lowcut, 1e-12)
        norm = np.clip((crop - entry.lowcut) / span, 0, 1)
        gray = np.where(np.isnan(crop), 0.12, norm)
        gray_u8 = (gray * 255).astype(np.uint8)
        rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

        if mask_crop.any():
            alpha = 0.55
            ar, ag, ab = ACCENT_RGB
            for ch, accent_v in enumerate((ar, ag, ab)):
                channel = rgb[..., ch].astype(np.float32)
                blended = channel * (1 - alpha) + accent_v * alpha
                rgb[..., ch] = np.where(mask_crop, blended, channel).astype(np.uint8)

        square = min(w, h)
        block = max(square // MAG_SIZE, 1)
        disp = block * MAG_SIZE
        pil_img = Image.fromarray(rgb, mode="RGB").resize((disp, disp), Image.NEAREST)
        self._mag_photo = ImageTk.PhotoImage(pil_img)
        ox, oy = (w - disp) // 2, (h - disp) // 2
        canvas.create_image(ox, oy, image=self._mag_photo, anchor="nw")

        cxp, cyp = ox + half * block, oy + half * block
        canvas.create_rectangle(cxp, cyp, cxp + block, cyp + block, outline=MAGNIFIER_GREEN, width=2)

    @staticmethod
    def _draw_grid(canvas: tk.Canvas, w: int, h: int, step: int = 10) -> None:
        for x in range(0, w, step):
            canvas.create_line(x, 0, x, h, fill=PANEL_BORDER)
        for y in range(0, h, step):
            canvas.create_line(0, y, w, y, fill=PANEL_BORDER)

    def render_tool_preview(self) -> None:
        if self.preview_canvas is None or not self.preview_canvas.winfo_exists():
            return
        self.preview_canvas.delete("all")
        w, h = PREVIEW_W, PREVIEW_H
        if self.tool.get() == "ellipse":
            a, b = self.axis_a.get(), self.axis_b.get()
            scale = (min(w, h) / 2 * 0.75) / max(a, b, 1)
            pts = ellipse_polygon_points(w / 2, h / 2, a * scale, b * scale, self.angle.get())
            self.preview_canvas.create_polygon(pts, fill=ACCENT, outline="")
        else:
            thickness = min(self.thickness.get(), h * 0.6)
            y0, y1 = h / 2 - thickness / 2, h / 2 + thickness / 2
            pts = rounded_rect_points(12, y0, w - 12, y1, thickness / 2)
            self.preview_canvas.create_polygon(pts, smooth=True, fill=ACCENT, outline="")

    # ---------------------------------------------------------------- input

    def _on_motion(self, event: tk.Event) -> None:
        ix, iy = self.canvas_to_img(event.x, event.y)
        self._cursor_img_pos = (ix, iy)
        self.render_magnifier()
        ix_i, iy_i = int(round(ix)), int(round(iy))
        self.readout["x"].config(text=str(ix_i))
        self.readout["y"].config(text=str(iy_i))

        image = self.image
        if image is not None and 0 <= iy_i < image.data.shape[0] and 0 <= ix_i < image.data.shape[1]:
            self.readout["value"].config(text=self._fmt(float(image.data[iy_i, ix_i])))
        else:
            self.readout["value"].config(text="")

        if image is not None and image.wcs is not None:
            try:
                sky = image.wcs.pixel_to_world(ix, iy)
                self.readout["RA"].config(text=sky.ra.to_string(unit="hourangle", sep=":", precision=2))
                self.readout["DEC"].config(text=sky.dec.to_string(sep=":", precision=1, alwaysign=True))
            except Exception:
                self.readout["RA"].config(text="")
                self.readout["DEC"].config(text="")
        else:
            self.readout["RA"].config(text="")
            self.readout["DEC"].config(text="")

        if self.tool.get() == "ellipse":
            self._update_shape_preview(event.x, event.y)
        elif self._line_anchor is not None and self.line_style.get() != "draw":
            self._update_click_line_preview(event.x, event.y)

    def _update_shape_preview(self, cx: float, cy: float) -> None:
        if self._preview_item is not None:
            self.canvas.delete(self._preview_item)
            self._preview_item = None
        if self.tool.get() != "ellipse" or self.image is None:
            return
        a = self.axis_a.get() * self.zoom
        b = self.axis_b.get() * self.zoom
        pts = ellipse_polygon_points(cx, cy, a, b, self.angle.get())
        self._preview_item = self.canvas.create_polygon(pts, outline=TEXT_DIM, fill="", width=2)

    def _update_click_line_preview(self, cx: float, cy: float) -> None:
        self.canvas.delete("line_preview")
        if self.image is None or self._line_anchor is None:
            return
        x0, y0 = self._line_anchor
        x1, y1 = self.canvas_to_img(cx, cy)
        ex0, ey0, ex1, ey1 = self._extend_for_style(x0, y0, x1, y1)
        sx0, sy0 = self.img_to_canvas(ex0, ey0)
        sx1, sy1 = self.img_to_canvas(ex1, ey1)
        self.canvas.create_line(sx0, sy0, sx1, sy1, fill=TEXT_DIM, width=2, dash=(4, 3), tags="line_preview")
        ax, ay = self.img_to_canvas(x0, y0)
        self.canvas.create_oval(ax - 4, ay - 4, ax + 4, ay + 4, fill=ACCENT, outline="", tags="line_preview")

    def _extend_for_style(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        if self.image is None:
            return x0, y0, x1, y1
        shape = self.image.data.shape
        style = self.line_style.get()
        if style == "arrow":
            return extend_ray_to_border(shape, x0, y0, x1, y1)
        if style == "line":
            return extend_line_to_borders(shape, x0, y0, x1, y1)
        return x0, y0, x1, y1

    def _on_button(self, event: tk.Event, erase: bool) -> None:
        if self.image is None:
            return
        tool = self.tool.get()
        if tool == "ellipse":
            self._push_undo()
            self._stamp_ellipse(event.x, event.y, erase)
        elif tool == "line":
            if self.line_style.get() == "draw":
                self._push_undo()
                self._line_drag_start = self.canvas_to_img(event.x, event.y)
                self._line_drag_erase = erase
            elif self._line_anchor is None:
                self._push_undo()
                self._line_anchor = self.canvas_to_img(event.x, event.y)
                self._line_anchor_erase = erase
            else:
                self._finalize_click_line(event.x, event.y)

    def _on_drag(self, event: tk.Event, erase: bool) -> None:
        tool = self.tool.get()
        if tool == "ellipse" and self.image is not None:
            self._stamp_ellipse(event.x, event.y, erase)
            self._update_shape_preview(event.x, event.y)
        elif tool == "line" and self.line_style.get() == "draw" and self._line_drag_start is not None:
            self.canvas.delete("line_preview")
            sx, sy = self.img_to_canvas(*self._line_drag_start)
            self.canvas.create_line(sx, sy, event.x, event.y, fill=TEXT_DIM, width=2, tags="line_preview")

    def _on_release(self, event: tk.Event, erase: bool) -> None:
        tool = self.tool.get()
        if tool == "line" and self.line_style.get() == "draw" and self._line_drag_start is not None and self.image is not None:
            self.canvas.delete("line_preview")
            x0, y0 = self._line_drag_start
            x1, y1 = self.canvas_to_img(event.x, event.y)
            stamp = line_mask(self.image.data.shape, x0, y0, x1, y1, self.thickness.get())
            self.image.mask = (self.image.mask & ~stamp) if self._line_drag_erase else (self.image.mask | stamp)
            self._line_drag_start = None
            self.render()

    def _finalize_click_line(self, cx: float, cy: float) -> None:
        if self.image is None or self._line_anchor is None:
            return
        x0, y0 = self._line_anchor
        x1, y1 = self.canvas_to_img(cx, cy)
        ex0, ey0, ex1, ey1 = self._extend_for_style(x0, y0, x1, y1)
        stamp = line_mask(self.image.data.shape, ex0, ey0, ex1, ey1, self.thickness.get())
        self.image.mask = (self.image.mask & ~stamp) if self._line_anchor_erase else (self.image.mask | stamp)
        self._line_anchor = None
        self.canvas.delete("line_preview")
        self.render()

    def _stamp_ellipse(self, cx: float, cy: float, erase: bool) -> None:
        image = self.image
        if image is None:
            return
        ix, iy = self.canvas_to_img(cx, cy)
        stamp = ellipse_mask(image.data.shape, ix, iy, self.axis_a.get(), self.axis_b.get(), self.angle.get())
        image.mask = (image.mask & ~stamp) if erase else (image.mask | stamp)
        self.render()

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_drag = (event.x, event.y, self.view_cx, self.view_cy)

    def _on_pan_drag(self, event: tk.Event) -> None:
        if self._pan_drag is None:
            return
        sx, sy, ocx, ocy = self._pan_drag
        self.view_cx = ocx - (event.x - sx) / self.zoom
        self.view_cy = ocy - (event.y - sy) / self.zoom
        self.render()

    def reset_zoom(self) -> None:
        self.zoom_mult = 1.0
        if self.image is not None:
            ny, nx = self.image.data.shape
            self.view_cx, self.view_cy = nx / 2, ny / 2
            self.fit_zoom = self._compute_fit_zoom()
        self._update_zoom_label()
        self.render()

    def zoom_in(self) -> None:
        self._zoom_at(self.canvas_w / 2, self.canvas_h / 2, ZOOM_STEP)

    def zoom_out(self) -> None:
        self._zoom_at(self.canvas_w / 2, self.canvas_h / 2, 1 / ZOOM_STEP)

    def _on_wheel(self, event: tk.Event) -> None:
        factor = ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        if self.image is None:
            return
        new_mult = max(min(self.zoom_mult * factor, ZOOM_MULT_MAX), ZOOM_MULT_MIN)
        if new_mult == self.zoom_mult:
            return
        ix, iy = self.canvas_to_img(cx, cy)
        self.zoom_mult = new_mult
        new_cx, new_cy = self.canvas_to_img(cx, cy)
        self.view_cx += ix - new_cx
        self.view_cy += iy - new_cy
        self._update_zoom_label()
        self.render()


def run_gui(paths: list[str]) -> int:
    root = tk.Tk()
    FitsEditApp(root, paths)
    root.mainloop()
    return 0
