"""Tkinter GUI: view FITS images and paint circular/elliptical or line (satellite trail) masks."""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np
from astropy.io import fits
from PIL import Image, ImageTk

from maskfits.binning import bin_func, bin_mask, unbin_mask
from maskfits.colormaps import COLORMAP_LUTS, COLORMAP_NAMES, mask_tint_for
from maskfits.cuts_dialog import ManualCutsWindow
from maskfits.imagedata import (
    PERCENTILE_PRESETS,
    STRETCH_NAMES,
    STRETCHES,
    FitsImage,
    gaussian_smooth,
    load_fits_image,
    minmax_cuts,
    percentile_cuts,
    zscale_cuts,
)
from maskfits.masking import (
    ellipse_mask,
    ellipse_polygon_points,
    extend_line_to_borders,
    extend_ray_to_border,
    line_mask,
)
from maskfits.theme import (
    ACCENT,
    APP_BG,
    BUTTON_BG,
    CANVAS_BG,
    FONT,
    FONT_SMALL,
    GREEN,
    PANEL_BG,
    PANEL_BORDER,
    TEXT,
    TEXT_DIM,
    set_theme,
)
from maskfits.widgets import RoundButton, RoundedPanel, RoundSlider, SegmentedControl, ThemeToggle

MAG_SIZE = 31
MAG_BLOCK = 6
PAN_W = PAN_H = MAG_SIZE * MAG_BLOCK
PREVIEW_W, PREVIEW_H = 210, 170
MAX_SHAPE_SIZE = 500
RADIUS_MIN = 0.5  # small enough to mask a single pixel
SIDEBAR_W = 240
ZOOM_STEP = 1.25
ZOOM_MULT_MIN = 1.0
ZOOM_MULT_MAX = 40.0

LINE_STYLES = [
    ("segment", "Segment"),
    ("arrow", "Arrow"),
    ("line", "Line"),
]

SCALE_OPTIONS = [(name, name.capitalize()) for name in STRETCH_NAMES]

HOTKEY_ENTRIES = [
    ("Left-click / drag", "Paint mask"),
    ("Right-click / drag", "Erase mask"),
    ("Ctrl + left-click drag", "Pan the view"),
    ("Mouse wheel", "Zoom in"),
    ("← / →", "Previous / next image"),
    ("Ctrl+Z / U", "Undo last mask stroke"),
    ("Ctrl+Shift+Z / Y", "Redo"),
    ("R", "Clear the whole mask"),
    ("E / W", "Grow / shrink radius or thickness"),
    ("C", "Cycle colormap"),
    ("I", "Invert colormap"),
    ("S", "Smooth image (Gaussian, current sigma)"),
    ("B", "Bin image (NxN, current factor)"),
    ("Esc", "Cancel a pending line click"),
]

ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "icon.png")


def _apply_icon(root: tk.Tk) -> None:
    """Set the window/taskbar icon cross-platform.

    On Windows/Linux, iconphoto() is enough to change the taskbar icon too.
    On macOS it only affects the window itself - a plain Python process's Dock
    icon stays the generic Python icon otherwise - so also set it live via
    PyObjC (AppKit) if available. That's an optional dependency: if it isn't
    installed, the Dock icon is simply left as-is rather than erroring.
    """
    if not os.path.exists(ICON_PATH):
        return
    try:
        icon_image = tk.PhotoImage(file=ICON_PATH)
        root.iconphoto(True, icon_image)
        root._icon_image_ref = icon_image  # keep a reference so Tk doesn't GC it
    except tk.TclError:
        pass

    if sys.platform == "darwin":
        try:
            from AppKit import NSApplication, NSImage

            ns_image = NSImage.alloc().initByReferencingFile_(ICON_PATH)
            NSApplication.sharedApplication().setApplicationIconImage_(ns_image)
        except Exception:
            pass


def _bring_to_front(root: tk.Tk) -> None:
    """Force the window to open in front of every other window.

    A freshly-created Tk window doesn't always land on top of whatever
    already has focus (e.g. the terminal it was launched from, especially on
    macOS) - briefly toggling -topmost forces it to the front once at
    startup, without leaving it permanently pinned above other windows.
    """
    root.lift()
    root.attributes("-topmost", True)
    root.after(250, lambda: root.attributes("-topmost", False))
    root.focus_force()


class Entry:
    """A single loaded (or not-yet-loaded) FITS file slot."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.image: Optional[FitsImage] = None
        self.lowcut = 0.0
        self.highcut = 1.0
        # Smoothing and binning are independent and composable: original_data
        # is the one pristine array (captured the first time either is turned
        # on), and up to three more arrays cache each combination actually
        # used - smoothed-only, binned-only, and smoothed-then-binned -
        # keyed on the sigma/factor they were built at, so toggling either
        # effect back on (or switching between combinations already seen)
        # reuses a cache instead of recomputing. A cache is only rebuilt when
        # the sigma and/or factor it depends on has actually changed.
        self.original_data: Optional[np.ndarray] = None

        self.is_smoothed = False
        self.smooth_sigma: Optional[float] = None
        self.smoothed_cache: Optional[np.ndarray] = None
        self._smoothed_cache_sigma: Optional[float] = None

        self.is_binned = False
        self.bin_factor: Optional[int] = None
        self.binned_cache: Optional[np.ndarray] = None
        self._binned_cache_factor: Optional[int] = None

        self.smoothed_binned_cache: Optional[np.ndarray] = None
        self._smoothed_binned_cache_sigma: Optional[float] = None
        self._smoothed_binned_cache_factor: Optional[int] = None

        # The mask, unlike the data, only ever changes shape via binning (never
        # smoothing). It's user-edited at whatever resolution is currently
        # active, so instead of a pure cache it's re-derived fresh on every
        # bin-related change - mask_backup holds the full-res mask while
        # currently binned (the source both to unbin back into and to fall
        # back on for whatever bottom/right remainder binning trimmed off).
        self.mask_backup: Optional[np.ndarray] = None

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
        elif stretch.startswith("pct"):
            percent = float(stretch[3:])
            self.lowcut, self.highcut = percentile_cuts(self.image.data, percent)
        else:
            self.lowcut, self.highcut = minmax_cuts(self.image.data)


class MaskFitsApp:
    def __init__(self, root: tk.Tk, paths: list[str]):
        self.root = root
        self.root.title("maskfits")
        self.root.geometry("1400x900")
        self.root.configure(bg=APP_BG)
        _apply_icon(self.root)
        _bring_to_front(self.root)

        self.entries: list[Entry] = [Entry(p) for p in paths] or [Entry(None)]
        self.index = 0

        self.stretch = "zscale"
        self.scale_function = tk.StringVar(value="log")
        self.colormap = tk.StringVar(value="Grayscale")
        self.invert_colormap = tk.BooleanVar(value=False)
        self.mask_alpha = tk.IntVar(value=100)
        self.tool = tk.StringVar(value="circle")
        self.ellipticity = tk.IntVar(value=0)
        self.angle = tk.IntVar(value=0)
        self.radius = tk.DoubleVar(value=40.0)
        self.thickness = tk.IntVar(value=15)
        self.line_style = tk.StringVar(value="segment")
        self.smooth_sigma_var = tk.StringVar(value="2")
        self.bin_factor_var = tk.StringVar(value="4")

        self.fit_zoom = 1.0
        self.zoom_mult = 1.0
        self.view_cx = 0.0
        self.view_cy = 0.0
        self.canvas_w = 1
        self.canvas_h = 1
        self._cursor_img_pos: Optional[tuple[float, float]] = None

        self._pan_drag: Optional[tuple[int, int, float, float]] = None
        self._line_anchor: Optional[tuple[float, float]] = None
        self._line_anchor_erase = False
        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._undo: Optional[tuple[int, np.ndarray]] = None
        self._redo: Optional[tuple[int, np.ndarray]] = None
        self.light_mode = False

        self._build_ui()

        self.tool.trace_add("write", lambda *_: self._on_tool_changed())
        self.line_style.trace_add("write", lambda *_: self._cancel_pending_line())
        self.scale_function.trace_add("write", lambda *_: self.render())
        self.colormap.trace_add("write", lambda *_: self.render())
        self.invert_colormap.trace_add("write", lambda *_: self.render())
        self.root.bind("<Control-z>", self._on_undo)
        self.root.bind("<Escape>", lambda e: self._cancel_pending_line())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<u>", self._on_undo)
        self.root.bind("<Control-Shift-Z>", self._on_redo)
        self.root.bind("<y>", self._on_redo)
        self.root.bind("<r>", lambda e: self.reset_mask())
        self.root.bind("<e>", lambda e: self._adjust_shape_size(1))
        self.root.bind("<w>", lambda e: self._adjust_shape_size(-1))
        self.root.bind("<c>", self._cycle_colormap)
        self.root.bind("<i>", lambda e: self.invert_colormap.set(not self.invert_colormap.get()))
        self.root.bind("<s>", self._toggle_smoothing)
        self.root.bind("<b>", self._toggle_binning)

        # A text box should give up keyboard focus once you're done with it -
        # otherwise every hotkey above keeps typing into it instead of firing.
        # These fire on the "all" bindtag, after any widget-specific handler
        # (e.g. an entry's own <Return> handler applies its value first, then
        # this runs and releases focus).
        self.root.bind_all("<Return>", self._defocus_active_entry, add="+")
        self.root.bind_all("<Button-1>", self._on_global_click, add="+")

        self._rebuild_tool_options()
        self.load_current(reset_view=True)

    # ---------------------------------------------------------------- theme

    def _build_ui(self) -> None:
        self._build_menu()
        self._build_layout()

    def _toggle_theme(self) -> None:
        self.light_mode = not self.light_mode
        set_theme("light" if self.light_mode else "dark")
        self._rebuild_ui()

    def _rebuild_ui(self) -> None:
        """Tear down and reconstruct the whole widget tree after a theme switch.

        The custom canvas-drawn widgets (RoundButton, RoundedPanel, RoundSlider)
        bake their colors into their canvas draw calls at construction time, so
        restyling them means rebuilding, not recoloring in place. All actual
        state - loaded images, masks, zoom, current tool, colormap, smoothing
        cache, etc. - lives in plain attributes and tk Variables on self, none
        of which this touches; only the widgets themselves are thrown away and
        rebuilt against the now-active palette.
        """
        for child in self.root.winfo_children():
            child.destroy()
        self.root.configure(bg=APP_BG)
        self._build_ui()
        self._rebuild_tool_options()
        self._update_cuts_display()
        self._update_smooth_button()
        self._update_bin_button()
        self.load_current(reset_view=False)

    # ---------------------------------------------------------------- menu

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", command=self.open_files)
        file_menu.add_command(label="Export Mask", command=self.export_mask)
        file_menu.add_command(label="Save Mask As...", command=self.export_mask_as)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_radiobutton(label="Ellipse Mask", variable=self.tool, value="ellipse")
        mode_menu.add_radiobutton(label="Circle Mask", variable=self.tool, value="circle")
        mode_menu.add_radiobutton(label="Line Mask (satellite trail)", variable=self.tool, value="line")
        menubar.add_cascade(label="Mode", menu=mode_menu)

        # Cuts and Scale used to be separate menus; combined here into one
        # "Scale" menu with the two groups kept visually distinct by a
        # separator, since both are about how pixel values map to the screen.
        scale_menu = tk.Menu(menubar, tearoff=0)
        scale_menu.add_command(label="Min/Max", command=lambda: self.set_stretch("minmax"))
        scale_menu.add_command(label="ZScale", command=lambda: self.set_stretch("zscale"))
        scale_menu.add_separator()
        for percent in PERCENTILE_PRESETS:
            scale_menu.add_command(label=f"{percent}%", command=lambda p=percent: self.set_stretch(f"pct{p}"))
        scale_menu.add_separator()
        scale_menu.add_command(label="Manual...", command=self.open_manual_cuts)
        scale_menu.add_separator()
        for value, label in SCALE_OPTIONS:
            scale_menu.add_radiobutton(label=label, variable=self.scale_function, value=value)
        scale_menu.add_separator()
        scale_menu.add_command(label="Reset", command=self.reset_scale)
        menubar.add_cascade(label="Scale", menu=scale_menu)

        color_menu = tk.Menu(menubar, tearoff=0)
        for name in COLORMAP_NAMES:
            color_menu.add_radiobutton(label=name, variable=self.colormap, value=name)
        color_menu.add_separator()
        color_menu.add_checkbutton(label="Invert Colormap", variable=self.invert_colormap)
        menubar.add_cascade(label="Color", menu=color_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_help)
        help_menu.add_separator()
        for keys, description in HOTKEY_ENTRIES:
            help_menu.add_command(label=f"{keys}    {description}", state="disabled")

        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _show_help(self) -> None:
        messagebox.showinfo(
            "maskfits",
            "maskfits IMAGE1 IMAGE2 ...\n\n"
            "Left-click / drag: paint mask with the current tool\n"
            "Right-click / drag: erase mask\n"
            "Ctrl + left-click drag: pan the view\n"
            "Mouse wheel: zoom in (zoom 1 shows the full image; you can only zoom in from there)\n"
            "Ctrl+Z or U: undo last mask stroke\n"
            "Ctrl+Shift+Z or Y: redo\n"
            "R: clear the whole mask\n"
            "← / →: previous / next image\n"
            "E / W: grow / shrink the active tool's radius or thickness\n"
            "Esc: cancel a pending line click\n\n"
            "Ellipse mode: stamp shapes sized by the radius, ellipticity, and angle sliders\n"
            "Circle mode: stamp circles sized by the radius slider\n\n"
            "Satellite mode styles:\n"
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
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_pan_drag", None))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._on_redo())
        self.canvas.bind("<ButtonPress-2>", lambda e: self._on_button(e, erase=True))
        self.canvas.bind("<B2-Motion>", lambda e: self._on_drag(e, erase=True))
        self.canvas.bind("<Control-Button-1>", self._on_pan_start)
        self.canvas.bind("<Control-B1-Motion>", self._on_pan_drag)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, ZOOM_STEP))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / ZOOM_STEP))

        status_container = tk.Frame(self.root, bg=APP_BG)
        status_container.pack(side="bottom", fill="x", padx=10, pady=(6, 10))
        status_panel = RoundedPanel(status_container, outer_bg=APP_BG, bg=PANEL_BG, radius=10, mode="hug")
        status_panel.pack(fill="x")
        self.status = tk.Label(status_panel.inner, text="new file", bg=PANEL_BG, fg=TEXT_DIM, anchor="w", font=FONT_SMALL)
        self.status.pack(side="left", fill="x", expand=True, padx=14, pady=8)
        tk.Label(status_panel.inner, text="\u00A9 Jan-Niklas Pippert 2026", bg=PANEL_BG, fg=TEXT_DIM,
                 anchor="e", font=FONT_SMALL).pack(side="right", padx=14, pady=8)

    def _build_toolbar(self, parent: tk.Frame) -> None:
        parent.configure(bg=PANEL_BG)
        pad = dict(padx=4, pady=10)

        ThemeToggle(parent, command=self._toggle_theme, light=self.light_mode, outer_bg=PANEL_BG).pack(
            side="left", padx=(14, 12), pady=10)

        tk.Label(parent, text="Zoom:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self.zoom_label = tk.Label(parent, text="1", bg=PANEL_BG, fg=TEXT, font=FONT, width=5)
        self.zoom_label.pack(side="left")
        RoundButton(parent, "-", command=self.zoom_out, outer_bg=PANEL_BG, width=32).pack(side="left", **pad)
        RoundButton(parent, "+", command=self.zoom_in, outer_bg=PANEL_BG, width=32).pack(side="left", **pad)
        RoundButton(parent, "reset", command=self.reset_zoom, outer_bg=PANEL_BG).pack(side="left", padx=(0, 16), pady=10)

        self.smooth_button = RoundButton(parent, "smooth", command=self._toggle_smoothing, outer_bg=PANEL_BG,
                                          toggle=True, width=72)
        self.smooth_button.pack(side="left", padx=(0, 4), pady=10)
        sigma_entry = tk.Entry(parent, textvariable=self.smooth_sigma_var, width=4, justify="center",
                                bg=BUTTON_BG, fg=TEXT, insertbackground=TEXT, relief="flat", highlightthickness=1,
                                highlightbackground=PANEL_BORDER, highlightcolor=ACCENT, font=FONT_SMALL)
        sigma_entry.pack(side="left", padx=(0, 16), pady=10)
        sigma_entry.bind("<Return>", self._apply_sigma_entry)
        sigma_entry.bind("<FocusOut>", self._apply_sigma_entry)

        self.bin_button = RoundButton(parent, "bin", command=self._toggle_binning, outer_bg=PANEL_BG,
                                       toggle=True, width=60)
        self.bin_button.pack(side="left", padx=(0, 4), pady=10)
        bin_entry = tk.Entry(parent, textvariable=self.bin_factor_var, width=4, justify="center",
                              bg=BUTTON_BG, fg=TEXT, insertbackground=TEXT, relief="flat", highlightthickness=1,
                              highlightbackground=PANEL_BORDER, highlightcolor=ACCENT, font=FONT_SMALL)
        bin_entry.pack(side="left", padx=(0, 16), pady=10)
        bin_entry.bind("<Return>", self._apply_bin_entry)
        bin_entry.bind("<FocusOut>", self._apply_bin_entry)

        SegmentedControl(parent, SCALE_OPTIONS, self.scale_function, outer_bg=PANEL_BG).pack(
            side="left", padx=(0, 16), pady=10)

        RoundButton(parent, "<-", command=self.prev_image, outer_bg=PANEL_BG, width=36).pack(side="left", **pad)
        self.counter_label = tk.Label(parent, text="1/1", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, width=6)
        self.counter_label.pack(side="left")
        RoundButton(parent, "->", command=self.next_image, outer_bg=PANEL_BG, width=36).pack(side="left", padx=(4, 16), pady=10)

        self.filename_label = tk.Label(parent, text="noname", bg=PANEL_BG, fg=TEXT, font=FONT)
        self.filename_label.pack(side="left")

        RoundButton(parent, "kill", command=self.kill_current, outer_bg=PANEL_BG, danger=True).pack(side="right", padx=14, pady=10)
        RoundButton(parent, "reset mask", command=self.reset_mask, outer_bg=PANEL_BG).pack(side="right", pady=10)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        parent.configure(bg=PANEL_BG)

        self.magnifier_canvas = tk.Canvas(parent, width=PAN_W, height=PAN_H, bg=CANVAS_BG, highlightthickness=0)
        self.magnifier_canvas.pack(padx=10, pady=10)

        info = tk.Frame(parent, bg=PANEL_BG)
        info.pack(fill="x", padx=14, pady=(0, 10))
        self.readout = {}
        for row, (left_key, right_key) in enumerate((("x", "RA"), ("y", "DEC"))):
            tk.Label(info, text=f"{left_key}:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).grid(
                row=row, column=0, sticky="w")
            left_lbl = tk.Label(info, text="", bg=PANEL_BG, fg=TEXT, font=FONT_SMALL)
            left_lbl.grid(row=row, column=1, sticky="w", padx=(8, 14))
            self.readout[left_key] = left_lbl

            tk.Label(info, text=f"{right_key}:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).grid(
                row=row, column=2, sticky="w")
            right_lbl = tk.Label(info, text="", bg=PANEL_BG, fg=TEXT, font=FONT_SMALL)
            right_lbl.grid(row=row, column=3, sticky="w", padx=(8, 0))
            self.readout[right_key] = right_lbl

        tk.Label(info, text="value:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).grid(row=2, column=0, sticky="w")
        value_lbl = tk.Label(info, text="", bg=PANEL_BG, fg=TEXT, font=FONT_SMALL)
        value_lbl.grid(row=2, column=1, columnspan=3, sticky="w", padx=(8, 0))
        self.readout["value"] = value_lbl

        self._divider(parent)

        self.lowcut_var = tk.StringVar(value="0")
        self.highcut_var = tk.StringVar(value="1")

        # Stacked label-left/entry-right rows (same pattern as the shape-size
        # boxes below) rather than a single row of two label+entry pairs -
        # the sidebar isn't wide enough to fit both entries side by side
        # without the highcut box getting clipped.
        lowcut_row = tk.Frame(parent, bg=PANEL_BG)
        lowcut_row.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(lowcut_row, text="lowcut:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, anchor="w").pack(side="left")
        lowcut_entry = tk.Entry(lowcut_row, textvariable=self.lowcut_var, width=8, justify="center", bg=BUTTON_BG,
                                 fg=TEXT, insertbackground=TEXT, relief="flat", highlightthickness=1,
                                 highlightbackground=PANEL_BORDER, highlightcolor=ACCENT, font=FONT_SMALL)
        lowcut_entry.pack(side="right")
        lowcut_entry.bind("<Return>", self._apply_lowcut_entry)
        lowcut_entry.bind("<FocusOut>", self._apply_lowcut_entry)

        highcut_row = tk.Frame(parent, bg=PANEL_BG)
        highcut_row.pack(fill="x", padx=14, pady=(0, 10))
        tk.Label(highcut_row, text="highcut:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, anchor="w").pack(
            side="left")
        highcut_entry = tk.Entry(highcut_row, textvariable=self.highcut_var, width=8, justify="center",
                                  bg=BUTTON_BG, fg=TEXT, insertbackground=TEXT, relief="flat", highlightthickness=1,
                                  highlightbackground=PANEL_BORDER, highlightcolor=ACCENT, font=FONT_SMALL)
        highcut_entry.pack(side="right")
        highcut_entry.bind("<Return>", self._apply_highcut_entry)
        highcut_entry.bind("<FocusOut>", self._apply_highcut_entry)

        alpha_frame = tk.Frame(parent, bg=PANEL_BG)
        alpha_frame.pack(fill="x", padx=14, pady=(0, 10))
        alpha_label = tk.Label(alpha_frame, bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, anchor="w")
        alpha_label.pack(fill="x")

        def _update_alpha_label(*_args: object) -> None:
            alpha_label.config(text=f"mask opacity: {self.mask_alpha.get()}%")

        self.mask_alpha.trace_add("write", _update_alpha_label)
        _update_alpha_label()
        alpha_slider = RoundSlider(alpha_frame, self.mask_alpha, 0, 100, width=SIDEBAR_W - 40, height=22,
                                    outer_bg=PANEL_BG)
        alpha_slider.pack(pady=(4, 0))
        # Re-render only once the slider is released, not on every drag step -
        # this drives a full main-canvas repaint, same reasoning as the manual
        # cuts dialog: continuous re-rendering during a drag is what feels slow.
        alpha_slider.bind("<ButtonRelease-1>", lambda e: self.render())

        RoundButton(parent, "export mask", command=self.export_mask, outer_bg=PANEL_BG, accent=True,
                    width=SIDEBAR_W - 28).pack(padx=14, pady=(4, 12))

        self._divider(parent)

        mode_frame = tk.Frame(parent, bg=PANEL_BG)
        mode_frame.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(mode_frame, text="mode", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w")
        SegmentedControl(
            mode_frame, [("ellipse", "Ellipse"), ("circle", "Circle"), ("line", "Satellite")], self.tool,
            outer_bg=PANEL_BG,
        ).pack(anchor="w", pady=(4, 0))

        self.tool_options_frame = tk.Frame(parent, bg=PANEL_BG)
        self.tool_options_frame.pack(fill="x")

        self.preview_canvas: Optional[tk.Canvas] = None

    def _divider(self, parent: tk.Frame) -> None:
        tk.Frame(parent, bg=PANEL_BORDER, height=1).pack(fill="x", padx=14)

    @staticmethod
    def _fmt_slider_value(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    def _build_slider_row(self, parent: tk.Frame, name: str, var: tk.Variable, lo: float, hi: float,
                           suffix: str = "") -> None:
        # Name (static, plus a unit hint) on the left; the value itself lives only
        # in the text box on the right - showing it in both the label and the box
        # would just be redundant. Slider goes on the row below.
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", padx=14, pady=(6, 2))

        unit_hint = f" ({suffix.strip()})" if suffix.strip() else ""
        tk.Label(row, text=f"{name}{unit_hint}:", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL, anchor="w").pack(
            side="left")

        # A text box for typing an exact value - handy for precise sizes the
        # slider itself is too coarse to hit reliably (e.g. a 0.5px radius to
        # mask a single pixel).
        entry_str = tk.StringVar(value=self._fmt_slider_value(var.get()))

        def sync_entry(*_args: object) -> None:
            if entry.winfo_exists():
                entry_str.set(self._fmt_slider_value(var.get()))

        entry_trace_id = var.trace_add("write", sync_entry)

        def apply_entry(_event: Optional[tk.Event] = None) -> None:
            try:
                value = float(entry_str.get())
            except ValueError:
                entry_str.set(self._fmt_slider_value(var.get()))
                return
            value = max(lo, min(value, hi))
            if isinstance(var, tk.IntVar):
                value = int(round(value))
            var.set(value)
            entry_str.set(self._fmt_slider_value(var.get()))

        entry = tk.Entry(row, textvariable=entry_str, width=6, justify="center", bg=BUTTON_BG, fg=TEXT,
                          insertbackground=TEXT, relief="flat", highlightthickness=1,
                          highlightbackground=PANEL_BORDER, highlightcolor=ACCENT, font=FONT_SMALL)
        entry.pack(side="right")
        entry.bind("<Return>", apply_entry)
        entry.bind("<FocusOut>", apply_entry)
        entry.bind("<Destroy>", lambda e: var.trace_remove("write", entry_trace_id), add="+")

        RoundSlider(parent, var, lo, hi, width=SIDEBAR_W - 40, height=22, outer_bg=PANEL_BG).pack(padx=14, pady=(0, 8))

    def _rebuild_tool_options(self) -> None:
        for child in list(self.tool_options_frame.winfo_children()):
            child.destroy()
        self._cancel_pending_line()

        if self.tool.get() == "ellipse":
            self._build_slider_row(self.tool_options_frame, "radius", self.radius, RADIUS_MIN, MAX_SHAPE_SIZE, suffix=" px")
            self._build_slider_row(self.tool_options_frame, "ellipticity", self.ellipticity, 0, 90, suffix="%")
            self._build_slider_row(self.tool_options_frame, "angle", self.angle, 0, 180, suffix="°")
        elif self.tool.get() == "circle":
            self._build_slider_row(self.tool_options_frame, "radius", self.radius, RADIUS_MIN, MAX_SHAPE_SIZE, suffix=" px")
        else:
            self._build_slider_row(self.tool_options_frame, "thickness", self.thickness, 1, MAX_SHAPE_SIZE, suffix=" px")
            style_frame = tk.Frame(self.tool_options_frame, bg=PANEL_BG)
            style_frame.pack(fill="x", padx=14, pady=(8, 4))
            tk.Label(style_frame, text="style", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w")
            SegmentedControl(style_frame, LINE_STYLES, self.line_style, outer_bg=PANEL_BG).pack(anchor="w", pady=(4, 0))

        self.preview_canvas = tk.Canvas(self.tool_options_frame, width=PREVIEW_W, height=PREVIEW_H,
                                         bg=PANEL_BG, highlightthickness=0)
        self.preview_canvas.pack(padx=14, pady=14)

        for var in (self.ellipticity, self.angle, self.radius, self.thickness):
            var.trace_add("write", lambda *_: self.render_tool_preview())
        self.render_tool_preview()

    def _on_tool_changed(self) -> None:
        self._rebuild_tool_options()
        self._update_shape_preview(-1000, -1000)

    def _cancel_pending_line(self) -> None:
        self._line_anchor = None
        if hasattr(self, "canvas"):
            self.canvas.delete("line_preview")

    def _current_round_params(self) -> tuple[float, float, float]:
        """(semi-major axis, semi-minor axis, angle) for the current ellipse/circle tool.

        Ellipse mode shares the same radius slider as circle mode for overall size
        (a); the ellipticity slider (0-90%) shrinks b = a * (1 - ellipticity/100),
        so ellipticity=0 is exactly a circle of that radius.
        """
        r = self.radius.get()
        if self.tool.get() == "circle":
            return r, r, 0
        b = r * (1 - self.ellipticity.get() / 100.0)
        return r, b, self.angle.get()

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
            messagebox.showerror("maskfits", f"Could not load {entry.path}:\n{exc}")
            entry.path = None

        if entry.image is not None and reset_view:
            self.reset_zoom()

        self.filename_label.config(text=os.path.basename(entry.path) if entry.path else "noname")
        self.counter_label.config(text=f"{self.index + 1}/{len(self.entries)}")
        self._update_cuts_display()
        self._update_smooth_button()
        self._update_bin_button()
        self.status.config(text=f"loaded {entry.path}" if entry.path else "new file")

        self.render()

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:.4g}"

    def _update_cuts_display(self) -> None:
        self.lowcut_var.set(self._fmt(self.entry.lowcut))
        self.highcut_var.set(self._fmt(self.entry.highcut))

    def _apply_lowcut_entry(self, _event: Optional[tk.Event] = None) -> None:
        entry = self.entry
        try:
            value = min(float(self.lowcut_var.get()), entry.highcut)
        except ValueError:
            self._update_cuts_display()
            return
        entry.lowcut = value
        self._update_cuts_display()
        self.render()

    def _apply_highcut_entry(self, _event: Optional[tk.Event] = None) -> None:
        entry = self.entry
        try:
            value = max(float(self.highcut_var.get()), entry.lowcut)
        except ValueError:
            self._update_cuts_display()
            return
        entry.highcut = value
        self._update_cuts_display()
        self.render()

    def set_stretch(self, stretch: str) -> None:
        self.stretch = stretch
        self.entry.apply_stretch(stretch)
        self._update_cuts_display()
        self.render()

    def reset_scale(self) -> None:
        self.scale_function.set("linear")

    def open_manual_cuts(self) -> None:
        if self.image is None:
            messagebox.showwarning("maskfits", "No image loaded.")
            return
        ManualCutsWindow(self.root, self.image.data, self.entry.lowcut, self.entry.highcut, self._apply_manual_cuts)

    def _apply_manual_cuts(self, lowcut: float, highcut: float) -> None:
        self.entry.lowcut = lowcut
        self.entry.highcut = highcut
        self._update_cuts_display()
        self.render()

    # ------------------------------------------------------------- navigation

    def _release_mask(self) -> None:
        """Drop the current entry's in-memory mask (and any smoothing/binning
        caches) before navigating away from it.

        Each mask - and each smoothing/binning cache array - is a full- (or
        near full-) resolution array; for a multi-image session with large
        FITS files, keeping every visited image's copies around adds up fast.
        Export first if you want to keep the mask - navigating back re-loads
        the image (fast, it stays cached) with a fresh empty mask and no
        smoothing/binning applied.
        """
        entry = self.entry
        image = entry.image
        if image is not None:
            if entry.original_data is not None:
                image.data = entry.original_data
            image.mask = np.zeros(image.data.shape, dtype=bool)
        entry.original_data = None
        entry.is_smoothed = False
        entry.smooth_sigma = None
        entry.smoothed_cache = None
        entry._smoothed_cache_sigma = None
        entry.is_binned = False
        entry.bin_factor = None
        entry.binned_cache = None
        entry._binned_cache_factor = None
        entry.smoothed_binned_cache = None
        entry._smoothed_binned_cache_sigma = None
        entry._smoothed_binned_cache_factor = None
        entry.mask_backup = None
        self._undo = None
        self._redo = None

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Open FITS files",
            filetypes=[("FITS files", "*.fits *.fit *.fts *.fits.gz"), ("All files", "*.*")],
        )
        if not paths:
            return
        self._release_mask()
        if len(self.entries) == 1 and self.entries[0].path is None:
            self.entries = []
        start = len(self.entries)
        self.entries.extend(Entry(p) for p in paths)
        self.index = start
        self.load_current(reset_view=True)

    def prev_image(self) -> None:
        if self.index > 0:
            self._release_mask()
            self.index -= 1
            self.load_current(reset_view=True)

    def next_image(self) -> None:
        if self.index < len(self.entries) - 1:
            self._release_mask()
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

    def reset_mask(self) -> None:
        if self.image is None:
            return
        self._push_undo()
        self.image.mask[:] = False
        self.render()
        self.status.config(text="mask cleared")

    def _ensure_original(self, entry: "Entry") -> None:
        """Capture entry's true, pristine data the first time smoothing or
        binning is ever used - every other cache is derived from this one
        array, which is never itself overwritten."""
        if entry.original_data is None:
            entry.original_data = entry.image.data

    def _smoothed_cache_for(self, entry: "Entry", sigma: float) -> np.ndarray:
        if entry.smoothed_cache is None or entry._smoothed_cache_sigma != sigma:
            entry.smoothed_cache = gaussian_smooth(entry.original_data, sigma)
            entry._smoothed_cache_sigma = sigma
        return entry.smoothed_cache

    def _binned_cache_for(self, entry: "Entry", factor: int) -> np.ndarray:
        if entry.binned_cache is None or entry._binned_cache_factor != factor:
            entry.binned_cache = bin_func(entry.original_data, factor)
            entry._binned_cache_factor = factor
        return entry.binned_cache

    def _smoothed_binned_cache_for(self, entry: "Entry", sigma: float, factor: int) -> np.ndarray:
        if (entry.smoothed_binned_cache is None
                or entry._smoothed_binned_cache_sigma != sigma
                or entry._smoothed_binned_cache_factor != factor):
            # Smooth first, then bin - low-pass filtering before downsampling
            # avoids the blockiness/aliasing that binning-then-smoothing a
            # much coarser grid would introduce. Reuses smoothed_cache when
            # it's already fresh at this sigma instead of reblurring.
            base = self._smoothed_cache_for(entry, sigma)
            entry.smoothed_binned_cache = bin_func(base, factor)
            entry._smoothed_binned_cache_sigma = sigma
            entry._smoothed_binned_cache_factor = factor
        return entry.smoothed_binned_cache

    def _current_display_data(self, entry: "Entry") -> np.ndarray:
        """The array for entry's current is_smoothed/is_binned combination -
        one of up to four cached arrays (original / smoothed / binned /
        smoothed-then-binned), computed and cached lazily."""
        if entry.is_smoothed and entry.is_binned:
            return self._smoothed_binned_cache_for(entry, entry.smooth_sigma, entry.bin_factor)
        if entry.is_smoothed:
            return self._smoothed_cache_for(entry, entry.smooth_sigma)
        if entry.is_binned:
            return self._binned_cache_for(entry, entry.bin_factor)
        return entry.original_data

    def _read_sigma(self) -> float:
        """Parse/clamp the sigma box, writing the cleaned-up value back to it."""
        try:
            sigma = float(self.smooth_sigma_var.get())
        except ValueError:
            entry = self.entry
            sigma = entry.smooth_sigma if entry.smooth_sigma is not None else 3.0
        sigma = max(sigma, 0.0)
        self.smooth_sigma_var.set(self._fmt(sigma))
        return sigma

    def _update_smooth_button(self) -> None:
        smoothed = self.image is not None and self.entry.is_smoothed
        self.smooth_button.set_text("unsmooth" if smoothed else "smooth")
        self.smooth_button.set_active(smoothed)

    def _toggle_smoothing(self, _event: Optional[tk.Event] = None) -> None:
        """Hotkey s / the smooth-unsmooth toolbar button.

        Composable with binning - toggling this only ever changes pixel
        values, never the array shape or the mask, so it can be freely
        combined with binning in either order. Whichever of the (up to four)
        cached arrays the current combination needs is computed once and
        reused until its sigma/factor actually changes.
        """
        entry = self.entry
        if entry.image is None:
            return
        if entry.is_smoothed:
            entry.is_smoothed = False
            entry.image.data = self._current_display_data(entry)
            self._update_smooth_button()
            self.render()
            self.status.config(text="smoothing removed")
            return
        sigma = self._read_sigma()
        if sigma <= 0:
            return
        self._ensure_original(entry)
        entry.smooth_sigma = sigma
        entry.is_smoothed = True
        entry.image.data = self._current_display_data(entry)
        self._update_smooth_button()
        self.render()
        self.status.config(text=f"smoothed (sigma={self._fmt(sigma)})")

    def _apply_sigma_entry(self, _event: Optional[tk.Event] = None) -> None:
        """Commit the sigma box. If the image is already smoothed, re-smooths
        from the untouched original at the new sigma (cached, so re-entering a
        previously-used sigma doesn't recompute) rather than compounding onto
        the already-blurred data; otherwise just formats/clamps the value for
        when "smooth" is next pressed."""
        entry = self.entry
        sigma = self._read_sigma()
        if entry.image is None or not entry.is_smoothed:
            return
        if sigma <= 0:
            entry.is_smoothed = False
            entry.image.data = self._current_display_data(entry)
            self._update_smooth_button()
            self.render()
            self.status.config(text="smoothing removed")
            return
        entry.smooth_sigma = sigma
        entry.image.data = self._current_display_data(entry)
        self.render()
        self.status.config(text=f"smoothed (sigma={self._fmt(sigma)})")

    # -------------------------------------------------------------- binning

    def _read_bin_factor(self) -> int:
        """Parse/clamp the bin-factor box, writing the cleaned-up value back to it."""
        try:
            factor = int(round(float(self.bin_factor_var.get())))
        except ValueError:
            entry = self.entry
            factor = entry.bin_factor if entry.bin_factor is not None else 3
        factor = max(factor, 1)
        if self.image is not None:
            factor = min(factor, max(min(self.image.data.shape), 1))
        self.bin_factor_var.set(str(factor))
        return factor

    def _update_bin_button(self) -> None:
        binned = self.image is not None and self.entry.is_binned
        self.bin_button.set_text("unbin" if binned else "bin")
        self.bin_button.set_active(binned)

    def _toggle_binning(self, _event: Optional[tk.Event] = None) -> None:
        """Hotkey b / the bin-unbin toolbar button.

        Composable with smoothing - see _toggle_smoothing/_current_display_data
        for the data side. The mask is additionally reshaped to match: going
        in, the current full-res mask is block-OR'd down to the binned grid so
        painting continues to work; coming back out, whatever's currently on
        the binned mask (including edits made while binned) is expanded back
        over the full-res mask. Export always uses the full-res mask
        regardless of which state this is currently in - see _full_res_mask /
        export_mask. Binning changes the array shape, so the view is
        recentered and reset to zoom 1 (matching the new size) every time it
        toggles or the factor changes.
        """
        entry = self.entry
        if entry.image is None:
            return
        if entry.is_binned:
            entry.image.mask = unbin_mask(entry.image.mask, entry.bin_factor, entry.mask_backup)
            entry.mask_backup = None
            entry.is_binned = False
            entry.image.data = self._current_display_data(entry)
            self._undo = None
            self._redo = None
            self._update_bin_button()
            self.reset_zoom()
            self.status.config(text="binning removed")
            return
        factor = self._read_bin_factor()
        if factor <= 1:
            return
        self._ensure_original(entry)
        entry.mask_backup = entry.image.mask
        entry.bin_factor = factor
        entry.is_binned = True
        entry.image.data = self._current_display_data(entry)
        entry.image.mask = bin_mask(entry.mask_backup, factor)
        self._undo = None
        self._redo = None
        self._update_bin_button()
        self.reset_zoom()
        self.status.config(text=f"binned {factor}x{factor}")

    def _apply_bin_entry(self, _event: Optional[tk.Event] = None) -> None:
        """Commit the bin-factor box. If the image is already binned, re-bins
        from the untouched original at the new factor (cached, so re-entering
        a previously-used factor doesn't recompute); otherwise just
        formats/clamps the value for when "bin" is next pressed.

        Changing the factor while binned first expands the current binned mask
        (which may hold edits made at the old factor) back over the full-res
        mask, then re-derives both data and mask fresh at the new factor - so
        edits are never lost and never compounded. The view is recentered and
        reset to zoom 1 since the array size just changed again.
        """
        entry = self.entry
        factor = self._read_bin_factor()
        if entry.image is None or not entry.is_binned:
            return
        if factor <= 1:
            entry.image.mask = unbin_mask(entry.image.mask, entry.bin_factor, entry.mask_backup)
            entry.mask_backup = None
            entry.is_binned = False
            entry.image.data = self._current_display_data(entry)
            self._undo = None
            self._redo = None
            self._update_bin_button()
            self.reset_zoom()
            self.status.config(text="binning removed")
            return
        full_res_mask = unbin_mask(entry.image.mask, entry.bin_factor, entry.mask_backup)
        entry.mask_backup = full_res_mask
        entry.bin_factor = factor
        entry.image.data = self._current_display_data(entry)
        entry.image.mask = bin_mask(full_res_mask, factor)
        self._undo = None
        self._redo = None
        self.reset_zoom()
        self.status.config(text=f"binned {factor}x{factor}")

    def _full_res_mask(self, entry: "Entry") -> np.ndarray:
        """entry.image.mask translated back to full (unbinned) resolution -
        what export_mask always writes, regardless of whether the display is
        currently binned for painting convenience."""
        if entry.is_binned:
            return unbin_mask(entry.image.mask, entry.bin_factor, entry.mask_backup)
        return entry.image.mask

    def _defocus_active_entry(self, _event: Optional[tk.Event] = None) -> None:
        if isinstance(self.root.focus_get(), tk.Entry):
            self.root.focus_set()

    def _on_global_click(self, event: tk.Event) -> None:
        """Release focus from whatever entry box was being edited as soon as
        the user clicks anywhere that isn't itself a text box."""
        if not isinstance(event.widget, tk.Entry):
            self._defocus_active_entry()

    def _adjust_shape_size(self, direction: int) -> None:
        """Hotkey e/w: grow/shrink the active tool's size (radius/thickness)."""
        step = 5
        tool = self.tool.get()
        if tool in ("circle", "ellipse"):
            var, lo, hi = self.radius, RADIUS_MIN, MAX_SHAPE_SIZE
        elif tool == "line":
            var, lo, hi = self.thickness, 1, MAX_SHAPE_SIZE
        else:
            return
        var.set(max(lo, min(var.get() + direction * step, hi)))
        if tool in ("ellipse", "circle") and self._cursor_img_pos is not None:
            cx, cy = self.img_to_canvas(*self._cursor_img_pos)
            self._update_shape_preview(cx, cy)

    # -------------------------------------------------------------- export

    @staticmethod
    def _mask_stem(path: str) -> str:
        stem = os.path.basename(path)
        if stem.endswith(".fits.gz"):
            return stem[: -len(".fits.gz")]
        return os.path.splitext(stem)[0]

    def _build_mask_hdu(self, entry: "Entry") -> "fits.PrimaryHDU":
        header = entry.image.header.copy()
        header["OBJECT"] = "MASK"
        # entry.image.mask is True where the user painted a mask, in WORKING (possibly
        # load-time-transposed, and possibly currently binned for painting
        # convenience) orientation. _full_res_mask undoes the binning (if any)
        # so the exported mask always matches the original file's resolution;
        # detranspose back to match the ORIGINAL file/header before writing.
        # Exported convention is inverted (0 = masked/excluded, 1 = kept),
        # matching typical good-pixel maps.
        full_res_mask = self._full_res_mask(entry)
        mask = full_res_mask.T if entry.image.rotated else full_res_mask
        exported = (~mask).astype("uint8")
        return fits.PrimaryHDU(data=exported, header=header)

    def export_mask(self) -> None:
        """Quick-save: the sidebar 'export mask' button. Always writes
        mask_<original filename> next to the source file, no prompt."""
        entry = self.entry
        if entry.image is None or entry.path is None:
            messagebox.showwarning("maskfits", "No image loaded to export a mask for.")
            return
        out_path = os.path.join(os.path.dirname(entry.path), f"mask_{self._mask_stem(entry.path)}.fits")
        self._build_mask_hdu(entry).writeto(out_path, overwrite=True)
        self.status.config(text=f"exported mask to {out_path}")

    def export_mask_as(self) -> None:
        """File > Save Mask As...: prompts for a location/name via a file
        browser, starting in the directory maskfits was launched from and
        pre-filled with the same mask_<original filename> default."""
        entry = self.entry
        if entry.image is None or entry.path is None:
            messagebox.showwarning("maskfits", "No image loaded to export a mask for.")
            return
        default_name = f"mask_{self._mask_stem(entry.path)}.fits"
        out_path = filedialog.asksaveasfilename(
            title="Save Mask As",
            initialdir=os.getcwd(),
            initialfile=default_name,
            defaultextension=".fits",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")],
        )
        if not out_path:
            return
        self._build_mask_hdu(entry).writeto(out_path, overwrite=True)
        self.status.config(text=f"exported mask to {out_path}")

    # --------------------------------------------------------------- undo

    def _push_undo(self) -> None:
        if self.image is None:
            return
        self._undo = (self.index, self.image.mask.copy())
        self._redo = None  # a fresh edit invalidates any pending redo

    def _on_undo(self, _event: Optional[tk.Event] = None) -> None:
        if self._undo is None:
            return
        idx, mask = self._undo
        if idx < len(self.entries) and self.entries[idx].image is not None:
            current = self.entries[idx].image.mask.copy()
            self.entries[idx].image.mask = mask
            self._redo = (idx, current)
            self._undo = None
            if idx == self.index:
                self.render()

    def _on_redo(self, _event: Optional[tk.Event] = None) -> None:
        if self._redo is None:
            return
        idx, mask = self._redo
        if idx < len(self.entries) and self.entries[idx].image is not None:
            current = self.entries[idx].image.mask.copy()
            self.entries[idx].image.mask = mask
            self._undo = (idx, current)
            self._redo = None
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
        self.render_tool_preview()
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
        crop_h, crop_w = y1 - y0, x1 - x0
        disp_w = max(int(round(crop_w * self.zoom)), 1)
        disp_h = max(int(round(crop_h * self.zoom)), 1)

        # When zoomed out, downsample toward display resolution BEFORE the
        # per-pixel stretch/colormap work, instead of computing it at full
        # source resolution and throwing most of it away in the final resize.
        # For a huge image at fit-to-window zoom this is the difference between
        # processing tens of millions of pixels and a couple million - the
        # actual source of the sluggishness, not rotation-related overhead.
        step_x = max(crop_w // max(disp_w, 1), 1)
        step_y = max(crop_h // max(disp_h, 1), 1)
        crop = data[y0:y1:step_y, x0:x1:step_x]
        mask_crop = image.mask[y0:y1:step_y, x0:x1:step_x]

        span = max(entry.highcut - entry.lowcut, 1e-12)
        norm = np.clip((crop - entry.lowcut) / span, 0, 1)
        norm = np.nan_to_num(norm, nan=0.0)
        rgb = self._scale_and_color(norm)

        self._tint_masked(rgb, mask_crop)

        pil_img = Image.fromarray(rgb, mode="RGB")
        resample = Image.NEAREST if self.zoom >= 1 else Image.BOX
        pil_img = pil_img.resize((disp_w, disp_h), resample)

        cx0, cy0 = self.img_to_canvas(x0, y0)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(cx0, cy0, image=self._photo, anchor="nw", tags="img")
        self.canvas.tag_lower("img")

    def _active_lut(self) -> np.ndarray:
        """The current colormap's LUT, reversed if Invert Colormap is on."""
        lut = COLORMAP_LUTS[self.colormap.get()]
        return lut[::-1] if self.invert_colormap.get() else lut

    def _cycle_colormap(self, _event: Optional[tk.Event] = None) -> None:
        idx = COLORMAP_NAMES.index(self.colormap.get())
        self.colormap.set(COLORMAP_NAMES[(idx + 1) % len(COLORMAP_NAMES)])

    def _scale_and_color(self, norm: np.ndarray) -> np.ndarray:
        """Map cut-normalized [0, 1] values (no NaN) to a uint8 RGB array via the
        current scale function (stretch curve) and colormap."""
        stretch = STRETCHES[self.scale_function.get()]
        stretched = np.clip(np.asarray(stretch(norm)), 0.0, 1.0)
        gray = (stretched * 255).astype(np.uint8)
        return self._active_lut()[gray]

    def _mask_tint_hex(self) -> str:
        """The current (possibly inverted) colormap's mask-tint color as a
        Tk-friendly hex string, for canvas outlines/lines that should visually
        match the actual mask."""
        return "#%02x%02x%02x" % mask_tint_for(self.colormap.get(), self._active_lut())

    def _tint_masked(self, rgb: np.ndarray, mask_crop: np.ndarray) -> None:
        """Blend the current colormap's complementary tint into masked pixels, in place."""
        if not mask_crop.any():
            return
        alpha = self.mask_alpha.get() / 100.0
        tint = mask_tint_for(self.colormap.get(), self._active_lut())
        for ch, tint_v in enumerate(tint):
            channel = rgb[..., ch].astype(np.float32)
            blended = channel * (1 - alpha) + tint_v * alpha
            rgb[..., ch] = np.where(mask_crop, blended, channel).astype(np.uint8)

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
        norm = np.where(np.isnan(crop), 0.12, norm)
        rgb = self._scale_and_color(norm)

        self._tint_masked(rgb, mask_crop)

        square = min(w, h)
        block = max(square // MAG_SIZE, 1)
        disp = block * MAG_SIZE
        pil_img = Image.fromarray(rgb, mode="RGB").resize((disp, disp), Image.NEAREST)
        self._mag_photo = ImageTk.PhotoImage(pil_img)
        ox, oy = (w - disp) // 2, (h - disp) // 2
        canvas.create_image(ox, oy, image=self._mag_photo, anchor="nw")

        cxp, cyp = ox + half * block, oy + half * block
        canvas.create_rectangle(cxp, cyp, cxp + block, cyp + block, outline=GREEN, width=2)

    @staticmethod
    def _draw_grid(canvas: tk.Canvas, w: int, h: int, step: int = 10) -> None:
        for x in range(0, w, step):
            canvas.create_line(x, 0, x, h, fill=PANEL_BORDER)
        for y in range(0, h, step):
            canvas.create_line(0, y, w, y, fill=PANEL_BORDER)

    def render_tool_preview(self) -> None:
        """Draw the shape at the same on-screen scale it currently has on the main canvas.

        Uses self.zoom (fit_zoom * zoom_mult), not a fixed 1:1 pixel scale - a radius
        of 100 should look exactly as big here as it does when you hover the image,
        so this panel is a true stand-in for "move the mouse onto the image and look".
        Deliberately not auto-fit-to-box either: it should really grow with the
        slider and clip against the panel edges once it outgrows it.
        """
        if self.preview_canvas is None or not self.preview_canvas.winfo_exists():
            return
        self.preview_canvas.delete("all")
        w, h = PREVIEW_W, PREVIEW_H
        zoom = self.zoom
        if self.tool.get() in ("ellipse", "circle"):
            a, b, angle = self._current_round_params()
            pts = ellipse_polygon_points(w / 2, h / 2, a * zoom, b * zoom, angle)
            self.preview_canvas.create_polygon(pts, fill=ACCENT, outline=ACCENT, smooth=True)
        else:
            thickness = max(self.thickness.get() * zoom, 1)
            self.preview_canvas.create_line(
                0, h / 2, w, h / 2,
                fill=ACCENT, width=thickness, capstyle=tk.ROUND, joinstyle=tk.ROUND,
            )

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
                # image.wcs describes the ORIGINAL (untransposed, unbinned) file;
                # ix/iy are in working (possibly load-time-transposed, and
                # possibly currently-binned) space, so swap them back and scale
                # back up to original-file pixels before the WCS lookup.
                wcs_ix, wcs_iy = (iy, ix) if image.rotated else (ix, iy)
                if self.entry.is_binned:
                    factor = self.entry.bin_factor
                    wcs_ix, wcs_iy = wcs_ix * factor, wcs_iy * factor
                sky = image.wcs.pixel_to_world(wcs_ix, wcs_iy)
                self.readout["RA"].config(text=sky.ra.to_string(unit="hourangle", sep=":", precision=2))
                self.readout["DEC"].config(text=sky.dec.to_string(sep=":", precision=1, alwayssign=True))
            except Exception:
                self.readout["RA"].config(text="")
                self.readout["DEC"].config(text="")
        else:
            self.readout["RA"].config(text="")
            self.readout["DEC"].config(text="")

        if self.tool.get() in ("ellipse", "circle"):
            self._update_shape_preview(event.x, event.y)
        elif self._line_anchor is not None:
            self._update_click_line_preview(event.x, event.y)

    def _update_shape_preview(self, cx: float, cy: float) -> None:
        self.canvas.delete("shape_preview")
        self._preview_photo = None
        if self.tool.get() not in ("ellipse", "circle") or self.image is None:
            return
        a, b, angle = self._current_round_params()
        disp_a, disp_b = a * self.zoom, b * self.zoom
        pts = ellipse_polygon_points(cx, cy, disp_a, disp_b, angle)

        # A translucent fill shows the actual area the shape covers, not just
        # its edge. Tkinter canvas fills have no real alpha, so this is drawn
        # as a small RGBA raster instead; a crisp polygon outline on top keeps
        # the boundary precise. Fixed opacity, independent of the mask-opacity
        # slider (that slider controls how painted mask pixels blend into the
        # image once applied, not this hover preview).
        tint = mask_tint_for(self.colormap.get(), self._active_lut())
        alpha = round(0.5 * 255)
        half = max(int(np.ceil(max(disp_a, disp_b))) + 2, 1)
        box = 2 * half
        local_mask = ellipse_mask((box, box), half, half, disp_a, disp_b, angle)
        rgba = np.zeros((box, box, 4), dtype=np.uint8)
        rgba[..., 0] = tint[0]
        rgba[..., 1] = tint[1]
        rgba[..., 2] = tint[2]
        rgba[..., 3] = np.where(local_mask, alpha, 0).astype(np.uint8)
        self._preview_photo = ImageTk.PhotoImage(Image.fromarray(rgba, mode="RGBA"))
        self.canvas.create_image(cx - half, cy - half, image=self._preview_photo, anchor="nw", tags="shape_preview")
        tint_hex = "#%02x%02x%02x" % tint
        self.canvas.create_polygon(pts, outline=tint_hex, fill="", width=2, tags="shape_preview")

    def _update_click_line_preview(self, cx: float, cy: float) -> None:
        self.canvas.delete("line_preview")
        if self.image is None or self._line_anchor is None:
            return
        x0, y0 = self._line_anchor
        x1, y1 = self.canvas_to_img(cx, cy)
        ex0, ey0, ex1, ey1 = self._extend_for_style(x0, y0, x1, y1)
        sx0, sy0 = self.img_to_canvas(ex0, ey0)
        sx1, sy1 = self.img_to_canvas(ex1, ey1)
        self.canvas.create_line(sx0, sy0, sx1, sy1, fill=self._mask_tint_hex(), width=2, dash=(4, 3), tags="line_preview")
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
        if tool in ("ellipse", "circle"):
            self._push_undo()
            self._stamp_round(event.x, event.y, erase)
        elif tool == "line":
            if self._line_anchor is None:
                self._push_undo()
                self._line_anchor = self.canvas_to_img(event.x, event.y)
                self._line_anchor_erase = erase
            else:
                self._finalize_click_line(event.x, event.y)

    def _on_drag(self, event: tk.Event, erase: bool) -> None:
        tool = self.tool.get()
        if tool in ("ellipse", "circle") and self.image is not None:
            self._stamp_round(event.x, event.y, erase)
            self._update_shape_preview(event.x, event.y)


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

    def _stamp_round(self, cx: float, cy: float, erase: bool) -> None:
        image = self.image
        if image is None:
            return
        ix, iy = self.canvas_to_img(cx, cy)
        a, b, angle = self._current_round_params()
        stamp = ellipse_mask(image.data.shape, ix, iy, a, b, angle)
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


MODE_FLAGS = {"s": "line", "e": "ellipse", "c": "circle"}


def run_gui(paths: list[str], zoom: Optional[float] = None, mode: Optional[str] = None) -> int:
    root = tk.Tk()
    app = MaskFitsApp(root, paths)
    if mode is not None:
        app.tool.set(MODE_FLAGS[mode])
    if zoom is not None:
        app.zoom_mult = max(min(zoom, ZOOM_MULT_MAX), ZOOM_MULT_MIN)
        app._update_zoom_label()
        app.render()
    root.mainloop()
    return 0
