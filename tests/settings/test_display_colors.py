"""Coverage for the two optional display colors in Settings: blank_color (a
flat color painted over pixels whose value is exactly 0 or NaN) and
cross_color (the center cross; see tests/gui/test_center_cross.py for how
it is drawn)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MAG_SIZE, MaskFitsApp  # noqa: E402
from maskfits.settings import Settings, is_hex_color, load_settings, save_settings  # noqa: E402
from maskfits.settings_window import SettingsWindow  # noqa: E402

BLUE = "#3355ff"
BLUE_RGB = [0x33, 0x55, 0xFF]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def _window(tmp_path, blank_color):
    data = np.full((40, 60), 100.0, dtype=np.float32)
    data[0, 0] = 0.0
    data[1, 0] = np.nan
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(data).writeto(path, overwrite=True)
    return MaskFitsApp([str(path)], settings=Settings(blank_color=blank_color))


# ------------------------------------------------------------- persistence


def test_defaults_to_off():
    assert Settings().blank_color is None


def test_round_trips_through_the_settings_store(store):
    save_settings(Settings(blank_color=BLUE), store)
    assert load_settings(store).blank_color == BLUE


def test_off_round_trips_as_none(store):
    save_settings(Settings(blank_color=None), store)
    assert load_settings(store).blank_color is None


@pytest.mark.parametrize("bad", ["blue", "#12345", "#gggggg", "3355ff"])
def test_an_invalid_stored_value_loads_as_off(store, bad):
    store.setValue("blank_color", bad)
    assert load_settings(store).blank_color is None


@pytest.mark.parametrize("value, ok", [("#3355ff", True), ("#ABCDEF", True), ("", False), (None, False), ("#fff", False)])
def test_is_hex_color(value, ok):
    assert is_hex_color(value) is ok


# --------------------------------------------------------------- rendering


def test_paints_exact_zero_and_nan_but_nothing_else(qapp, tmp_path):
    win = _window(tmp_path, BLUE)
    values = np.array([[0.0, np.nan, 1e-30, -0.0, 5.0]])
    rgb = np.full((1, 5, 3), 7, dtype=np.uint8)

    win._paint_blank(rgb, values)

    assert rgb[0, 0].tolist() == BLUE_RGB
    assert rgb[0, 1].tolist() == BLUE_RGB
    assert rgb[0, 2].tolist() == [7, 7, 7]  # tiny but non-zero
    assert rgb[0, 3].tolist() == BLUE_RGB  # -0.0 is zero
    assert rgb[0, 4].tolist() == [7, 7, 7]


def test_is_a_no_op_when_off(qapp, tmp_path):
    win = _window(tmp_path, None)
    rgb = np.full((1, 2, 3), 7, dtype=np.uint8)
    win._paint_blank(rgb, np.array([[0.0, np.nan]]))
    assert (rgb == 7).all()


def test_within_restricts_what_gets_painted(qapp, tmp_path):
    win = _window(tmp_path, BLUE)
    rgb = np.full((1, 2, 3), 7, dtype=np.uint8)
    win._paint_blank(rgb, np.array([[np.nan, np.nan]]), within=np.array([[True, False]]))
    assert rgb[0, 0].tolist() == BLUE_RGB
    assert rgb[0, 1].tolist() == [7, 7, 7]


def _canvas_colors(win):
    """Every distinct color the main canvas render produced."""
    captured = {}
    real = win._paint_blank

    def spy(rgb, values, within=None):
        real(rgb, values, within)
        captured.setdefault("calls", []).append((rgb.copy(), within))

    win._paint_blank = spy
    return captured


def test_main_canvas_render_uses_the_blank_color(qapp, tmp_path):
    win = _window(tmp_path, BLUE)
    win.show()
    win.canvas.resize(600, 400)
    win.canvas_w, win.canvas_h = 600, 400
    win.fit_zoom, win.zoom_mult = 10.0, 1.0
    win.view_cx, win.view_cy = 30.0, 20.0
    captured = _canvas_colors(win)

    win.render()

    rgb, _within = captured["calls"][-1]
    assert rgb[0, 0].tolist() == BLUE_RGB  # the zero pixel
    assert rgb[1, 0].tolist() == BLUE_RGB  # the NaN pixel
    assert rgb[2, 0].tolist() != BLUE_RGB
    assert (rgb.reshape(-1, 3) == BLUE_RGB).all(axis=1).sum() == 2


def test_magnifier_does_not_paint_the_padding_past_the_image_edge(qapp, tmp_path):
    win = _window(tmp_path, BLUE)
    win.show()
    win._cursor_img_pos = (0.5, 0.5)  # corner: most of the crop is off-image
    captured = _canvas_colors(win)

    win.magnifier.grab()  # forces a real paintEvent

    rgb, within = captured["calls"][-1]
    assert within is not None and within.shape == (MAG_SIZE, MAG_SIZE)
    blue = (rgb.reshape(-1, 3) == BLUE_RGB).all(axis=1).sum()
    assert blue == 2  # only the real zero + NaN pixels, none of the padding


def test_mask_tint_still_shows_on_top(qapp, tmp_path):
    win = _window(tmp_path, BLUE)
    rgb = np.full((1, 1, 3), 7, dtype=np.uint8)
    win._paint_blank(rgb, np.array([[0.0]]))
    win._tint_masked(rgb, np.array([[True]]))
    assert rgb[0, 0].tolist() != BLUE_RGB


# ---------------------------------------------------------- settings window


def test_settings_window_row_edits_and_resets_the_value(qapp):
    dialog = SettingsWindow(Settings(blank_color=BLUE))
    assert dialog._blank_picker.value() == BLUE
    assert dialog._current_settings().blank_color == BLUE

    dialog._blank_picker.setValue("#00ff00")
    assert dialog._current_settings().blank_color == "#00ff00"

    dialog._blank_picker.setValue(None)
    assert dialog._current_settings().blank_color is None


def test_saving_applies_it_to_the_running_window(qapp, tmp_path):
    win = _window(tmp_path, None)
    win._apply_settings_live(Settings(blank_color=BLUE))
    rgb = np.full((1, 1, 3), 7, dtype=np.uint8)
    win._paint_blank(rgb, np.array([[0.0]]))
    assert rgb[0, 0].tolist() == BLUE_RGB


# -------------------------------------------------------- center cross color


def test_cross_color_defaults_to_auto():
    assert Settings().cross_color is None


def test_cross_color_round_trips_through_the_settings_store(store):
    save_settings(Settings(cross_color="#ff3030"), store)
    assert load_settings(store).cross_color == "#ff3030"


def test_an_invalid_stored_cross_color_loads_as_auto(store):
    store.setValue("cross_color", "red")
    assert load_settings(store).cross_color is None


def test_settings_window_cross_color_row(qapp):
    dialog = SettingsWindow(Settings(cross_color="#ff3030"))
    assert dialog._cross_picker.value() == "#ff3030"
    dialog._cross_picker.setValue("#00ff00")
    assert dialog._current_settings().cross_color == "#00ff00"
    dialog._cross_picker.setValue(None)
    assert dialog._current_settings().cross_color is None
    # The two optional color rows are independent of each other.
    dialog._blank_picker.setValue(BLUE)
    assert dialog._current_settings().blank_color == BLUE
    assert dialog._current_settings().cross_color is None


# ------------------------------------------- Save keeps the session's choices


def test_saving_an_unrelated_setting_keeps_the_session_choices(qapp, tmp_path):
    """Regression: Save used to re-apply every default, so changing only
    the zero/NaN color reset a colormap picked from the menu."""
    win = _window(tmp_path, None)
    win.set_colormap("Viridis")
    win.set_scale_function("log")
    win.set_stretch("minmax")
    win.set_tool("line")
    win.zoom_mult = 2.5
    before = (win.colormap, win.scale_function, win.stretch, win.tool, win.zoom_mult)

    win._apply_settings_live(Settings(blank_color=BLUE))

    assert (win.colormap, win.scale_function, win.stretch, win.tool, win.zoom_mult) == before
    assert win.settings.blank_color == BLUE


def test_saving_a_changed_default_still_applies_it(qapp, tmp_path):
    win = _window(tmp_path, None)
    win.set_scale_function("log")

    win._apply_settings_live(Settings(colormap="Viridis"))

    assert win.colormap == "Viridis"
    assert win.scale_function == "log"  # untouched default, session choice kept
