"""Coverage for per-colormap mask colors: Settings.mask_colors, the
"Edit Mask Colors..." window (MaskColorsWindow) and the main window's
mask tint picking them up."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.colormaps import COLORMAP_NAMES, mask_tint_for  # noqa: E402
from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.mask_colors_window import MaskColorsWindow, auto_mask_color, colormap_lut  # noqa: E402
from maskfits.settings import Settings, load_settings, sanitize_mask_colors, save_settings  # noqa: E402
from maskfits.settings_window import SettingsWindow  # noqa: E402
from maskfits.theme import current_theme, rgb_to_hex  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


# ------------------------------------------------------------- persistence


def test_defaults_to_all_automatic():
    assert Settings().mask_colors == {}


def test_round_trips_through_the_settings_store(store):
    save_settings(Settings(mask_colors={"Viridis": "#ff2266", "Inferno": "#00ff00"}), store)
    assert load_settings(store).mask_colors == {"Viridis": "#ff2266", "Inferno": "#00ff00"}


def test_sanitize_drops_unknown_colormaps_and_bad_colors():
    raw = {"Viridis": "#ff2266", "NoSuchMap": "#000000", "Inferno": "red", "Grayscale": 5}
    assert sanitize_mask_colors(raw) == {"Viridis": "#ff2266"}
    assert sanitize_mask_colors(["not", "a", "dict"]) == {}


def test_corrupt_stored_value_loads_as_empty(store):
    store.setValue("mask_colors_json", "{not json")
    assert load_settings(store).mask_colors == {}


# ----------------------------------------------------------- main window


def test_custom_color_is_used_for_its_colormap_only(qapp):
    win = MaskFitsApp([], settings=Settings(mask_colors={"Viridis": "#ff2266"}))
    win.set_colormap("Viridis")
    assert win._mask_tint() == (255, 34, 102)

    win.set_colormap("Inferno")
    assert win._mask_tint() == mask_tint_for("Inferno", win._active_lut(), current_theme().accent)


def test_custom_color_ignores_colormap_inversion(qapp):
    win = MaskFitsApp([], settings=Settings(mask_colors={"Viridis": "#ff2266"}))
    win.set_colormap("Viridis")
    win.invert_colormap = True
    assert win._mask_tint() == (255, 34, 102)


def test_saving_settings_applies_new_mask_colors_live(qapp):
    win = MaskFitsApp([], settings=Settings())
    win.set_colormap("Viridis")
    win._apply_settings_live(Settings(mask_colors={"Viridis": "#123456"}))
    assert win._mask_tint() == (0x12, 0x34, 0x56)
    assert win.colormap == "Viridis"


# ------------------------------------------------------------------ window


def test_window_has_a_row_per_colormap(qapp):
    window = MaskColorsWindow({})
    assert list(window.pickers) == COLORMAP_NAMES
    assert list(window.previews) == COLORMAP_NAMES


def test_preview_shows_the_automatic_color_when_unset(qapp):
    window = MaskColorsWindow({})
    for name in COLORMAP_NAMES:
        assert window.previews[name].mask_rgb == auto_mask_color(name)
        assert window.pickers[name]._empty_color == rgb_to_hex(auto_mask_color(name))


def test_preview_follows_the_picker(qapp):
    window = MaskColorsWindow({"Viridis": "#ff2266"})
    assert window.pickers["Viridis"].value() == "#ff2266"
    assert window.previews["Viridis"].mask_rgb == (255, 34, 102)

    window.pickers["Viridis"].setValue("#00ff00")
    assert window.previews["Viridis"].mask_rgb == (0, 255, 0)

    window.pickers["Viridis"].setValue(None)  # Reset
    assert window.previews["Viridis"].mask_rgb == auto_mask_color("Viridis")
    assert "Viridis" not in window.mask_colors


def test_save_emits_only_the_custom_colors(qapp):
    window = MaskColorsWindow({})
    window.pickers["Inferno"].setValue("#abcdef")
    received = []
    window.mask_colors_saved.connect(received.append)
    window._save()
    assert received == [{"Inferno": "#abcdef"}]


def test_isopy_preview_uses_a_default_calibration(qapp):
    lut = colormap_lut("IsoPy")
    assert lut.shape == (256, 3)


def test_settings_window_keeps_the_saved_colors(qapp):
    dialog = SettingsWindow(Settings(mask_colors={"Viridis": "#ff2266"}))
    assert dialog._current_settings().mask_colors == {"Viridis": "#ff2266"}

    dialog._edit_mask_colors()
    editor = dialog._mask_colors_window
    assert editor is not None
    editor.pickers["Grayscale"].setValue("#010203")
    editor._save()

    assert dialog._current_settings().mask_colors == {"Viridis": "#ff2266", "Grayscale": "#010203"}
