"""Coverage for the K-hotkey-toggled center cross - a crosshair marking the
image's exact center, drawn in gui.MaskFitsApp._paint_center_cross as four
gapped arms with a contrasting outline. Colored via _center_cross_color():
Settings.cross_color if set, else black or white, whichever contrasts with
what is displayed over the central 15x15 pixels."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.shortcuts import SHORTCUT_ACTIONS_BY_ID  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_fits(path, size=64):
    rng = np.random.default_rng(0)
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)).writeto(path, overwrite=True)


def _paint(win):
    """Actually exercises _paint_overlays via a real QPainter, rather than
    just calling the pixel-agnostic helper directly, so a broken paintEvent
    wiring would fail these tests too."""
    img = QImage(win.canvas_w or 200, win.canvas_h or 200, QImage.Format.Format_RGB32)
    painter = QPainter(img)
    win._paint_overlays(painter)
    painter.end()


def test_defaults_to_off(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    assert win.show_center_cross is False


def test_toggle_flips_the_flag(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    win._toggle_center_cross()
    assert win.show_center_cross is True

    win._toggle_center_cross()
    assert win.show_center_cross is False


def test_k_hotkey_is_bound_to_toggle_center_cross():
    assert SHORTCUT_ACTIONS_BY_ID["toggle_center_cross"].defaults == ("K",)


def test_k_qshortcut_is_wired_up(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    keys = [sc.key().toString() for sc in win._shortcuts]
    assert "K" in keys


def test_paint_overlays_does_not_crash_when_off(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    win.canvas_w, win.canvas_h = 200, 200

    _paint(win)  # just must not raise


def test_paint_overlays_does_not_crash_when_on(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    win.canvas_w, win.canvas_h = 200, 200
    win.show_center_cross = True

    _paint(win)


def _window(tmp_path, value, **settings_kwargs):
    """A flat image whose center displays at the low (value=0) or high
    (value=1) end of a fixed 0..1 cut range."""
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.full((64, 64), value, dtype=np.float32)).writeto(path, overwrite=True)
    win = MaskFitsApp([str(path)], settings=Settings(colormap="Grayscale", **settings_kwargs))
    win.show()
    win.entry.lowcut, win.entry.highcut = 0.0, 1.0
    return win


def test_auto_color_is_white_on_a_dark_center(qapp, tmp_path):
    assert _window(tmp_path, 0.0)._center_cross_color() == (255, 255, 255)


def test_auto_color_is_black_on_a_bright_center(qapp, tmp_path):
    assert _window(tmp_path, 1.0)._center_cross_color() == (0, 0, 0)


def test_auto_color_is_never_a_mid_gray(qapp, tmp_path):
    """Regression: the old "inverse of what's underneath" rule turned
    mid-gray sky into an equally mid-gray, near-invisible cross."""
    assert _window(tmp_path, 0.5)._center_cross_color() in ((0, 0, 0), (255, 255, 255))


def test_auto_color_follows_a_mask_painted_over_the_center(qapp, tmp_path):
    """The auto color samples what is actually displayed, mask tint
    included - so it flips when the mask changes the center's brightness."""
    win = _window(tmp_path, 0.0)
    assert win._center_cross_color() == (255, 255, 255)
    win.image.mask[:, :] = True
    tint = win._mask_tint()
    expected = (0, 0, 0) if win._luminance(tint) >= 128 else (255, 255, 255)
    assert win._center_cross_color() == expected


def test_custom_color_setting_overrides_auto(qapp, tmp_path):
    for value in (0.0, 1.0):
        assert _window(tmp_path, value, cross_color="#ff3030")._center_cross_color() == (255, 48, 48)


def _painted_colors(win):
    win.canvas_w, win.canvas_h = 200, 200
    win.view_cx, win.view_cy = 32.0, 32.0
    win.show_center_cross = True
    img = QImage(200, 200, QImage.Format.Format_RGB32)
    img.fill(0x808080)
    painter = QPainter(img)
    win._paint_overlays(painter)
    painter.end()
    return img


def test_cross_has_a_gap_at_the_center_and_an_outline(qapp, tmp_path):
    win = _window(tmp_path, 0.0)
    img = _painted_colors(win)
    cx, cy = (round(v) for v in win.img_to_canvas(32.0, 32.0))

    assert img.pixelColor(cx, cy).getRgb()[:3] == (128, 128, 128)  # gap: untouched
    on_arm = img.pixelColor(cx + 12, cy).getRgb()[:3]
    assert min(on_arm) > 200  # white line
    beside_arm = img.pixelColor(cx + 12, cy - 2).getRgb()[:3]
    assert max(beside_arm) < 80  # black outline next to it


def test_custom_color_gets_a_contrasting_outline(qapp, tmp_path):
    win = _window(tmp_path, 0.0, cross_color="#101010")  # dark line -> white outline
    img = _painted_colors(win)
    cx, cy = (round(v) for v in win.img_to_canvas(32.0, 32.0))
    assert max(img.pixelColor(cx + 12, cy).getRgb()[:3]) < 80
    assert min(img.pixelColor(cx + 12, cy - 2).getRgb()[:3]) > 200


def test_cross_color_handles_a_tiny_image_smaller_than_the_patch(qapp, tmp_path):
    """The 15x15 sample region must clamp to the actual data bounds for an
    image smaller than that, not index out of range."""
    path = tmp_path / "tiny.fits"
    rng = np.random.default_rng(0)
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(6, 6)).astype(np.float32)).writeto(path, overwrite=True)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    color = win._center_cross_color()  # must not raise
    assert all(0 <= c <= 255 for c in color)


def test_no_crash_without_a_loaded_image(qapp):
    win = MaskFitsApp([], settings=Settings())
    win.show()
    win.canvas_w, win.canvas_h = 200, 200
    win.show_center_cross = True

    _paint(win)  # image is None - _paint_overlays must bail out cleanly
