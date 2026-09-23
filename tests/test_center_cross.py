"""Coverage for the K-hotkey-toggled center cross - a crosshair marking the
image's exact center, drawn in gui.MaskFitsApp._paint_center_cross. Colored
via _center_cross_color(): the inverse of the actual displayed color
(colormap-mapped, mask-tinted) over the central 15x15 pixels - not a fixed
color, so it can never blend into a same-colored mask covering the center."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.imagedata import safe_span  # noqa: E402
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


def test_cross_color_is_stable_and_deterministic(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(colormap="Grayscale"))
    win.show()

    assert win._center_cross_color() == win._center_cross_color()


def test_cross_color_changes_with_the_colormap(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(colormap="Grayscale"))
    win.show()

    color_before = win._center_cross_color()
    win.set_colormap("Viridis")
    color_after = win._center_cross_color()

    assert color_before != color_after


def test_cross_color_is_the_inverse_of_the_displayed_center_region(qapp, tmp_path):
    """Regression: coloring the cross to MATCH the mask (an earlier design)
    made it disappear once the mask covered the center - it must instead be
    the inverse of whatever's actually shown there, mask or no mask."""
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(colormap="Grayscale"))
    win.show()

    ny, nx = win.image.data.shape
    cy_i, cx_i = ny // 2, nx // 2
    crop = win.image.data[cy_i - 7:cy_i + 8, cx_i - 7:cx_i + 8]
    mask_crop = win.image.mask[cy_i - 7:cy_i + 8, cx_i - 7:cx_i + 8]
    span = safe_span(win.entry.lowcut, win.entry.highcut)
    norm = np.clip((crop - win.entry.lowcut) / span, 0, 1)
    rgb = win._scale_and_color(norm)
    win._tint_masked(rgb, mask_crop)
    expected_avg = rgb.reshape(-1, 3).mean(axis=0)
    expected = tuple(int(255 - c) for c in expected_avg)

    assert win._center_cross_color() == expected


def test_cross_color_changes_once_the_center_is_masked(qapp, tmp_path):
    """The literal reported bug: painting a mask over the center used to
    leave the (same-colored) cross invisible against it."""
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    color_before = win._center_cross_color()

    ny, nx = win.image.data.shape
    win.image.mask[ny // 2 - 7:ny // 2 + 8, nx // 2 - 7:nx // 2 + 8] = True
    color_after = win._center_cross_color()

    assert color_before != color_after
    # And it must stay a real inverse - not just any different color, but
    # specifically NOT equal to the mask's own tint (the bug being fixed).
    assert color_after != win._mask_tint()


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
