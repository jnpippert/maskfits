"""Regression: the ellipse hover preview (main canvas and magnifier) must
show the same orientation a click actually masks. The mask rotates in image
coordinates (y up); the previews are drawn in screen coordinates (y down)
and used the same angle, so they showed the mirror image."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits import gui  # noqa: E402
from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.masking import ellipse_mask  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, tmp_path):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.zeros((200, 200), dtype=np.float32)).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings(mode="ellipse"))
    win.show()
    win.canvas.resize(400, 400)
    win.canvas_w, win.canvas_h = 400, 400
    win.view_cx, win.view_cy = 100.0, 100.0
    win.fit_zoom, win.zoom_mult = 2.0, 1.0
    win.radius, win.ellipticity, win.angle = 30.0, 60, 30
    win._on_motion(*win.img_to_canvas(100.0, 100.0))
    return win


def _capture_polygon(monkeypatch):
    captured = {}
    real = gui.ellipse_polygon_points

    def spy(*args, **kwargs):
        pts = real(*args, **kwargs)
        captured["pts"] = np.asarray(pts).reshape(-1, 2)
        return pts

    monkeypatch.setattr(gui, "ellipse_polygon_points", spy)
    return captured


def test_canvas_preview_outline_lies_on_the_masked_ellipse(win, monkeypatch):
    captured = _capture_polygon(monkeypatch)
    img = QImage(400, 400, QImage.Format.Format_RGB32)
    painter = QPainter(img)
    win._paint_overlays(painter)
    painter.end()

    a, b, angle = win._current_round_params()
    mask = ellipse_mask(win.image.data.shape, 100.0, 100.0, a, b, angle)
    theta = np.deg2rad(angle)
    for cx, cy in captured["pts"]:
        ix, iy = win.canvas_to_img(cx, cy)
        # Just inside the outline must be masked, just outside must not.
        for scale, expected in ((0.9, True), (1.1, False)):
            px, py = 100.0 + (ix - 100.0) * scale, 100.0 + (iy - 100.0) * scale
            assert mask[int(py), int(px)] == expected
    # And the same ellipse equation the mask itself uses holds on the outline.
    ix, iy = np.array([win.canvas_to_img(x, y) for x, y in captured["pts"]]).T
    xr, yr = ix - 100.0, iy - 100.0
    xrot = xr * np.cos(theta) + yr * np.sin(theta)
    yrot = -xr * np.sin(theta) + yr * np.cos(theta)
    assert np.allclose((xrot / a) ** 2 + (yrot / b) ** 2, 1.0, atol=1e-6)


def test_magnifier_outline_is_rotated_counter_clockwise_on_screen(win, monkeypatch):
    captured = _capture_polygon(monkeypatch)
    win.magnifier.grab()  # forces a real paintEvent

    pts = captured["pts"]
    rel = pts - pts.mean(axis=0)
    far = rel[np.argmax(np.hypot(rel[:, 0], rel[:, 1]))]
    # Major axis at +30 deg in image space (y up) points up-right or
    # down-left on screen (y down): dx and dy have opposite signs.
    assert far[0] * far[1] < 0
