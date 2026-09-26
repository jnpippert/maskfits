"""Coverage for the magnifier's hover-preview shape outline
(MaskFitsApp._paint_shape_outline, wired into MagnifierWidget.paintEvent) -
an outline-only (no fill) rendering of the active tool's ellipse/satellite-
line preview, so it doesn't obscure the already-tiny zoomed pixels the fill
used on the main canvas would."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_fits(path, size=64):
    rng = np.random.default_rng(0)
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)).writeto(path, overwrite=True)


def _hovering_window(tmp_path, **settings_kwargs):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings(**settings_kwargs))
    win.show()
    win.canvas.resize(400, 400)
    win.canvas_w, win.canvas_h = win.canvas.width(), win.canvas.height()
    win._on_motion(win.canvas_w / 2, win.canvas_h / 2)
    return win


def test_magnifier_paints_without_crash_for_ellipse_tool(qapp, tmp_path):
    win = _hovering_window(tmp_path, mode="ellipse")
    win.magnifier.grab()  # forces a real paintEvent; must not raise


def test_magnifier_paints_without_crash_for_line_tool_with_anchor(qapp, tmp_path):
    win = _hovering_window(tmp_path, mode="line")
    win._line_anchor = win.canvas_to_img(win.canvas_w / 2 - 20, win.canvas_h / 2 - 20)
    win.magnifier.grab()  # forces a real paintEvent; must not raise


def test_magnifier_paints_without_crash_for_line_tool_without_anchor(qapp, tmp_path):
    win = _hovering_window(tmp_path, mode="line")
    assert win._line_anchor is None
    win.magnifier.grab()  # forces a real paintEvent; must not raise - no pending line means nothing to outline


def test_magnifier_paints_without_crash_without_a_cursor_position(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    assert win._cursor_img_pos is None
    win.magnifier.grab()  # early-return path (no cursor over the canvas)


def test_shape_size_change_refreshes_the_magnifier_without_a_mouse_move(qapp, tmp_path):
    """Regression: the magnifier used to only repaint on _on_motion, so
    changing the shape's radius/ellipticity/angle/thickness (or switching
    tools) while the mouse stayed still left its outline stale until the
    next mouse move."""
    win = _hovering_window(tmp_path, mode="ellipse")

    with patch.object(win.magnifier, "update") as mock_update:
        win._adjust_shape_size(1)
        assert mock_update.called


def test_toggle_tool_refreshes_the_magnifier_without_a_mouse_move(qapp, tmp_path):
    win = _hovering_window(tmp_path, mode="ellipse")

    with patch.object(win.magnifier, "update") as mock_update:
        win._toggle_tool()
        assert mock_update.called


def test_outline_transform_is_centered_on_the_cursor_pixel(qapp, tmp_path):
    """Regression: MagnifierWidget.paintEvent's to_widget() y-mapping must
    counter the array row-flip used when rendering the zoomed pixel crop
    (rgb[::-1]) - a naive mirror formula (matching the x-axis one, which
    needs no such correction) lands exactly one magnifier pixel too high,
    so the ellipse/line outline appeared shifted relative to the green
    cursor-pixel box it's supposed to be centered on."""
    path = tmp_path / "img.fits"
    _write_fits(path, size=101)
    win = MaskFitsApp([str(path)], settings=Settings(mode="ellipse"))
    win.show()
    win.canvas.resize(400, 400)
    win.canvas_w, win.canvas_h = win.canvas.width(), win.canvas.height()
    win.view_cx, win.view_cy = 50.5, 50.5
    win.zoom_mult = 1.0
    win.fit_zoom = win._compute_fit_zoom()
    cx, cy = win.img_to_canvas(50.5, 50.5)  # dead center of pixel (50, 50)
    win._on_motion(cx, cy)
    assert win._cursor_img_pos == pytest.approx((50.5, 50.5))

    captured = {}
    real_paint = MaskFitsApp._paint_shape_outline

    def spy(self, painter, to_widget, scale):
        captured["to_widget"] = to_widget
        return real_paint(self, painter, to_widget, scale)

    with patch.object(MaskFitsApp, "_paint_shape_outline", spy):
        win.magnifier.grab()  # forces a real, synchronous paintEvent

    from maskfits.gui import MAG_SIZE

    half = MAG_SIZE // 2
    square = min(win.magnifier.width(), win.magnifier.height())
    block = max(square // MAG_SIZE, 1)
    disp = block * MAG_SIZE
    ox, oy = (win.magnifier.width() - disp) // 2, (win.magnifier.height() - disp) // 2
    # Same rect the green cursor-pixel box is drawn at (see paintEvent) -
    # its center is where a shape centered on the cursor must land too.
    green_box_center = (ox + half * block + block / 2, oy + half * block + block / 2)

    assert captured["to_widget"](*win._cursor_img_pos) == pytest.approx(green_box_center)


def test_paint_shape_outline_ellipse_uses_current_round_params(qapp, tmp_path):
    """Exercises MaskFitsApp._paint_shape_outline directly with a trivial
    identity-like transform, so the geometry it builds can be checked
    without depending on the magnifier's own pixel layout."""
    win = _hovering_window(tmp_path, mode="ellipse")
    win.radius = 12
    win.ellipticity = 0
    win.angle = 0

    from PySide6.QtGui import QImage, QPainter

    img = QImage(200, 200, QImage.Format.Format_RGB32)
    painter = QPainter(img)

    seen = {}

    def to_widget(ix, iy):
        seen["center"] = (ix, iy)
        return (100.0, 100.0)

    win._paint_shape_outline(painter, to_widget, scale=2.0)
    painter.end()

    assert seen["center"] == win._cursor_img_pos
