"""Coverage for the sidebar's cursor readout (x/y/value left, RA/DEC/shape
right, right column anchored at the panel's center) and the RoundSlider
handle inset that keeps the handle unclipped at min/max."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import SIDEBAR_W, MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.widgets import RoundSlider  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write(path, data):
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32)).writeto(path, overwrite=True)


def test_shape_readout_for_a_landscape_image(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write(path, np.zeros((30, 50)))  # NAXIS1=50 (x), NAXIS2=30 (y)
    win = MaskFitsApp([str(path)], settings=Settings())
    assert win.readout["shape"].text() == "50 × 30"


def test_shape_readout_for_a_portrait_image_is_the_on_disk_shape(qapp, tmp_path):
    # Portrait frames are transposed in memory - the readout must still show
    # the file's own NAXIS1 x NAXIS2, not the transposed array's shape.
    path = tmp_path / "img.fits"
    _write(path, np.zeros((50, 30)))
    win = MaskFitsApp([str(path)], settings=Settings())
    assert win.image.rotated
    assert win.readout["shape"].text() == "30 × 50"


def test_shape_readout_for_a_cube(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    _write(path, np.zeros((4, 30, 50)))
    win = MaskFitsApp([str(path)], settings=Settings())
    assert win.readout["shape"].text() == "50 × 30 × 4"


def test_shape_readout_empty_without_an_image(qapp):
    win = MaskFitsApp([], settings=Settings())
    assert win.readout["shape"].text() == ""


def test_right_column_is_anchored_at_the_panel_center(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write(path, np.zeros((30, 50)))
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    qapp.processEvents()

    assert win.sidebar_container.width() == SIDEBAR_W
    right_half = win.readout["RA"].parentWidget()
    # The panel's scrollable content widget (a vertical scrollbar, when
    # shown, narrows it) - its margins are symmetric, so its center is
    # where the right column must start.
    content = right_half.parentWidget()
    right_x = right_half.x()
    assert abs(right_x - content.width() / 2) <= 1

    # Independent of how long the left column's values get.
    win.readout["value"].setText("-1.23456789e-20 extra long text")
    qapp.processEvents()
    assert right_half.x() == right_x


@pytest.mark.parametrize("value", [0, 100])
def test_slider_handle_is_fully_inside_the_widget_at_the_extremes(qapp, value):
    slider = RoundSlider(0, 100, value)
    slider.resize(200, slider.height())
    cx = slider._handle_center_x(slider._track_rect())
    outline = RoundSlider._RADIUS + 1  # 2px pen straddles the circle edge
    assert cx - outline >= 0
    assert cx + outline <= slider.width()
