"""The magnifier's empty-state grid is closed on all four edges (it used to
end in stubs on the right and bottom) and matches the zoom view's pixel
blocks."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MAG_SIZE, MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.theme import current_theme  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_empty_grid_is_closed(qapp):
    win = MaskFitsApp([], settings=Settings(theme="dark"))
    win.show()
    img = win.magnifier.grab().toImage()
    line = QColor(current_theme().panel_border).rgb()
    w, h = img.width(), img.height()
    assert all(img.pixel(0, y) == line for y in range(h))
    assert all(img.pixel(w - 1, y) == line for y in range(h))
    assert all(img.pixel(x, 0) == line for x in range(w))
    assert all(img.pixel(x, h - 1) == line for x in range(w))
    # One cell per zoom-view pixel: MAG_SIZE + 1 vertical lines along a row
    # through the middle of the first cell row.
    block = w // MAG_SIZE
    row = [img.pixel(x, block // 2) == line for x in range(w)]
    lines = sum(1 for x in range(w) if row[x] and (x == 0 or not row[x - 1]))
    assert lines == MAG_SIZE + 1
