"""Coverage for the sidebar's fixed width and its collapse/expand toggle
(SidebarToggle, replacing the old drag-to-resize ResizeGrip) - the sidebar
used to be draggable down to 220px, narrow enough to clip its own buttons/
histogram/cut-level boxes (no horizontal scrollbar to reveal what got cut
off). It's now a fixed width; the strip next to it only shows/hides it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import SIDEBAR_W, MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.widgets import SidebarToggle  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_fits(path, size=32):
    rng = np.random.default_rng(0)
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)).writeto(path, overwrite=True)


def _mouse_event(event_type, pos, button, buttons):
    return QMouseEvent(event_type, pos, pos, button, buttons, Qt.KeyboardModifier.NoModifier)


def _click(widget):
    center = QPointF(widget.width() / 2, widget.height() / 2)
    widget.mousePressEvent(_mouse_event(
        QEvent.Type.MouseButtonPress, center, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
    ))
    widget.mouseReleaseEvent(_mouse_event(
        QEvent.Type.MouseButtonRelease, center, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
    ))


# --------------------------------------------------------------- SidebarToggle


def test_click_emits_clicked(qapp):
    toggle = SidebarToggle()
    toggle.resize(10, 40)
    received = []
    toggle.clicked.connect(lambda: received.append(True))

    _click(toggle)

    assert received == [True]


def test_press_without_release_inside_does_not_emit(qapp):
    toggle = SidebarToggle()
    toggle.resize(10, 40)
    received = []
    toggle.clicked.connect(lambda: received.append(True))

    toggle.mousePressEvent(_mouse_event(
        QEvent.Type.MouseButtonPress, QPointF(5, 20), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
    ))
    # Released far outside the widget's own bounds - not a click.
    toggle.mouseReleaseEvent(_mouse_event(
        QEvent.Type.MouseButtonRelease, QPointF(500, 500), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
    ))

    assert received == []


def test_set_collapsed_does_not_emit_clicked(qapp):
    toggle = SidebarToggle()
    received = []
    toggle.clicked.connect(lambda: received.append(True))

    toggle.set_collapsed(True)

    assert received == []


def test_has_no_dragged_signal():
    """The old ResizeGrip's dragged(int)/released() drag API is gone -
    SidebarToggle is click-only."""
    assert not hasattr(SidebarToggle, "dragged")


# -------------------------------------------------------- MaskFitsApp integration


def test_sidebar_is_a_fixed_width(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    assert win.sidebar_container.width() == SIDEBAR_W
    assert win.sidebar_container.minimumWidth() == win.sidebar_container.maximumWidth()


def test_toggle_hides_and_shows_the_sidebar(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    assert win.sidebar_collapsed is False
    assert win.sidebar_container.isVisible() is True

    win._toggle_sidebar()
    assert win.sidebar_collapsed is True
    assert win.sidebar_container.isVisible() is False

    win._toggle_sidebar()
    assert win.sidebar_collapsed is False
    assert win.sidebar_container.isVisible() is True


def test_toggle_keeps_the_sidebar_width_fixed(qapp, tmp_path):
    """Collapsing hides the sidebar rather than shrinking it - its width
    stays exactly SIDEBAR_W throughout, ready to reappear unclipped."""
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    win._toggle_sidebar()
    assert win.sidebar_container.width() == SIDEBAR_W

    win._toggle_sidebar()
    assert win.sidebar_container.width() == SIDEBAR_W


def test_clicking_the_toggle_strip_collapses_the_sidebar(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    _click(win.sidebar_toggle)

    assert win.sidebar_collapsed is True
    assert win.sidebar_container.isVisible() is False


def test_toggle_chevron_state_matches_collapsed_state(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    assert win.sidebar_toggle._collapsed is False
    win._toggle_sidebar()
    assert win.sidebar_toggle._collapsed is True
