"""Regression coverage for a real bug: dragging one of the histogram's
lowcut/highcut handles updated the drawn line and self.vmin/vmax, but left
the numeric entry box it corresponds to showing the stale value until the
drag ended - _HistogramCanvas.mouseMoveEvent passed from_entry="lo"/"hi" to
set_cuts(), a flag meant to stop set_cuts() from overwriting whichever entry
box the user is actively TYPING into (see _on_lo_entry/_on_hi_entry), which
doesn't apply to a canvas drag - it just suppressed updating the very box
whose handle was being dragged."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.cuts_histogram import CutsHistogram  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _mouse_event(event_type, x, y=50.0):
    pos = QPointF(x, y)
    return QMouseEvent(
        event_type, pos, pos, Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )


def _make_histogram(qapp, lowcut=10.0, highcut=90.0):
    rng = np.random.default_rng(0)
    data = rng.normal(50.0, 15.0, size=1000)
    hist = CutsHistogram(data, lowcut, highcut)
    hist.resize(300, 150)
    hist.canvas.resize(300, 100)
    qapp.processEvents()
    return hist


def _drag_handle(hist, which, target_value):
    canvas = hist.canvas
    disp_lo, disp_hi = hist.compute_disp_range()
    start_x = canvas._value_to_x(hist.vmin if which == "lo" else hist.vmax, disp_lo, disp_hi)
    canvas.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, start_x))
    assert canvas._drag == which  # sanity - the press actually grabbed the handle

    target_x = canvas._value_to_x(target_value, disp_lo, disp_hi)
    canvas.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, target_x))


def test_dragging_the_lowcut_handle_updates_its_own_entry_box(qapp):
    hist = _make_histogram(qapp)
    _drag_handle(hist, "lo", 30.0)

    assert hist.vmin == pytest.approx(30.0, abs=1.0)
    assert float(hist._lo_entry.text()) == pytest.approx(hist.vmin)


def test_dragging_the_highcut_handle_updates_its_own_entry_box(qapp):
    hist = _make_histogram(qapp)
    _drag_handle(hist, "hi", 70.0)

    assert hist.vmax == pytest.approx(70.0, abs=1.0)
    assert float(hist._hi_entry.text()) == pytest.approx(hist.vmax)


def test_dragging_the_lowcut_handle_does_not_touch_the_highcut_box(qapp):
    hist = _make_histogram(qapp)
    before = hist._hi_entry.text()
    _drag_handle(hist, "lo", 30.0)

    assert hist._hi_entry.text() == before


def test_dragging_the_highcut_handle_does_not_touch_the_lowcut_box(qapp):
    hist = _make_histogram(qapp)
    before = hist._lo_entry.text()
    _drag_handle(hist, "hi", 70.0)

    assert hist._lo_entry.text() == before
