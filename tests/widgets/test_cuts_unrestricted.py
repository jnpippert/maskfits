"""Lowcut/highcut have no limits of their own: typed values far outside
the data range are kept, the view follows them, and a drag past the plot
edge keeps going. The only rule left is lowcut <= highcut."""

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


def _mouse(event_type, x, y=50.0):
    pos = QPointF(x, y)
    return QMouseEvent(event_type, pos, pos, Qt.MouseButton.LeftButton,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


@pytest.fixture
def hist(qapp):
    data = np.random.default_rng(0).normal(50.0, 15.0, size=1000)  # roughly 0..100
    widget = CutsHistogram(data, 10.0, 90.0)
    widget.resize(360, 220)
    widget.show()
    applied = []
    widget.cuts_applied.connect(lambda lo, hi: applied.append((lo, hi)))
    widget.applied = applied
    return widget


def _type(entry, text):
    entry.setText(text)
    entry.editingFinished.emit()


def test_typed_values_far_outside_the_data_are_kept(hist):
    _type(hist._hi_entry, "1e6")
    assert hist.vmax == 1e6
    _type(hist._lo_entry, "-5e5")
    assert hist.vmin == -5e5
    assert hist.applied[-1] == (-5e5, 1e6)


def test_view_follows_the_cuts_beyond_the_data(hist):
    _type(hist._hi_entry, "1000")
    lo, hi = hist.compute_disp_range()
    assert lo < 10.0 and hi > 1000.0  # cuts plus a margin, not capped at the data


def test_lowcut_still_cannot_pass_highcut(hist):
    _type(hist._lo_entry, "500")
    assert hist.vmin == 90.0
    _type(hist._hi_entry, "-500")
    assert hist.vmax == hist.vmin


def test_drag_past_the_plot_edge_keeps_going(hist):
    canvas = hist.canvas
    lo, hi = hist.compute_disp_range()
    x_hi = canvas._value_to_x(hist.vmax, lo, hi)
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, x_hi))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, canvas.width() + 300))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, canvas.width() + 300))
    assert hist.vmax > hi  # beyond what the view showed when the drag began
    # ... and the next drag starts from a view that includes the new cut
    assert hist.compute_disp_range()[1] > hist.vmax

    lo, hi = hist.compute_disp_range()
    x_lo = canvas._value_to_x(hist.vmin, lo, hi)
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, x_lo))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, -300))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, -300))
    assert hist.vmin < lo
