"""Coverage for widgets.RangeSlider - one track, two handles (lower/upper)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.widgets import RangeSlider  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _slider(lower=20, upper=80):
    slider = RangeSlider(0, 100, lower, upper, integer=True)
    slider.resize(220, slider.height())
    return slider


def _mouse(slider, kind, value, button=Qt.MouseButton.LeftButton):
    x = slider._x_for(value, slider._track_rect())
    point = QPointF(x, slider.height() / 2)
    event = QMouseEvent(kind, point, point, button, button, Qt.KeyboardModifier.NoModifier)
    {QEvent.Type.MouseButtonPress: slider.mousePressEvent, QEvent.Type.MouseMove: slider.mouseMoveEvent,
     QEvent.Type.MouseButtonRelease: slider.mouseReleaseEvent}[kind](event)


def test_initial_values(qapp):
    assert _slider().values() == (20.0, 80.0)


def test_values_are_ordered_and_clamped(qapp):
    slider = RangeSlider(0, 100, 90, 10, integer=True)
    assert slider.values() == (10.0, 90.0)
    slider.setValues(-5, 500)
    assert slider.values() == (0.0, 100.0)
    slider.setValues(70, 30)
    assert slider.values() == (30.0, 70.0)


def test_set_values_emits_only_on_change(qapp):
    slider = _slider()
    received = []
    slider.valuesChanged.connect(lambda lo, hi: received.append((lo, hi)))
    slider.setValues(20, 80)
    slider.setValues(25, 80)
    assert received == [(25.0, 80.0)]


def test_dragging_the_lower_handle(qapp):
    slider = _slider()
    _mouse(slider, QEvent.Type.MouseButtonPress, 20)
    _mouse(slider, QEvent.Type.MouseMove, 40)
    _mouse(slider, QEvent.Type.MouseButtonRelease, 40)
    assert slider.values() == (40.0, 80.0)


def test_dragging_the_upper_handle(qapp):
    slider = _slider()
    _mouse(slider, QEvent.Type.MouseButtonPress, 80)
    _mouse(slider, QEvent.Type.MouseMove, 60)
    assert slider.values() == (20.0, 60.0)


def test_a_handle_cannot_cross_the_other(qapp):
    slider = _slider()
    _mouse(slider, QEvent.Type.MouseButtonPress, 20)
    _mouse(slider, QEvent.Type.MouseMove, 95)
    assert slider.values() == (80.0, 80.0)


def test_click_on_the_track_moves_the_nearest_handle(qapp):
    slider = _slider()
    _mouse(slider, QEvent.Type.MouseButtonPress, 30)
    assert slider.values() == (30.0, 80.0)
    _mouse(slider, QEvent.Type.MouseButtonRelease, 30)
    _mouse(slider, QEvent.Type.MouseButtonPress, 70)
    assert slider.values() == (30.0, 70.0)


def test_stacked_handles_click_side_decides(qapp):
    slider = _slider(50, 50)
    assert slider._handle_for(slider._x_for(40, slider._track_rect())) == "lower"
    assert slider._handle_for(slider._x_for(60, slider._track_rect())) == "upper"


def test_disabled_ignores_the_mouse(qapp):
    slider = _slider()
    slider.setEnabled(False)
    _mouse(slider, QEvent.Type.MouseButtonPress, 50)
    assert slider.values() == (20.0, 80.0)


@pytest.mark.parametrize("values", [(0, 0), (100, 100), (0, 100)])
def test_handles_never_clipped_at_the_extremes(qapp, values):
    slider = _slider(*values)
    track = slider._track_rect()
    outline = RangeSlider._RADIUS + 1
    for value in values:
        x = slider._x_for(value, track)
        assert x - outline >= 0 and x + outline <= slider.width()
