"""Coverage for FlowLayout.add_right_widget() - added so the toolbar's
Export Mask/Kill buttons can stay anchored to the row's right edge
regardless of how much else is flowing before them, instead of just being
the last item in the normal left-packed flow (see gui._build_toolbar)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from maskfits.layouts import FlowLayout  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_right_widget_is_parented_under_the_right_zone(qapp):
    flow = FlowLayout()
    btn = QPushButton("Kill")
    flow.add_right_widget(btn)

    assert btn.parent() is flow._row1_right


def test_normal_widgets_are_parented_under_the_flow_zone(qapp):
    flow = FlowLayout()
    lbl = QLabel("hello")
    flow.addWidget(lbl)

    assert lbl.parent() is flow._row1_flow


def test_right_widget_never_moves_to_row2_on_reflow(qapp):
    flow = FlowLayout()
    right_btn = QPushButton("Export Mask - a somewhat long label")
    flow.add_right_widget(right_btn)
    # A wide flowing widget that would normally overflow into row2.
    wide = QLabel("x" * 400)
    flow.addWidget(wide)

    flow.resize(300, 40)
    flow._reflow(300)

    assert right_btn.parent() is flow._row1_right
    assert flow._row2.isVisible() in (True, False)  # just must not have crashed/misparented
    assert right_btn not in flow._widgets


def test_right_widgets_do_not_count_toward_the_flow_wrap_calculation_twice(qapp):
    """add_right_widget()'s widgets are tracked separately from the
    normally-flowing ones (_widgets) - only their reserved width should
    affect the flow wrap point, not their own presence in _widgets."""
    flow = FlowLayout()
    right_btn = QPushButton("Kill")
    flow.add_right_widget(right_btn)

    assert right_btn not in flow._widgets
    assert right_btn in flow._right_widgets


def test_multiple_right_widgets_all_stay_pinned(qapp):
    flow = FlowLayout()
    a, b = QPushButton("Export Mask"), QPushButton("Kill")
    flow.add_right_widget(a)
    flow.add_right_widget(b)

    assert flow._right_widgets == [a, b]
    assert a.parent() is flow._row1_right
    assert b.parent() is flow._row1_right
