"""A left-to-right, top-to-bottom wrapping row of widgets, used for the
toolbar so its controls wrap onto a second row instead of overflowing or
hiding behind a chevron - matching the Tkinter version's manual two-row
reflow.

This is deliberately NOT a custom QLayout subclass (an earlier version was,
following the standard Qt "Flow Layout" cookbook pattern) - that turned out
to cause an intermittent native access-violation crash under PySide6 6.11 /
Python 3.14 on this machine, confirmed by bisection (a plain QHBoxLayout in
the same position never crashed across dozens of runs; swapping in the
custom QLayout reproduced the crash reliably). Custom QLayout subclasses are
a documented fragile spot in PySide6/PyQt's Python bindings, so this instead
reflows plain widgets between two ordinary QHBoxLayout rows on resize - the
same two-row reassignment the Tkinter version did via pack(in_=...), just
using QWidget.setParent()/QHBoxLayout.addWidget() instead.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget


class FlowLayout(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, *, h_spacing: int = 12, v_spacing: int = 6):
        super().__init__(parent)
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self._widgets: list[QWidget] = []
        self._margins = (0, 0, 0, 0)
        self._last_split: Optional[int] = None
        self._row_height = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(v_spacing)

        # Named so theme.build_qss can target them - a plain QWidget would
        # otherwise pick up the global `QWidget { background-color: app_bg }`
        # rule and paint a visibly darker rectangle over whatever panel
        # (panel_bg) this FlowLayout is actually sitting inside of.
        self.setObjectName("flowLayout")

        # row1 is itself split into a flowing zone (unchanged behavior -
        # wraps to row2 under its own trailing stretch) and a right-anchored
        # zone next to it (add_right_widget) that never wraps - for controls
        # (Export Mask/Kill) that should always stay pinned to the
        # toolbar's right edge regardless of how much else is flowing.
        self._row1 = QWidget(self)
        self._row1.setObjectName("flowRow")
        row1_outer = QHBoxLayout(self._row1)
        row1_outer.setContentsMargins(0, 0, 0, 0)
        row1_outer.setSpacing(h_spacing)

        self._row1_flow = QWidget(self._row1)
        self._row1_layout = QHBoxLayout(self._row1_flow)
        self._row1_layout.setContentsMargins(0, 0, 0, 0)
        self._row1_layout.setSpacing(h_spacing)
        self._row1_layout.addStretch(0)

        self._row1_right = QWidget(self._row1)
        self._row1_right_layout = QHBoxLayout(self._row1_right)
        self._row1_right_layout.setContentsMargins(0, 0, 0, 0)
        self._row1_right_layout.setSpacing(h_spacing)
        self._right_widgets: list[QWidget] = []

        row1_outer.addWidget(self._row1_flow, 1)
        row1_outer.addWidget(self._row1_right, 0)

        self._row2 = QWidget(self)
        self._row2.setObjectName("flowRow")
        self._row2_layout = QHBoxLayout(self._row2)
        self._row2_layout.setContentsMargins(0, 0, 0, 0)
        self._row2_layout.setSpacing(h_spacing)
        self._row2_layout.addStretch(0)

        outer.addWidget(self._row1)
        outer.addWidget(self._row2)
        self._row2.hide()
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def setContentsMargins(self, left: int, top: int, right: int, bottom: int) -> None:  # noqa: N802
        self._margins = (left, top, right, bottom)
        self.layout().setContentsMargins(left, top, right, bottom)

    def addWidget(self, widget: QWidget) -> None:  # noqa: N802
        widget.setParent(self._row1_flow)
        self._widgets.append(widget)
        # Provisionally in row1 (last position, before the trailing stretch)
        # - the next resizeEvent reflows properly based on actual widths.
        self._row1_layout.insertWidget(self._row1_layout.count() - 1, widget)
        self._last_split = None
        self._update_row_height()

    def add_right_widget(self, widget: QWidget) -> None:
        """Adds `widget` pinned to row1's right edge - unlike addWidget(),
        it never wraps to row2 and never shares row1's usable width with
        the normally-flowing widgets (their wrap point accounts for the
        space this reserves - see _reflow)."""
        widget.setParent(self._row1_right)
        self._right_widgets.append(widget)
        self._row1_right_layout.addWidget(widget)
        self._update_row_height()

    def _update_row_height(self) -> None:
        """Pin both rows to a fixed height matching the tallest control -
        without this, the surrounding QVBoxLayout (root_layout in
        MaskFitsApp._build_ui) can end up granting this widget less height
        than its buttons/labels actually need whenever the window is
        resized (nothing here declares height-for-width, so the ancestor
        layout doesn't always re-query sizeHint() before repainting), which
        visibly compresses every control's text instead of leaving it
        readable."""
        all_widgets = self._widgets + self._right_widgets
        if not all_widgets:
            return
        self._row_height = max(w.sizeHint().height() for w in all_widgets)
        self._row1.setFixedHeight(self._row_height)
        self._row1_flow.setFixedHeight(self._row_height)
        self._row1_right.setFixedHeight(self._row_height)
        self._row2.setFixedHeight(self._row_height)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reflow(event.size().width())

    def _reflow(self, available_width: int) -> None:
        if not self._widgets:
            return
        left, _top, right, _bottom = self._margins
        right_reserved = sum(w.sizeHint().width() + self._h_spacing for w in self._right_widgets)
        usable = max(available_width - left - right - right_reserved, 1)
        total = 0
        split = len(self._widgets)
        for i, w in enumerate(self._widgets):
            width = w.sizeHint().width()
            add = width if i == 0 else width + self._h_spacing
            if total + add > usable and i > 0:
                split = i
                break
            total += add
        if split == self._last_split:
            return
        self._last_split = split
        for w in self._widgets:
            w.setParent(None)
        for w in self._widgets[:split]:
            self._row1_layout.insertWidget(self._row1_layout.count() - 1, w)
        for w in self._widgets[split:]:
            self._row2_layout.insertWidget(self._row2_layout.count() - 1, w)
        self._row2.setVisible(split < len(self._widgets))
        self.updateGeometry()

    # No sizeHint()/minimumSizeHint() override needed: with both rows
    # pinned to a fixed height (see _update_row_height) and `outer` (this
    # widget's own QVBoxLayout) correctly excluding row2 from its sum
    # whenever it's hidden, Qt's default QWidget behavior of delegating to
    # `self.layout().sizeHint()` already reports the right height.
