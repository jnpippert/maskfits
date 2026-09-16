"""Small reusable Qt widgets used by the maskfits GUI, themed via maskfits.theme.

Unlike the old Tkinter version (which had to hand-draw every rounded shape on
a Canvas, since Tk has no real widget styling), most of these are thin
QWidget/QPushButton/QSlider subclasses whose actual look comes from the QSS
in theme.build_qss() - selected either by Qt's built-in widget-class
selectors (QPushButton, QSlider, ...) or by these classes' own Python class
names (RoundedPanel). `ResizeGrip` is the one exception that still needs a
real paintEvent, since no native widget matches its drag-affordance look.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QEnterEvent, QMouseEvent, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from maskfits.theme import current_theme, theme_manager


class RoundedPanel(QFrame):
    """A card-like panel with rounded corners (all styling via the QSS
    `RoundedPanel` selector - see theme.build_qss). Add children to `.inner`'s
    layout, not the panel itself when scrollable.

    mode="fill" (default): takes whatever size its parent gives it.
    mode="hug": sizes itself to its content's natural height instead (a
      toolbar or status bar that should stay compact).
    scrollable: only meaningful with mode="fill" - scrolls instead of
      clipping when content exceeds the available height.
    """

    def __init__(self, parent: Optional[QWidget] = None, *, mode: str = "fill", scrollable: bool = False):
        super().__init__(parent)
        self.setObjectName("RoundedPanel")

        if scrollable and mode == "fill":
            # Only the scrollable case needs a layout of its own here (to
            # host the QScrollArea) - the non-scrollable case leaves `self`
            # layout-less on purpose, since `self.inner` IS `self` there and
            # callers (e.g. MaskFitsApp._build_toolbar) install their own
            # layout directly on `.inner`. Installing one here too would
            # give the widget two layouts - Qt silently refuses the second
            # one, so nothing added to it would ever actually display (this
            # is exactly what made the toolbar/status bar disappear before
            # this was fixed).
            outer_layout = QVBoxLayout(self)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.inner = QWidget()
            self.inner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            scroll.setWidget(self.inner)
            outer_layout.addWidget(scroll)
        else:
            self.inner = self

        if mode == "hug":
            self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)


class RoundButton(QPushButton):
    """A themed push button - checkable (for toggle/segmented-control use),
    or with an accent/danger/flat variant (see the QSS dynamic-property
    selectors in theme.build_qss)."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None, *, checkable: bool = False,
                 accent: bool = False, danger: bool = False, flat: bool = False):
        super().__init__(text, parent)
        self.setCheckable(checkable)
        if accent:
            self.setProperty("accent", True)
        if danger:
            self.setProperty("danger", True)
        if flat:
            self.setProperty("flat", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class SegmentedControl(QWidget):
    """A row of toggle buttons acting as a single-select group.

    Stays in sync regardless of what changes the value - not just its own
    button clicks - via `set_value()`; emits `valueChanged(str)` only for
    changes that originate from a button click, mirroring how the old
    Tkinter version's StringVar trace worked from either direction.
    """

    valueChanged = Signal(str)

    def __init__(self, options: list[tuple[str, str]], value: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, RoundButton] = {}
        self._value = value
        for val, label in options:
            btn = RoundButton(label, checkable=True)
            btn.setChecked(val == value)
            layout.addWidget(btn)
            self._group.addButton(btn)
            self._buttons[val] = btn
            btn.clicked.connect(lambda _checked=False, v=val: self._on_clicked(v))

    def _on_clicked(self, value: str) -> None:
        if value != self._value:
            self._value = value
            self.valueChanged.emit(value)

    def value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        if value not in self._buttons or value == self._value:
            return
        self._value = value
        self._buttons[value].setChecked(True)


class RoundSlider(QWidget):
    """A horizontal slider over an arbitrary float (or int) range.

    QSlider is integer-only, so a float range is represented internally as
    an integer 0..STEPS and mapped to/from the real [lo, hi] range - callers
    only ever see real float values via `value()`/`setValue()`/`valueChanged`.
    `sliderPressed`/`sliderReleased` are forwarded from the underlying
    QSlider for the "cheap update while dragging, expensive work only on
    release" pattern used by the cuts histogram and mask-alpha slider.
    """

    valueChanged = Signal(float)
    _STEPS = 1000

    def __init__(self, lo: float, hi: float, value: float, *, integer: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._lo, self._hi, self._integer = lo, hi, integer
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setCursor(Qt.CursorShape.PointingHandCursor)
        # QSlider is a "complex control" Qt paints via QStyle::drawComplexControl
        # - without WA_StyledBackground, it still erases its own bounding box
        # with the palette's Window color before drawing groove/handle/etc,
        # regardless of the QSS `background: transparent` rule for it. That
        # erase is exactly app_bg, visibly darker than whatever panel (panel_bg)
        # the slider sits on - this makes Qt actually respect the transparency.
        self._slider.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        if integer:
            self._slider.setMinimum(int(round(lo)))
            self._slider.setMaximum(int(round(hi)))
        else:
            self._slider.setMinimum(0)
            self._slider.setMaximum(self._STEPS)
        layout.addWidget(self._slider)
        self.setValue(value)
        self._slider.valueChanged.connect(self._on_raw_changed)
        self.sliderPressed = self._slider.sliderPressed
        self.sliderReleased = self._slider.sliderReleased

    def _raw_to_value(self, raw: int) -> float:
        if self._integer:
            return float(raw)
        frac = raw / self._STEPS
        return self._lo + frac * (self._hi - self._lo)

    def _value_to_raw(self, value: float) -> int:
        if self._integer:
            return int(round(value))
        span = self._hi - self._lo
        frac = (value - self._lo) / span if span else 0.0
        frac = min(max(frac, 0.0), 1.0)
        return int(round(frac * self._STEPS))

    def _on_raw_changed(self, raw: int) -> None:
        self.valueChanged.emit(self._raw_to_value(raw))

    def value(self) -> float:
        return self._raw_to_value(self._slider.value())

    def setValue(self, value: float) -> None:  # noqa: N802 - matches Qt naming convention
        value = max(self._lo, min(value, self._hi))
        raw = self._value_to_raw(value)
        if raw != self._slider.value():
            self._slider.setValue(raw)
        else:
            # setValue with an unchanged raw step wouldn't otherwise emit -
            # callers (e.g. a linked spin-box) still expect the float value
            # to be current on return, so this just ensures state agreement,
            # no signal needed since nothing actually moved.
            pass

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        self._slider.setEnabled(enabled)
        super().setEnabled(enabled)


class ThemeToggle(QPushButton):
    """A sun/moon emoji button for switching between light and dark mode."""

    SUN = "☀️"
    MOON = "\U0001F319"

    def __init__(self, parent: Optional[QWidget] = None, *, light: bool = False):
        super().__init__(cls_text(light), parent)
        self.setProperty("flat", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(34)

    def set_light(self, light: bool) -> None:
        self.setText(cls_text(light))


def cls_text(light: bool) -> str:
    return ThemeToggle.SUN if light else ThemeToggle.MOON


class ResizeGrip(QWidget):
    """A thin vertical drag handle for resizing a panel next to it, with a
    small rounded grabber mark at its vertical center as a visual affordance
    that the gap between the two panels is draggable.

    Emits `dragged(int)` as a delta (dx) for each drag step, and `released()`
    once dragging ends - mirroring the old Tkinter callback shape so callers
    just accumulate the delta into whatever width they're tracking, and can
    defer expensive work (like re-laying-out a child widget) to `released()`.
    """

    dragged = Signal(int)
    released = Signal()

    def __init__(self, parent: Optional[QWidget] = None, *, width: int = 10):
        super().__init__(parent)
        self.setFixedWidth(width)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._hovering = False
        self._drag_last_x: Optional[int] = None
        theme_manager().theme_changed.connect(lambda _t: self.update())

    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802
        self._hovering = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovering = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_last_x = event.globalPosition().toPoint().x()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_last_x is None:
            return
        x = event.globalPosition().toPoint().x()
        dx = x - self._drag_last_x
        self._drag_last_x = x
        if dx:
            self.dragged.emit(dx)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_last_x is None:
            return
        self._drag_last_x = None
        self.released.emit()

    def paintEvent(self, event) -> None:  # noqa: N802
        theme = current_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx = w / 2
        cy = h / 2
        bar_w = 4
        bar_h = min(36, max(h - 8, 0))
        color = QColor(theme.text) if self._hovering else QColor(theme.panel_border)
        path = QPainterPath()
        path.addRoundedRect(QRectF(cx - bar_w / 2, cy - bar_h / 2, bar_w, bar_h), bar_w / 2, bar_w / 2)
        painter.fillPath(path, color)
