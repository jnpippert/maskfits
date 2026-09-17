"""Small reusable Qt widgets used by the maskfits GUI, themed via maskfits.theme.

Unlike the old Tkinter version (which had to hand-draw every rounded shape on
a Canvas, since Tk has no real widget styling), most of these are thin
QWidget/QPushButton subclasses whose actual look comes from the QSS in
theme.build_qss() - selected either by Qt's built-in widget-class selectors
(QPushButton, ...) or by these classes' own Python class names
(RoundedPanel). `ResizeGrip` and `RoundSlider` are the exceptions that need a
real paintEvent: no native widget matches ResizeGrip's drag-affordance look,
and QSlider's native complex-control painting (even under Fusion, even with
every sub-control QSS-styled) leaves residual decoration a stylesheet has no
documented hook to fully suppress - see RoundSlider's own docstring.
"""

from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QEnterEvent, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from maskfits.theme import current_theme, theme_manager

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


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
    """A horizontal slider over an arbitrary float (or int) range, fully
    custom-painted rather than a styled QSlider.

    QSlider under Fusion (needed in the first place for the app-wide QSS to
    have any effect on it at all - see run_gui's app.setStyle) still paints
    residual native decoration around its QSS-styled groove - a visibly
    different-toned band - that WA_StyledBackground/WA_NoSystemBackground
    and a fully-overridden stylesheet don't fully suppress; it comes from
    the style's own drawComplexControl step, not a background-erase this
    project's tools have any documented QSS hook to turn off. Painting the
    track/fill/handle directly sidesteps that ambiguity entirely - what's
    drawn here is the whole visual, nothing native left to leak through.

    `sliderPressed`/`sliderReleased` mirror the "cheap update while
    dragging, expensive work only on release" pattern used by the cuts
    histogram and the mask-alpha slider.
    """

    valueChanged = Signal(float)
    sliderPressed = Signal()
    sliderReleased = Signal()

    _RADIUS = 7  # handle radius, px - also sets the track's own corner/inset

    def __init__(self, lo: float, hi: float, value: float, *, integer: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._lo, self._hi, self._integer = lo, hi, integer
        self._value = self._clamp(value)
        self._dragging = False
        self.setFixedHeight(2 * self._RADIUS + 6)
        self.setMinimumWidth(60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        theme_manager().theme_changed.connect(lambda _t: self.update())

    def _clamp(self, value: float) -> float:
        value = max(self._lo, min(value, self._hi))
        return float(int(round(value))) if self._integer else value

    def _frac(self) -> float:
        span = self._hi - self._lo
        return (self._value - self._lo) / span if span else 0.0

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:  # noqa: N802 - matches Qt naming convention
        value = self._clamp(value)
        if value != self._value:
            self._value = value
            self.update()
            self.valueChanged.emit(self._value)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(120, self.height())

    # -------------------------------------------------------------- layout

    def _track_rect(self) -> QRectF:
        r = self._RADIUS
        h = 6
        y = (self.height() - h) / 2
        return QRectF(r, y, max(self.width() - 2 * r, 1), h)

    def _handle_center_x(self, track: QRectF) -> float:
        return track.left() + track.width() * min(max(self._frac(), 0.0), 1.0)

    def _x_to_value(self, x: float) -> float:
        track = self._track_rect()
        frac = (x - track.left()) / max(track.width(), 1)
        frac = min(max(frac, 0.0), 1.0)
        return self._lo + frac * (self._hi - self._lo)

    # -------------------------------------------------------------- paint

    def paintEvent(self, event) -> None:  # noqa: N802
        theme = current_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = self._track_rect()
        radius = track.height() / 2
        enabled = self.isEnabled()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.panel_border if not enabled else theme.track))
        painter.drawRoundedRect(track, radius, radius)

        cx = self._handle_center_x(track)
        fill_w = cx - track.left()
        if fill_w > 0:
            fill_color = QColor(theme.text_dim if not enabled else theme.accent)
            painter.setBrush(fill_color)
            painter.drawRoundedRect(QRectF(track.left(), track.top(), fill_w, track.height()), radius, radius)

        cy = track.center().y()
        painter.setPen(QPen(QColor(theme.text_dim if not enabled else theme.accent), 2))
        painter.setBrush(QColor(theme.text))
        painter.drawEllipse(QPointF(cx, cy), self._RADIUS, self._RADIUS)

    # -------------------------------------------------------------- input

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = True
        self.sliderPressed.emit()
        self.setValue(self._x_to_value(event.position().x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._dragging:
            return
        self.setValue(self._x_to_value(event.position().x()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._dragging or event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = False
        self.sliderReleased.emit()


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


class HexColorPicker(QWidget):
    """A "#rrggbb" text entry plus a clickable color swatch (opens
    QColorDialog) - used for the accent-color override and, with
    `allow_empty=False`, every field of a custom theme (see
    theme_editor_window.py).

    Only ever emits colorChanged for a *valid* 6-digit hex string (or "" when
    allow_empty and the field is cleared) - an in-progress edit like "#ab"
    just doesn't emit yet, rather than emitting garbage.
    """

    colorChanged = Signal(str)

    def __init__(self, initial: Optional[str], *, allow_empty: bool = False, placeholder: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._allow_empty = allow_empty
        self._value: Optional[str] = initial or None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._entry = QLineEdit(initial or "")
        self._entry.setFixedWidth(84)
        if placeholder:
            self._entry.setPlaceholderText(placeholder)
        self._entry.textChanged.connect(self._on_text_changed)
        layout.addWidget(self._entry)

        self._swatch = QPushButton()
        self._swatch.setFixedSize(22, 22)
        self._swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self._swatch.clicked.connect(self._pick)
        layout.addWidget(self._swatch)

        theme_manager().theme_changed.connect(lambda _t: self._update_swatch())
        self._update_swatch()

    def value(self) -> Optional[str]:
        return self._value

    def setValue(self, hex_color: Optional[str]) -> None:  # noqa: N802 - matches Qt naming convention
        self._entry.setText(hex_color or "")

    def _on_text_changed(self, text: str) -> None:
        text = text.strip()
        if not text:
            if self._allow_empty:
                self._value = None
                self._update_swatch()
                self.colorChanged.emit("")
            return
        if not _HEX_RE.match(text):
            return
        self._value = text
        self._update_swatch()
        self.colorChanged.emit(text)

    def _update_swatch(self) -> None:
        theme = current_theme()
        color = self._value or theme.accent
        self._swatch.setStyleSheet(
            f"QPushButton {{ background-color: {color}; border: 1px solid {theme.panel_border}; "
            f"border-radius: 5px; }}"
        )

    def _pick(self) -> None:
        initial = QColor(self._value or current_theme().accent)
        # Two things needed fixing here, not just one:
        # 1. Parent to the top-level window (SettingsWindow/ThemeEditorWindow),
        #    not just this HexColorPicker child widget - a non-toplevel parent
        #    left Qt/Cocoa unsure which window opened the (modal) dialog.
        # 2. DontUseNativeDialog - macOS's native color picker is backed by
        #    NSColorPanel, a single OS-level shared panel Qt reuses across
        #    calls rather than creating fresh each time. Its internal
        #    window-activation bookkeeping falls out of sync after repeated
        #    open/close cycles in one session (fine the first few times,
        #    then the main window steals focus back on OK) - Qt's own
        #    cross-platform dialog is a plain, single-owned QDialog with no
        #    such shared-panel state to accumulate.
        color = QColorDialog.getColor(initial, self.window(), "Pick a color",
                                       QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if color.isValid():
            self._entry.setText(color.name())
