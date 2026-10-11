"""Coverage for hold-to-accelerate on the shape hotkeys (size/thickness,
ellipticity, angle): the step ramps from 1x to HOLD_MAX_FACTOR the longer
the key stays held, and a real key release resets it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits import gui  # noqa: E402
from maskfits.gui import HOLD_GAP_S, HOLD_MAX_FACTOR, HOLD_RAMP_S, MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(gui.time, "monotonic", lambda: now[0])
    return now


@pytest.fixture
def win(qapp):
    win = MaskFitsApp([], settings=Settings(mode="ellipse"))
    win.zoom_mult = 1.0
    return win


def _hold(clock, press, seconds, tick=0.05):
    """Simulates key auto-repeat: calls press() every `tick` seconds for
    `seconds`, returning the last call's result."""
    result = None
    for _ in range(round(seconds / tick)):
        clock[0] += tick
        result = press()
    return result


def test_factor_ramps_from_1x_to_the_max_while_held(win, clock):
    assert win._hold_factor("size1") == 1.0
    half = _hold(clock, lambda: win._hold_factor("size1"), HOLD_RAMP_S / 2)
    assert half == pytest.approx(1 + (HOLD_MAX_FACTOR - 1) / 2)
    full = _hold(clock, lambda: win._hold_factor("size1"), HOLD_RAMP_S / 2)
    assert full == pytest.approx(HOLD_MAX_FACTOR)


def test_factor_never_exceeds_the_max(win, clock):
    win._hold_factor("size1")
    for _ in range(100):
        clock[0] += 0.1
        factor = win._hold_factor("size1")
    assert factor == HOLD_MAX_FACTOR == 3.0


def test_a_pause_longer_than_the_gap_starts_over(win, clock):
    win._hold_factor("size1")
    clock[0] += 0.5
    assert win._hold_factor("size1") > 1.0
    clock[0] += HOLD_GAP_S + 0.01
    assert win._hold_factor("size1") == 1.0


def test_a_different_key_starts_over(win, clock):
    win._hold_factor("size1")
    clock[0] += 0.5
    assert win._hold_factor("size-1") == 1.0


def test_a_real_key_release_resets_but_an_auto_repeat_one_does_not(qapp, win, clock):
    win._hold_factor("size1")
    clock[0] += 0.5
    repeat = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_E, Qt.KeyboardModifier.NoModifier, "e", True)
    qapp.sendEvent(win, repeat)
    assert win._hold_factor("size1") > 1.0

    release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_E, Qt.KeyboardModifier.NoModifier, "e", False)
    qapp.sendEvent(win, release)
    assert win._hold_factor("size1") == 1.0


def test_held_size_key_takes_3x_steps(win, clock):
    win.radius = 10.0
    win._adjust_shape_size(1)
    assert win.radius == 15.0  # base step 5 at zoom 1
    _hold(clock, lambda: win._hold_factor("size1"), HOLD_RAMP_S)
    before = win.radius
    win._adjust_shape_size(1)
    assert win.radius == before + 15.0  # 3 x 5


def test_held_angle_and_ellipticity_keys_take_3x_steps(win, clock):
    win.angle, win.ellipticity = 0, 0
    win._adjust_angle(1)
    win._adjust_ellipticity(1)
    assert (win.angle, win.ellipticity) == (2, 2)
    _hold(clock, lambda: win._hold_factor("angle1"), HOLD_RAMP_S)
    win._adjust_angle(1)
    assert win.angle == 2 + 6
    _hold(clock, lambda: win._hold_factor("ellipticity1"), HOLD_RAMP_S)
    win._adjust_ellipticity(1)
    assert win.ellipticity == 2 + 6


def test_held_thickness_key_takes_3x_steps(win, clock):
    win.set_tool("line")
    win.thickness = 100
    win._adjust_shape_size(-1)
    assert win.thickness == 95
    _hold(clock, lambda: win._hold_factor("size-1"), HOLD_RAMP_S)
    win._adjust_shape_size(-1)
    assert win.thickness == 80
