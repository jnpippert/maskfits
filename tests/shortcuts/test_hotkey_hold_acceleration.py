"""Coverage for hold-to-accelerate on the shape hotkeys: ellipticity and
angle ramp from 1x to HOLD_MAX_FACTOR, size/thickness follow the faster
SIZE_HOLD_RAMP, and a real key release resets either."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits import gui  # noqa: E402
from maskfits.gui import HOLD_GAP_S, HOLD_MAX_FACTOR, HOLD_RAMP_S, SIZE_HOLD_RAMP, MaskFitsApp  # noqa: E402
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


def test_size_ramp_speeds_up_early_and_keeps_going():
    assert SIZE_HOLD_RAMP[0] == (0.0, 1.0)
    factors = [f for _, f in SIZE_HOLD_RAMP]
    assert factors == sorted(factors)  # never slows down
    assert dict(SIZE_HOLD_RAMP)[1.0] >= 5.0  # well past the old 3x max within 1 s
    assert SIZE_HOLD_RAMP[-1][1] >= 10.0


def test_size_hold_factor_follows_the_ramp(win, clock):
    start = clock[0]
    assert win._size_hold_factor("size1") == 1.0
    for t, expected in [(0.4, 2.0), (0.7, 3.5), (1.0, 5.0), (2.0, 10.0), (5.0, 10.0)]:
        factor = None
        while clock[0] < start + t - 1e-9:  # auto-repeat every 50 ms
            clock[0] = min(clock[0] + 0.05, start + t)
            factor = win._size_hold_factor("size1")
        assert factor == pytest.approx(expected)


def test_held_size_key_steps(win, clock):
    win.radius = 10.0
    win._adjust_shape_size(1)
    assert win.radius == 15.0  # base step 5 at zoom 1
    _hold(clock, lambda: win._size_hold_factor("size1"), 1.0)
    before = win.radius
    win._adjust_shape_size(1)
    assert win.radius == before + 25.0  # 5x at 1 s
    _hold(clock, lambda: win._size_hold_factor("size1"), 1.0)
    before = win.radius
    win._adjust_shape_size(1)
    assert win.radius == before + 50.0  # 10x from 2 s on


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


def test_held_thickness_key_uses_the_size_ramp(win, clock):
    win.set_tool("line")
    win.thickness = 100
    win._adjust_shape_size(-1)
    assert win.thickness == 95
    _hold(clock, lambda: win._size_hold_factor("size-1"), 1.0)
    win._adjust_shape_size(-1)
    assert win.thickness == 70  # 5x at 1 s
