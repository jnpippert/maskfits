"""Coverage for hold-to-accelerate on the shape hotkeys (size/thickness,
ellipticity, angle): the step follows SHAPE_HOLD_RAMP the longer the key
stays held, and a real key release resets it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits import gui  # noqa: E402
from maskfits.gui import HOLD_GAP_S, SHAPE_HOLD_RAMP, MaskFitsApp  # noqa: E402
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


def test_factor_never_exceeds_the_max(win, clock):
    win._hold_factor("size1")
    for _ in range(100):
        clock[0] += 0.1
        factor = win._hold_factor("size1")
    assert factor == SHAPE_HOLD_RAMP[-1][1] == 10.0


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


def test_ramp_speeds_up_early_and_keeps_going():
    assert SHAPE_HOLD_RAMP[0] == (0.0, 1.0)
    factors = [f for _, f in SHAPE_HOLD_RAMP]
    assert factors == sorted(factors)  # never slows down
    assert dict(SHAPE_HOLD_RAMP)[1.0] >= 5.0  # well past the old 3x max within 1 s
    assert SHAPE_HOLD_RAMP[-1][1] >= 10.0


def test_hold_factor_follows_the_ramp(win, clock):
    start = clock[0]
    assert win._hold_factor("size1") == 1.0
    for t, expected in [(0.4, 2.0), (0.7, 3.5), (1.0, 5.0), (2.0, 10.0), (5.0, 10.0)]:
        factor = None
        while clock[0] < start + t - 1e-9:  # auto-repeat every 50 ms
            clock[0] = min(clock[0] + 0.05, start + t)
            factor = win._hold_factor("size1")
        assert factor == pytest.approx(expected)


def test_held_size_key_steps(win, clock):
    win.radius = 10.0
    win._adjust_shape_size(1)
    assert win.radius == 15.0  # base step 5 at zoom 1
    _hold(clock, lambda: win._hold_factor("size1"), 1.0)
    before = win.radius
    win._adjust_shape_size(1)
    assert win.radius == before + 25.0  # 5x at 1 s
    _hold(clock, lambda: win._hold_factor("size1"), 1.0)
    before = win.radius
    win._adjust_shape_size(1)
    assert win.radius == before + 50.0  # 10x from 2 s on


@pytest.mark.parametrize("attr, adjust, key", [
    ("angle", "_adjust_angle", "angle1"),
    ("ellipticity", "_adjust_ellipticity", "ellipticity1"),
])
def test_held_angle_and_ellipticity_keys_use_the_same_ramp(win, clock, attr, adjust, key):
    setattr(win, attr, 0)
    getattr(win, adjust)(1)
    assert getattr(win, attr) == 2  # base step 2
    _hold(clock, lambda: win._hold_factor(key), 1.0)
    getattr(win, adjust)(1)
    assert getattr(win, attr) == 2 + 10  # 5x at 1 s
    _hold(clock, lambda: win._hold_factor(key), 1.0)
    getattr(win, adjust)(1)
    assert getattr(win, attr) == 12 + 20  # 10x from 2 s


def test_held_thickness_key_uses_the_ramp(win, clock):
    win.set_tool("line")
    win.thickness = 100
    win._adjust_shape_size(-1)
    assert win.thickness == 95
    _hold(clock, lambda: win._hold_factor("size-1"), 1.0)
    win._adjust_shape_size(-1)
    assert win.thickness == 70  # 5x at 1 s
