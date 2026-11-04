"""Holding Left/Right in a cube steps through slices faster: 1 slice per
step, ramping to 5 after 2 seconds of holding and on to 10 after 5; a key
release resets it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits import gui  # noqa: E402
from maskfits.gui import SLICE_HOLD_RAMP, MaskFitsApp  # noqa: E402
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
def win(qapp, tmp_path):
    path = tmp_path / "cube.fits"
    # Long enough not to hit the last slice during a 5 s+ hold at up to 10/step.
    fits.PrimaryHDU(np.zeros((4000, 4, 4), dtype=np.float32)).writeto(path)
    return MaskFitsApp([str(path)], settings=Settings())


def _hold(win, clock, press, seconds, tick=0.05):
    """Key auto-repeat: press() every `tick` seconds for `seconds`."""
    for _ in range(round(seconds / tick)):
        clock[0] += tick
        press()


RAMP_S = 2.0  # seconds to 5x


def test_limits():
    assert SLICE_HOLD_RAMP == ((0.0, 1.0), (2.0, 5.0), (5.0, 10.0))


def _step_after(win, clock, seconds, press=None):
    """Slices moved by one more key repeat after holding for `seconds`."""
    press = press or win.next_image
    press()
    _hold(win, clock, press, seconds - 0.05)
    before = win.image.slice_index
    clock[0] += 0.05
    press()
    return abs(win.image.slice_index - before)


def test_reaches_ten_slices_per_step_after_five_seconds(win, clock):
    assert _step_after(win, clock, 5.0) == 10


def test_second_stage_ramps_from_five_to_ten(win, clock):
    assert _step_after(win, clock, 3.8) == 8  # 5 + 1.8 / 3 * 5 = 8


def test_stays_at_ten_past_five_seconds(qapp, tmp_path, clock):
    path = tmp_path / "big.fits"
    fits.PrimaryHDU(np.zeros((6000, 4, 4), dtype=np.float32)).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    assert _step_after(win, clock, 8.0) == 10


def test_a_single_press_steps_one_slice(win, clock):
    win.next_image()
    assert win.image.slice_index == 1
    clock[0] += 5  # a separate press later
    win.prev_image()
    assert win.image.slice_index == 0


def test_holding_ramps_up_to_five_slices_per_step_at_two_seconds(win, clock):
    assert _step_after(win, clock, RAMP_S) == 5


def test_halfway_through_the_ramp_is_about_3x(win, clock):
    win.next_image()
    _hold(win, clock, win.next_image, RAMP_S / 2 - 0.05)
    before = win.image.slice_index
    clock[0] += 0.05
    win.next_image()
    assert win.image.slice_index - before == 3


def test_holding_left_accelerates_too(win, clock):
    win.switch_slice(300)
    win.prev_image()
    _hold(win, clock, win.prev_image, RAMP_S)
    before = win.image.slice_index
    clock[0] += 0.05
    win.prev_image()
    assert before - win.image.slice_index == 5


def test_a_key_release_resets_the_ramp(win, clock, qapp):
    win.next_image()
    _hold(win, clock, win.next_image, RAMP_S)
    qapp.sendEvent(win, QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier, "", False))
    before = win.image.slice_index
    clock[0] += 0.05
    win.next_image()
    assert win.image.slice_index - before == 1


def test_stops_at_the_last_slice(win, clock):
    win.switch_slice(3997)
    win.next_image()
    _hold(win, clock, win.next_image, RAMP_S)
    assert win.image.slice_index == 3999


def test_without_a_cube_left_right_still_change_file(qapp, tmp_path, clock):
    paths = []
    for name in ("a.fits", "b.fits"):
        path = tmp_path / name
        fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(path)
        paths.append(str(path))
    win = MaskFitsApp(paths, settings=Settings())
    win.next_image()
    assert win.index == 1
