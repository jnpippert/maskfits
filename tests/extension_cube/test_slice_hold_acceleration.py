"""Holding Left/Right in a cube steps through slices faster: 1 slice per
step, ramping to 5 after 2 seconds of holding; a key release resets it."""

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
from maskfits.gui import SLICE_HOLD_MAX_FACTOR, SLICE_HOLD_RAMP_S, MaskFitsApp  # noqa: E402
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
    fits.PrimaryHDU(np.zeros((400, 8, 8), dtype=np.float32)).writeto(path)
    return MaskFitsApp([str(path)], settings=Settings())


def _hold(win, clock, press, seconds, tick=0.05):
    """Key auto-repeat: press() every `tick` seconds for `seconds`."""
    for _ in range(round(seconds / tick)):
        clock[0] += tick
        press()


def test_limits():
    assert (SLICE_HOLD_MAX_FACTOR, SLICE_HOLD_RAMP_S) == (5.0, 2.0)


def test_a_single_press_steps_one_slice(win, clock):
    win.next_image()
    assert win.image.slice_index == 1
    clock[0] += 5  # a separate press later
    win.prev_image()
    assert win.image.slice_index == 0


def test_holding_ramps_up_to_five_slices_per_step(win, clock):
    win.next_image()  # press
    _hold(win, clock, win.next_image, SLICE_HOLD_RAMP_S)
    before = win.image.slice_index
    clock[0] += 0.05
    win.next_image()
    assert win.image.slice_index - before == 5
    _hold(win, clock, win.next_image, 1.0)  # stays capped at 5
    before = win.image.slice_index
    clock[0] += 0.05
    win.next_image()
    assert win.image.slice_index - before == 5


def test_halfway_through_the_ramp_is_about_3x(win, clock):
    win.next_image()
    _hold(win, clock, win.next_image, SLICE_HOLD_RAMP_S / 2 - 0.05)
    before = win.image.slice_index
    clock[0] += 0.05
    win.next_image()
    assert win.image.slice_index - before == 3


def test_holding_left_accelerates_too(win, clock):
    win.switch_slice(300)
    win.prev_image()
    _hold(win, clock, win.prev_image, SLICE_HOLD_RAMP_S)
    before = win.image.slice_index
    clock[0] += 0.05
    win.prev_image()
    assert before - win.image.slice_index == 5


def test_a_key_release_resets_the_ramp(win, clock, qapp):
    win.next_image()
    _hold(win, clock, win.next_image, SLICE_HOLD_RAMP_S)
    qapp.sendEvent(win, QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier, "", False))
    before = win.image.slice_index
    clock[0] += 0.05
    win.next_image()
    assert win.image.slice_index - before == 1


def test_stops_at_the_last_slice(win, clock):
    win.switch_slice(397)
    win.next_image()
    _hold(win, clock, win.next_image, SLICE_HOLD_RAMP_S)
    assert win.image.slice_index == 399


def test_without_a_cube_left_right_still_change_file(qapp, tmp_path, clock):
    paths = []
    for name in ("a.fits", "b.fits"):
        path = tmp_path / name
        fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(path)
        paths.append(str(path))
    win = MaskFitsApp(paths, settings=Settings())
    win.next_image()
    assert win.index == 1
