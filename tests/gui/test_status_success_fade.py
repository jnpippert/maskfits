"""Coverage for the status bar's green "success" tint (e.g. after a mask
export) fading back to the normal text color after STATUS_SUCCESS_MS."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import STATUS_SUCCESS_MS, MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    return MaskFitsApp([], settings=Settings())


def test_success_status_is_green_and_schedules_the_fade(win):
    win._set_status("exported mask to x.fits", success=True)
    assert win.status.property("state") == "success"
    assert win._status_timer.isActive()
    assert win._status_timer.interval() == STATUS_SUCCESS_MS == 1500


def test_fade_restores_the_normal_color_but_keeps_the_message(win):
    win._set_status("exported mask to x.fits", success=True)
    win._status_timer.timeout.emit()
    assert win.status.property("state") == ""
    assert win.status.text() == "exported mask to x.fits"


def test_a_later_plain_status_cancels_the_pending_fade(win):
    win._set_status("exported mask to x.fits", success=True)
    win._set_status("loaded y.fits")
    assert win.status.property("state") == ""
    assert not win._status_timer.isActive()


def test_a_second_success_restarts_the_full_duration(win):
    win._set_status("exported mask to x.fits", success=True)
    first = win._status_timer.remainingTime()
    win._set_status("exported mask to y.fits", success=True)
    assert win._status_timer.isActive()
    assert win._status_timer.remainingTime() >= first
    assert win.status.property("state") == "success"
