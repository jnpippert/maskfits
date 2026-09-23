"""Regression coverage for a real bug: the main window opened at a fixed
1400x980, which was fine on the MacBook Pro used for development but was
larger than a Full HD (1920x1080, with taskbar/menu eating into that)
screen's available space on another machine, especially under Windows/WSL.
MaskFitsApp._size_to_screen() now clamps to the screen's actual available
geometry and centers the window, instead of always using the fixed size.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeScreen:
    def __init__(self, rect: QRect):
        self._rect = rect

    def availableGeometry(self) -> QRect:
        return self._rect


def test_window_shrinks_to_fit_a_small_full_hd_screen(qapp, monkeypatch, tmp_path):
    win = MaskFitsApp([], settings=Settings())
    monkeypatch.setattr(win, "screen", lambda: _FakeScreen(QRect(0, 0, 1920, 1040)))

    win._size_to_screen(1400, 980)

    assert win.width() <= 1920 - 60
    assert win.height() <= 1040 - 60


def test_window_uses_preferred_size_on_a_large_screen(qapp, monkeypatch):
    win = MaskFitsApp([], settings=Settings())
    monkeypatch.setattr(win, "screen", lambda: _FakeScreen(QRect(0, 0, 2560, 1600)))

    win._size_to_screen(1400, 980)

    assert win.width() == 1400
    assert win.height() == 980


def test_window_never_shrinks_below_a_sane_minimum(qapp, monkeypatch):
    win = MaskFitsApp([], settings=Settings())
    monkeypatch.setattr(win, "screen", lambda: _FakeScreen(QRect(0, 0, 640, 480)))

    win._size_to_screen(1400, 980)

    assert win.width() >= 640
    assert win.height() >= 480


def test_window_is_centered_on_the_available_geometry(qapp, monkeypatch):
    win = MaskFitsApp([], settings=Settings())
    monkeypatch.setattr(win, "screen", lambda: _FakeScreen(QRect(0, 0, 1920, 1040)))

    win._size_to_screen(1400, 980)

    expected_x = (1920 - win.width()) // 2
    expected_y = (1040 - win.height()) // 2
    assert abs(win.x() - expected_x) <= 1
    assert abs(win.y() - expected_y) <= 1


def test_window_gets_an_explicit_minimum_size(qapp, monkeypatch):
    """A window manager maximizing/resizing the window to the real screen
    size must never get a bigger buffer back than it asked for - without an
    explicit minimum, QMainWindow can refuse to shrink below whatever its
    layout naturally needs, which triggered a Wayland protocol violation
    under WSLg (see this file's module docstring)."""
    win = MaskFitsApp([], settings=Settings())
    monkeypatch.setattr(win, "screen", lambda: _FakeScreen(QRect(0, 0, 1920, 1040)))

    win._size_to_screen(1400, 980)

    assert win.minimumWidth() <= 640
    assert win.minimumHeight() <= 480
