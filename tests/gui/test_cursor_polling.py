"""Coverage for MaskFitsApp._poll_cursor: the shape outline, magnifier and
readouts follow the cursor even when no mouse-move events arrive (macOS
only delivers plain mouse moves to the active app, and an app launched by
another application may never become active; a cursor already resting on
the image at startup also sends none) - but only while maskfits is the
active app, so the cursor moving over another app in front of it doesn't
move anything in the background."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QCursor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import CURSOR_POLL_MS, MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def cursor(qapp):
    """Moves the (global) cursor for a test, then parks it far away again so
    later tests in this worker never find it over their own canvas."""
    def move(point: QPoint) -> None:
        QCursor.setPos(point)

    yield move
    QCursor.setPos(QPoint(-10000, -10000))


@pytest.fixture
def win(qapp, tmp_path):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.zeros((64, 64), dtype=np.float32)).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.resize(1000, 700)
    win.show()
    qapp.processEvents()
    return win


def test_poll_timer_runs(win):
    assert win._cursor_timer.isActive()
    assert win._cursor_timer.interval() == CURSOR_POLL_MS


def test_cursor_over_the_canvas_updates_without_a_move_event(win, cursor):
    assert win._cursor_canvas_pos is None
    cursor(win.canvas.mapToGlobal(QPoint(200, 150)))

    win._poll_cursor()

    assert win._cursor_canvas_pos == (200.0, 150.0)
    assert win._cursor_img_pos == pytest.approx(win.canvas_to_img(200, 150))
    assert win.readout["x"].text() != ""


def test_unchanged_position_does_not_recompute(win, cursor, monkeypatch):
    cursor(win.canvas.mapToGlobal(QPoint(200, 150)))
    win._poll_cursor()
    calls = []
    monkeypatch.setattr(win, "_on_motion", lambda *a: calls.append(a))

    win._poll_cursor()

    assert calls == []


def test_leaving_the_canvas_keeps_the_last_position(win, cursor):
    """The outline stays put while the sidebar sliders are adjusted."""
    cursor(win.canvas.mapToGlobal(QPoint(200, 150)))
    win._poll_cursor()
    cursor(win.sidebar_container.mapToGlobal(QPoint(20, 20)))

    win._poll_cursor()

    assert win._cursor_canvas_pos == (200.0, 150.0)


def test_own_window_on_top_of_the_canvas_is_ignored(win, cursor, qapp):
    from PySide6.QtWidgets import QWidget

    cover = QWidget()
    cover.setGeometry(win.canvas.mapToGlobal(QPoint(150, 100)).x(), win.canvas.mapToGlobal(QPoint(150, 100)).y(),
                      200, 200)
    cover.show()
    qapp.processEvents()
    cursor(win.canvas.mapToGlobal(QPoint(200, 150)))

    win._poll_cursor()

    assert win._cursor_canvas_pos is None
    cover.close()


def test_does_nothing_without_an_image(qapp, cursor):
    win = MaskFitsApp([], settings=Settings())
    win.resize(1000, 700)
    win.show()
    qapp.processEvents()
    cursor(win.canvas.mapToGlobal(QPoint(200, 150)))

    win._poll_cursor()

    assert win._cursor_canvas_pos is None


# ------------------------------------------------------- another app in front


@pytest.fixture
def inactive(monkeypatch):
    """Pretends another application is in front."""
    from PySide6.QtCore import Qt

    state = {"value": Qt.ApplicationState.ApplicationInactive}
    monkeypatch.setattr(QApplication, "applicationState", staticmethod(lambda: state["value"]))
    return state


def test_does_not_follow_the_cursor_while_another_app_is_in_front(win, cursor, inactive):
    cursor(win.canvas.mapToGlobal(QPoint(200, 150)))
    win._poll_cursor()
    assert win._cursor_canvas_pos is None


def test_follows_again_once_maskfits_is_back_in_front(win, cursor, inactive):
    from PySide6.QtCore import Qt

    cursor(win.canvas.mapToGlobal(QPoint(200, 150)))
    win._poll_cursor()
    inactive["value"] = Qt.ApplicationState.ApplicationActive
    win._poll_cursor()
    assert win._cursor_canvas_pos == (200.0, 150.0)


def test_a_click_on_the_image_while_inactive_turns_it_back_on(win, cursor, inactive):
    """The launched-by-another-app case: clicks arrive although macOS never
    makes maskfits the active app."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.mouseClick(win.canvas, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 10))
    cursor(win.canvas.mapToGlobal(QPoint(200, 150)))
    win._poll_cursor()
    assert win._cursor_canvas_pos == (200.0, 150.0)


def test_leaving_the_window_ends_that_until_the_next_click(win, cursor, inactive):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.mouseClick(win.canvas, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 10))
    cursor(win.canvas.mapToGlobal(QPoint(200, 150)))
    win._poll_cursor()
    cursor(win.frameGeometry().bottomRight() + QPoint(50, 50))  # outside the window
    win._poll_cursor()
    cursor(win.canvas.mapToGlobal(QPoint(220, 160)))  # back over the image, no new click
    win._poll_cursor()
    assert win._cursor_canvas_pos == (200.0, 150.0)
