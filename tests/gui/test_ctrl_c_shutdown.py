"""Coverage for Ctrl+C in the launching console (gui._exec_with_sigint_handling):
the running app shuts down with a message and exit code 130 instead of
ignoring it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import signal
import sys
import threading

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits import gui  # noqa: E402
from maskfits.gui import SIGINT_EXIT_CODE, SIGINT_MESSAGE, _exec_with_sigint_handling  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGINT"), reason="no SIGINT on this platform")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def app(qapp, monkeypatch):
    # Don't close every other test's windows in this worker process.
    closed = []
    monkeypatch.setattr(qapp, "closeAllWindows", lambda: closed.append(True))
    qapp.closed = closed
    # Fail instead of hanging forever if Ctrl+C is never handled.
    QTimer.singleShot(5000, lambda: qapp.exit(-1))
    return qapp


# On Windows os.kill(pid, SIGINT) terminates the process instead of delivering a signal.
@pytest.mark.skipif(sys.platform == "win32", reason="os.kill can't deliver SIGINT on Windows")
def test_a_real_sigint_stops_the_event_loop_with_a_message(app, capfd):
    # Sent from another thread, like a real Ctrl+C arriving while Qt's C++
    # event loop has control - only the wakeup timer lets Python see it.
    threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()

    code = _exec_with_sigint_handling(app)

    assert code == SIGINT_EXIT_CODE == 130
    assert SIGINT_MESSAGE in capfd.readouterr().err
    assert SIGINT_MESSAGE == "maskfits: shutdown by keyboard input (Ctrl+C)"
    assert app.closed == [True]


def test_previous_handler_is_restored_afterwards(app):
    before = signal.getsignal(signal.SIGINT)
    QTimer.singleShot(50, lambda: app.exit(0))

    assert _exec_with_sigint_handling(app) == 0
    assert signal.getsignal(signal.SIGINT) is before


def test_during_shutdown_a_second_ctrl_c_kills_the_default_way(app, monkeypatch):
    seen = {}

    def exit_and_record(code):
        seen["handler_during_shutdown"] = signal.getsignal(signal.SIGINT)
        QApplication.exit(code)

    monkeypatch.setattr(app, "exit", exit_and_record)
    QTimer.singleShot(50, lambda: signal.raise_signal(signal.SIGINT))

    _exec_with_sigint_handling(app)

    assert seen["handler_during_shutdown"] is signal.SIG_DFL


def test_run_gui_uses_it(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gui, "_exec_with_sigint_handling", lambda app: calls.append(app) or 0)
    monkeypatch.setattr(gui.MaskFitsApp, "start_background_update_check", lambda self: None)
    assert gui.run_gui([]) == 0
    assert len(calls) == 1
