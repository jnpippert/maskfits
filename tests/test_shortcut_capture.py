import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.widgets import ShortcutCapture  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _press(key, mods=Qt.KeyboardModifier.NoModifier, *, auto_repeat=False):
    return QKeyEvent(QEvent.Type.KeyPress, key, mods, autorep=auto_repeat)


def _release(key, mods=Qt.KeyboardModifier.NoModifier, *, auto_repeat=False):
    return QKeyEvent(QEvent.Type.KeyRelease, key, mods, autorep=auto_repeat)


def test_initial_value_shown(qapp):
    cap = ShortcutCapture("Ctrl+Z")
    assert cap.text() == "Ctrl+Z"
    assert cap.value() == "Ctrl+Z"


def test_empty_initial_value_shows_placeholder(qapp):
    cap = ShortcutCapture("")
    assert cap.text() == "(None)"


def test_click_enters_listening_state(qapp):
    cap = ShortcutCapture("Ctrl+Z")
    cap._start_listening()
    assert cap._listening is True
    assert cap.text() == "Press A Key..."


def test_pressing_a_plain_key_completes_capture(qapp):
    cap = ShortcutCapture("R")
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_X))
    assert cap.value() == "X"
    assert cap.text() == "X"
    assert cap._listening is False


def test_pressing_a_combo_completes_capture(qapp):
    cap = ShortcutCapture("R")
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))
    assert cap.value() == "Ctrl+Shift+A"


def test_bare_modifier_keeps_listening(qapp):
    cap = ShortcutCapture("C")
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_Shift, Qt.KeyboardModifier.ShiftModifier))
    assert cap._listening is True
    assert cap.value() == "C"  # unchanged


def test_key_sequence_changed_signal_emits_new_value(qapp):
    cap = ShortcutCapture("R")
    received = []
    cap.keySequenceChanged.connect(received.append)
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_X))
    assert received == ["X"]


def test_key_press_while_not_listening_does_not_change_value(qapp):
    cap = ShortcutCapture("R")
    cap.keyPressEvent(_press(Qt.Key.Key_X))
    assert cap.value() == "R"


def test_set_value_updates_text_and_stops_listening(qapp):
    cap = ShortcutCapture("R")
    cap._start_listening()
    cap.setValue("Ctrl+Q")
    assert cap.value() == "Ctrl+Q"
    assert cap.text() == "Ctrl+Q"
    assert cap._listening is False


def test_focus_out_while_listening_reverts_display(qapp):
    from PySide6.QtGui import QFocusEvent

    cap = ShortcutCapture("R")
    cap._start_listening()
    assert cap.text() == "Press A Key..."
    cap.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
    assert cap._listening is False
    assert cap.text() == "R"


# --------------------------------------------------------- Escape handling
#
# Escape is a real, bindable key (the default for "cancel a pending line
# click") - a quick tap must bind it like any other key; only a HELD Escape
# (>= ESCAPE_HOLD_MS) cancels listening. The timer itself is never actually
# waited out in these tests (no real 1s sleep) - "held past the threshold"
# is simulated by invoking the timeout handler directly, exactly what
# QTimer would do when it fires for real.


def test_escape_tap_starts_a_hold_timer_without_completing(qapp):
    cap = ShortcutCapture("B")
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_Escape))
    assert cap._listening is True  # not yet resolved either way
    assert cap._escape_timer is not None
    assert cap.value() == "B"  # unchanged so far


def test_escape_quick_tap_binds_escape_itself(qapp):
    cap = ShortcutCapture("B")
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_Escape))
    cap.keyReleaseEvent(_release(Qt.Key.Key_Escape))
    assert cap._listening is False
    assert cap.value() == "Esc"
    assert cap.text() == "Esc"


def test_escape_quick_tap_emits_key_sequence_changed(qapp):
    cap = ShortcutCapture("B")
    received = []
    cap.keySequenceChanged.connect(received.append)
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_Escape))
    cap.keyReleaseEvent(_release(Qt.Key.Key_Escape))
    assert received == ["Esc"]


def test_escape_held_past_threshold_cancels_back_to_previous_value(qapp):
    cap = ShortcutCapture("B")
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_Escape))
    assert cap._escape_timer is not None
    # Simulate the hold timer firing (the real 1s elapsing) rather than
    # actually sleeping in the test.
    cap._on_escape_held()
    assert cap._listening is False
    assert cap.value() == "B"
    assert cap.text() == "B"


def test_escape_auto_repeat_presses_do_not_restart_the_timer(qapp):
    cap = ShortcutCapture("B")
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_Escape))
    first_timer = cap._escape_timer
    cap.keyPressEvent(_press(Qt.Key.Key_Escape, auto_repeat=True))
    cap.keyPressEvent(_press(Qt.Key.Key_Escape, auto_repeat=True))
    assert cap._escape_timer is first_timer
    assert cap._listening is True


def test_escape_release_after_held_cancel_does_not_reopen_capture(qapp):
    """Once the hold timer has already cancelled listening, a release event
    for the (still physically held) Escape key that arrives afterward must
    not be treated as a fresh quick-tap completion."""
    cap = ShortcutCapture("B")
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_Escape))
    cap._on_escape_held()
    cap.keyReleaseEvent(_release(Qt.Key.Key_Escape))
    assert cap.value() == "B"


def test_non_escape_key_is_unaffected_by_escape_handling(qapp):
    cap = ShortcutCapture("B")
    cap._start_listening()
    cap.keyPressEvent(_press(Qt.Key.Key_X))
    assert cap.value() == "X"
    assert cap._escape_timer is None
