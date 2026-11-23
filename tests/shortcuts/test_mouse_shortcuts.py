"""Extra mouse buttons (Mouse 4 and up - Back/Forward and further side
buttons) bindable to any shortcut action: the binding format, storage, the
capture widget and editor column, and dispatch in the main window."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from maskfits import gui  # noqa: E402
from maskfits.gui import MOUSE_REPEAT_INTERVAL_MS, MaskFitsApp  # noqa: E402
from maskfits.hotkeys_window import HotkeysWindow  # noqa: E402
from maskfits.settings import Settings, load_settings, save_settings  # noqa: E402
from maskfits.settings_window import SettingsWindow  # noqa: E402
from maskfits.shortcuts import (  # noqa: E402
    actions_for_mouse,
    find_mouse_conflict,
    is_mouse_binding,
    mouse_binding,
    mouse_binding_label,
    sanitize_mouse_bindings,
)
from maskfits.shortcuts_window import ShortcutsWindow  # noqa: E402
from maskfits.widgets import MouseCapture  # noqa: E402

MB = Qt.MouseButton
NO_MOD = Qt.KeyboardModifier.NoModifier


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _mouse(button, event_type=QEvent.Type.MouseButtonPress, modifiers=NO_MOD):
    pos = QPointF(5, 5)
    buttons = MB.NoButton if event_type == QEvent.Type.MouseButtonRelease else button
    return QMouseEvent(event_type, pos, pos, button, buttons, modifiers)


def _native(widget):
    """Real input reaches a window's QWindow first, then the widget under
    the pointer - the bindings listen at that first stop."""
    handle = widget.window().windowHandle()
    assert handle is not None, "window must be shown"
    return handle


def _press(qapp, widget, button, modifiers=NO_MOD):
    event = _mouse(button, modifiers=modifiers)
    qapp.sendEvent(_native(widget), event)
    return event


def _release(qapp, widget, button):
    qapp.sendEvent(_native(widget), _mouse(button, QEvent.Type.MouseButtonRelease))


# ------------------------------------------------------------- format


def test_mouse_binding_from_button_flags():
    assert mouse_binding(MB.BackButton.value) == "Mouse4"
    assert mouse_binding(MB.ForwardButton.value) == "Mouse5"
    assert mouse_binding(MB.TaskButton.value) == "Mouse6"
    assert mouse_binding(MB.ExtraButton24.value) == "Mouse27"
    for fixed in (MB.LeftButton, MB.RightButton, MB.MiddleButton, MB.NoButton):
        assert mouse_binding(fixed.value) is None


def test_modifiers_are_written_in_a_fixed_order():
    assert mouse_binding(MB.BackButton.value, ("Shift", "Ctrl")) == "Ctrl+Shift+Mouse4"


def test_is_mouse_binding():
    for good in ("Mouse4", "Shift+Mouse5", "Meta+Ctrl+Alt+Shift+Mouse27"):
        assert is_mouse_binding(good)
    for bad in ("Mouse3", "Mouse28", "Shift+Shift+Mouse4", "Hyper+Mouse4", "Mouse", "", None, 4):
        assert not is_mouse_binding(bad)


def test_label():
    assert mouse_binding_label("Mouse4") == "Mouse 4 (Back)"
    assert mouse_binding_label("Shift+Mouse5") == "Shift+Mouse 5 (Forward)"
    assert mouse_binding_label("Mouse7") == "Mouse 7"
    assert mouse_binding_label("") == ""


def test_sanitize_drops_unknown_actions_and_bad_bindings():
    raw = {"undo": "Mouse4", "gone_action": "Mouse5", "redo": "Mouse1", "reset_mask": 7}
    assert sanitize_mouse_bindings(raw) == {"undo": "Mouse4"}
    assert sanitize_mouse_bindings(["Mouse4"]) == {}


def test_lookup_and_conflicts():
    bindings = {"undo": "Mouse4", "reset_zoom": "Mouse4", "redo": "Mouse5"}
    assert actions_for_mouse("Mouse4", bindings) == ["undo", "reset_zoom"]
    assert find_mouse_conflict("redo", "Mouse4", bindings) == "Undo"
    assert find_mouse_conflict("redo", "Mouse5", bindings) is None


def test_settings_round_trip(qapp, tmp_path):
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    save_settings(Settings(mouse_shortcuts={"undo": "Mouse4", "redo": "Shift+Mouse5"}), store)
    assert load_settings(store).mouse_shortcuts == {"undo": "Mouse4", "redo": "Shift+Mouse5"}
    assert Settings().mouse_shortcuts == {}  # nothing bound by default


# ------------------------------------------------------------ capture


@pytest.fixture
def capture(qapp):
    host = QLabel("host")
    cap = MouseCapture("", host)
    host.show()
    changes = []
    cap.bindingChanged.connect(changes.append)
    cap.changes = changes
    cap.other = QLabel("elsewhere", host)
    yield cap
    host.close()


def test_capture_binds_a_side_button_pressed_anywhere_in_the_window(qapp, capture):
    assert capture.text() == "(None)"
    capture.click()
    assert capture.isListening()
    _press(qapp, capture.other, MB.ForwardButton, Qt.KeyboardModifier.ShiftModifier)
    assert capture.value() == "Shift+Mouse5"
    assert capture.text() == "Shift+Mouse 5 (Forward)"
    assert capture.changes == ["Shift+Mouse5"]
    assert not capture.isListening()


def test_capture_ignores_presses_when_not_listening(qapp, capture):
    _press(qapp, capture.other, MB.BackButton)
    assert capture.value() == "" and capture.changes == []


def test_right_click_clears_and_left_click_cancels(qapp, capture):
    capture.setValue("Mouse4")
    capture.click()
    _press(qapp, capture.other, MB.LeftButton)
    assert capture.value() == "Mouse4" and not capture.isListening()
    capture.click()
    _press(qapp, capture.other, MB.RightButton)
    assert capture.value() == "" and capture.changes == [""]


def test_backspace_clears_and_escape_cancels(qapp, capture):
    capture.setValue("Mouse4")
    capture.click()
    # Called directly: through QApplication, a leftover window from another
    # test could take Escape as its own shortcut first.
    capture.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, NO_MOD))
    assert capture.value() == "Mouse4" and not capture.isListening()
    capture.click()
    capture.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, NO_MOD))
    assert capture.value() == ""


# ------------------------------------------------------------- editor


def test_editor_has_a_mouse_column_and_saves_it(qapp):
    win = ShortcutsWindow({}, mouse_bindings={"undo": "Mouse4"})
    assert win._mouse_captures["undo"].value() == "Mouse4"
    assert win._mouse_captures["redo"].value() == ""
    saved = []
    win.mouse_shortcuts_saved.connect(saved.append)
    win._mouse_captures["redo"].bindingChanged.emit("Mouse5")
    win._save()
    assert saved == [{"undo": "Mouse4", "redo": "Mouse5"}]


def test_editor_warns_about_a_shared_mouse_button(qapp):
    win = ShortcutsWindow({}, mouse_bindings={"undo": "Mouse4"})
    win._mouse_captures["redo"].bindingChanged.emit("Mouse4")
    assert "Undo" in win._conflict_label.text()
    assert "Mouse 4 (Back)" in win._conflict_label.text()


def test_editor_reset_clears_mouse_bindings(qapp):
    win = ShortcutsWindow({}, mouse_bindings={"undo": "Mouse4"})
    win._reset_all()
    assert win.mouse_bindings == {}
    assert win._mouse_captures["undo"].value() == ""


def test_settings_window_keeps_the_mouse_bindings(qapp):
    win = SettingsWindow(Settings(mouse_shortcuts={"undo": "Mouse4"}))
    assert win._current_settings().mouse_shortcuts == {"undo": "Mouse4"}
    win._on_mouse_shortcuts_saved({"redo": "Mouse5"})
    assert win._current_settings().mouse_shortcuts == {"redo": "Mouse5"}


def test_hotkeys_popup_lists_bound_buttons(qapp):
    win = HotkeysWindow({}, mouse_bindings={"reset_zoom": "Mouse5"})
    texts = {lbl.text() for lbl in win.findChildren(QLabel)}
    assert "Mouse 5 (Forward)" in texts


# ----------------------------------------------------------- dispatch


def _app(**bindings):
    win = MaskFitsApp([], settings=Settings(mouse_shortcuts=bindings))
    win.show()
    return win


def test_bound_button_runs_its_action_and_is_consumed(qapp):
    win = _app(toggle_center_cross="Mouse4")
    before = win.show_center_cross
    event = _press(qapp, win.canvas, MB.BackButton)
    assert win.show_center_cross is (not before)
    assert event.isAccepted()
    _release(qapp, win.canvas, MB.BackButton)


def test_quick_second_press_arrives_as_double_click_and_still_counts(qapp):
    win = _app(toggle_center_cross="Mouse4")
    before = win.show_center_cross
    qapp.sendEvent(_native(win), _mouse(MB.BackButton, QEvent.Type.MouseButtonDblClick))
    assert win.show_center_cross is (not before)


def test_unbound_button_and_wrong_modifiers_do_nothing(qapp):
    win = _app(toggle_center_cross="Shift+Mouse4")
    before = win.show_center_cross
    _press(qapp, win.canvas, MB.BackButton)  # no Shift
    _press(qapp, win.canvas, MB.ForwardButton, Qt.KeyboardModifier.ShiftModifier)
    assert win.show_center_cross is before
    _press(qapp, win.canvas, MB.BackButton, Qt.KeyboardModifier.ShiftModifier)
    assert win.show_center_cross is (not before)


def test_bindings_only_fire_in_their_own_window(qapp):
    win = _app(toggle_center_cross="Mouse4")
    other = QLabel("another window")
    other.show()
    before = win.show_center_cross
    _press(qapp, other, MB.BackButton)
    assert win.show_center_cross is before
    other.close()


def test_menu_bound_action_can_be_on_a_mouse_button(qapp):
    win = _app(show_hotkeys="Mouse5")
    _press(qapp, win.canvas, MB.ForwardButton)
    assert win._hotkeys_window is not None
    win._hotkeys_window.close()


def test_saved_settings_apply_without_restart(qapp):
    win = _app()
    win.settings.mouse_shortcuts = {"toggle_center_cross": "Mouse6"}
    before = win.show_center_cross
    _press(qapp, win.canvas, MB.TaskButton)
    assert win.show_center_cross is (not before)


def test_holding_a_grow_button_repeats_and_speeds_up(qapp, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(gui.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: MB.BackButton))
    win = _app(grow_shape="Mouse4")
    win.zoom_mult = 1.0
    win.radius = 10.0
    _press(qapp, win.canvas, MB.BackButton)
    assert win.radius == 15.0
    assert win._mouse_repeat_timer.isActive()
    for _ in range(20):  # 1 s of repeats
        now[0] += MOUSE_REPEAT_INTERVAL_MS / 1000
        win._repeat_mouse_actions()
    assert win._mouse_repeat_timer.interval() == MOUSE_REPEAT_INTERVAL_MS
    before = win.radius
    now[0] += MOUSE_REPEAT_INTERVAL_MS / 1000
    win._repeat_mouse_actions()
    assert win.radius - before > 5.0  # accelerated past the 5 px base step

    _release(qapp, win.canvas, MB.BackButton)
    assert not win._mouse_repeat_timer.isActive()
    assert win._hold_state is None  # the next press starts slow again


def test_repeat_stops_if_the_release_was_missed(qapp, monkeypatch):
    win = _app(grow_shape="Mouse4")
    _press(qapp, win.canvas, MB.BackButton)
    monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: MB.NoButton))
    radius = win.radius
    win._repeat_mouse_actions()
    assert win.radius == radius
    assert not win._mouse_repeat_timer.isActive()


def test_non_repeating_actions_do_not_start_the_timer(qapp):
    win = _app(toggle_center_cross="Mouse4")
    _press(qapp, win.canvas, MB.BackButton)
    assert not win._mouse_repeat_timer.isActive()
