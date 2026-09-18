"""Regression coverage for a real bug: pressing the 1/2/3/4 hotkeys changed
the underlying value correctly (ellipticity/angle in ellipse mode, line
style in satellite mode) and the mask itself came out right, but the UI
didn't reflect it - the ellipticity/angle sliders stayed at their old
position, the satellite style buttons didn't highlight the new selection,
and the preview shape only actually redrew once the mouse moved next (since
nothing triggered a repaint from the hotkey itself).

Root causes, both in gui.py:
- _sync_tool_option_widgets() only ever updated the radius/thickness slider,
  never ellipticity/angle/line_style - because it was never taught to.
- _adjust_ellipticity()/_adjust_angle() (the 1/2/3/4 handlers in ellipse
  mode) never called _refresh_active_preview() at all, unlike every other
  tool-option change (_on_radius_changed, _on_thickness_changed, ...) or
  even the E/W shape-size hotkey (_adjust_shape_size).
- The satellite mode's line-style SegmentedControl was built as a local
  variable in _rebuild_tool_options() and never stored anywhere, so nothing
  could have updated it even if _sync_tool_option_widgets() had tried to.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_fits(path, size=64):
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)
    fits.PrimaryHDU(data).writeto(path, overwrite=True)


@pytest.fixture
def win(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    window = MaskFitsApp([str(path)], settings=Settings())
    window.show()
    qapp.processEvents()
    return window


def test_ellipticity_hotkey_updates_the_slider(win):
    before = win.ellipticity
    win._hotkey_digit(2)  # raise ellipticity
    assert win.ellipticity != before
    assert win._ellipticity_slider.value() == win.ellipticity


def test_ellipticity_hotkey_decrease_updates_the_slider(win):
    win._hotkey_digit(2)  # move off 0 first so a decrease is observable
    before = win.ellipticity
    win._hotkey_digit(1)
    assert win.ellipticity != before
    assert win._ellipticity_slider.value() == win.ellipticity


def test_angle_hotkey_updates_the_slider(win):
    before = win.angle
    win._hotkey_digit(4)  # raise angle
    assert win.angle != before
    assert win._angle_slider.value() == win.angle


def test_angle_hotkey_decrease_updates_the_slider(win):
    win._hotkey_digit(4)
    before = win.angle
    win._hotkey_digit(3)
    assert win.angle != before
    assert win._angle_slider.value() == win.angle


def test_ellipticity_hotkey_triggers_an_immediate_repaint(win):
    calls = []
    win.canvas.update = lambda: calls.append(True)
    win._hotkey_digit(2)
    assert calls, "the preview must redraw immediately, not wait for the next mouse move"


def test_angle_hotkey_triggers_an_immediate_repaint(win):
    calls = []
    win.canvas.update = lambda: calls.append(True)
    win._hotkey_digit(4)
    assert calls


def test_line_style_control_reference_is_stored(win):
    win.set_tool("line")
    assert win._line_style_control is not None
    assert win._line_style_control.value() == win.line_style


def test_line_style_hotkey_updates_the_segmented_control(win):
    win.set_tool("line")
    win._hotkey_digit(2)  # "arrow"
    assert win.line_style == "arrow"
    assert win._line_style_control.value() == "arrow"


def test_line_style_hotkey_visually_checks_the_right_button(win):
    win.set_tool("line")
    win._hotkey_digit(2)  # "arrow"
    assert win._line_style_control._buttons["arrow"].isChecked() is True
    assert win._line_style_control._buttons["segment"].isChecked() is False


def test_line_style_hotkey_triggers_an_immediate_repaint(win):
    win.set_tool("line")
    calls = []
    win.canvas.update = lambda: calls.append(True)
    win._hotkey_digit(3)  # "line"
    assert calls


def test_line_style_control_cleared_on_switch_back_to_ellipse(win):
    win.set_tool("line")
    assert win._line_style_control is not None
    win.set_tool("ellipse")
    assert win._line_style_control is None


def test_switching_tools_hides_the_old_tool_options_immediately(qapp, win):
    """Regression test for a real visual bug: _rebuild_tool_options() used
    to only takeAt() + deleteLater() the old widgets - takeAt() detaches
    from the layout but does not hide the widget, and deleteLater()'s
    actual deletion is a deferred event with no guarantee of running before
    the next paint. Switching tools back and forth could leave the old
    tool's labels ("radius (px):", "ellipticity (%):", ...) visibly
    overlapping the new tool's ("thickness (px):", "style", ...) until the
    deferred delete eventually caught up."""
    from PySide6.QtWidgets import QLabel

    win.set_tool("line")
    qapp.processEvents()
    labels = {lbl.text(): lbl.isVisible() for lbl in win.tool_options_layout.parentWidget().findChildren(QLabel)}
    assert labels.get("thickness (px):") is True
    assert labels.get("radius (px):") in (False, None)
    assert labels.get("ellipticity (%):") in (False, None)
    assert labels.get("angle (°):") in (False, None)
