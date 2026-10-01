"""Coverage for theme.on_theme_changed - theme-change callbacks tied to a
widget's lifetime. Regression: widgets used to connect a bare lambda to
theme_changed, which Qt never disconnects when the widget is destroyed, so
every later theme change raised "Internal C++ object (RoundSlider) already
deleted" once per dead widget (the tool-options sliders are destroyed and
rebuilt on every ellipse <-> satellite switch)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.theme import on_theme_changed, theme_manager  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def flip_theme():
    """Emits theme_changed twice, ending back on the mode it started in."""
    manager = theme_manager()
    original = manager._mode

    def flip() -> None:
        manager.set_mode("light" if original != "light" else "dark")
        manager.set_mode(original)

    yield flip
    manager.set_mode(original)


def _flush_deferred_deletes(qapp) -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_callback_fires_while_the_owner_is_alive(qapp, flip_theme):
    owner = QWidget()
    calls = []
    on_theme_changed(owner, lambda: calls.append(1))

    flip_theme()

    assert len(calls) == 2


def test_callback_stops_once_the_owner_is_destroyed(qapp, flip_theme):
    owner = QWidget()
    calls = []
    on_theme_changed(owner, lambda: calls.append(1))

    owner.deleteLater()
    _flush_deferred_deletes(qapp)
    flip_theme()

    assert calls == []


def test_theme_change_after_tool_switches_does_not_touch_deleted_sliders(qapp, tmp_path, flip_theme, capfd):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.zeros((32, 32), dtype=np.float32)).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()

    for _ in range(3):
        win._toggle_tool()  # destroys the previous tool's sliders
        _flush_deferred_deletes(qapp)
    capfd.readouterr()

    flip_theme()

    assert "already deleted" not in capfd.readouterr().err
