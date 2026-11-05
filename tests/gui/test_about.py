"""Help -> About text."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from maskfits import __version__  # noqa: E402
from maskfits.gui import MaskFitsApp, about_text  # noqa: E402
from maskfits.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_about_text():
    text = about_text()
    assert "Tool to mask FITS images." in text
    assert "Inspired by Johannes Koppenhöfer's fitsedit." in text
    assert "Developed by Jan-Niklas Pippert © 2026" in text
    assert text.endswith(f"Version {__version__}")


def test_about_menu_shows_it(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "about", lambda parent, title, text: shown.append((title, text)))
    win = MaskFitsApp([], settings=Settings())
    win._show_help()
    assert shown == [("About maskfits", about_text())]


def test_settings_menu_entry_has_no_ellipsis(qapp):
    win = MaskFitsApp([], settings=Settings())
    texts = [a.text() for menu in win.menuBar().actions() if menu.menu() for a in menu.menu().actions()]
    assert "Settings" in texts
    assert "Settings..." not in texts
