"""The welcome screen shown when no file is open: sidebar folded away,
logo + name + basic shortcuts painted on the canvas, and the three new
rebindable shortcuts (open file, settings, keyboard shortcuts)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtGui import QColor, QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.shortcuts import SHORTCUT_ACTIONS_BY_ID, find_conflict  # noqa: E402
from maskfits.welcome import key_caps, watermark_image  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_fits(path, size=32):
    rng = np.random.default_rng(0)
    fits.PrimaryHDU(rng.normal(100.0, 5.0, size=(size, size)).astype(np.float32)).writeto(path, overwrite=True)


def _open_via_dialog(win, monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(path)], ""))
    win.open_files()


def test_new_shortcut_defaults():
    assert SHORTCUT_ACTIONS_BY_ID["open_file"].defaults == ("Ctrl+O",)
    assert SHORTCUT_ACTIONS_BY_ID["open_settings"].defaults == ("Ctrl+,",)
    assert SHORTCUT_ACTIONS_BY_ID["show_hotkeys"].defaults == ("Ctrl+K",)
    for action_id in ("open_file", "open_settings", "show_hotkeys"):
        key = SHORTCUT_ACTIONS_BY_ID[action_id].defaults[0]
        assert find_conflict(action_id, key, {}) is None


def test_key_caps():
    assert key_caps("Ctrl+O", mac=False) == ["Ctrl", "O"]
    assert key_caps("Ctrl+Shift+Z", mac=False) == ["Ctrl", "Shift", "Z"]
    assert key_caps("Ctrl+O", mac=True) == ["⌘", "O"]
    assert key_caps("Ctrl+Shift+Z", mac=True) == ["⇧", "⌘", "Z"]
    assert key_caps("Meta+Alt+X", mac=True) == ["⌃", "⌥", "X"]
    assert key_caps("Ctrl++", mac=False) == ["Ctrl", "+"]
    assert key_caps("Ctrl+,", mac=False) == ["Ctrl", ","]
    assert key_caps("Tab", mac=True) == ["Tab"]


def test_watermark_keeps_only_galaxy_and_line(qapp):
    image = watermark_image(QColor(255, 255, 255, 40), 128)
    assert image is not None and image.width() == 128
    alphas = {image.pixelColor(x, y).alpha() for x in range(0, 128, 4) for y in range(0, 128, 4)}
    assert max(alphas) <= 40  # never stronger than the requested tint
    assert max(alphas) >= 35  # the spiral/line strokes are there
    # Plain tile area (inside the rounded square, away from the spiral and
    # the line): transparent, so no icon outline shows.
    assert image.pixelColor(22, 100).alpha() == 0
    assert image.pixelColor(64, 4).alpha() == 0  # outside the tile too


def test_sidebar_folded_when_no_file_and_unfolds_on_open(qapp, tmp_path, monkeypatch):
    win = MaskFitsApp([], settings=Settings())
    win.show()
    assert win.sidebar_collapsed
    assert not win.sidebar_container.isVisible()

    path = tmp_path / "img.fits"
    _write_fits(path)
    _open_via_dialog(win, monkeypatch, path)
    assert win.image is not None
    assert not win.sidebar_collapsed
    assert win.sidebar_container.isVisible()


def test_sidebar_open_when_started_with_a_file(qapp, tmp_path):
    path = tmp_path / "img.fits"
    _write_fits(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.show()
    assert not win.sidebar_collapsed


def test_manual_toggle_wins_over_auto_unfold(qapp, tmp_path, monkeypatch):
    win = MaskFitsApp([], settings=Settings())
    win.show()
    win._toggle_sidebar()  # user unfolds it while empty...
    win._toggle_sidebar()  # ...and folds it again themselves
    path = tmp_path / "img.fits"
    _write_fits(path)
    _open_via_dialog(win, monkeypatch, path)
    assert win.sidebar_collapsed


def test_welcome_painted_only_without_a_file(qapp, tmp_path, monkeypatch):
    win = MaskFitsApp([], settings=Settings(theme="dark"))
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    empty = win.canvas.grab().toImage()
    bg = QColor("#0a0a0b").rgb()
    w, h = empty.width(), empty.height()
    # Something besides the plain canvas color in the middle (logo/text)
    middle = [empty.pixel(x, y) for x in range(w // 3, 2 * w // 3, 3) for y in range(h // 4, 3 * h // 4, 3)]
    assert any(p != bg for p in middle)

    with_image = MaskFitsApp([], settings=Settings(theme="dark"))
    called = []
    monkeypatch.setattr("maskfits.gui.paint_welcome", lambda *a, **k: called.append(1))
    path = tmp_path / "img.fits"
    _write_fits(path)
    _open_via_dialog(with_image, monkeypatch, path)
    with_image.show()
    with_image.canvas.grab()
    assert not called


def test_menu_actions_carry_the_rebindable_keys(qapp):
    win = MaskFitsApp([], settings=Settings())
    actions = win._menu_shortcut_actions
    assert actions["open_file"].shortcut() == QKeySequence("Ctrl+O")
    assert actions["open_settings"].shortcut() == QKeySequence("Ctrl+,")
    assert actions["show_hotkeys"].shortcut() == QKeySequence("Ctrl+K")

    win.settings.shortcuts = {"open_file": ["Ctrl+Shift+O"]}
    win._rebuild_shortcuts()
    assert actions["open_file"].shortcut() == QKeySequence("Ctrl+Shift+O")


def test_menu_actions_open_their_windows(qapp):
    win = MaskFitsApp([], settings=Settings())
    win._menu_shortcut_actions["show_hotkeys"].trigger()
    assert win._hotkeys_window is not None
    win._hotkeys_window.close()
    win._menu_shortcut_actions["open_settings"].trigger()
    assert win._settings_window is not None
    win._settings_window.close()
