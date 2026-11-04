"""Space is the default key for resetting the zoom."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.gui import MaskFitsApp  # noqa: E402
from maskfits.settings import Settings  # noqa: E402
from maskfits.shortcuts import SHORTCUT_ACTIONS_BY_ID, find_conflict  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_space_is_the_default(qapp):
    assert SHORTCUT_ACTIONS_BY_ID["reset_zoom"].defaults == ("Space",)
    # Stored the way ShortcutCapture records a rebinding, so they compare equal.
    assert QKeySequence("Space").toString() == "Space"
    assert find_conflict("reset_zoom", "Space", {}) is None


def test_the_space_shortcut_resets_the_zoom(qapp, tmp_path):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.zeros((40, 40), dtype=np.float32)).writeto(path)
    win = MaskFitsApp([str(path)], settings=Settings())
    win.zoom_mult = 4.0
    space = next(sc for sc in win._shortcuts if sc.key().toString() == "Space")
    space.activated.emit()
    assert win.zoom_mult == 1.0
