"""Coverage for spectrum_view: axis labels, ticks and painting."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from maskfits.imagedata import SpectralAxis  # noqa: E402
from maskfits.spectrum_view import SpectrumView, display_unit, nice_ticks  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _axis(quantity="wavelength", unit="Angstrom", n=5):
    return SpectralAxis(np.linspace(3500, 5500, n), quantity, unit, ("",) * n)


@pytest.mark.parametrize("unit, shown", [("Angstrom", "Å"), ("angstrom", "Å"), ("AA", "Å"),
                                         ("erg/s/cm2/Angstrom", "erg/s/cm2/Å"), ("nm", "nm"), ("", "")])
def test_display_unit(unit, shown):
    assert display_unit(unit) == shown


def test_titles(qapp):
    view = SpectrumView()
    view.set_axis(_axis(), "erg/s/cm2/Angstrom")
    assert view.x_title() == "λ [Å]"
    assert view.y_title() == "Flux [erg/s/cm2/Å]"
    view.set_axis(_axis(), "")
    assert view.y_title() == "Flux"
    view.set_axis(_axis("frequency", "GHz"), "")
    assert view.x_title() == "ν [GHz]"
    view.set_axis(_axis("slice", ""), "")
    assert view.x_title() == "slice"


def test_nice_ticks():
    assert nice_ticks(3470, 5540) == [3500, 4000, 4500, 5000, 5500]
    assert nice_ticks(0.0, 1.0) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = nice_ticks(-3e-18, 1.1e-18)
    assert ticks[0] >= -3e-18 and ticks[-1] <= 1.1e-18 and 0.0 in ticks


def test_y_range_ignores_nan(qapp):
    view = SpectrumView()
    view.set_axis(_axis(n=4), "")
    view.set_spectrum(np.array([1.0, np.nan, 3.0, 2.0]))
    lo, hi = view.y_range()
    assert lo < 1.0 and hi > 3.0
    view.set_spectrum(np.array([np.nan] * 4))
    assert view.y_range() is None


@pytest.mark.parametrize("spectrum", [None, np.array([1.0, np.nan, 3.0, 2.0, 5.0]), np.full(5, 7.0)])
def test_paints_without_errors(qapp, spectrum):
    view = SpectrumView()
    view.resize(600, 200)
    view.set_axis(_axis(), "Jy")
    view.set_spectrum(spectrum)
    view.set_marker(2)
    view.grab()  # real paintEvent
