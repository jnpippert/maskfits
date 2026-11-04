"""Coverage for imagedata.cube_spectral_axis / header_flux_unit: a cube's
spectral axis and flux unit from its header."""

import numpy as np
import pytest
from astropy.io import fits

from maskfits.imagedata import cube_spectral_axis, header_flux_unit


def _header(**cards):
    return fits.Header({"NAXIS": 3, "NAXIS1": 10, "NAXIS2": 10, "NAXIS3": 4, **cards})


def test_kcwi_style_header():
    # As written by the KCWI pipeline: a lowercase-ish "Wave" CTYPE3 that
    # astropy itself doesn't treat as spectral, plus a celestial WCS.
    header = _header(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=49.7, CRVAL2=41.5, CRPIX1=1.0, CRPIX2=1.0,
                     CDELT1=-1e-4, CDELT2=1e-4, CUNIT1="deg", CUNIT2="deg",
                     CTYPE3="Wave", CRPIX3=1.0, CRVAL3=3470.0, CDELT3=2.0, CUNIT3="Angstrom")
    axis = cube_spectral_axis(header, 4)
    assert axis.quantity == "wavelength"
    assert axis.unit == "Angstrom"
    assert np.allclose(axis.values, [3470, 3472, 3474, 3476])


def test_values_are_in_the_headers_own_unit_not_si():
    # WCSLIB hands WAVE axes back in metres - must come back as Angstrom.
    axis = cube_spectral_axis(_header(CTYPE3="WAVE", CUNIT3="Angstrom", CRVAL3=4750.0, CD3_3=1.25, CRPIX3=1.0), 4)
    assert np.allclose(axis.values, [4750, 4751.25, 4752.5, 4753.75])
    assert axis.unit == "Angstrom"


def test_log_wavelength_axis():
    axis = cube_spectral_axis(_header(CTYPE3="WAVE-LOG", CUNIT3="nm", CRVAL3=500.0, CDELT3=1.0, CRPIX3=1.0), 4)
    assert axis.values[0] == pytest.approx(500.0)
    assert axis.values[1] == pytest.approx(500.0 * np.exp(1.0 / 500.0))
    assert axis.unit == "nm"


@pytest.mark.parametrize("ctype, unit, quantity", [("FREQ", "GHz", "frequency"), ("VRAD", "km/s", "velocity"),
                                                   ("AWAV", "Angstrom", "wavelength")])
def test_quantity_comes_from_ctype3(ctype, unit, quantity):
    axis = cube_spectral_axis(_header(CTYPE3=ctype, CUNIT3=unit, CRVAL3=1.0, CDELT3=1.0, CRPIX3=1.0), 4)
    assert axis.quantity == quantity
    assert axis.unit == unit
    assert np.allclose(axis.values, [1, 2, 3, 4])


def test_missing_crpix3_uses_the_fits_default_of_zero():
    axis = cube_spectral_axis(_header(CRVAL3=5000.0, CDELT3=10.0, CUNIT3="Angstrom"), 4)
    assert np.allclose(axis.values, [5010, 5020, 5030, 5040])


def test_per_slice_wave_keywords():
    cards = {f"WAVE{i}": w for i, w in enumerate([3551.0, 4686.0, 6165.0, 7481.0])}
    cards.update({f"BAND{i}": b for i, b in enumerate("ugri")})
    axis = cube_spectral_axis(_header(**cards), 4)
    assert np.allclose(axis.values, [3551, 4686, 6165, 7481])
    assert axis.unit == "Angstrom"
    assert axis.labels == ("u", "g", "r", "i")


def test_partial_wave_keywords_override_the_wcs_axis():
    axis = cube_spectral_axis(_header(CRVAL3=1.0, CDELT3=1.0, CRPIX3=1.0, WAVE1=4686.0), 4)
    assert axis.values[1] == 4686.0
    assert axis.quantity == "wavelength"


def test_no_axis_information_falls_back_to_slice_numbers():
    axis = cube_spectral_axis(_header(), 4)
    assert axis.quantity == "slice"
    assert axis.unit == ""
    assert np.allclose(axis.values, [1, 2, 3, 4])


def test_flux_unit():
    assert header_flux_unit(fits.Header({"BUNIT": "10**(-20)*erg/s/cm**2/Angstrom"})) == "10**(-20)*erg/s/cm**2/Angstrom"
    assert header_flux_unit(fits.Header({"FLUXUNIT": "Jy"})) == "Jy"
    assert header_flux_unit(fits.Header()) == ""
