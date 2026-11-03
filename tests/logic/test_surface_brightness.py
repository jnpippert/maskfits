"""Coverage for the photometric calibration helpers in maskfits.imagedata:
zero point / pixel scale lookup across common header keywords (pixel scale
falling back to the WCS) and surface brightness from a pixel value."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from maskfits.imagedata import (
    PXSCALE_KEYWORDS,
    ZP_KEYWORDS,
    header_pixel_scale,
    header_zeropoint,
    isopy_cuts_and_stops,
    sb_calibration,
    surface_brightness,
)


def _wcs(arcsec_per_px=0.25):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.cdelt = [-arcsec_per_px / 3600, arcsec_per_px / 3600]
    return wcs


@pytest.mark.parametrize("key", ZP_KEYWORDS)
# >8-character names exist only as HIERARCH cards - still found, just warned about.
@pytest.mark.filterwarnings("ignore::astropy.io.fits.verify.VerifyWarning")
def test_every_zero_point_keyword_is_recognized(key):
    assert header_zeropoint(fits.Header({key: 28.5})) == 28.5


@pytest.mark.parametrize("key", PXSCALE_KEYWORDS)
# >8-character names exist only as HIERARCH cards - still found, just warned about.
@pytest.mark.filterwarnings("ignore::astropy.io.fits.verify.VerifyWarning")
def test_every_pixel_scale_keyword_is_recognized(key):
    assert header_pixel_scale(fits.Header({key: 0.4}), None) == 0.4


def test_keyword_order_decides_when_several_are_present():
    header = fits.Header({"MAGZP": 25.0, "ZP": 30.0})
    assert header_zeropoint(header) == 30.0


def test_unparseable_keyword_falls_through_to_the_next():
    header = fits.Header({"ZP": "n/a", "MAGZP": 27.0})
    assert header_zeropoint(header) == 27.0


def test_missing_zero_point_is_none():
    assert header_zeropoint(fits.Header()) is None


def test_negative_pixel_scale_keyword_is_taken_as_its_magnitude():
    assert header_pixel_scale(fits.Header({"PIXSCALE": -0.3}), None) == 0.3


def test_pixel_scale_falls_back_to_the_wcs():
    assert header_pixel_scale(fits.Header(), _wcs(0.25)) == pytest.approx(0.25)


def test_header_pixel_scale_wins_over_the_wcs():
    assert header_pixel_scale(fits.Header({"PXSCALE": 0.5}), _wcs(0.25)) == 0.5


def test_no_pixel_scale_anywhere_is_none():
    assert header_pixel_scale(fits.Header(), None) is None


def test_calibration_needs_both():
    assert sb_calibration(fits.Header({"ZP": 30.0}), None) is None
    assert sb_calibration(fits.Header({"PXSCALE": 0.5}), _wcs()) is None
    assert sb_calibration(fits.Header({"ZP": 30.0}), _wcs(0.25)) == pytest.approx((30.0, 0.25))


def test_surface_brightness_formula():
    # 1 count in a 1"x1" pixel at ZP 30 is exactly 30 mag/arcsec^2; a 0.5"
    # pixel packs it into a quarter of the area: 2.5*log10(4) brighter.
    assert surface_brightness(1.0, 30.0, 1.0) == pytest.approx(30.0)
    assert surface_brightness(1.0, 30.0, 0.5) == pytest.approx(30.0 - 2.5 * np.log10(4))
    assert surface_brightness(100.0, 30.0, 1.0) == pytest.approx(25.0)


@pytest.mark.parametrize("flux", [0.0, -5.0, np.nan, np.inf])
def test_surface_brightness_undefined_values(flux):
    assert surface_brightness(flux, 30.0, 1.0) is None


def test_isopy_cuts_use_the_same_keyword_lookup():
    via_alias = isopy_cuts_and_stops(fits.Header({"MAGZP": 28.0, "PIXSCALE": 0.5}), None)
    via_primary = isopy_cuts_and_stops(fits.Header({"ZP": 28.0, "PXSCALE": 0.5}), None)
    assert via_alias == via_primary
