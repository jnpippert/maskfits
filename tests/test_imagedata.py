import numpy as np
import pytest
from astropy.io import fits

from maskfits.imagedata import (
    ISOPY_DEFAULT_PXSCALE,
    ISOPY_DEFAULT_ZP,
    isopy_cuts_and_stops,
    list_image_extensions,
    load_fits_image,
)


def _write_multi_ext(path, with_object_names=True):
    primary = fits.PrimaryHDU()  # empty primary - no image data
    sci = fits.ImageHDU(data=np.full((10, 20), 1.0, dtype=np.float32), name="SCI")
    if with_object_names:
        sci.header["OBJECT"] = "GALAXY-A"
    err = fits.ImageHDU(data=np.full((10, 20), 2.0, dtype=np.float32), name="ERR")
    # no OBJECT keyword on this one
    table = fits.BinTableHDU.from_columns([fits.Column(name="x", format="D", array=[1.0, 2.0])])
    fits.HDUList([primary, sci, err, table]).writeto(path)


def test_list_image_extensions_skips_non_image_hdus(tmp_path):
    path = tmp_path / "multi.fits"
    _write_multi_ext(path)
    extensions = list_image_extensions(str(path))
    # primary (empty) and the table HDU should both be skipped
    assert [i for i, _ in extensions] == [1, 2]


def test_list_image_extensions_label_uses_object_keyword(tmp_path):
    path = tmp_path / "multi.fits"
    _write_multi_ext(path)
    extensions = list_image_extensions(str(path))
    labels = dict(extensions)
    assert labels[1] == "1 - GALAXY-A"
    assert labels[2] == "2 - NO HDU NAME"


def test_load_fits_image_with_explicit_ext(tmp_path):
    path = tmp_path / "multi.fits"
    _write_multi_ext(path)
    img_sci = load_fits_image(str(path), ext=1)
    assert np.allclose(img_sci.data, 1.0)
    img_err = load_fits_image(str(path), ext=2)
    assert np.allclose(img_err.data, 2.0)


def test_load_fits_image_invalid_ext_raises(tmp_path):
    path = tmp_path / "multi.fits"
    _write_multi_ext(path)
    with pytest.raises(ValueError):
        load_fits_image(str(path), ext=0)  # empty primary HDU
    with pytest.raises(ValueError):
        load_fits_image(str(path), ext=99)  # out of range


def test_load_fits_image_default_ext_picks_first_image_hdu(tmp_path):
    path = tmp_path / "multi.fits"
    _write_multi_ext(path)
    img = load_fits_image(str(path))  # ext=None -> first 2D-image HDU
    assert np.allclose(img.data, 1.0)


def test_single_extension_file_has_exactly_one_entry(tmp_path):
    path = tmp_path / "single.fits"
    fits.PrimaryHDU(data=np.zeros((10, 10))).writeto(path)
    extensions = list_image_extensions(str(path))
    assert len(extensions) == 1
    assert extensions[0][0] == 0


def test_isopy_cuts_default_zp_and_pxscale_when_absent():
    header = fits.Header()  # no ZP, no PXSCALE
    vmin, vmax, stops = isopy_cuts_and_stops(header, None)
    minflux = 10 ** (-0.4 * (27.0 - ISOPY_DEFAULT_ZP)) * ISOPY_DEFAULT_PXSCALE**2
    assert vmin == pytest.approx(-minflux)
    assert vmax == pytest.approx(10 ** (-0.4 * (22.0 - ISOPY_DEFAULT_ZP)) * ISOPY_DEFAULT_PXSCALE**2)
    assert stops == sorted(stops), "stop positions must be monotonically increasing for LUT interpolation"
    assert stops[0] == 0.0 and stops[-1] == 1.0


def test_isopy_cuts_uses_header_zp():
    header = fits.Header()
    header["ZP"] = 25.0
    vmin, _vmax, _stops = isopy_cuts_and_stops(header, None)
    default_vmin, _, _ = isopy_cuts_and_stops(fits.Header(), None)
    assert vmin != default_vmin


def test_isopy_cuts_pxscale_header_keyword_takes_priority():
    header = fits.Header()
    header["PXSCALE"] = 0.5
    vmin, vmax, _ = isopy_cuts_and_stops(header, None)
    default_vmin, default_vmax, _ = isopy_cuts_and_stops(fits.Header(), None)
    # flux scales with pxscale^2, so halving it should scale vmin/vmax by 0.25
    assert vmin == pytest.approx(default_vmin * 0.25)
    assert vmax == pytest.approx(default_vmax * 0.25)


def test_isopy_cuts_different_headers_give_different_cuts():
    h1, h2 = fits.Header(), fits.Header()
    h1["ZP"], h2["ZP"] = 25.0, 30.0
    vmin1, vmax1, _ = isopy_cuts_and_stops(h1, None)
    vmin2, vmax2, _ = isopy_cuts_and_stops(h2, None)
    assert (vmin1, vmax1) != (vmin2, vmax2)
