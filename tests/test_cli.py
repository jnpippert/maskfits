import numpy as np
from astropy.io import fits

from fitsedit.cli import main


def make_fits(path):
    hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
    hdu.header["OBJECT"] = "TEST"
    hdu.writeto(path)


def test_get(tmp_path, capsys):
    path = tmp_path / "test.fits"
    make_fits(path)

    assert main(["get", str(path), "OBJECT"]) == 0
    assert capsys.readouterr().out.strip() == "TEST"


def test_set(tmp_path):
    path = tmp_path / "test.fits"
    make_fits(path)

    assert main(["set", str(path), "OBJECT", "UPDATED"]) == 0
    with fits.open(path) as hdul:
        assert hdul[0].header["OBJECT"] == "UPDATED"
