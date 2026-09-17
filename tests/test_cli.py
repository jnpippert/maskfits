import argparse

import numpy as np
import pytest
from astropy.io import fits

from maskfits.cli import COLORMAP_CLI_CHOICES, SCALE_CLI_CHOICES, _parse_cuts, build_gui_parser, main


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


# ------------------------------------------------------- GUI argument parsing


@pytest.mark.parametrize("value, expected", [("zscale", "zscale"), ("minmax", "minmax"), ("99.5", "pct99.5"),
                                              ("99", "pct99.0"), ("ZSCALE", "zscale")])
def test_parse_cuts_accepts_known_forms(value, expected):
    assert _parse_cuts(value) == expected


@pytest.mark.parametrize("value", ["bogus", "150", "0", "-5"])
def test_parse_cuts_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_cuts(value)


def test_build_gui_parser_defaults_to_none():
    args = build_gui_parser().parse_args([])
    assert args.files == []
    for name in ("zoom", "mode", "vmin", "vmax", "scale", "cuts", "colormap", "binning", "smooth"):
        assert getattr(args, name) is None


def test_build_gui_parser_parses_every_new_flag():
    args = build_gui_parser().parse_args([
        "-c", "isopy", "--cuts", "99.5", "--scale", "log", "-b", "4", "-s", "2",
        "--vmin", "0.1", "--vmax", "0.9", "img.fits",
    ])
    assert args.files == ["img.fits"]
    assert args.colormap == "isopy"
    assert args.cuts == "pct99.5"
    assert args.scale == "log"
    assert args.binning == 4
    assert args.smooth == 2
    assert args.vmin == 0.1
    assert args.vmax == 0.9


def test_build_gui_parser_rejects_unknown_colormap():
    with pytest.raises(SystemExit):
        build_gui_parser().parse_args(["-c", "bogus"])


def test_colormap_cli_choices_map_to_real_internal_names():
    from maskfits.colormaps import COLORMAP_NAMES

    assert set(COLORMAP_CLI_CHOICES.values()) <= set(COLORMAP_NAMES)


def test_scale_cli_choices_map_to_real_internal_names():
    from maskfits.imagedata import STRETCH_NAMES

    assert set(SCALE_CLI_CHOICES.values()) == set(STRETCH_NAMES)
