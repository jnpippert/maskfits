"""Coverage for maskfits.regions: mask -> polygons (one per 4-connected
group, holes as excluded polygons) -> DS9 region file text."""

import numpy as np
import pytest
from astropy.wcs import WCS

from maskfits.regions import (
    RegionPolygon,
    format_region_file,
    mask_to_polygons,
    write_region_file,
)


def _rasterize(polygons, shape):
    """Exact inverse of mask_to_polygons: winding number of every pixel,
    built from each outline's vertical edges (+1 heading down... the sign
    just has to be consistent), then a running sum along x."""
    ny, nx = shape
    delta = np.zeros((ny, nx + 1), dtype=int)
    for polygon in polygons:
        v = polygon.vertices.astype(int)
        for (x0, y0), (x1, y1) in zip(v, np.roll(v, -1, axis=0)):
            if x0 == x1 and y0 != y1:
                lo, hi = sorted((y0, y1))
                delta[lo:hi, x0] += 1 if y1 < y0 else -1
    return np.cumsum(delta, axis=1)[:, :nx] > 0


def test_empty_mask_gives_no_polygons():
    assert mask_to_polygons(np.zeros((5, 5), dtype=bool)) == []


def test_single_pixel_is_its_own_unit_square():
    mask = np.zeros((4, 6), dtype=bool)
    mask[2, 3] = True
    (polygon,) = mask_to_polygons(mask)
    assert not polygon.hole
    assert sorted(map(tuple, polygon.vertices)) == [(3, 2), (3, 3), (4, 2), (4, 3)]


def test_rectangle_has_only_its_four_corners():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:7, 1:9] = True
    (polygon,) = mask_to_polygons(mask)
    assert len(polygon.vertices) == 4  # collinear edge points are dropped


def test_separate_groups_get_separate_polygons():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 5:8] = True
    polygons = mask_to_polygons(mask)
    assert [p.hole for p in polygons] == [False, False]
    assert len({p.group for p in polygons}) == 2


def test_pixels_touching_only_at_a_corner_are_separate_polygons():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = mask[1, 1] = True
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 2
    assert all(len(p.vertices) == 4 for p in polygons)


def test_a_hole_becomes_an_excluded_polygon_after_its_group():
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    mask[3, 3] = False
    outline, hole = mask_to_polygons(mask)
    assert (outline.hole, hole.hole) == (False, True)
    assert outline.group == hole.group
    assert sorted(map(tuple, hole.vertices)) == [(3, 3), (3, 4), (4, 3), (4, 4)]


def test_mask_touching_every_image_edge():
    mask = np.ones((3, 5), dtype=bool)
    (polygon,) = mask_to_polygons(mask)
    assert sorted(map(tuple, polygon.vertices)) == [(0, 0), (0, 3), (5, 0), (5, 3)]


@pytest.mark.parametrize("seed", range(12))
def test_polygons_rasterize_back_to_exactly_the_mask(seed):
    rng = np.random.default_rng(seed)
    # Dense random masks are full of holes, islands in holes, and pixels
    # touching only at corners - every awkward case at once.
    mask = rng.random((40, 55)) < (0.25 + 0.05 * seed)
    polygons = mask_to_polygons(mask)
    assert np.array_equal(_rasterize(polygons, mask.shape), mask)
    assert all(len(p.vertices) >= 4 for p in polygons)


def test_one_polygon_per_scipy_group_plus_holes():
    from scipy import ndimage

    rng = np.random.default_rng(3)
    mask = rng.random((30, 30)) < 0.45
    polygons = mask_to_polygons(mask)
    _labels, count = ndimage.label(mask)
    outlines = [p for p in polygons if not p.hole]
    assert len(outlines) == count
    assert sorted(p.group for p in outlines) == list(range(1, count + 1))


# ------------------------------------------------------------------ format


def _square():
    return [RegionPolygon(np.array([[3.0, 2.0], [4.0, 2.0], [4.0, 3.0], [3.0, 3.0]]), group=1, hole=False)]


def test_image_format_is_classic_ds9_with_fits_pixel_coordinates():
    text = format_region_file(_square(), color="red", coords="image")
    lines = text.splitlines()
    assert lines[0] == "# Region file format: DS9 version 4.1"
    assert lines[1].startswith("global color=red ")
    assert lines[2] == "image"
    # Array pixel (row 2, col 3) is FITS pixel (4, 3): edges at 3.5..4.5 / 2.5..3.5.
    assert lines[3] == "polygon(3.5,2.5,4.5,2.5,4.5,3.5,3.5,3.5)"
    assert text.endswith("\n")


def test_regions_have_no_labels():
    text = format_region_file(_square())
    assert "text=" not in text
    assert "#" not in text.splitlines()[-1]


def test_holes_are_written_as_excluded_polygons():
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    mask[3, 3] = False
    lines = format_region_file(mask_to_polygons(mask)).splitlines()[3:]
    assert lines[0].startswith("polygon(")
    assert lines[1].startswith("-polygon(")


def _wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [4.0, 3.0]  # FITS pixel (4, 3) = array (row 2, col 3)
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    wcs.wcs.radesys = "FK5"
    wcs.wcs.equinox = 2000.0
    return wcs


def test_wcs_format_is_fk5_degrees():
    lines = format_region_file(_square(), coords="wcs", wcs=_wcs()).splitlines()
    assert lines[2] == "fk5"
    numbers = [float(v) for v in lines[3][len("polygon("):-1].split(",")]
    ra, dec = numbers[0::2], numbers[1::2]
    # The square is centered on the reference pixel, half a pixel (0.5") out.
    assert np.mean(ra) == pytest.approx(150.0, abs=1e-6)
    assert np.mean(dec) == pytest.approx(2.0, abs=1e-6)
    assert max(dec) - min(dec) == pytest.approx(1 / 3600, rel=1e-3)


def test_wcs_without_a_wcs_is_an_error():
    with pytest.raises(ValueError):
        format_region_file(_square(), coords="wcs", wcs=None)


def test_unknown_coordinate_system_is_an_error():
    with pytest.raises(ValueError):
        format_region_file(_square(), coords="galactic")


def test_write_region_file_returns_the_polygon_count(tmp_path):
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 5:8] = True
    path = tmp_path / "out.reg"
    assert write_region_file(str(path), mask, color="cyan") == 2
    text = path.read_text()
    assert text.count("polygon(") == 2
    assert "color=cyan" in text
