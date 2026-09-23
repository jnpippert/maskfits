import numpy as np

from maskfits.masking import (
    ellipse_mask,
    extend_line_to_borders,
    extend_ray_to_border,
    line_mask,
)


def test_ellipse_mask_circle_centered():
    # Pixel j's center sits at continuous coordinate j + 0.5 (it spans
    # [j, j+1)), matching how click coordinates and the canvas renderer treat
    # pixels - so the reference formula checks distance from there, not from
    # the bare index.
    mask = ellipse_mask((21, 21), 10, 10, 5, 5, 0)
    assert mask[10, 10]
    assert mask[10, 14]
    assert not mask[10, 16]
    yy, xx = np.mgrid[0:21, 0:21]
    assert mask.sum() == np.sum((yy + 0.5 - 10) ** 2 + (xx + 0.5 - 10) ** 2 <= 25)


def test_ellipse_mask_independent_axes():
    circle = ellipse_mask((41, 41), 20, 20, 10, 10, 0)
    ellipse = ellipse_mask((41, 41), 20, 20, 10, 5, 0)
    assert ellipse.sum() < circle.sum()


def test_ellipse_mask_out_of_bounds_is_empty():
    mask = ellipse_mask((10, 10), 100, 100, 5, 5, 0)
    assert not mask.any()


def test_line_mask_covers_endpoints_and_band():
    mask = line_mask((20, 20), 2, 2, 17, 17, width=4)
    assert mask[2, 2]
    assert mask[10, 10]  # well inside the band along the diagonal
    assert not mask[0, 19]  # off to the side, outside the band width
    # Flat-ended rectangle, not a rounded capsule: a pixel whose center sits
    # past the segment's own endpoint (here, just beyond (17, 17)) is outside
    # the band - a rounded cap would have reached it, a square-cut one doesn't.
    assert not mask[19, 19]


def test_line_mask_zero_length_is_a_dot():
    mask = line_mask((20, 20), 5, 5, 5, 5, width=4)
    assert mask[5, 5]
    assert not mask[5, 15]


def test_extend_ray_to_border_only_extends_forward():
    x0, y0, x1, y1 = extend_ray_to_border((100, 100), 50, 50, 60, 50)
    assert (x0, y0) == (50, 50)
    assert x1 == 100
    assert y1 == 50


def test_extend_ray_to_border_diagonal():
    x0, y0, x1, y1 = extend_ray_to_border((100, 100), 10, 10, 20, 20)
    assert (x0, y0) == (10, 10)
    assert x1 == 100 and y1 == 100


def test_extend_line_to_borders_both_ends():
    x0, y0, x1, y1 = extend_line_to_borders((100, 100), 40, 50, 60, 50)
    assert x0 == 0 and y0 == 50
    assert x1 == 100 and y1 == 50


def test_extend_line_to_borders_vertical():
    x0, y0, x1, y1 = extend_line_to_borders((100, 100), 50, 30, 50, 70)
    assert x0 == 50 and y0 == 0
    assert x1 == 50 and y1 == 100
