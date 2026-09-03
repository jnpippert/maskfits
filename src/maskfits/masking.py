"""Pixel-mask geometry helpers: circular/elliptical stamps and satellite-trail lines."""

import math
from typing import Optional

import numpy as np


def ellipse_mask(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    a: float,
    b: float,
    angle_deg: float,
) -> np.ndarray:
    """Boolean mask of `shape` (ny, nx) that is True inside a rotated ellipse.

    a and b are the semi-major and semi-minor axis lengths in pixels.
    angle_deg is the rotation of the major axis from the x-axis.
    """
    ny, nx = shape
    a = max(a, 0.5)
    b = max(b, 0.5)
    half = max(a, b) + 1
    x0 = max(int(cx - half), 0)
    x1 = min(int(cx + half) + 1, nx)
    y0 = max(int(cy - half), 0)
    y1 = min(int(cy + half) + 1, ny)

    mask = np.zeros(shape, dtype=bool)
    if x0 >= x1 or y0 >= y1:
        return mask

    # Pixel index j spans continuous image coordinates [j, j+1) (matching how
    # the canvas renderer and click coordinates already treat pixels), so its
    # center - what actually needs to fall within the ellipse - is at j + 0.5,
    # not at j itself. Comparing against the bare index would shift every
    # mask half a pixel off from the click position (most visible at radius
    # 0.5, where clicking a pixel's center would then miss it entirely).
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xr = (xx + 0.5) - cx
    yr = (yy + 0.5) - cy
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    xrot = xr * ct + yr * st
    yrot = -xr * st + yr * ct
    mask[y0:y1, x0:x1] = (xrot / a) ** 2 + (yrot / b) ** 2 <= 1.0
    return mask


def line_mask(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
) -> np.ndarray:
    """Boolean mask of `shape` marking a flat-ended rectangular band of `width`
    pixels along a segment - a true rotated rectangle, not a rounded-cap
    capsule: the ends are cut off square exactly at (x0, y0) and (x1, y1),
    with no semicircular bulge past them.

    Intended for masking satellite trails: draw a line from (x0, y0) to (x1, y1).
    """
    ny, nx = shape
    half_w = width / 2.0 + 1
    xlo = max(int(min(x0, x1) - half_w), 0)
    xhi = min(int(max(x0, x1) + half_w) + 1, nx)
    ylo = max(int(min(y0, y1) - half_w), 0)
    yhi = min(int(max(y0, y1) + half_w) + 1, ny)

    mask = np.zeros(shape, dtype=bool)
    if xlo >= xhi or ylo >= yhi:
        return mask

    # Same pixel-center-is-at-index+0.5 convention as ellipse_mask - see there.
    yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
    px, py = xx + 0.5, yy + 0.5
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    if length2 == 0:
        # No direction to build a rectangle from - falls back to a round dot.
        dist = np.hypot(px - x0, py - y0)
        mask[ylo:yhi, xlo:xhi] = dist <= (width / 2.0)
        return mask
    # t is NOT clipped to [0, 1] here (unlike a capsule test) - it's used
    # directly to require the projection fall strictly within the segment's
    # own length, which is what gives the band its flat, square-cut ends.
    t = ((px - x0) * dx + (py - y0) * dy) / length2
    perp_dist = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
    mask[ylo:yhi, xlo:xhi] = (perp_dist <= (width / 2.0)) & (t >= 0.0) & (t <= 1.0)
    return mask


def _clip_param_range_to_bounds(
    px: float, py: float, dx: float, dy: float, shape: tuple[int, int]
) -> Optional[tuple[float, float]]:
    """Liang-Barsky clip of the infinite line px+t*dx, py+t*dy against [0, nx] x [0, ny].

    Returns the (tmin, tmax) parametric range that stays inside the image bounds,
    or None if the line never crosses the image at all.
    """
    ny, nx = shape
    t0, t1 = -math.inf, math.inf
    for p, q in ((-dx, px), (dx, nx - px), (-dy, py), (dy, ny - py)):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return t0, t1


def extend_ray_to_border(
    shape: tuple[int, int], x0: float, y0: float, x1: float, y1: float
) -> tuple[float, float, float, float]:
    """Extend the ray starting at (x0, y0) through (x1, y1) to the far image border.

    The start point is kept fixed; only the far end is pushed out to the edge.
    """
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return x0, y0, x1, y1
    clipped = _clip_param_range_to_bounds(x0, y0, dx, dy, shape)
    if clipped is None:
        return x0, y0, x1, y1
    _, t1 = clipped
    t_far = max(t1, 1.0)
    return x0, y0, x0 + t_far * dx, y0 + t_far * dy


def extend_line_to_borders(
    shape: tuple[int, int], x0: float, y0: float, x1: float, y1: float
) -> tuple[float, float, float, float]:
    """Extend the infinite line through (x0, y0) and (x1, y1) to both image borders."""
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return x0, y0, x1, y1
    clipped = _clip_param_range_to_bounds(x0, y0, dx, dy, shape)
    if clipped is None:
        return x0, y0, x1, y1
    t0, t1 = clipped
    return x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy


def ellipse_polygon_points(
    cx: float,
    cy: float,
    a: float,
    b: float,
    angle_deg: float,
    n: int = 48,
) -> list[float]:
    """Flat [x0, y0, x1, y1, ...] point list approximating a rotated ellipse outline.

    Used for canvas drawing, since Tkinter ovals cannot be rotated.
    """
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    t = np.linspace(0, 2 * np.pi, n)
    ex = a * np.cos(t)
    ey = b * np.sin(t)
    xs = cx + ex * ct - ey * st
    ys = cy + ex * st + ey * ct
    return list(np.column_stack([xs, ys]).flatten())
