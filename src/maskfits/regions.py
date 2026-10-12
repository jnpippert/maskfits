"""Mask -> DS9 region file: one polygon per connected group of masked pixels.

No Qt in here - the options dialog lives in region_export_window.py.

Polygons follow the pixel edges exactly, so rasterizing the region file
reproduces the mask. Groups are 4-connected (scipy.ndimage.label's
default): pixels touching only at a corner are separate polygons, which
keeps every outline free of bow-tie crossings. A DS9 polygon can't contain
a hole, so a hole in a group is written as an excluded polygon
("-polygon(...)") right after the group's own outline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

# Named colors DS9 itself offers in its Region -> Color menu.
DS9_COLORS = ["green", "red", "blue", "cyan", "magenta", "yellow", "black", "white"]
COORD_SYSTEMS = ["image", "wcs"]

# Step (dx, dy) per direction code; +1 mod 4 is a left turn.
_STEPS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass(frozen=True)
class RegionPolygon:
    """One closed outline in pixel-edge coordinates: pixel (row r, col c)
    spans x in [c, c+1], y in [r, r+1]. `vertices` is (N, 2) as (x, y),
    corners only. `hole` marks an outline of a hole inside `group`."""

    vertices: np.ndarray
    group: int
    hole: bool


def mask_to_polygons(mask: np.ndarray) -> list[RegionPolygon]:
    """Every masked group's outline, then its holes, ordered by group."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    labels, _count = ndimage.label(mask)
    ny, nx = mask.shape
    padded = np.pad(mask, 1)
    core = padded[1:-1, 1:-1]

    # Bitmask of boundary edges leaving each pixel corner, each directed
    # with the masked pixel on its left (so outlines run counter-clockwise
    # and holes clockwise). Bit d is direction _STEPS[d].
    edges = np.zeros((ny + 1, nx + 1), dtype=np.uint8)
    edges[:-1, :-1][core & ~padded[:-2, 1:-1]] |= 1  # bottom side, heading +x
    edges[:-1, 1:][core & ~padded[1:-1, 2:]] |= 2    # right side, heading +y
    edges[1:, 1:][core & ~padded[2:, 1:-1]] |= 4     # top side, heading -x
    edges[1:, :-1][core & ~padded[1:-1, :-2]] |= 8   # left side, heading -y

    remaining = edges.copy()
    polygons: list[RegionPolygon] = []
    for start_y, start_x in np.argwhere(edges):
        start_y, start_x = int(start_y), int(start_x)
        while remaining[start_y, start_x]:
            bits = int(remaining[start_y, start_x])
            start_dir = (bits & -bits).bit_length() - 1
            vertices = _trace(edges, remaining, start_x, start_y, start_dir)
            pixel_y = start_y - (1 if start_dir in (2, 3) else 0)
            pixel_x = start_x - (1 if start_dir in (1, 2) else 0)
            polygons.append(RegionPolygon(
                vertices=vertices, group=int(labels[pixel_y, pixel_x]), hole=_signed_area(vertices) < 0,
            ))
    polygons.sort(key=lambda p: (p.group, p.hole))
    return polygons


def _trace(edges: np.ndarray, remaining: np.ndarray, x: int, y: int, direction: int) -> np.ndarray:
    """Follows one closed loop of boundary edges, consuming them from
    `remaining`, and returns its corner vertices."""
    start = (x, y, direction)
    corners: list[tuple[int, int]] = []
    while True:
        remaining[y, x] &= ~(1 << direction) & 0xF
        dx, dy = _STEPS[direction]
        x, y = x + dx, y + dy
        # Where two masked pixels touch only at this corner, two loops pass
        # through it - preferring the left turn keeps following the same
        # pixel. Looked up in `edges` (not `remaining`) so the choice never
        # depends on what was already traced.
        available = int(edges[y, x])
        for turn in (1, 0, 3):
            candidate = (direction + turn) % 4
            if available & (1 << candidate):
                break
        if candidate != direction:
            corners.append((x, y))
        direction = candidate
        if (x, y, direction) == start:
            return np.array(corners, dtype=float)


def _signed_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def format_region_file(polygons: list[RegionPolygon], *, color: str = "green", coords: str = "image",
                       wcs=None) -> str:
    """The region file text, in DS9's classic (version 4.1) layout, with no
    text labels on any region.

    coords="image": 1-based FITS pixel coordinates (a pixel's center is at
    an integer). coords="wcs": fk5 degrees via `wcs` (an astropy WCS for
    the image the mask belongs to).
    """
    if coords not in COORD_SYSTEMS:
        raise ValueError(f"unknown coordinate system: {coords!r}")
    if coords == "wcs" and wcs is None:
        raise ValueError("WCS coordinates need a WCS")
    lines = [
        "# Region file format: DS9 version 4.1",
        f'global color={color} dashlist=8 3 width=1 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "fk5" if coords == "wcs" else "image",
    ]
    for polygon in polygons:
        if coords == "wcs":
            # pixel_to_world takes 0-based pixel-center coordinates.
            sky = wcs.celestial.pixel_to_world(polygon.vertices[:, 0] - 0.5, polygon.vertices[:, 1] - 0.5)
            sky = sky.transform_to("fk5")
            pairs = zip(sky.ra.deg, sky.dec.deg)
            numbers = ",".join(f"{ra:.7f},{dec:.7f}" for ra, dec in pairs)
        else:
            numbers = ",".join(f"{x + 0.5:g},{y + 0.5:g}" for x, y in polygon.vertices)
        lines.append(f"{'-' if polygon.hole else ''}polygon({numbers})")
    return "\n".join(lines) + "\n"


def write_region_file(path: str, mask: np.ndarray, *, color: str = "green", coords: str = "image",
                      wcs: Optional[object] = None) -> int:
    """Writes `mask` (in the FITS file's own orientation) as a DS9 region
    file and returns how many polygons it holds."""
    polygons = mask_to_polygons(mask)
    with open(path, "w", encoding="ascii") as handle:
        handle.write(format_region_file(polygons, color=color, coords=coords, wcs=wcs))
    return len(polygons)
