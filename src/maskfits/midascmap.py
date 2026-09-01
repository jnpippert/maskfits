"""
Module that inhabits the translated MIDAS colormap, which enhances isophotes visibility, to matplotlib. (Jan-Niklas Pippert 2024)

Example
-------
>>> from midascmap import midascmap
>>> data = fits.getdata(FILE)
>>> cmap,norm,ticks = midascmap(data,vmin=VMIN,vmax=VMAX,return_ticks=True)
>>> fig, ax = plt.subplots(figsize=(10,10))
>>> iax = ax.imshow(d,cmap=cmap,norm=norm,origin="lower")
>>> cbar=fig.colorbar(iax,cax=fig.add_axes([0.01,0.01,0.98,0.05]),
>>>                   spacing="proportional",orientation="horizontal")
>>> cbar.set_ticks(ticks)
>>> plt.show()
"""

import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from numpy.typing import ArrayLike
from typing import Tuple
__all__ = ["midascmap"]


# The hex code for each color. (35 entries)
_COLORS = [
    "#000000",
    "#1500e6",
    "#fd02ff",
    "#ffb500",
    "#fcff0e",
    "#00ffaa",
    "#58caff",
    "#ffadff",
    "#fffeff",
    "#fedbf9",
    "#ffaefd",
    "#d370fe",
    "#a19dfe",
    "#5bc9fe",
    "#00ffff",
    "#00ffad",
    "#00fd0c",
    "#99fe00",
    "#feff01",
    "#ffd800",
    "#ffc500",
    "#ffb600",
    "#ff6e01",
    "#ff4e00",
    "#fe0000",
    "#ff00b6",
    "#ff00fe",
    "#c500ee",
    "#9800e8",
    "#8500e5",
    "#1500e5",
    "#1301d3",
    "#0f00bb",
    "#000124",
    "#000000",
]

# The mapped percentages for each color. (35 entries)
_PERCENTAGES = [
    0.0,
    0.25157233,
    0.62893082,
    1.00628931,
    1.3836478,
    1.88679245,
    2.26415094,
    2.64150943,
    3.01886792,
    3.39622642,
    4.1509434,
    5.40880503,
    6.5408805,
    8.17610063,
    9.3081761,
    11.32075472,
    13.20754717,
    15.59748428,
    17.61006289,
    20.75471698,
    24.27672956,
    24.65408805,
    28.17610063,
    32.0754717,
    33.33333333,
    37.61006289,
    43.52201258,
    50.56603774,
    57.98742138,
    59.24528302,
    65.91194969,
    76.10062893,
    77.35849057,
    87.54716981,
    89.05660377,
    100.0,
]


def _getbounds(
    data: ArrayLike, vmin: float = None, vmax: float = None, return_ticks: bool = False, percentiles : tuple = None
):
    
    if isinstance(percentiles,tuple) and vmin is None and vmax is None:
        vmin,vmax = np.percentile(data,percentiles)
    else:
        vmin = np.nanmin(data) if vmin is None else vmin
        vmax = np.nanmax(data) if vmax is None else vmax
        
    ticks = np.linspace(vmin, vmax, 10) if return_ticks else None

    vmax -= vmin
    bounds = np.array(_PERCENTAGES) / 100 * vmax
    bounds += vmin
    print(bounds)
    return bounds, ticks


def midascmap(
    data: ArrayLike, vmin: float = None, vmax: float = None, return_ticks: bool = False, percentiles : tuple = None
) -> Tuple[ListedColormap, BoundaryNorm]:
    """
    Creates a colormap that mimcis the MIDAS rainbow colormap.
    This cmap is useful to enhance the visibility of isophotes,
    especially in the center.

    If no vmin or vmax is parsed the min and max values of the
    data are used respectively.

    Parameters
    ----------
    data : ArrayLike
        The data/image array.

    vmin : float, optional
        The lower cut level of the data to color mapping.

    vmax : float, optional
        The upper cut level of the data to color mapping.

    return_ticks : bool, optional
        If ``True``, the method returns an additional array of 10 ticks, distributed between vmin and vmax.
        Used if one wants to plot the colorbar with ticks.

    Returns
    -------
        Colormap : ListedColormap
            A matplotlib colormap.

        Normalization : BoundaryNorm
            The calculated normalization depending on vmin and vmax.
    """
    _cmap = ListedColormap(_COLORS)
    _bounds, _ticks = _getbounds(data, vmin, vmax, return_ticks, percentiles)
    _norm = BoundaryNorm(_bounds, _cmap.N)
    if return_ticks:
        return _cmap, _norm, _ticks
    return _cmap, _norm
