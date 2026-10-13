# Copyright (C) 2019-2026 by Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of hydropack.
#
# hydropack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# hydropack source directory or at <http://www.gnu.org/licenses/>.

r"""North Lake (Greenland) lake-drainage set-up of Sect. 4.2

The experiment uses BedMachine Greenland v6 bed and ice thickness on a
1 km grid covering about 68.25-69.02 N and 49.05-51.47 W.  All four
boundaries are no-flux, and melt comes from the temperature-index model of
Eqs. (24)-(25).  North Lake (68.72 N, 49.50 W) drains 44e6 m^3 over 1.4 h,
and the initial condition is a 400-day spin-up.  BedMachine is not shipped
with hydropack; pass the path of the NetCDF file to :func:`load_bedmachine`.
"""

import numpy as np

from hydropack.glads2.constants import seconds_per_hour
from hydropack.glads2.forcing import parabolic_pulse, temperature_index_melt
from hydropack.glads2.model import BoundedHydrologyModel, PointSource, StaggeredGrid

#: location of North Lake (degrees north, degrees east)
north_lake_latlon = (68.72, -49.50)

#: approximate model domain (south, north, west, east) in degrees
north_lake_bounds_latlon = (68.25, 69.02, -51.47, -49.05)

_WGS84_A = 6378137.0
_WGS84_E = 0.0818191908426215


def latlon_to_epsg3413(lat, lon):
    r"""Project WGS84 latitude/longitude to NSIDC polar stereographic north (EPSG:3413)

    Ellipsoidal polar stereographic projection (Snyder, 1987, Eqs. 15-9,
    14-15 and 21-34) with true scale at 70 N and central meridian 45 W.
    """
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    lat_c, lon_0 = np.radians(70.0), np.radians(-45.0)
    e = _WGS84_E

    def t(phi):
        return np.tan(np.pi / 4 - phi / 2) / ((1 - e * np.sin(phi)) / (1 + e * np.sin(phi))) ** (e / 2)

    m_c = np.cos(lat_c) / np.sqrt(1 - e**2 * np.sin(lat_c) ** 2)
    rho = _WGS84_A * m_c * t(lat) / t(lat_c)
    return rho * np.sin(lon - lon_0), -rho * np.cos(lon - lon_0)


def north_lake_bounds(margin=0.0):
    r"""Return ``(xmin, xmax, ymin, ymax)`` in EPSG:3413 enclosing the domain corners"""
    south, north, west, east = north_lake_bounds_latlon
    lats = np.array([south, south, north, north])
    lons = np.array([west, east, west, east])
    x, y = latlon_to_epsg3413(lats, lons)
    return x.min() - margin, x.max() + margin, y.min() - margin, y.max() + margin


def load_bedmachine(path, bounds=None, resolution=1000.0):
    r"""Read and subsample BedMachine Greenland onto a regular grid

    Parameters
    ----------
    path : str
        Path to a BedMachine Greenland NetCDF file (variables ``x``, ``y``,
        ``bed``, ``thickness``, ``surface``)
    bounds : tuple, optional
        ``(xmin, xmax, ymin, ymax)`` in EPSG:3413; defaults to
        :func:`north_lake_bounds`
    resolution : float
        Target grid spacing (m); 1000 m in the paper

    Returns
    -------
    dict
        ``x`` and ``y`` cell-centre coordinates, and ``bed``, ``thickness``
        and ``surface`` arrays indexed as ``[i_x, j_y]``
    """
    import xarray

    xmin, xmax, ymin, ymax = bounds or north_lake_bounds()
    with xarray.open_dataset(path) as ds:
        sub = ds[["bed", "thickness", "surface"]].sel(x=slice(xmin, xmax))
        y = ds["y"].values
        sub = sub.sel(y=slice(ymax, ymin) if y[0] > y[-1] else slice(ymin, ymax))
        x_new = np.arange(xmin + resolution / 2, xmax, resolution)
        y_new = np.arange(ymin + resolution / 2, ymax, resolution)
        sub = sub.interp(x=x_new, y=y_new)
        return {
            "x": x_new,
            "y": y_new,
            "bed": sub["bed"].transpose("x", "y").values.astype(float),
            "thickness": np.maximum(sub["thickness"].transpose("x", "y").values.astype(float), 0.0),
            "surface": sub["surface"].transpose("x", "y").values.astype(float),
        }


def north_lake_drainage(
    topography,
    *,
    lake_volume=44e6,
    lake_duration=1.4 * seconds_per_hour,
    lake_start=0.0,
    lake_location=None,
    melt=None,
    **kwargs,
):
    r"""Return a model of the North Lake drainage experiment (Sect. 4.2)

    Parameters
    ----------
    topography : dict
        Output of :func:`load_bedmachine`, or any dictionary with the same
        keys on a uniform grid
    lake_volume, lake_duration, lake_start : float
        Lake drainage pulse; use ``lake_volume=0`` for the 400-day spin-up
    lake_location : tuple, optional
        ``(x, y)`` in EPSG:3413; defaults to North Lake
    melt : dict, optional
        Keyword overrides for
        :func:`~hydropack.glads2.forcing.temperature_index_melt`
    **kwargs
        Passed to :class:`~hydropack.glads2.model.BoundedHydrologyModel`
    """
    x, y = np.asarray(topography["x"]), np.asarray(topography["y"])
    dx, dy = x[1] - x[0], y[1] - y[0]
    grid = StaggeredGrid(x.size, y.size, dx, dy, x0=x[0] - dx / 2, y0=y[0] - dy / 2)
    H = np.nan_to_num(topography["thickness"], nan=0.0)
    B = np.nan_to_num(topography["bed"], nan=0.0)
    surface = np.nan_to_num(topography["surface"], nan=0.0)
    melt_rate = temperature_index_melt(np.maximum(surface, B + H), **(melt or {}))
    if lake_location is None:
        lake_location = latlon_to_epsg3413(*north_lake_latlon)
    sources = []
    if lake_volume > 0.0:
        sources.append(PointSource(*lake_location, parabolic_pulse(lake_volume, lake_duration, lake_start)))
    return BoundedHydrologyModel(
        grid,
        bed=B,
        thickness=H,
        melt_rate=melt_rate,
        point_sources=sources,
        boundary_conditions={side: "no_flux" for side in ("west", "east", "south", "north")},
        **kwargs,
    )
