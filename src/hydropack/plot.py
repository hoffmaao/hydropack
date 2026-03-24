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

r"""Visualization helpers for subglacial hydrology fields

This module provides functions for plotting CR1 (edge) fields on Firedrake
meshes.  The main function :func:`tripcolor_cr` mirrors the calling
convention of :func:`firedrake.tripcolor` for Crouzeix-Raviart functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import firedrake as fd


def _edge_segments(mesh):
    """Extract (n_edges, 2, 2) array of edge endpoint coordinates from DMPlex."""
    dm = mesh.topology_dm
    sec = dm.getCoordinateSection()
    coords_local = dm.getCoordinatesLocal()
    arr = coords_local.array
    vStart, vEnd = dm.getDepthStratum(0)
    gdim = sec.getDof(vStart)

    XY = np.empty((vEnd - vStart, gdim), dtype=float)
    for p in range(vStart, vEnd):
        off = sec.getOffset(p)
        XY[p - vStart, :] = arr[off:off + gdim]

    eStart, eEnd = dm.getDepthStratum(1)
    segments = np.empty((eEnd - eStart, 2, 2), dtype=float)
    for i, e in enumerate(range(eStart, eEnd)):
        cone = dm.getCone(e)
        segments[i, 0, :] = XY[int(cone[0]) - vStart, :2]
        segments[i, 1, :] = XY[int(cone[1]) - vStart, :2]
    return segments


def _cr_values(func):
    """Extract CR1 DOF values in DMPlex edge order."""
    V = func.function_space()
    family = V.ufl_element().family()
    if family != "Crouzeix-Raviart":
        raise ValueError(
            f"Expected a Crouzeix-Raviart function, got {family}. "
            "Project to CR1 first."
        )
    return func.dat.data_ro.copy()


def tripcolor_cr(function, *args, **kwargs):
    r"""Create a pseudo-color plot of a CR1 (edge) field

    This is the edge-field analogue of :func:`firedrake.tripcolor`.
    Each mesh edge is drawn as a colored line segment whose color is
    set by the CR1 function value on that edge.

    Parameters
    ----------
    function : firedrake.Function
        A function on a Crouzeix-Raviart space.
    axes : matplotlib.axes.Axes or None
        Target axes (created if ``None``).
    threshold : float
        Minimum absolute value to display (default 0).
    lw_scale : float
        Linewidth scaling factor (default 2.0).
    lw_min : float
        Minimum linewidth for visible edges (default 0.3).
    vmin, vmax : float or None
        Color limits (auto-scaled if ``None``).
    cmap : str
        Colormap name (default ``"inferno"``).

    Returns
    -------
    matplotlib.collections.LineCollection
    """
    axes = kwargs.pop("axes", None)
    threshold = kwargs.pop("threshold", 0.0)
    lw_scale = kwargs.pop("lw_scale", 2.0)
    lw_min = kwargs.pop("lw_min", 0.3)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)

    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    mesh = function.function_space().mesh()
    segments = _edge_segments(mesh)
    vals = np.abs(_cr_values(function))

    if threshold > 0:
        mask = vals >= threshold
        segments = segments[mask]
        vals = vals[mask]

    if vmin is None:
        vmin = vals.min() if len(vals) > 0 else 0.0
    if vmax is None:
        vmax = vals.max() if len(vals) > 0 else 1.0

    log_v = np.log10(np.maximum(vals, 1e-30))
    log_lo = np.log10(max(vmin, 1e-30))
    log_hi = np.log10(max(vmax, 1e-30))
    log_range = log_hi - log_lo
    if log_range > 0:
        lw = lw_min + lw_scale * (log_v - log_lo) / log_range
    else:
        lw = np.full_like(vals, lw_min + lw_scale / 2)

    kwargs["linewidths"] = lw
    lc = LineCollection(segments, **kwargs)
    lc.set_array(vals)
    lc.set_clim(vmin, vmax)

    axes.add_collection(lc)
    axes.autoscale_view()

    return lc
