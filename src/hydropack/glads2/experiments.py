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

r"""Experiment set-ups from Wells et al. (2026)

:func:`synthetic_lake_drainage` builds the synthetic land-terminating
glacier of Sect. 4.1 (Figs. 4-6).  The paper describes it as a 50 x 15 km
domain with a constant-slope bed and a quadratically decreasing ice
thickness, with the last ~3 km ice free.  It does not give the formulas,
so the defaults below were digitised from the potentials in Fig. 4b:

* :math:`B = B_0 (1 - x/L_B)`, with :math:`B_0 = 200` m and
  :math:`L_B = 50` km
* :math:`H = \max(0, H_0 + a x - c x^2)`, with :math:`H_0 = 900` m,
  :math:`a = 2.646\times10^{-3}` and :math:`c = 4.5255\times10^{-7}`
  m^-1, which puts the terminus at :math:`x \approx 47.6` km.  This fits
  the digitised overburden to within about 9 m.

All of them can be overridden.

:func:`stepped_glacier` builds the steady-state comparison of Sect. 3
(Fig. 3) against Schoof et al. (2012) and Hewitt et al. (2012).  Its bed and
ice thickness were digitised from the potentials shaded in Fig. 3a and are
stored in ``data/stepped_glacier.csv``, together with the digitised
reference curves for the sheet-only case.
"""

import os

import numpy as np

from hydropack.glads2.constants import seconds_per_day, seconds_per_hour
from hydropack.glads2.forcing import parabolic_pulse
from hydropack.glads2.model import BoundedHydrologyModel, PointSource, StaggeredGrid


def synthetic_geometry(grid, B0=200.0, bed_length=50e3, H0=900.0, a=2.646e-3, c=4.5255e-7):
    r"""Return bed and ice thickness for the synthetic topography of Sect. 4.1"""
    B = B0 * (1.0 - grid.X / bed_length)
    H = np.maximum(0.0, H0 + a * grid.X - c * grid.X**2)
    return B, H


def synthetic_lake_drainage(
    *,
    dx=500.0,
    length=50e3,
    width=15e3,
    melt_rate=1e-3 / seconds_per_day,
    inflow=1e-6,
    lake_volume=44e6,
    lake_duration=1.4 * seconds_per_hour,
    lake_location=(10e3, 7.5e3),
    lake_start=0.0,
    geometry=None,
    channels=True,
    **kwargs,
):
    r"""Return a model of the synthetic lake-drainage experiment (Sect. 4.1)

    Parameters
    ----------
    dx : float
        Grid spacing in both directions (500 m in the paper)
    melt_rate : float
        Uniform distributed melt (1 mm/d in Fig. 4, 20 mm/d in Fig. 6)
    inflow : float
        Sheet inflow per unit width on the upstream boundary (1 mm^2/s)
    lake_volume, lake_duration : float
        Lake volume (44e6 m^3; 20e6 m^3 in Fig. 6) drained as a parabolic
        pulse over ``lake_duration`` (1.4 h).  Use ``lake_volume=0`` to
        spin up the initial steady state.
    lake_start : float
        Model time at which the drainage begins
    geometry : dict, optional
        Keyword overrides for :func:`synthetic_geometry`
    **kwargs
        Passed to :class:`~hydropack.glads2.model.BoundedHydrologyModel`
        (e.g. ``e_v``, ``h_init``, ``h_g_init``, ``S_x_init``)
    """
    nx, ny = int(round(length / dx)), int(round(width / dx))
    grid = StaggeredGrid(nx, ny, dx, dx)
    B, H = synthetic_geometry(grid, **(geometry or {}))
    sources = []
    if lake_volume > 0.0:
        sources.append(PointSource(*lake_location, parabolic_pulse(lake_volume, lake_duration, lake_start)))
    bcs = {"west": ("flux", inflow), "east": "atmospheric", "south": "no_flux", "north": "no_flux"}
    return BoundedHydrologyModel(
        grid,
        bed=B,
        thickness=H,
        melt_rate=melt_rate,
        point_sources=sources,
        boundary_conditions=bcs,
        channels=channels,
        **kwargs,
    )


_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_stepped_glacier():
    r"""Return the digitised stepped-glacier data as a dictionary of arrays

    Keys are ``x`` (m), ``bed`` (m), ``thickness`` (m), and the reference
    sheet-only steady state for :math:`m = 20` mm/d: ``phi_ref`` (Pa) and
    ``h_ref`` (m), which are NaN where they were hidden in the figure.
    """
    data = np.loadtxt(os.path.join(_DATA_DIR, "stepped_glacier.csv"), delimiter=",")
    return {
        "x": data[:, 0],
        "bed": data[:, 1],
        "thickness": data[:, 2],
        "phi_ref": data[:, 3] * 1e6,
        "h_ref": data[:, 4] * 1e-2,
    }


def stepped_glacier(*, melt_rate=20e-3 / seconds_per_day, channels=True, dx=500.0, dy=200.0, **kwargs):
    r"""Return a model of the stepped-glacier steady states of Sect. 3 (Fig. 3)

    The glacier is 50 km long and uniform across flow.  As in the paper,
    channels are spaced by the grid spacing ``dy = 200`` m, and the domain
    is a single cell wide.  The upstream boundary has no flux and the
    terminus is at atmospheric pressure.  The paper uses
    :math:`e_v = 10^{-5}` and melt inputs of 20 and 5 mm/d.

    Parameters
    ----------
    melt_rate : float
        Uniform melt input (m/s)
    channels : bool
        ``False`` for the sheet-only case (Fig. 3, left column)
    **kwargs
        Passed to :class:`~hydropack.glads2.model.BoundedHydrologyModel`
    """
    data = load_stepped_glacier()
    length = data["x"][-1]
    nx = int(round(length / dx))
    grid = StaggeredGrid(nx, 1, dx, dy)
    B = np.interp(grid.X, data["x"], data["bed"])
    H = np.interp(grid.X, data["x"], data["thickness"])
    kwargs.setdefault("e_v", 1e-5)
    bcs = {"west": "no_flux", "east": "atmospheric", "south": "no_flux", "north": "no_flux"}
    return BoundedHydrologyModel(
        grid,
        bed=B,
        thickness=H,
        melt_rate=melt_rate,
        boundary_conditions=bcs,
        channels=channels,
        **kwargs,
    )


def spin_up(model, duration, e_v_spin_up=1e-3, dt_max=3600.0, final_duration=0.0):
    r"""Approach a steady state quickly using a large englacial void ratio

    Steady states do not depend on :math:`e_v` (Sect. 4.1.1), while the
    explicit time step scales with it (Sect. 2.5).  This integrates for
    ``duration`` with ``e_v_spin_up``, restores the original :math:`e_v`
    while keeping the water pressure fixed, and optionally integrates for
    ``final_duration`` more.
    """
    e_v = model.e_v
    model.set_englacial_void_ratio(max(e_v, e_v_spin_up))
    model.run(model.t + duration, dt_max=dt_max)
    model.set_englacial_void_ratio(e_v)
    if final_duration > 0.0:
        model.run(model.t + final_duration, dt_max=dt_max)
    return model
