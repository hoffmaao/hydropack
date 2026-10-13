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

r"""Coarse-resolution versions of the experiments in Wells et al. (2026)

These check the qualitative behaviour reported in Sect. 4 on grids small
enough to run in a few seconds.  The full-resolution figure scripts are in
``examples/glads2``.
"""

import numpy as np
import pytest

from hydropack.glads2 import experiments, north_lake, physics
from hydropack.glads2.constants import seconds_per_day, seconds_per_hour


@pytest.fixture(scope="module")
def drained():
    model = experiments.synthetic_lake_drainage(
        dx=2500.0, lake_volume=0.0, e_v=1e-2, h_g_init=0.02
    )
    model.h = model.h_g + 0.7 * model.delta_h
    model.diagnose()
    model.run(120 * seconds_per_day, dt_max=seconds_per_hour)
    steady = model.state()

    e_v = 1e-3
    lake = experiments.synthetic_lake_drainage(
        dx=2500.0,
        e_v=e_v,
        h_g_init=steady["h_g"],
        S_x_init=steady["S_x"],
        S_y_init=steady["S_y"],
        lake_volume=44e6,
    )
    lake.h = steady["h_w"] + e_v * steady["p_w"] / (1000.0 * 9.81)
    lake.diagnose()
    V0 = lake.total_water_volume()
    history = {}

    def record(m):
        history[round(m.t / seconds_per_hour)] = (m.state(), m.blister_mask(), m.blister_volume())

    lake.run(48 * seconds_per_hour, dt_max=60.0, callback=record, output_times=[0.0, 2 * 3600.0, 12 * 3600.0, 48 * 3600.0])
    return steady, lake, history, V0


def test_steady_state_has_pressurised_interior_and_free_surface_margin(drained):
    steady, lake, history, _ = drained
    regime = physics.sheet_regime(steady["h"], steady["h_g"], steady["delta_h"])
    ice = lake.H > 0
    # pressurised flow inland, free-surface flow at the ice margin (Fig. 4, t = 0)
    assert np.all(regime[:8][ice[:8]] == physics.PRESSURISED)
    terminus = np.nonzero(ice[:, 0])[0].max()
    assert np.all(regime[terminus] == physics.FREE_SURFACE)
    # atmospheric pressure in the proglacial zone
    assert np.all(steady["p_w"][~ice] == 0.0)


def test_lake_drainage_creates_blister_that_moves_downstream(drained):
    _, lake, history, _ = drained
    i_lake, j_lake = lake.grid.cell_index(10e3, 7.5e3)
    _, blister_2, volume_2 = history[2]
    _, blister_12, volume_12 = history[12]
    _, blister_48, volume_48 = history[48]
    # ice uplift at the injection site just after drainage
    assert blister_2[i_lake, j_lake]
    assert volume_2 > 0.5 * 44e6
    # the blister leaks into the surrounding system and its volume decays
    assert volume_48 < volume_12 < volume_2
    # and it travels down-glacier
    x = lake.grid.X

    def centroid(mask):
        return x[mask].mean() if mask.any() else np.inf

    assert centroid(blister_12) > centroid(blister_2)


def test_lake_drainage_conserves_mass(drained):
    _, lake, _, V0 = drained
    assert lake.clipped_volume == 0.0
    assert abs(lake.mass_balance_residual(V0)) < 1e-9 * (V0 + 44e6)
    # the lake volume and the background melt were all added; channel melt
    # adds a little more, and can be negative where the pressure-melting term
    # of Eq. (12) exceeds dissipation (supercooling at the blister edge)
    melt = 1e-3 / seconds_per_day * lake.grid.cell_area * lake.grid.nx * lake.grid.ny * lake.t
    channel = lake.source_volume - 44e6 - melt
    assert abs(channel) < 0.02 * 44e6


def test_stepped_glacier_geometry():
    data = experiments.load_stepped_glacier()
    assert data["x"][0] == 0.0 and data["x"][-1] == pytest.approx(50e3)
    assert np.all(np.diff(data["bed"]) <= 1e-6)
    assert data["thickness"][-1] == 0.0 and np.all(data["thickness"][:-1] > 0.0)
    model = experiments.stepped_glacier(channels=False)
    assert model.grid.shape == (100, 1)
    assert model.e_v == 1e-5
    assert model.boundary_conditions["east"][0] == "atmospheric"


def test_north_lake_projection_and_setup():
    # reference values from PROJ for EPSG:4326 -> EPSG:3413
    x, y = north_lake.latlon_to_epsg3413(68.72, -49.50)
    assert x == pytest.approx(-182888.467, abs=0.01)
    assert y == pytest.approx(-2323818.305, abs=0.01)
    x, y = north_lake.latlon_to_epsg3413(60.0, 0.0)
    assert x == pytest.approx(2349829.162, abs=0.01)
    assert y == pytest.approx(-2349829.162, abs=0.01)

    # a synthetic stand-in for BedMachine around the lake
    x0, y0 = north_lake.latlon_to_epsg3413(*north_lake.north_lake_latlon)
    xs = x0 + np.arange(-5, 5) * 1000.0 + 500.0
    ys = y0 + np.arange(-4, 4) * 1000.0 + 500.0
    X, _ = np.meshgrid(xs, ys, indexing="ij")
    bed = 300.0 - 0.01 * (X - xs[0])
    thickness = np.full(X.shape, 1000.0)
    thickness[0, :] = np.nan
    topo = {"x": xs, "y": ys, "bed": bed, "thickness": thickness, "surface": bed + np.nan_to_num(thickness)}
    model = north_lake.north_lake_drainage(topo, e_v=1e-2)
    assert model.grid.cell_index(x0, y0) == (5, 4)
    assert all(bc[0] == "no_flux" for bc in model.boundary_conditions.values())
    assert np.all(model.H[0] == 0.0)
    melt = model.melt(0.0)
    assert melt[1, 0] == pytest.approx(2e-8 * (11.0 - 0.006 * (model.B[1, 0] + 1000.0 - 50.0)))
    V0 = model.total_water_volume()
    model.run(3 * seconds_per_hour, dt_max=60.0)
    assert abs(model.mass_balance_residual(V0)) < 1e-9 * model.total_water_volume()
    assert model.blister_mask()[5, 4]
