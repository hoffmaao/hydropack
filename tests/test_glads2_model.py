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

r"""Tests of the GlaDS-2 finite-difference model

Steady states are checked against analytic solutions in each of the three
sheet regimes (free-surface, pressurised and ice uplift), and the
discretisation is checked for exact mass conservation and symmetry.
"""

import numpy as np
import pytest

from hydropack.glads2 import BoundedHydrologyModel, StaggeredGrid, PointSource, physics
from hydropack.glads2.constants import gravity as g, water_density as rho_w, ice_density as rho_i
from hydropack.glads2.forcing import parabolic_pulse


def run_to_steady_state(model, tol, t_max, chunk=86400.0, dt_max=np.inf):
    while model.t < t_max:
        model.run(model.t + chunk, dt_max=dt_max)
        if np.abs(model.dhdt()).max() < tol:
            return model
    raise AssertionError(f"no steady state after {t_max} s; max|dh/dt| = {np.abs(model.dhdt()).max()}")


def test_free_surface_flow_on_proglacial_slope():
    r"""Kinematic free-surface sheet on an ice-free slope.

    With :math:`H = 0` the pressure is atmospheric everywhere and
    :math:`\phi = \rho_w g (B + h)`.  For a bed of slope :math:`s` and a
    divide at :math:`x = 0`, the steady flux is :math:`q = m x`, so
    :math:`h = (m x / (k\sqrt{\rho_w g s}))^{1/\alpha}` when
    :math:`|h'| \ll s`.  Upwinding evaluates the face flux with the upstream
    cell value, so the cell value matches the analytic solution at the
    downstream face.
    """
    nx, dx, s, m, k = 40, 250.0, 0.01, 1e-6, 0.01
    grid = StaggeredGrid(nx, 1, dx, 100.0)
    model = BoundedHydrologyModel(
        grid,
        bed=-s * grid.X,
        thickness=0.0,
        melt_rate=m,
        h_init=0.01,
        boundary_conditions={"east": "atmospheric"},
        channels=False,
        k=k,
        phi_reg=1e-3,
    )
    run_to_steady_state(model, tol=1e-12, t_max=50 * 86400.0, chunk=0.5 * 86400.0)
    x_face = grid.x_faces[1:]
    h_exact = (m * x_face / (k * np.sqrt(rho_w * g * s))) ** (1.0 / 1.25)
    assert np.all(model.p_w == 0.0)
    assert np.allclose(model.h[:, 0], h_exact, rtol=5e-3)


def test_pressurised_flow_matches_analytic_potential():
    r"""Pressurised sheet with fixed cavity height on a flat bed.

    With :math:`A = 0` the cavities stay at :math:`h_g = h_r` and the
    flowing thickness is :math:`h_r`.  An inflow :math:`q_0` at :math:`x = 0`
    keeps the gradient away from zero, and the steady flux
    :math:`q = q_0 + m x` then requires

    .. math::
        p_w(x) = \frac{(q_0 + m L)^3 - (q_0 + m x)^3}{3 m K^2},
        \qquad K = k h_r^\alpha

    for atmospheric pressure at :math:`x = L`.  The ghost-point method of
    Sect. 2.3 imposes :math:`p_w = 0` at the centre of the ghost cell, so
    :math:`L` is half a cell beyond the last face.
    """
    nx, dx, m, q_0, k, h_r = 20, 500.0, 1e-6, 5e-3, 0.01, 0.1
    L = nx * dx + dx / 2
    K = k * h_r**1.25
    grid = StaggeredGrid(nx, 1, dx, 100.0)
    model = BoundedHydrologyModel(
        grid,
        bed=0.0,
        thickness=800.0,
        sliding_speed=1e-6,
        melt_rate=m,
        h_g_init=h_r,
        boundary_conditions={"west": ("flux", q_0), "east": "atmospheric"},
        channels=False,
        k=k,
        A=0.0,
        h_r=h_r,
        e_v=1e-2,
        phi_reg=1e-3,
    )
    model.h = model.h_g + 0.5 * model.delta_h
    model.diagnose()
    run_to_steady_state(model, tol=1e-10, t_max=1000 * 86400.0, chunk=5 * 86400.0)
    x = grid.x
    p_exact = ((q_0 + m * L) ** 3 - (q_0 + m * x) ** 3) / (3.0 * m * K**2)
    assert np.all(model.sheet_regime() == physics.PRESSURISED)
    assert np.allclose(model.h_w[:, 0], h_r)
    assert np.allclose(model.p_w[:, 0], p_exact, rtol=1e-2)


def test_ice_uplift_matches_kinematic_solution():
    r"""Sheet under uniformly sloping ice at overburden pressure.

    Under ice uplift :math:`p_w = p_i`, so :math:`\phi = \phi_0 + \rho_w g h_w`
    and the steady flux :math:`q = m x` gives the kinematic solution
    :math:`h_w = (m x / (k\sqrt{|\nabla\phi_0|}))^{1/\alpha}` with
    :math:`h = h_w + \Delta h`.
    """
    nx, dx, m, k, e_v = 40, 250.0, 1e-6, 0.01, 1e-4
    grid = StaggeredGrid(nx, 1, dx, 100.0)
    H0, slope_H, slope_B = 600.0, 0.01, 0.005
    H = H0 - slope_H * grid.X
    B = -slope_B * grid.X
    grad_phi0 = rho_w * g * slope_B + rho_i * g * slope_H
    model = BoundedHydrologyModel(
        grid,
        bed=B,
        thickness=H,
        sliding_speed=0.0,
        melt_rate=m,
        h_g_init=0.01,
        boundary_conditions={"east": "atmospheric"},
        channels=False,
        k=k,
        e_v=e_v,
        phi_reg=1e-3,
    )
    model.h = model.h_g + model.delta_h + 0.05
    model.diagnose()
    run_to_steady_state(model, tol=1e-12, t_max=50 * 86400.0, chunk=0.5 * 86400.0)
    # the first cell, which receives almost no water, stays pressurised and
    # the last two adjust to atmospheric pressure at the terminus
    interior = slice(1, nx - 2)
    x_face = grid.x_faces[1:]
    h_w_exact = (m * x_face / (k * np.sqrt(grad_phi0))) ** (1.0 / 1.25)
    assert np.all(model.sheet_regime()[interior] == physics.UPLIFT)
    assert np.allclose(model.p_w[interior], model.p_i[interior])
    assert np.allclose(model.h_w[interior, 0], h_w_exact[interior], rtol=2e-3)
    assert np.allclose(model.h[interior, 0] - model.h_w[interior, 0], model.delta_h[interior, 0])


@pytest.mark.parametrize("channels", [False, True])
def test_mass_conservation_in_closed_domain(channels):
    r"""The finite-volume update conserves water to round-off (Eq. 17)."""
    rng = np.random.default_rng(1)
    grid = StaggeredGrid(12, 9, 500.0, 500.0)
    H = 400.0 + 50.0 * rng.random(grid.shape)
    B = 100.0 - 0.002 * grid.X + 5.0 * rng.random(grid.shape)
    model = BoundedHydrologyModel(
        grid,
        bed=B,
        thickness=H,
        melt_rate=lambda t: 1e-7 * (1.0 + np.sin(t / 3600.0)),
        point_sources=[PointSource(3e3, 2e3, parabolic_pulse(1e6, 3600.0))],
        h_g_init=0.03 + 0.02 * rng.random(grid.shape),
        S_x_init=0.1 * rng.random(grid.x_face_shape) if channels else None,
        channels=channels,
        e_v=1e-3,
    )
    model.h = model.h_g + rng.uniform(-0.02, 1.5, grid.shape) * model.delta_h
    model.diagnose()
    V0 = model.total_water_volume()
    model.run(2 * 3600.0)
    added = model.source_volume
    assert added > 1e6
    assert model.boundary_inflow_volume == 0.0
    assert model.clipped_volume == 0.0
    assert abs(model.mass_balance_residual(V0)) < 1e-9 * (V0 + added)
    # all three regimes are present for this initial state
    assert set(np.unique(model.sheet_regime())) >= {physics.PRESSURISED, physics.UPLIFT}


def test_flux_boundary_condition_and_outflow_accounting():
    grid = StaggeredGrid(10, 4, 500.0, 500.0)
    q_in = 1e-3
    model = BoundedHydrologyModel(
        grid,
        bed=50.0 - 0.005 * grid.X,
        thickness=300.0,
        boundary_conditions={"west": ("flux", q_in), "east": "atmospheric"},
        channels=True,
        e_v=1e-3,
    )
    V0 = model.total_water_volume()
    model.run(86400.0)
    assert np.allclose(model.q_x[0, :], q_in)
    assert np.allclose(model.q_y[:, [0, -1]], 0.0)
    assert abs(model.mass_balance_residual(V0)) < 1e-9 * model.total_water_volume()


def test_blister_spreads_symmetrically():
    r"""A lake drained into the centre of a flat, closed domain stays symmetric."""
    grid = StaggeredGrid(15, 15, 500.0, 500.0)
    model = BoundedHydrologyModel(
        grid,
        bed=0.0,
        thickness=800.0,
        point_sources=[PointSource(3750.0, 3750.0, parabolic_pulse(5e6, 3600.0))],
        h_g_init=0.05,
        channels=True,
        e_v=1e-3,
    )
    model.h = model.h_g + 0.5 * model.delta_h
    model.diagnose()
    V0 = model.total_water_volume()
    model.run(3 * 3600.0)
    h = model.h
    assert np.allclose(h, h[::-1, :], rtol=1e-9, atol=1e-12)
    assert np.allclose(h, h[:, ::-1], rtol=1e-9, atol=1e-12)
    assert np.allclose(h, h.T, rtol=1e-9, atol=1e-12)
    assert model.blister_mask()[7, 7]
    assert model.blister_volume() > 0.0
    assert abs(model.mass_balance_residual(V0)) < 1e-9 * model.total_water_volume()


def test_englacial_void_ratio_change_preserves_pressure():
    grid = StaggeredGrid(5, 3, 500.0, 500.0)
    model = BoundedHydrologyModel(grid, bed=0.0, thickness=500.0, h_g_init=0.05, e_v=1e-2)
    model.h = model.h_g + np.linspace(-0.01, 1.5, 15).reshape(5, 3) * model.delta_h
    model.diagnose()
    p_w, h_w = model.p_w.copy(), model.h_w.copy()
    model.set_englacial_void_ratio(1e-5)
    assert np.allclose(model.p_w, p_w)
    assert np.allclose(model.h_w, h_w)
    with pytest.raises(ValueError):
        model.set_englacial_void_ratio(0.0)


def test_constitutive_relations_are_swappable():
    calls = []

    def water_pressure(h, h_g, p_i, delta_h):
        calls.append(True)
        return physics.water_pressure(h, h_g, p_i, delta_h)

    grid = StaggeredGrid(4, 4, 500.0, 500.0)
    model = BoundedHydrologyModel(grid, bed=0.0, thickness=100.0, water_pressure=water_pressure, k=0.02)
    model.step()
    assert calls and model.params["k"] == 0.02
    with pytest.raises(TypeError):
        BoundedHydrologyModel(grid, bed=0.0, thickness=100.0, not_a_parameter=1.0)
    with pytest.raises(ValueError):
        BoundedHydrologyModel(grid, bed=0.0, thickness=100.0, boundary_conditions={"west": "open"})
