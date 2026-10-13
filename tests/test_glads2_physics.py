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

r"""Unit tests for the GlaDS-2 constitutive relations (Wells et al., 2026)"""

import numpy as np
import pytest

from hydropack.glads2 import physics, forcing
from hydropack.glads2.constants import (
    gravity as g,
    ice_density as rho_i,
    water_density as rho_w,
    latent_heat as L,
    heat_capacity_water as c_w,
    clapeyron_slope as c_t,
    default_parameters as P,
)


H = 500.0
p_i = rho_i * g * H
e_v = 1e-3
delta_h = e_v * rho_i / rho_w * H
h_g = 0.05


def test_storage_capacity():
    # Eq. (3): delta_h = e_v p_i / (rho_w g) = e_v rho_i H / rho_w
    assert physics.storage_capacity(H, e_v) == pytest.approx(e_v * p_i / (rho_w * g))
    assert physics.storage_capacity(0.0, e_v) == 0.0


def test_closure_relation_regimes():
    # Eq. (4) in each of the three regimes
    h = np.array([0.0, 0.5 * h_g, h_g, h_g + 0.25 * delta_h, h_g + delta_h, h_g + 2 * delta_h, 10.0])
    p_w = physics.water_pressure(h, h_g, p_i, delta_h)
    expected = np.array([0.0, 0.0, 0.0, 0.25 * p_i, p_i, p_i, p_i])
    assert np.allclose(p_w, expected)
    regime = physics.sheet_regime(h, h_g, delta_h)
    F, P_, U = physics.FREE_SURFACE, physics.PRESSURISED, physics.UPLIFT
    assert regime.tolist() == [F, F, P_, P_, P_, U, U]


def test_closure_is_continuous_and_monotone():
    h = np.linspace(0.0, h_g + 3 * delta_h, 2001)
    p_w = physics.water_pressure(h, h_g, p_i, delta_h)
    assert np.all(np.diff(p_w) >= 0.0)
    assert np.max(np.abs(np.diff(p_w))) <= p_i * (h[1] - h[0]) / delta_h * (1 + 1e-9)
    assert p_w.min() >= 0.0 and p_w.max() <= p_i


def test_proglacial_pressure_is_atmospheric():
    # with p_i = 0 both bounds collapse to atmospheric pressure
    h = np.linspace(0.0, 2.0, 11)
    assert np.all(physics.water_pressure(h, h_g, 0.0, 0.0) == 0.0)
    assert np.allclose(physics.flowing_thickness(h, h_g, 0.0), h)


def test_partition_of_water_thickness():
    # Eqs. (1)-(2): h = h_w + h_e and h_e = e_v p_w / (rho_w g)
    h = np.linspace(0.0, h_g + 3 * delta_h, 101)
    p_w = physics.water_pressure(h, h_g, p_i, delta_h)
    h_e = physics.englacial_storage(h, h_g, delta_h)
    h_w = physics.flowing_thickness(h, h_g, delta_h)
    assert np.allclose(h_w + h_e, h)
    assert np.allclose(h_e, e_v * p_w / (rho_w * g))
    regime = physics.sheet_regime(h, h_g, delta_h)
    assert np.allclose(h_w[regime == physics.FREE_SURFACE], h[regime == physics.FREE_SURFACE])
    assert np.allclose(h_w[regime == physics.PRESSURISED], h_g)
    up = regime == physics.UPLIFT
    assert np.allclose(h_w[up], h[up] - delta_h)


def test_hydraulic_potential_bounds():
    # phi_m <= phi - rho_w g h_w <= phi_0 (Sect. 2.1)
    rng = np.random.default_rng(0)
    B = rng.uniform(-100, 500, 200)
    h = rng.uniform(0, 1, 200)
    p_w = physics.water_pressure(h, h_g, p_i, delta_h)
    h_w = physics.flowing_thickness(h, h_g, delta_h)
    phi = physics.hydraulic_potential(B, h_w, p_w)
    reduced = phi - rho_w * g * h_w
    assert np.all(reduced >= rho_w * g * B - 1e-6)
    assert np.all(reduced <= rho_w * g * B + p_i + 1e-6)


def test_cavity_height_update():
    u_b, N = 3.2e-6, 2e6
    kw = dict(A=P["A"], h_r=P["h_r"], l_r=P["l_r"])
    # steady state of Eq. (5)
    closure = 2.0 / 27.0 * P["A"] * N**3
    h_steady = u_b * P["h_r"] / P["l_r"] / (u_b / P["l_r"] + closure)
    assert physics.cavity_height_rate(h_steady, N, u_b, **kw) == pytest.approx(0.0, abs=1e-20)
    assert physics.update_cavity_height(h_steady, N, u_b, 1e9, **kw) == pytest.approx(h_steady)
    # consistent with the rate for a small step
    dt = 1.0
    h1 = physics.update_cavity_height(0.02, N, u_b, dt, **kw)
    assert (h1 - 0.02) / dt == pytest.approx(physics.cavity_height_rate(0.02, N, u_b, **kw), rel=1e-4)
    # bounded between 0 and h_r for any time step when N >= 0
    for dt in [1.0, 1e5, 1e12]:
        for N_ in [0.0, 1e5, 1e7]:
            h = physics.update_cavity_height(np.array([0.0, 0.05, 0.1]), N_, u_b, dt, **kw)
            assert np.all(h >= 0.0) and np.all(h <= P["h_r"] + 1e-15)


def test_water_filled_area():
    S = 2.0
    R = physics.channel_radius(S)
    assert np.pi * R**2 / 2 == pytest.approx(S)
    # empty at atmospheric pressure, full when p_w >= rho_w g R
    assert physics.water_filled_area(S, 0.0) == 0.0
    assert physics.water_filled_area(S, rho_w * g * R) == pytest.approx(S)
    assert physics.water_filled_area(S, 1e6) == pytest.approx(S)
    # circular segment for R_w = R/2 (Eq. 15)
    S_w = physics.water_filled_area(S, rho_w * g * R / 2)
    expected = S - R**2 * np.arccos(0.5) + 0.5 * R * np.sqrt(R**2 - R**2 / 4)
    assert S_w == pytest.approx(expected)
    # monotone in pressure and never larger than S
    p = np.linspace(0, 2 * rho_w * g * R, 101)
    S_w = physics.water_filled_area(np.full_like(p, S), p)
    assert np.all(np.diff(S_w) >= -1e-14) and np.all(S_w <= S)
    assert physics.water_filled_area(0.0, 1e5) == 0.0


def test_channel_regime():
    S = np.array([1.0, 1.0, 1.0])
    S_w = np.array([0.0, 0.4, 1.0])
    assert physics.channel_regime(S, S_w).tolist() == [physics.EMPTY, physics.PARTIALLY_FILLED, physics.FULLY_FILLED]


def test_flux_laws():
    kw = dict(k=0.01, phi_reg=0.0)
    h_w, grad = 0.1, -100.0
    q = physics.sheet_flux(h_w, grad, grad**2, **kw)
    assert q == pytest.approx(0.01 * 0.1**1.25 * 100.0**0.5)
    assert physics.sheet_flux(0.0, grad, grad**2, **kw) == 0.0
    Q = physics.channel_discharge(1.0, grad, k_c=0.1, phi_reg=0.0)
    assert Q == pytest.approx(0.1 * 100.0**0.5)
    # flow is down the potential gradient
    assert physics.channel_discharge(1.0, -grad, k_c=0.1, phi_reg=0.0) == pytest.approx(-Q)


def test_dissipation_and_pressure_melting():
    grad, dpw = -50.0, -20.0
    Q = physics.channel_discharge(0.5, grad, k_c=0.1, phi_reg=0.0)
    q_c = physics.sheet_flux(0.05, grad, grad**2, k=0.01, phi_reg=0.0)
    Xi = physics.channel_dissipation(Q, q_c, grad, l_c=2.0)
    # Eq. (11): dissipation is non-negative and matches its definition
    assert Xi > 0.0
    assert Xi == pytest.approx(-(Q + 2.0 * q_c) * grad)
    Pi = physics.channel_pressure_melting(Q, q_c, dpw, 0.5, l_c=2.0)
    # Eq. (12) with c_t < 0: flow towards lower pressure cools the water, Pi < 0
    assert Pi == pytest.approx(-c_t * c_w * rho_w * (Q + 2.0 * q_c) * dpw)
    assert Pi < 0.0
    assert abs(Pi) < Xi


def test_pressure_melting_switch():
    # Eq. (13)
    S = np.array([0.0, 0.0, 1.0])
    q_c = np.array([1.0, 1.0, 1.0])
    dpw = np.array([-1.0, 1.0, -1.0])
    assert physics.pressure_melting_switch(S, q_c, dpw).tolist() == [0.0, 1.0, 1.0]


def test_channel_area_update():
    N, A_c = 1e6, P["A_c"]
    closure = 2.0 / 27.0 * A_c * N**3
    Xi, Pi = 10.0, -1.0
    S_steady = (Xi + Pi) / (rho_i * L) / closure
    assert physics.channel_area_rate(S_steady, Xi, Pi, N, A_c=A_c) == pytest.approx(0.0, abs=1e-18)
    assert physics.update_channel_area(S_steady, Xi, Pi, N, 1e10, A_c=A_c) == pytest.approx(S_steady)
    # never negative, even when freezing exceeds the available area
    assert physics.update_channel_area(1e-3, 0.0, -1e3, N, 1e6, A_c=A_c) == 0.0


def test_channel_melt_source():
    assert physics.channel_melt_source(334.0, 0.0) == pytest.approx(334.0 / (rho_w * L))


def test_effective_coefficients_and_timestep():
    grad2 = np.array([100.0**2])
    kw = dict(k=0.01, k_c=0.1, e_v=1e-4, phi_reg=0.0)
    D_uf = physics.effective_diffusivity(0.1, 0.0, grad2, False, 0.0, **kw)
    D_p = physics.effective_diffusivity(0.1, 0.0, grad2, True, 0.0, **kw)
    # Eqs. (19)-(20)
    assert D_uf == pytest.approx(rho_w * g * 0.01 * 0.1**1.25 / 10.0)
    assert D_p == pytest.approx(D_uf / 1e-4)
    # Eq. (23)
    dt = physics.stable_timestep(D_p[0], 0.0, 500.0, 500.0, C=0.5)
    assert dt == pytest.approx(0.5 * 500.0**2 / (4.0 * D_p[0]))
    assert physics.stable_timestep(0.0, 2.0, 500.0, 200.0, C=0.5) == pytest.approx(0.5 * 200.0 / 4.0)


def test_parabolic_pulse():
    V, T = 44e6, 1.4 * 3600.0
    rate = forcing.parabolic_pulse(V, T, t_start=100.0)
    t = np.linspace(100.0, 100.0 + T, 20001)
    r = np.array([rate(s) for s in t])
    assert np.trapezoid(r, t) == pytest.approx(V, rel=1e-6)
    assert t[np.argmax(r)] == pytest.approx(100.0 + T / 2, abs=T / 1000)
    assert rate(50.0) == 0.0 and rate(200.0 + T) == 0.0
    assert forcing.parabolic_pulse_volume(V, T, 100.0 + T / 2, 100.0) == pytest.approx(V / 2)


def test_temperature_index_melt():
    # Eqs. (24)-(25)
    m = forcing.temperature_index_melt(np.array([50.0, 1050.0, 3000.0]))
    assert m[0] == pytest.approx(2e-8 * 11.0)
    assert m[1] == pytest.approx(2e-8 * 5.0)
    assert m[2] == 0.0
