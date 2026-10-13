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

r"""Constitutive relations for the bounded-pressure GlaDS-2 formulation

This module contains the constitutive relations of GlaDS-2 (Wells et al.,
2026), which extends GlaDS (Werder et al., 2013) with ice uplift and
free-surface flow by bounding the water pressure between atmospheric and
ice-overburden pressure.  The prognostic sheet variable :math:`h` is the
total water volume per unit bed area, split into a flowing part
:math:`h_w` and an englacially stored part :math:`h_e`.  A piecewise
closure (Eq. 4) maps :math:`h` to the water pressure and implicitly selects
one of three regimes:

* ice uplift, :math:`h > h_g + \Delta h`
* pressurised flow, :math:`h_g \le h \le h_g + \Delta h`
* free-surface flow, :math:`h < h_g`

Every function operates on NumPy arrays so that it can be used on the
structured staggered grid of :mod:`hydropack.glads2.model`, and each one
can be swapped for a custom implementation at model construction time.
Equation numbers refer to Wells et al. (2026).
"""

import numpy as np

from hydropack.glads2.constants import (
    gravity,
    ice_density,
    water_density,
    latent_heat,
    heat_capacity_water,
    clapeyron_slope,
    glen_exponent,
    flux_alpha,
    flux_beta,
    default_parameters,
)

#: regime labels returned by :func:`sheet_regime`
FREE_SURFACE, PRESSURISED, UPLIFT = -1, 0, 1

#: regime labels returned by :func:`channel_regime`
EMPTY, PARTIALLY_FILLED, FULLY_FILLED = 0, 1, 2


def overburden_pressure(H, rho_i=ice_density, g=gravity):
    r"""Return the hydrostatic ice-overburden pressure :math:`p_i = \rho_i g H`"""
    return rho_i * g * np.asarray(H, dtype=float)


def storage_capacity(H, e_v, rho_i=ice_density, rho_w=water_density):
    r"""Return the englacial storage capacity thickness (Eq. 3)

    .. math::
        \Delta h = e_v\,\frac{p_i}{\rho_w g} = e_v\,\frac{\rho_i}{\rho_w}\,H
    """
    return e_v * rho_i / rho_w * np.asarray(H, dtype=float)


def water_pressure(h, h_g, p_i, delta_h):
    r"""Return the bounded water pressure from the closure relation (Eq. 4)

    .. math::
        p_w = \begin{cases}
            p_i & h > h_g + \Delta h \\
            p_i\,(h - h_g)/\Delta h & h_g \le h \le h_g + \Delta h \\
            0 & h < h_g
        \end{cases}

    Where :math:`p_i = 0` (proglacial zone) both bounds collapse to
    atmospheric pressure and :math:`p_w = 0` for any :math:`h`.
    """
    h = np.asarray(h, dtype=float)
    p_i = np.asarray(p_i, dtype=float)
    delta_h = np.asarray(delta_h, dtype=float)
    excess = h - h_g
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(delta_h > 0.0, excess / delta_h, np.where(excess > 0.0, 1.0, 0.0))
    return p_i * np.clip(fraction, 0.0, 1.0)


def englacial_storage(h, h_g, delta_h):
    r"""Return the stored water thickness :math:`h_e` (Eqs. 2 and 4)

    :math:`h_e` is 0 in free-surface flow, :math:`h - h_g` in pressurised
    flow and :math:`\Delta h` under ice uplift.  This is identical to
    :math:`e_v p_w / (\rho_w g)`.
    """
    return np.clip(np.asarray(h, dtype=float) - h_g, 0.0, delta_h)


def flowing_thickness(h, h_g, delta_h):
    r"""Return the flowing sheet thickness :math:`h_w = h - h_e` (Eq. 1)

    This equals :math:`h` in free-surface flow, :math:`h_g` in pressurised
    flow and :math:`h - \Delta h` under ice uplift.
    """
    return np.asarray(h, dtype=float) - englacial_storage(h, h_g, delta_h)


def hydraulic_potential(B, h_w, p_w, rho_w=water_density, g=gravity):
    r"""Return the hydraulic potential (Eq. 7)

    .. math::
        \phi = \rho_w g\,(B + h_w) + p_w

    Unlike GlaDS, the flowing sheet thickness contributes to the potential,
    so that :math:`p_w` is the pressure at the top of the flowing sheet.
    """
    return rho_w * g * (B + h_w) + p_w


def sheet_regime(h, h_g, delta_h):
    r"""Classify each point as free-surface (-1), pressurised (0) or uplift (1)"""
    h = np.asarray(h, dtype=float)
    regime = np.full(h.shape, PRESSURISED, dtype=int)
    regime[h < h_g] = FREE_SURFACE
    regime[h > h_g + delta_h] = UPLIFT
    return regime


def cavity_height_rate(h_g, N, u_b, **kwargs):
    r"""Return :math:`\partial h_g/\partial t` (Eq. 5)

    .. math::
        \frac{\partial h_g}{\partial t}
        = \frac{u_b}{l_r}(h_r - h_g) - \frac{2}{n^n} A\, h_g N^n

    Parameters
    ----------
    h_g : ndarray
        Mean cavity height
    N : ndarray
        Effective pressure, non-negative under the closure of Eq. 4
    u_b : ndarray
        Basal sliding speed
    **kwargs
        ``A``, ``h_r``, ``l_r``
    """
    A = kwargs.get("A", default_parameters["A"])
    h_r = kwargs.get("h_r", default_parameters["h_r"])
    l_r = kwargs.get("l_r", default_parameters["l_r"])
    n = glen_exponent
    return u_b / l_r * (h_r - h_g) - 2.0 / n**n * A * h_g * N**n


def update_cavity_height(h_g, N, u_b, dt, **kwargs):
    r"""Advance Eq. (5) by one semi-implicit Euler step

    Opening and closure are both linear in :math:`h_g`, so treating them
    implicitly gives

    .. math::
        h_g^{k+1} = \frac{h_g^k + \Delta t\,u_b h_r / l_r}
                         {1 + \Delta t\,(u_b/l_r + 2 A N^n / n^n)}

    which keeps :math:`0 \le h_g \le h_r` for any :math:`\Delta t`.
    """
    A = kwargs.get("A", default_parameters["A"])
    h_r = kwargs.get("h_r", default_parameters["h_r"])
    l_r = kwargs.get("l_r", default_parameters["l_r"])
    n = glen_exponent
    numerator = h_g + dt * u_b * h_r / l_r
    denominator = 1.0 + dt * (u_b / l_r + 2.0 / n**n * A * N**n)
    return numerator / denominator


def regularised_gradient_factor(grad_squared, phi_reg):
    r"""Return :math:`(|\nabla\phi|^2 + \epsilon^2)^{(\beta-2)/2}`"""
    return (grad_squared + phi_reg**2) ** ((flux_beta - 2.0) / 2.0)


def sheet_flux(h_w, dphi_dn, grad_squared, **kwargs):
    r"""Return the normal sheet flux on a cell face (Eq. 6)

    .. math::
        q = -k\, h_w^\alpha\, |\nabla\phi|^{\beta-2}\,\nabla\phi

    Parameters
    ----------
    h_w : ndarray
        Flowing sheet thickness on the face
    dphi_dn : ndarray
        Normal derivative of the hydraulic potential on the face
    grad_squared : ndarray
        :math:`|\nabla\phi|^2` on the face (normal plus tangential parts)
    **kwargs
        ``k`` : sheet conductivity, ``phi_reg`` : gradient regularisation
    """
    k = kwargs.get("k", default_parameters["k"])
    phi_reg = kwargs.get("phi_reg", default_parameters["phi_reg"])
    h_w = np.maximum(h_w, 0.0)
    return -k * h_w**flux_alpha * regularised_gradient_factor(grad_squared, phi_reg) * dphi_dn


def channel_radius(S):
    r"""Return the radius of a semicircular channel, :math:`S = \pi R^2/2`"""
    return np.sqrt(2.0 * np.maximum(S, 0.0) / np.pi)


def water_filled_height(p_w, R, rho_w=water_density, g=gravity):
    r"""Return the water-filled channel height (Eq. 14)

    .. math::
        R_w = \min\left(\frac{p_w}{\rho_w g},\, R\right)
    """
    return np.clip(p_w / (rho_w * g), 0.0, R)


def water_filled_area(S, p_w, rho_w=water_density, g=gravity):
    r"""Return the water-filled channel cross-section (Eqs. 14-15)

    .. math::
        S_w = \min\left(S - R^2\arccos\frac{R_w}{R} + R_w\sqrt{R^2-R_w^2},\, S\right)

    The channel is empty (:math:`S_w = 0`) at atmospheric pressure, and
    fully filled (:math:`S_w = S`) once :math:`p_w \ge \rho_w g R`.
    """
    S = np.maximum(np.asarray(S, dtype=float), 0.0)
    R = channel_radius(S)
    R_w = water_filled_height(p_w, R, rho_w, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(R > 0.0, R_w / R, 1.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    segment = S - R**2 * np.arccos(ratio) + R_w * np.sqrt(np.maximum(R**2 - R_w**2, 0.0))
    # remove round-off at the empty and fully-filled limits
    segment = np.where(R_w <= 0.0, 0.0, np.where(ratio >= 1.0, S, segment))
    return np.clip(segment, 0.0, S)


def channel_regime(S, S_w, rtol=1e-10):
    r"""Classify channels as empty (0), partially (1) or fully filled (2)"""
    S = np.asarray(S, dtype=float)
    regime = np.full(S.shape, PARTIALLY_FILLED, dtype=int)
    regime[S_w <= rtol * np.maximum(S, 1e-300)] = EMPTY
    regime[S_w >= (1.0 - rtol) * S] = FULLY_FILLED
    return regime


def channel_discharge(S_w, dphi_ds, **kwargs):
    r"""Return the channel discharge (Eq. 16)

    .. math::
        Q = -k_c\, S_w^\alpha\,
            \left|\frac{\partial\phi}{\partial s}\right|^{\beta-2}
            \frac{\partial\phi}{\partial s}

    **kwargs
        ``k_c`` : channel conductivity, ``phi_reg`` : gradient regularisation
    """
    k_c = kwargs.get("k_c", default_parameters["k_c"])
    phi_reg = kwargs.get("phi_reg", default_parameters["phi_reg"])
    S_w = np.maximum(S_w, 0.0)
    return -k_c * S_w**flux_alpha * regularised_gradient_factor(dphi_ds**2, phi_reg) * dphi_ds


def channel_dissipation(Q, q_c, dphi_ds, **kwargs):
    r"""Return the dissipation power per unit channel length (Eq. 11)

    .. math::
        \Xi = -Q\,\frac{\partial\phi}{\partial s} - l_c\, q_c\,\frac{\partial\phi}{\partial s}
    """
    l_c = kwargs.get("l_c", default_parameters["l_c"])
    return -Q * dphi_ds - l_c * q_c * dphi_ds


def pressure_melting_switch(S, q_c, dpw_ds):
    r"""Return the switch :math:`f` of Eq. (13)

    :math:`f = 1` if :math:`S > 0` or :math:`q_c\,\partial p_w/\partial s > 0`,
    and :math:`f = 0` otherwise.  This prevents the sheet contribution from
    driving the channel area negative.
    """
    return np.where((S > 0.0) | (q_c * dpw_ds > 0.0), 1.0, 0.0)


def channel_pressure_melting(Q, q_c, dpw_ds, S, **kwargs):
    r"""Return the pressure-melting power per unit channel length (Eq. 12)

    .. math::
        \Pi = -c_t c_w \rho_w\,(Q + f\, l_c q_c)\,\frac{\partial p_w}{\partial s}

    Note the sign convention of Wells et al. (2026): :math:`c_t < 0` and the
    term enters the channel evolution as :math:`\Xi + \Pi`.
    """
    l_c = kwargs.get("l_c", default_parameters["l_c"])
    c_t = kwargs.get("c_t", clapeyron_slope)
    c_w = kwargs.get("c_w", heat_capacity_water)
    rho_w = kwargs.get("rho_w", water_density)
    f = pressure_melting_switch(S, q_c, dpw_ds)
    return -c_t * c_w * rho_w * (Q + f * l_c * q_c) * dpw_ds


def channel_area_rate(S, Xi, Pi, N, **kwargs):
    r"""Return :math:`\partial S/\partial t` (Eq. 10)

    .. math::
        \frac{\partial S}{\partial t}
        = \frac{\Xi + \Pi}{\rho_i L} - \frac{2}{n^n} A_c\, S N^n
    """
    A_c = kwargs.get("A_c", default_parameters["A_c"])
    n = glen_exponent
    return (Xi + Pi) / (ice_density * latent_heat) - 2.0 / n**n * A_c * S * N**n


def update_channel_area(S, Xi, Pi, N, dt, **kwargs):
    r"""Advance Eq. (10) by one semi-implicit Euler step

    Melt opening is explicit and creep closure, which is linear in
    :math:`S`, is implicit:

    .. math::
        S^{k+1} = \max\left(0,\;
            \frac{S^k + \Delta t\,(\Xi + \Pi)/(\rho_i L)}
                 {1 + \Delta t\, 2 A_c N^n / n^n}\right)
    """
    A_c = kwargs.get("A_c", default_parameters["A_c"])
    n = glen_exponent
    numerator = S + dt * (Xi + Pi) / (ice_density * latent_heat)
    denominator = 1.0 + dt * 2.0 / n**n * A_c * N**n
    return np.maximum(numerator / denominator, 0.0)


def channel_melt_source(Xi, Pi, rho_w=water_density):
    r"""Return the channel melt water source per unit length, :math:`(\Xi+\Pi)/(\rho_w L)`

    This is the right-hand-side line source in the mass balance (Eq. 17).
    """
    return (Xi + Pi) / (rho_w * latent_heat)


def effective_diffusivity(h_w, S_w, grad_squared, pressurised, channel_density, **kwargs):
    r"""Return the effective diffusivity used for the time-step bound (Eqs. 19-20)

    .. math::
        D_{\mathrm{eff},uf} = \rho_w g\,(k h_w^\alpha + k_c S_w^\alpha\delta)
            |\nabla\phi|^{\beta-2}, \qquad
        D_{\mathrm{eff},p} = e_v^{-1} D_{\mathrm{eff},uf}

    Parameters
    ----------
    pressurised : ndarray of bool
        Whether the face lies in the pressurised regime
    channel_density : float
        Discrete version of :math:`\delta(x_c)`: one over the channel spacing
    """
    k = kwargs.get("k", default_parameters["k"])
    k_c = kwargs.get("k_c", default_parameters["k_c"])
    e_v = kwargs.get("e_v", default_parameters["e_v"])
    phi_reg = kwargs.get("phi_reg", default_parameters["phi_reg"])
    conductance = k * np.maximum(h_w, 0.0) ** flux_alpha
    conductance = conductance + k_c * np.maximum(S_w, 0.0) ** flux_alpha * channel_density
    D = water_density * gravity * conductance * regularised_gradient_factor(grad_squared, phi_reg)
    return np.where(pressurised, D / e_v, D)


def effective_velocity(h_w, S_w, q, Q, dphi_dn, grad_squared, pressurised, channel_density, **kwargs):
    r"""Return the effective advective speed used for the time-step bound (Eqs. 21-22)

    .. math::
        V_{\mathrm{eff},uf} = k\alpha h_w^{\alpha-1}|\nabla\phi|^{\beta-2}\nabla\phi
            + h_w^{-1} Q\delta\,\hat{s}, \qquad
        V_{\mathrm{eff},p} = h_w^{-1} q
            + e_v^{-1} k_c \alpha R S_w^{\alpha-1}|\nabla\phi|^{\beta-2}\nabla\phi\,\delta

    Only the magnitude of the normal component on each face is returned.
    """
    k = kwargs.get("k", default_parameters["k"])
    k_c = kwargs.get("k_c", default_parameters["k_c"])
    e_v = kwargs.get("e_v", default_parameters["e_v"])
    phi_reg = kwargs.get("phi_reg", default_parameters["phi_reg"])
    h_min = kwargs.get("h_min", 1e-3)
    alpha = flux_alpha
    gfac = regularised_gradient_factor(grad_squared, phi_reg) * np.abs(dphi_dn)
    h_w = np.maximum(h_w, 0.0)
    h_safe = np.maximum(h_w, h_min)
    S_w = np.maximum(S_w, 0.0)
    V_uf = k * alpha * h_w ** (alpha - 1.0) * gfac + np.abs(Q) * channel_density / h_safe
    R = channel_radius(S_w)
    V_p = np.abs(q) / h_safe + k_c * alpha * R * S_w ** (alpha - 1.0) * gfac * channel_density / e_v
    return np.where(pressurised, V_p, V_uf)


def stable_timestep(D_max, V_max, dx, dy, C=0.5):
    r"""Return the explicit time-step bound (Eq. 23)

    .. math::
        \Delta t = C\min\left(\frac{\min(\Delta x,\Delta y)^2}{4\max D_{\mathrm{eff}}},
                              \frac{\min(\Delta x,\Delta y)}{2\max\|V_{\mathrm{eff}}\|_\infty}\right)
    """
    d = min(dx, dy)
    bounds = [np.inf]
    if D_max > 0.0:
        bounds.append(d**2 / (4.0 * D_max))
    if V_max > 0.0:
        bounds.append(d / (2.0 * V_max))
    return C * min(bounds)
