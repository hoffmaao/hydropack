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

r"""Constitutive relations for the GlaDS subglacial hydrology model

This module contains the constitutive relations and source/sink terms that
define the GlaDS coupled sheet-channel drainage system (Werder et al., 2013).
Each function returns a UFL expression or NumPy array and can be swapped for
a custom implementation at model construction time.
"""

import numpy as np
import firedrake
from firedrake import Constant, dot, grad, max_value, conditional, gt

from hydropack.constants import (
    ice_density,
    water_density,
    latent_heat,
    heat_capacity_water,
    clapeyron_slope,
    glen_exponent,
    nye_closure_factor,
    sheet_flux_alpha as alpha,
    sheet_flux_beta as beta,
    sheet_flux_delta as delta,
    default_sheet_conductivity,
    default_channel_conductivity,
    default_flow_rate_factor,
    default_bump_height,
    default_bump_spacing,
    default_channel_sheet_width,
)


def sheet_flux(phi, h, **kwargs):
    r"""Return the sheet water flux vector

    The sheet flux follows a nonlinear Darcy-type law (Werder et al., 2013):

    .. math::
        \mathbf{q} = -k\, h^\alpha\,
        \bigl(|\nabla\phi|^2 + \epsilon^2\bigr)^{\delta/2}\,\nabla\phi

    where :math:`\alpha = 5/4`, :math:`\delta = \beta - 2 = -1/2`, and
    :math:`\epsilon` is a small regularisation parameter.

    Parameters
    ----------
    phi : firedrake.Function
        Hydraulic potential
    h : firedrake.Function
        Sheet water thickness
    **kwargs
        ``k`` : sheet conductivity (default 0.005)
        ``phi_reg`` : gradient regularisation (default 1e-10)
        ``h_min`` : minimum sheet thickness floor (default 1e-6)
    """
    k = kwargs.get("k", default_sheet_conductivity)
    phi_reg = kwargs.get("phi_reg", Constant(1e-10))
    h_min = kwargs.get("h_min", Constant(1e-6))

    h_eff = max_value(h, h_min)
    return (
        -Constant(k)
        * h_eff ** alpha
        * (dot(grad(phi), grad(phi)) + phi_reg ** 2) ** (delta / 2.0)
        * grad(phi)
    )


def sheet_opening(h, u_b, **kwargs):
    r"""Return the cavity opening rate due to basal sliding

    .. math::
        w = \frac{u_b\,(h_r - h)}{l_r} \quad\text{if } h < h_r,
        \quad 0 \;\text{otherwise}

    Parameters
    ----------
    h : firedrake.Function
        Sheet water thickness
    u_b : firedrake.Function
        Basal sliding speed
    **kwargs
        ``h_r`` : bump height (default 0.1 m)
        ``l_r`` : bump spacing (default 2.0 m)
        ``h_min`` : minimum sheet thickness floor (default 1e-6)
    """
    h_r = kwargs.get("h_r", default_bump_height)
    l_r = kwargs.get("l_r", default_bump_spacing)
    h_min = kwargs.get("h_min", Constant(1e-6))

    h_eff = max_value(h, h_min)
    return conditional(
        gt(Constant(h_r) - h_eff, Constant(0.0)),
        u_b * (Constant(h_r) - h_eff) / Constant(l_r),
        Constant(0.0),
    )


def sheet_closure(h, N, **kwargs):
    r"""Return the creep closure rate for the distributed sheet

    The Nye (1953) closure rate under Glen's flow law is

    .. math::
        v = \frac{2A}{n^n}\, h\, N^3

    The sign-preserving form :math:`N^3` ensures the Newton Jacobian
    remains well-conditioned when :math:`N < 0`.

    Parameters
    ----------
    h : firedrake.Function
        Sheet water thickness
    N : ufl.core.expr.Expr
        Effective pressure, may be negative
    **kwargs
        ``A`` : Glen's flow-rate factor (default 3.375e-24)
    """
    A = kwargs.get("A", default_flow_rate_factor)
    A_closure = nye_closure_factor * A
    return Constant(A_closure) * h * N ** glen_exponent


def channel_discharge(phi, S, t, **kwargs):
    r"""Return the channel water discharge along a mesh edge

    .. math::
        Q = -k_c\, S^\alpha\,
        \bigl((\partial\phi/\partial s)^2 + \epsilon^2\bigr)^{\delta/2}\,
        \frac{\partial\phi}{\partial s}

    Parameters
    ----------
    phi : firedrake.Function
        Hydraulic potential
    S : firedrake.Function
        Channel cross-sectional area (CR1)
    t : ufl vector
        Unit tangent vector along the edge
    **kwargs
        ``k_c`` : channel conductivity (default 0.1)
        ``phi_reg`` : gradient regularisation (default 1e-10)
    """
    k_c = kwargs.get("k_c", default_channel_conductivity)
    phi_reg = kwargs.get("phi_reg", Constant(1e-10))

    dphi_ds = dot(grad(phi), t)
    S_alpha = max_value(S, Constant(0.0)) ** alpha
    abs_dphi_delta = abs((dphi_ds ** 2 + phi_reg ** 2) ** (delta / 2.0))
    return -Constant(k_c) * S_alpha * abs_dphi_delta * dphi_ds


def channel_sheet_flux(phi, h, t, **kwargs):
    r"""Return the sheet flux component along a channel edge

    .. math::
        q_c = -k\, h^\alpha\,
        \bigl((\partial\phi/\partial s)^2 + \epsilon^2\bigr)^{\delta/2}\,
        \frac{\partial\phi}{\partial s}

    Parameters
    ----------
    phi : firedrake.Function
        Hydraulic potential
    h : firedrake.Function
        Sheet water thickness
    t : ufl vector
        Unit tangent vector along the edge
    **kwargs
        ``k`` : sheet conductivity (default 0.005)
        ``phi_reg`` : gradient regularisation (default 1e-10)
        ``h_min`` : minimum sheet thickness floor (default 1e-6)
    """
    k = kwargs.get("k", default_sheet_conductivity)
    phi_reg = kwargs.get("phi_reg", Constant(1e-10))
    h_min = kwargs.get("h_min", Constant(1e-6))

    dphi_ds = dot(grad(phi), t)
    h_eff = max_value(h, h_min)
    abs_dphi_delta = abs((dphi_ds ** 2 + phi_reg ** 2) ** (delta / 2.0))
    return -Constant(k) * h_eff ** alpha * abs_dphi_delta * dphi_ds


def channel_dissipation(Q_c, q_c, dphi_ds, **kwargs):
    r"""Return the energy dissipation rate along a channel

    .. math::
        \Xi = |Q\,\partial\phi/\partial s|
            + |l_c\, q_c\,\partial\phi/\partial s|

    Parameters
    ----------
    Q_c : ufl expression
        Channel discharge
    q_c : ufl expression
        Sheet flux along the channel edge
    dphi_ds : ufl expression
        Tangential derivative of the hydraulic potential
    **kwargs
        ``l_c`` : sheet width beneath channel (default 2.0 m)
    """
    l_c = kwargs.get("l_c", default_channel_sheet_width)
    return abs(Q_c * dphi_ds) + abs(Constant(l_c) * q_c * dphi_ds)


def channel_pressure_melting(Q_c, phi, phi_m, t):
    r"""Return the pressure-melting (Clausius-Clapeyron) correction

    .. math::
        \Pi = -c_t\, c_w\, \rho_w\, Q\,
        \frac{\partial(\phi - \phi_m)}{\partial s}

    Parameters
    ----------
    Q_c : ufl expression
        Channel discharge
    phi : firedrake.Function
        Hydraulic potential
    phi_m : firedrake.Function
        Bed-elevation potential :math:`\rho_w g B`
    t : ufl vector
        Unit tangent vector along the edge
    """
    C = clapeyron_slope * heat_capacity_water * water_density
    return -Constant(C) * Q_c * dot(grad(phi - phi_m), t)


def channel_melt_opening(Xi, Pi):
    r"""Return the channel opening rate for the water-balance equation

    .. math::
        w_c = \frac{\Xi - \Pi}{L}
              \left(\frac{1}{\rho_i} - \frac{1}{\rho_w}\right)

    Parameters
    ----------
    Xi : ufl expression
        Energy dissipation rate
    Pi : ufl expression
        Pressure-melting correction
    """
    density_factor = 1.0 / ice_density - 1.0 / water_density
    return (Xi - Pi) / Constant(latent_heat) * Constant(density_factor)


def channel_closure(S, N, **kwargs):
    r"""Return the creep closure rate for a channel

    .. math::
        v_c = \frac{2A}{n^n}\, S\, N^3

    Sign-preserving in :math:`N`, matching :func:`sheet_closure`.

    Parameters
    ----------
    S : firedrake.Function
        Channel cross-sectional area (CR1)
    N : ufl.core.expr.Expr
        Effective pressure, may be negative
    **kwargs
        ``A`` : Glen's flow-rate factor (default 3.375e-24)
    """
    A = kwargs.get("A", default_flow_rate_factor)
    A_closure = nye_closure_factor * A
    return Constant(A_closure) * S * N ** glen_exponent


def h_ode_coefficients(h_old, N_u, u_b, **kwargs):
    r"""Return linearised ODE coefficients for the sheet-height equation

    The sheet thickness evolves as

    .. math::
        \frac{\mathrm{d}h}{\mathrm{d}t} = w - v
        = \frac{u_b(h_r - h)}{l_r} - \frac{2A}{n^n}\,h\,|N|^{n-1}N

    which is recast as :math:`\dot{h} = \alpha\,h + \beta` for
    backward-Euler integration.

    Parameters
    ----------
    h_old : ndarray
        Sheet thickness at the current time (CG1 DOFs)
    N_u : ndarray
        Effective pressure at CG1 DOFs
    u_b : ndarray
        Basal sliding speed at CG1 DOFs
    **kwargs
        ``A`` : Glen's flow-rate factor
        ``h_r`` : bump height
        ``l_r`` : bump spacing

    Returns
    -------
    alpha, beta : ndarray pair
        Coefficients for the linearised ODE
    """
    A = kwargs.get("A", default_flow_rate_factor)
    h_r = kwargs.get("h_r", default_bump_height)
    l_r = kwargs.get("l_r", default_bump_spacing)
    n = glen_exponent

    A_closure = nye_closure_factor * A
    N_abs_nm1 = np.abs(N_u) ** (n - 1.0)
    closure_coeff = A_closure * N_abs_nm1 * N_u

    below = h_old < h_r
    alpha_h = np.where(below, -u_b / l_r - closure_coeff, -closure_coeff)
    beta_h = np.where(below, u_b * h_r / l_r, 0.0)
    return alpha_h, beta_h


def S_ode_coefficients(S_cur, S_old, phi_s, h_cr, N_cr, pw_s, **kwargs):
    r"""Return linearised ODE coefficients for the channel-area equation

    The channel cross-section evolves as

    .. math::
        \frac{\mathrm{d}S}{\mathrm{d}t}
        = \frac{\Xi - \Pi}{\rho_i L} - \frac{2A}{n^n}\,S\,|N|^{n-1}N

    which is recast as :math:`\dot{S} = \alpha\,S + \beta` for
    Crank-Nicolson integration with iteration.

    Parameters
    ----------
    S_cur : ndarray
        Channel area at current iteration (CR1 DOFs)
    S_old : ndarray
        Channel area at the start of the timestep
    phi_s : ndarray
        Signed tangential derivative of phi on CR1 edges
    h_cr : ndarray
        Sheet thickness at edge midpoints (CR1)
    N_cr : ndarray
        Effective pressure at edge midpoints (CR1)
    pw_s : ndarray
        Signed tangential derivative of (phi - phi_m) on CR1 edges
    **kwargs
        ``A`` : Glen's flow-rate factor
        ``k`` : sheet conductivity
        ``k_c`` : channel conductivity
        ``l_c`` : sheet width beneath channel

    Returns
    -------
    alpha_S, beta_sheet : ndarray pair
        Coefficients for the linearised ODE
    """
    A = kwargs.get("A", default_flow_rate_factor)
    k = kwargs.get("k", default_sheet_conductivity)
    k_c = kwargs.get("k_c", default_channel_conductivity)
    l_c = kwargs.get("l_c", default_channel_sheet_width)
    n = glen_exponent
    rho_i = ice_density
    L = latent_heat
    c_t = clapeyron_slope
    c_w = heat_capacity_water
    rho_w = water_density

    A_closure = nye_closure_factor * A
    Bfactor = 1.0 / (rho_i * L)
    C = c_w * c_t * rho_w

    Ngrad = np.maximum(np.abs(phi_s), 1e-16)

    Ks = k * np.power(h_cr, alpha) * np.power(Ngrad, beta - 2.0)
    qc = -Ks * phi_s

    dphimds = phi_s - pw_s
    dPw = phi_s - dphimds

    Qprime = -k_c * np.power(Ngrad, beta - 2.0) * phi_s

    N_abs_nm1_N = A_closure * np.power(np.abs(N_cr), n - 1.0) * N_cr

    S_cur = np.maximum(S_cur, 0.0)
    S_am1 = np.where(S_cur > 0, np.power(S_cur, alpha - 1.0), 0.0)

    alpha_S = (
        Bfactor * (np.abs(Qprime * S_am1 * phi_s) + C * Qprime * S_am1 * dPw)
        - N_abs_nm1_N
    )

    beta_sheet = Bfactor * np.abs(l_c * qc * phi_s)

    fF = np.zeros_like(S_old)
    fF_mask = (S_old > 0.0) | (qc * dPw > 0.0)
    fF[fF_mask] = l_c * qc[fF_mask]
    beta_sheet += Bfactor * C * fF * dPw

    return alpha_S, beta_sheet
