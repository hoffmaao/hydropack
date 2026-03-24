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

r"""Solver for the hydraulic potential equation

This module contains :class:`PhiSolver`, which assembles and solves the
nonlinear variational problem for :math:`\phi` with sheet thickness
:math:`h` and channel area :math:`S` held fixed.
"""

import firedrake

from hydropack.constants import water_density, gravity
from hydropack import physics as _physics


class PhiSolver:
    r"""Solve for hydraulic potential with h and S fixed.

    Parameters
    ----------
    model : SubglacialHydrologyModel
        The parent model instance.
    sheet_flux : callable or None
        Replacement for :func:`~hydropack.physics.sheet_flux`.
    sheet_opening : callable or None
        Replacement for :func:`~hydropack.physics.sheet_opening`.
    sheet_closure : callable or None
        Replacement for :func:`~hydropack.physics.sheet_closure`.
    channel_discharge : callable or None
        Replacement for :func:`~hydropack.physics.channel_discharge`.
    channel_sheet_flux : callable or None
        Replacement for :func:`~hydropack.physics.channel_sheet_flux`.
    channel_dissipation : callable or None
        Replacement for :func:`~hydropack.physics.channel_dissipation`.
    channel_pressure_melting : callable or None
        Replacement for :func:`~hydropack.physics.channel_pressure_melting`.
    channel_melt_opening : callable or None
        Replacement for :func:`~hydropack.physics.channel_melt_opening`.
    channel_closure : callable or None
        Replacement for :func:`~hydropack.physics.channel_closure`.
    """

    def __init__(
        self,
        model,
        sheet_flux=None,
        sheet_opening=None,
        sheet_closure=None,
        channel_discharge=None,
        channel_sheet_flux=None,
        channel_dissipation=None,
        channel_pressure_melting=None,
        channel_melt_opening=None,
        channel_closure=None,
    ):
        _sheet_flux = sheet_flux or _physics.sheet_flux
        _sheet_opening = sheet_opening or _physics.sheet_opening
        _sheet_closure = sheet_closure or _physics.sheet_closure
        _channel_discharge = channel_discharge or _physics.channel_discharge
        _channel_sheet_flux = channel_sheet_flux or _physics.channel_sheet_flux
        _channel_dissipation = channel_dissipation or _physics.channel_dissipation
        _channel_pressure_melting = channel_pressure_melting or _physics.channel_pressure_melting
        _channel_melt_opening = channel_melt_opening or _physics.channel_melt_opening
        _channel_closure = channel_closure or _physics.channel_closure

        physics_kwargs = dict(
            k=model.k,
            k_c=model.k_c,
            A=model.A,
            h_r=model.h_r,
            l_r=model.l_r,
            l_c=model.l_c,
        )

        m = model.m
        h = model.h
        S = model.S
        phi_m = model.phi_m
        u_b = model.u_b
        phi = model.phi
        phi_prev = model.phi_prev
        phi_0 = model.phi_0

        self._gamma = firedrake.Constant(1.0)
        c1 = model.e_v / (water_density * gravity)
        dt = firedrake.Constant(1.0)

        N = phi_0 - phi
        q_s = _sheet_flux(phi, h, **physics_kwargs)
        w = _sheet_opening(h, u_b, **physics_kwargs)
        v = _sheet_closure(h, N, **physics_kwargs)

        n = firedrake.FacetNormal(model.mesh)
        t = firedrake.as_vector([n[1], -n[0]])
        dphi_ds = firedrake.dot(firedrake.grad(phi), t)

        Q_c = _channel_discharge(phi, S, t, **physics_kwargs)
        q_c = _channel_sheet_flux(phi, h, t, **physics_kwargs)
        Xi = _channel_dissipation(Q_c, q_c, dphi_ds, **physics_kwargs)
        Pi = _channel_pressure_melting(Q_c, phi, phi_m, t)
        w_c = _channel_melt_opening(Xi, Pi)
        v_c = _channel_closure(S, N, **physics_kwargs)

        theta = firedrake.TestFunction(model.U)

        C1 = firedrake.Constant(c1)
        F_s = C1 * (phi - phi_prev) * theta * firedrake.dx
        F_s += dt * (
            (-firedrake.dot(firedrake.grad(theta), q_s) + (w - v - m) * theta)
            * firedrake.dx
        )

        F_c = dt * (
            -firedrake.dot(firedrake.grad(theta)('+'), t('+')) * Q_c('+')
            + (w_c('+') - v_c('+')) * theta('+')
        ) * firedrake.dS

        F = F_s + F_c

        dphi = firedrake.TrialFunction(model.U)
        J = firedrake.derivative(F, phi, dphi)

        self.F = F
        self.J = J
        self.model = model
        self.dt = dt

    def step(self, dt):
        r"""Step the potential forward by *dt*.

        Returns ``True`` if the primary Newton solve converged, ``False``
        if the fallback solver was needed.
        """
        self.dt.assign(dt)

        try:
            firedrake.solve(
                self.F == 0,
                self.model.phi,
                self.model.d_bcs,
                J=self.J,
                solver_parameters={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_linesearch_damping": 1.0,
                    "snes_rtol": 1.0e-5,
                    "snes_atol": 1.0e-5,
                    "snes_max_it": 30,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "umfpack",
                    "mat_type": "aij",
                },
            )
            self.model.update_phi()
        except:
            firedrake.solve(
                self.F == 0,
                self.model.phi,
                self.model.d_bcs,
                J=self.J,
                solver_parameters={
                    "snes_type": "newtonls",
                    "snes_rtol": 5e-11,
                    "snes_atol": 5e-10,
                    "pc_type": "lu",
                    "snes_max_it": 50,
                    "mat_type": "aij",
                },
            )
            self.model.update_phi()
            return False

        return True
