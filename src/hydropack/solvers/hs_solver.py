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

r"""Implicit ODE integrator for sheet height and channel area

This module contains :class:`HSSolver`, which advances the distributed
sheet height :math:`h` and channel cross-sectional area :math:`S` forward
in time while the hydraulic potential :math:`\phi` is held fixed.
"""

import numpy as np
import firedrake as fd

from hydropack import physics as _physics


class HSSolver:
    r"""Advance h and S by one timestep with phi fixed.

    Uses backward Euler for :math:`h` and iterated Crank-Nicolson for
    :math:`S`, following the ISSM operator-split approach.

    Parameters
    ----------
    model : SubglacialHydrologyModel
        The parent model instance.
    h_ode_coefficients : callable or None
        Replacement for :func:`~hydropack.physics.h_ode_coefficients`.
    S_ode_coefficients : callable or None
        Replacement for :func:`~hydropack.physics.S_ode_coefficients`.
    """

    def __init__(self, model, h_ode_coefficients=None, S_ode_coefficients=None):
        self.model = model
        self._h_ode = h_ode_coefficients or _physics.h_ode_coefficients
        self._S_ode = S_ode_coefficients or _physics.S_ode_coefficients

        self._physics_kwargs = dict(
            A=model.A,
            h_r=model.h_r,
            l_r=model.l_r,
            k=model.k,
            k_c=model.k_c,
            l_c=model.l_c,
        )

        self.S_max_iter = 10
        self.S_rtol = 1e-8
        self.S_cap = 500.0

        self.u_b_u = model.u_b.dat.data_ro.copy()

        try:
            mask = model.mask.dat.data_ro
            self.mask_cr = mask.copy() if mask is not None else None
        except Exception:
            self.mask_cr = None

    def _cache(self):
        """Snapshot phi-derived fields that are frozen during the HS step."""
        m = self.model
        N_u = m.N.dat.data_ro.copy()
        N_cr = m.N_cr.dat.data_ro.copy()
        h_cr = m.h_cr.dat.data_ro.copy()
        phi_s = m.dphi_ds_cr.dat.data_ro.copy()
        pw_s = m.utilities.ds_array(m.phi - m.phi_m)
        mask = self.mask_cr if self.mask_cr is not None else 1.0
        return dict(
            N_u=N_u, N_cr=N_cr, h_cr=h_cr,
            phi_s=phi_s, pw_s=pw_s,
            mask=mask, u_b=self.u_b_u,
        )

    def _step_h(self, dt, cache):
        """Advance h by backward Euler on the linearised ODE."""
        h_old = self.model.h.dat.data_ro.copy()

        alpha_h, beta_h = self._h_ode(
            h_old, cache["N_u"], cache["u_b"], **self._physics_kwargs
        )

        h_new = (h_old + dt * beta_h) / (1.0 - dt * alpha_h)
        h_new = np.maximum(h_new, 1e-16)

        self.model.h.dat.data[:] = h_new

    def _step_S(self, dt, cache):
        """Advance S by iterated Crank-Nicolson."""
        N_cr = cache["N_cr"]
        h_cr = cache["h_cr"]
        phi_s = cache["phi_s"]
        pw_s = cache["pw_s"]
        mask = cache["mask"]

        S_old = self.model.S.dat.data_ro.copy()
        S_old = np.maximum(S_old, 0.0)

        S_cur = S_old.copy()
        for _iter in range(self.S_max_iter):
            S_prev = S_cur.copy()
            S_cur = np.maximum(S_cur, 0.0)

            alpha_S, beta_sheet = self._S_ode(
                S_cur, S_old, phi_s, h_cr, N_cr, pw_s, **self._physics_kwargs
            )

            numer = S_old + 0.5 * dt * (alpha_S * S_old + 2.0 * beta_sheet)
            denom = 1.0 - 0.5 * dt * alpha_S
            denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
            S_cur = numer / denom

            S_cur = np.where(np.isfinite(S_cur), S_cur, 0.0)
            S_cur = np.clip(S_cur, 0.0, self.S_cap)

            rel_change = np.abs(S_cur - S_prev) / (np.abs(S_prev) + 1e-30)
            if np.max(rel_change) < self.S_rtol:
                break

        S_cur *= mask
        self.model.S.dat.data[:] = S_cur

    def step(self, dt):
        """Advance h (CG1) and S (CR1) by *dt* using implicit methods."""
        cache = self._cache()
        self._step_h(dt, cache)
        self._step_S(dt, cache)
        self.model.update_S_alpha()
        self.model.update_h_cr()
        self.model.t += dt
