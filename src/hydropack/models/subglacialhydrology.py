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

r"""Coupled sheet-channel subglacial hydrology model

This module contains :class:`SubglacialHydrologyModel`, the top-level model
class that couples a distributed sheet drainage system to a channelised
drainage system on a two-dimensional mesh (Werder et al., 2013).
"""

import os
import numpy as np
import firedrake
import firedrake as fd

from hydropack.utilities import CRTools
from hydropack.solvers.phi_solver import PhiSolver
from hydropack.solvers.hs_solver import HSSolver
from hydropack.constants import (
    default_sheet_conductivity,
    default_channel_conductivity,
    default_flow_rate_factor,
    default_bump_height,
    default_bump_spacing,
    default_channel_sheet_width,
    default_englacial_void_ratio,
    sheet_flux_alpha,
)


class SubglacialHydrologyModel:
    r"""Coupled sheet-channel subglacial hydrology model.

    Parameters
    ----------
    mesh : firedrake.Mesh
        The computational mesh.
    thickness : firedrake.Function
        Ice thickness :math:`H` (CG1).
    bed : firedrake.Function
        Bed elevation :math:`B` (CG1).
    sliding_speed : firedrake.Function
        Basal sliding speed :math:`u_b` (CG1).
    melt_rate : firedrake.Function
        Basal melt / recharge rate :math:`m` (CG1).
    phi_init : firedrake.Function
        Initial hydraulic potential (CG1).
    h_init : firedrake.Function
        Initial sheet thickness (CG1).
    S_init : firedrake.Function
        Initial channel cross-sectional area (CR1).
    phi_m : firedrake.Function
        Bed-elevation potential :math:`\rho_w g B` (CG1).
    p_i : firedrake.Function
        Ice overburden pressure (CG1).
    phi_0 : firedrake.Function
        Potential at overburden pressure (CG1).
    dirichlet_bcs : list
        List of Firedrake DirichletBC objects.
    sheet_conductivity : float
        Sheet conductivity :math:`k`.
    channel_conductivity : float
        Channel conductivity :math:`k_c`.
    flow_rate_factor : float
        Glen's flow-rate factor :math:`A`.
    bump_height : float
        Average bump height :math:`h_r`.
    bump_spacing : float
        Typical bump spacing :math:`l_r`.
    channel_sheet_width : float
        Sheet width beneath a channel :math:`l_c`.
    englacial_void_ratio : float
        Englacial void ratio :math:`e_v`.
    sheet_flux : callable or None
        Custom sheet flux function.
    sheet_opening : callable or None
        Custom sheet opening function.
    sheet_closure : callable or None
        Custom sheet closure function.
    channel_discharge : callable or None
        Custom channel discharge function.
    channel_sheet_flux : callable or None
        Custom channel sheet flux function.
    channel_dissipation : callable or None
        Custom channel dissipation function.
    channel_pressure_melting : callable or None
        Custom channel pressure-melting function.
    channel_melt_opening : callable or None
        Custom channel melt-opening function.
    channel_closure : callable or None
        Custom channel closure function.
    h_ode_coefficients : callable or None
        Custom h ODE coefficients function.
    S_ode_coefficients : callable or None
        Custom S ODE coefficients function.
    out_dir : str
        Output directory for PVD files.
    """

    def __init__(
        self,
        mesh,
        *,
        thickness,
        bed,
        sliding_speed,
        melt_rate,
        phi_init,
        h_init,
        S_init,
        phi_m,
        p_i,
        phi_0,
        dirichlet_bcs,
        sheet_conductivity=default_sheet_conductivity,
        channel_conductivity=default_channel_conductivity,
        flow_rate_factor=default_flow_rate_factor,
        bump_height=default_bump_height,
        bump_spacing=default_bump_spacing,
        channel_sheet_width=default_channel_sheet_width,
        englacial_void_ratio=default_englacial_void_ratio,
        sheet_flux=None,
        sheet_opening=None,
        sheet_closure=None,
        channel_discharge=None,
        channel_sheet_flux=None,
        channel_dissipation=None,
        channel_pressure_melting=None,
        channel_melt_opening=None,
        channel_closure=None,
        h_ode_coefficients=None,
        S_ode_coefficients=None,
        out_dir="./outputs",
    ):
        self.mesh = mesh
        self.U = firedrake.FunctionSpace(self.mesh, "CG", 1)
        self.V = firedrake.VectorFunctionSpace(self.mesh, "CG", 1)
        self.CR = firedrake.FunctionSpace(self.mesh, "CR", 1)
        self.utilities = CRTools(self.mesh, self.U, self.CR)

        self.k = sheet_conductivity
        self.k_c = channel_conductivity
        self.A = flow_rate_factor
        self.h_r = bump_height
        self.l_r = bump_spacing
        self.l_c = channel_sheet_width
        self.e_v = englacial_void_ratio

        self._out_dir = out_dir

        self.H = thickness
        self.B = bed
        self.u_b = sliding_speed
        self.m = melt_rate
        self.h = h_init
        self.phi_prev = phi_init
        self.phi_m = phi_m
        self.p_i = p_i
        self.phi_0 = phi_0
        self.d_bcs = dirichlet_bcs
        self.S = S_init
        self.n_bcs = []

        self.phi = firedrake.Function(self.U)
        self.dphi_ds_cr = firedrake.Function(self.CR)
        self.N = firedrake.Function(self.U)
        self.N_cr = firedrake.Function(self.CR)
        self.h_cr = firedrake.Function(self.CR)
        self.update_h_cr()
        self.S_alpha = firedrake.Function(self.CR)
        self.update_S_alpha()
        self.p_w = firedrake.Function(self.U)
        self.pfo = firedrake.Function(self.U)
        self.t = 0.0

        self.phi_solver = PhiSolver(
            self,
            sheet_flux=sheet_flux,
            sheet_opening=sheet_opening,
            sheet_closure=sheet_closure,
            channel_discharge=channel_discharge,
            channel_sheet_flux=channel_sheet_flux,
            channel_dissipation=channel_dissipation,
            channel_pressure_melting=channel_pressure_melting,
            channel_melt_opening=channel_melt_opening,
            channel_closure=channel_closure,
        )
        self.hs_solver = HSSolver(
            self,
            h_ode_coefficients=h_ode_coefficients,
            S_ode_coefficients=S_ode_coefficients,
        )

    def step(self, dt, max_picard=3, picard_tol=1e-3):
        r"""Step the potential, gap height, and channel area forward by *dt*.

        Parameters
        ----------
        dt : float
            Timestep in seconds.
        max_picard : int
            Maximum Picard iterations for the phi solve.
        picard_tol : float
            Relative tolerance for Picard convergence.
        """
        for picard_iter in range(max_picard):
            phi_old = self.phi.dat.data_ro.copy()
            self.phi_solver.step(dt)
            dphi = np.linalg.norm(self.phi.dat.data_ro - phi_old)
            phi_norm = np.linalg.norm(self.phi.dat.data_ro) + 1e-30
            if dphi / phi_norm < picard_tol:
                break

        self.hs_solver.step(dt)

    def update_N(self):
        """Update effective pressure to reflect current phi."""
        self.N.assign(self.phi_0 - self.phi)

    def update_pw(self):
        """Update water pressure to reflect current phi."""
        self.p_w.assign(self.phi - self.phi_m)

    def update_pfo(self):
        """Update pressure as fraction of overburden."""
        self.update_pw()
        self.update_pw()
        eps = firedrake.Constant(1.0)
        self.pfo.interpolate(self.p_w / firedrake.max_value(self.p_i, eps))

        p_w_tmp = firedrake.interpolate(self.p_w, self.U)
        p_i_tmp = firedrake.interpolate(self.p_i, self.U)
        pfo_tmp = firedrake.interpolate(p_w_tmp / p_i_tmp, self.U)
        self.pfo.vector().set_local(pfo_tmp.vector().array())
        self.pfo.vector().apply("insert")

    def update_N_cr(self):
        """Update effective pressure on edge midpoints."""
        self.update_N()
        self.utilities.midpoint(self.N, self.N_cr)

    def update_dphi_ds_cr(self):
        """Update tangential derivative of phi on edges."""
        self.utilities.ds_assemble(self.phi, self.dphi_ds_cr)

    def update_phi(self):
        """Update all fields derived from phi."""
        self.phi_prev.assign(self.phi)
        self.update_N_cr()
        self.update_dphi_ds_cr()
        self.update_pfo()

    def update_h_cr(self):
        """Update edge midpoint values of h."""
        self.utilities.midpoint(self.h, self.h_cr)

    def update_S_alpha(self):
        r"""Update :math:`S^\alpha` to reflect current S."""
        self.S_alpha.interpolate(self.S ** sheet_flux_alpha)

    def compute_flux_fields(self):
        """Compute sheet and channel flux fields from the current state."""
        from hydropack.constants import sheet_flux_delta

        mesh = self.mesh
        _alpha = sheet_flux_alpha
        _delta = sheet_flux_delta
        phi_reg = fd.Constant(1e-10)

        Vvec = fd.VectorFunctionSpace(mesh, "CG", 1)
        self.q_s = fd.Function(Vvec, name="q_s")
        Ks = fd.Constant(self.k)
        h = self.h
        self.q_s.project(
            -Ks * fd.max_value(h, 0.0) ** _alpha
            * (fd.inner(fd.grad(self.phi), fd.grad(self.phi)) + phi_reg ** 2) ** (_delta / 2.0)
            * fd.grad(self.phi)
        )

        Vdg = fd.FunctionSpace(mesh, "DG", 0)
        self.q_s_mag = fd.Function(Vdg, name="q_s_mag")
        self.q_s_mag.project(fd.sqrt(fd.inner(self.q_s, self.q_s) + fd.Constant(1e-30)))

        self.update_dphi_ds_cr()

        Vcr = self.dphi_ds_cr.function_space()
        self.Q_ch = fd.Function(Vcr)

        Kc = fd.Constant(self.k_c)
        dphi = self.dphi_ds_cr
        self.Q_ch.interpolate(
            -Kc * fd.max_value(self.S, 0.0) ** fd.Constant(_alpha)
            * (dphi ** 2 + phi_reg ** 2) ** (fd.Constant(_delta) / 2.0)
            * dphi
        )

    def write_pvds(self):
        """Write h, S, pfo, and phi to PVD files."""
        if not hasattr(self, "S_out"):
            out_dir = self._out_dir
            os.makedirs(out_dir, exist_ok=True)
            self.S_out = firedrake.File(os.path.join(out_dir, "S.pvd"))
            self.h_out = firedrake.File(os.path.join(out_dir, "h.pvd"))
            self.phi_out = firedrake.File(os.path.join(out_dir, "phi.pvd"))
            self.pfo_out = firedrake.File(os.path.join(out_dir, "pfo.pvd"))

        self.S_out << self.S
        self.h_out << self.h
        self.phi_out << self.phi
        self.pfo_out << self.pfo

    def write_checkpoint(self, filename="hydrology_state.h5"):
        """Save the current model state to an HDF5 checkpoint file.

        Parameters
        ----------
        filename : str
            Path to the output HDF5 file.
        """
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        with firedrake.CheckpointFile(filename, "w") as checkpoint:
            checkpoint.save_mesh(self.mesh)
            checkpoint.save_function(self.h, name="h")
            checkpoint.save_function(self.S, name="S")
            checkpoint.save_function(self.phi, name="phi")
            checkpoint.save_function(self.pfo, name="pfo")
            checkpoint.save_function(self.N, name="N")
            checkpoint.save_function(self.N_cr, name="N_cr")
            checkpoint.save_function(self.h_cr, name="h_cr")
            checkpoint.save_function(self.S_alpha, name="S_alpha")
            checkpoint.save_function(self.p_w, name="p_w")
