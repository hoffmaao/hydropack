# Copyright (C) 2019 by Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of Andrew's 1D hydrology model.
#
# This model is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found at 
# <http://www.gnu.org/licenses/>.


import firedrake
from hydropack.cr_tools import *
from hydropack.solvers.hs_solver import *
from hydropack.solvers.phi_solver import *
from hydropack.constants import *




class Glads2DModel:
    def __init__(self, model_inputs, in_dir=None):

        r"""Initialize model inputs and 
        Parameters
        ----------
        model_inputs: model input object
        """

        self.mesh = model_inputs["mesh"]
        self.U = firedrake.FunctionSpace(self.mesh, "CG", 1)
        self.V = firedrake.VectorFunctionSpace(self.mesh,"CG",1)
        self.CR = firedrake.FunctionSpace(self.mesh,"CR",1) 


        self.model_inputs = model_inputs

        # If an input directory is specified, load model inputs from there.
        # Otherwise use the specified model inputs dictionary.
        if in_dir:
            model_inputs = self.load_inputs(in_dir)

        # Ice thickness
        self.H = self.model_inputs["H"]
        # Bed elevation
        self.B = self.model_inputs["B"]
        # Basal sliding speed
        self.u_b = self.model_inputs["u_b"]
        # Melt rate
        self.m = self.model_inputs["m"]
        # Cavity gap height
        self.h = self.model_inputs["h_init"]
        # Potential
        self.phi_prev = self.model_inputs["phi_init"]
        # Potential at 0 pressure
        self.phi_m = self.model_inputs["phi_m"]
        # Ice overburden pressure
        self.p_i = self.model_inputs["p_i"]
        # Potential at overburden pressure
        self.phi_0 = self.model_inputs["phi_0"]
        # Dirichlet boundary conditions
        self.d_bcs = self.model_inputs["d_bcs"]
        # Channel areas
        self.S = self.model_inputs["S_init"]

        # If there is a dictionary of physical constants specified, use it.
        # Otherwise use the defaults.
        if "constants" in self.model_inputs:
            self.pcs = self.model_inputs["constants"]
        else:
            self.pcs = pcs

        self.n_bcs = []
        if "n_bcs" in self.model_inputs:
            self.n_bcs = model_inputs["n_bcs"]

        ### Create some fields


        # Potential
        self.phi = firedrake.Function(self.U)
        # Effective pressure at nodes
        self.N_n = firedrake.Function(self.U)
        # Effective pressure at edges
        self.N_e = firedrake.Function(self.CR)
        # Stores the value of S**alpha.
        self.S_alpha = firedrake.Function(self.CR)
        self.update_S_alpha()
        # Water pressure
        self.p_w = firedrake.Function(self.U)
        # Pressure as a fraction of overburden
        self.pfo = firedrake.Function(self.U)
        # Current time
        self.t = 0.0

        ### Output files
        self.S_out = firedrake.File(self.out_dir + "S.pvd")
        self.h_out = firedrake.File(self.out_dir + "h.pvd")
        self.phi_out = firedrake.File(self.out_dir + "phi.pvd")
        self.pfo_out = firedrake.File(self.out_dir + "pfo.pvd")

        ### Create the solver objects
        # Potential solver
        self.phi_solver = PhiSolver(self)
        # Gap height solver
        self.hs_solver = HSSolver(self)

    # Steps the potential, gap height, and water height forward by dt
    def step(self, dt):
        # Step the potential forward by dt with h fixed
        self.phi_solver.step(dt)
        # Step h forward by dt with phi fixed
        self.hs_solver.step(dt)

    # Load all model inputs from a directory except for the mesh and initial
    # conditions on h, h_w, and phi
    def load_inputs(self, in_dir):
        # Bed
        B = firedrake.Function(self.U)
        firedrake.File(in_dir + "B.pvd") >> B
        # Ice thickness
        H = firedrake.Function(self.U)
        firedrake.File(in_dir + "H.pvd") >> H
        # Melt
        m = firedrake.Function(self.U)
        firedrake.File(in_dir + "m.pvd") >> m
        # Sliding speed
        u_b = firedrake.Function(self.U)
        firedrake.File(in_dir + "u_b.pvd") >> u_b
        # Potential at 0 pressure
        phi_m = firedrake.Function(self.U)
        firedrake.File(in_dir + "phi_m.pvd") >> phi_m
        # Potential at overburden pressure
        phi_0 = firedrake.Function(self.U)
        firedrake.File(in_dir + "phi_0.pvd") >> phi_0
        # Ice overburden pressure
        p_i = firedrake.Function(self.U)
        firedrake.File(in_dir + "p_i.pvd") >> p_i

        self.model_inputs["B"] = B
        self.model_inputs["H"] = H
        self.model_inputs["m"] = m
        self.model_inputs["u_b"] = u_b
        self.model_inputs["phi_m"] = phi_m
        self.model_inputs["phi_0"] = phi_0
        self.model_inputs["p_i"] = p_i

    # Update the effective pressure to reflect current value of phi
    def update_N(self):
        phi_0_tmp = firedrake.interpolate(self.phi_0, self.U)
        phi_tmp = firedrake.interpolate(self.phi, self.U)
        N_tmp = firedrake.assemble(phi_0_tmp - phi_tmp)
        self.N.vector().set_local(N_tmp.vector().array())
        self.N.vector().apply("insert")

    # Update the water pressure to reflect current value of phi
    def update_pw(self):
        phi_tmp = firedrake.interpolate(self.phi, self.U)
        phi_m_tmp = firedrake.interpolate(self.phi_m, self.U)
        p_w_tmp = firedrake.assemble(phi_tmp - phi_m_tmp)
        self.p_w.vector().set_local(p_w_tmp.vector().array())
        self.p_w.vector().apply("insert")

    # Update the pressure as a fraction of overburden to reflect the current
    # value of phi
    def update_pfo(self):
        # Update water pressure
        self.update_pw()

        # Compute overburden pressure
        p_w_tmp = firedrake.interpolate(self.p_w, self.U)
        p_i_tmp = firedrake.interpolate(self.p_i, self.U)
        pfo_tmp = firedrake.assemble(p_w_tmp / p_i_tmp)
        self.pfo.vector().set_local(pfo_tmp.vector().array())
        self.pfo.vector().apply("insert")

    # update effective pressure on edge midpoints to reflect current value of phi
    def update_N_cr(self):
        self.update_N()
        self.cr_tools.midpoint(self.N,self.N_cr)

    def update_dphi_ds_cr(self):
        self.cr_tools.ds_assemble(self.phi,self.dphi_ds_cr)

    # Updates all fields derived from phi
    def update_phi(self):
        # phi_tmp=firedrake.interpolate(self.phi,self.U)
        self.phi_prev = firedrake.interpolate(self.phi_prev, self.U)
        phi_tmp = firedrake.interpolate(self.phi, self.U)
        self.phi_prev.assign(phi_tmp)
        self.update_N_cr()
        self.update_dphi_ds_cr()
        self.update_pfo()

    # Update the edge midpoint values h_cr to reflect the current value of h
    def update_h_cr(self):
        self.cr_tools.midpoint(self.h,self.h_cr)

    # Update S**alpha to reflect current value of S
    def update_S_alpha(self):
        alpha = self.pcs["alpha"]
        self.S_alpha.vector().set_local(self.S.vector().array() ** alpha)
        self.S_alpha.vector().apply('insert')
    # Write h, S, pfo, and phi to pvd files
    def write_pvds(self):
        self.S.assign(firedrake.interpolate(self.S, self.CR))
        self.h.assign(firedrake.interpolate(self.h, self.U))
        self.S_out << self.S
        self.h_out << self.h
        self.phi_out << self.phi
        self.pfo_out << self.pfo