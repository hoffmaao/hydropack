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
import os



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
        self.cr_tools = CRTools(self.mesh, self.U, self.CR)


        self.model_inputs = model_inputs

        # If an input directory is specified, load model inputs from there.
        # Otherwise use the specified model inputs dictionary.
        if in_dir:
            model_inputs = self.load_inputs(in_dir)

        # Ice thickness
        self.H = self.model_inputs["thickness"]
        # Bed elevation
        self.B = self.model_inputs["bed"]
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
        # Derivative of potential over channel edges
        self.dphi_ds_cr = firedrake.Function(self.CR)
        # Effective pressure at nodes
        self.N = firedrake.Function(self.U)
        # Effective pressure at edges
        self.N_cr = firedrake.Function(self.CR)
        # Sheet height on edges
        self.h_cr = firedrake.Function(self.CR)
        self.update_h_cr()
        # Stores the value of S**alpha.
        self.S_alpha = firedrake.Function(self.CR)
        self.update_S_alpha()
        # Water pressure
        self.p_w = firedrake.Function(self.U)
        # Pressure as a fraction of overburden
        self.pfo = firedrake.Function(self.U)
        # Current time
        self.t = 0.0

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


    # Update the effective pressure to reflect current value of phi
    def update_N(self):
        #phi_0_tmp = firedrake.interpolate(self.phi_0, self.U)
        #phi_tmp = firedrake.interpolate(self.phi, self.U)
        #N_tmp = firedrake.assemble(phi_0_tmp - phi_tmp)
        self.N.assign(self.phi_0 - self.phi)
        #self.N.vector().set_local(N_tmp.vector().array())
        #self.N.vector().apply("insert")

    # Update the water pressure to reflect current value of phi
    def update_pw(self):
        #phi_tmp = firedrake.interpolate(self.phi, self.U)
        #phi_m_tmp = firedrake.interpolate(self.phi_m, self.U)
        #p_w_tmp = firedrake.assemble(phi_tmp - phi_m_tmp)
        #self.p_w.vector().set_local(p_w_tmp.vector().array())
        #self.p_w.vector().apply("insert")

        self.p_w.assign(self.phi - self.phi_m)

    # Update the pressure as a fraction of overburden to reflect the current
    # value of phi
    def update_pfo(self):
        # Update water pressure
        self.update_pw()
        self.update_pw()
        # Safer division (in case p_i has tiny values anywhere)
        eps = firedrake.Constant(1.0)  # Pa; any small positive works since p_i ~ rho*g*H
        self.pfo.interpolate(self.p_w / firedrake.max_value(self.p_i, eps))


        # Compute overburden pressure
        p_w_tmp = firedrake.interpolate(self.p_w, self.U)
        p_i_tmp = firedrake.interpolate(self.p_i, self.U)
        pfo_tmp = firedrake.interpolate(p_w_tmp / p_i_tmp, self.U)
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
        #self.phi_prev = firedrake.interpolate(self.phi_prev, self.U)
        #phi_tmp = firedrake.interpolate(self.phi, self.U)
        self.phi_prev.assign(self.phi)
        self.update_N_cr()
        self.update_dphi_ds_cr()
        self.update_pfo()

    # Update the edge midpoint values h_cr to reflect the current value of h
    def update_h_cr(self):
        self.cr_tools.midpoint(self.h,self.h_cr)

    # Update S**alpha to reflect current value of S
    def update_S_alpha(self):
        #alpha = self.pcs["alpha"]
        #self.S_alpha.vector().set_local(self.S.vector().array() ** alpha)
        #self.S_alpha.vector().apply('insert')
        self.S_alpha.interpolate(self.S ** self.pcs["alpha"])

    import firedrake as fd

    def compute_flux_fields(self):
        mesh = self.mesh

        # --- Sheet flux vector q_s  (units m^2/s): q_s = -K_s * h^3 * grad(phi)
        Vvec = fd.VectorFunctionSpace(mesh, "CG", 1)   # or "DG", 0 if you prefer cellwise
        self.q_s = fd.Function(Vvec, name="q_s")
        Ks = fd.Constant(self.pcs.get("k"))    # plug in your law/params
        h  = self.h
        self.q_s.project(Ks * fd.max_value(h, 0.0)**self.pcs["alpha"] * (fd.inner(fd.grad(self.phi),fd.grad(self.phi))+ fd.Constant(1e-30))**(self.pcs["delta"]/2.0) * fd.grad(self.phi))

        # Optionally save magnitude (cellwise) too
        Vdg = fd.FunctionSpace(mesh, "DG", 0)
        self.q_s_mag = fd.Function(Vdg, name="q_s_mag")
        self.q_s_mag.project(fd.sqrt(fd.inner(self.q_s, self.q_s) + fd.Constant(1e-30)))

        # |∂phi/∂s| averaged per interior facet, abs on boundary
        self.update_dphi_ds_cr() 

        Vcr = self.dphi_ds_cr.function_space()
        self.Q_ch = fd.Function(Vcr)

        # Now build Q_ch from your law
        Kc = fd.Constant(self.pcs.get("k_c"))    # plug in your coefficient
        # pointwise on CR DOFs:
        self.Q_ch.interpolate(Kc * fd.max_value(self.S, 0.0)**(fd.Constant(self.pcs["alpha"])) *
                              fd.max_value(self.dphi_ds_cr, 0.0)**(fd.Constant(self.pcs["delta"])))


    # Write h, S, pfo, and phi to pvd files
    def write_pvds(self):
        # create file handles on first call
        if not hasattr(self, "S_out"):
            out_dir = self.model_inputs.get("out_dir", "./outputs")
            os.makedirs(out_dir, exist_ok=True)
            self.S_out  = firedrake.File(os.path.join(out_dir, "S.pvd"))
            self.h_out  = firedrake.File(os.path.join(out_dir, "h.pvd"))
            self.phi_out = firedrake.File(os.path.join(out_dir, "phi.pvd"))
            self.pfo_out = firedrake.File(os.path.join(out_dir, "pfo.pvd"))

        # You don't need to re-interpolate; S is already CR, h/phi/pfo are U.
        self.S_out << self.S
        self.h_out << self.h
        self.phi_out << self.phi
        self.pfo_out << self.pfo

    def write_checkpoint(self, filename="glads_state.h5"):
        """
        Save the current model state to an HDF5 checkpoint file.
        All functions are saved on their native function spaces.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        with firedrake.CheckpointFile(filename, "w") as checkpoint:
            # Save the mesh first
            checkpoint.save_mesh(self.mesh)

            # Save all relevant fields
            checkpoint.save_function(self.h, name="h")               # sheet height (U)
            checkpoint.save_function(self.S, name="S")               # channel area (CR)
            checkpoint.save_function(self.phi, name="phi")           # potential (U)
            checkpoint.save_function(self.pfo, name="pfo")           # fraction overburden (U)
            checkpoint.save_function(self.N, name="N")               # effective pressure (U)
            checkpoint.save_function(self.N_cr, name="N_cr")         # effective pressure (CR)
            checkpoint.save_function(self.h_cr, name="h_cr")         # sheet height (CR)
            checkpoint.save_function(self.S_alpha, name="S_alpha")   # S^alpha (CR)
            checkpoint.save_function(self.p_w, name="p_w")       