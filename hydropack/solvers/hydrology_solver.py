# Copyright (C) 2021 by Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of hydropack.
#
# hydropack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.


import firedrake
from firedrake import dx, inner, Constant
from hydropack.optimization import MinimizationProblem, NewtonSolver
from ..utilities import default_solver_parameters
from hydropack.calculus import grad, div, FacetNormal



class HydrologySolver(object):
    def __init__(self, model):
        r"""Solves the diagnostic models for subglacial hydrology

        This class is responsible for efficiently solving the physics
        problem you have chosen.

        Parameters
        ----------
        model
            The effective pressure model object -- glads, etc.
        dirichlet_ids : list of int, optional
            Numerical IDs of the boundary segments where effective pressure
            should be fixed
        side_wall_ids :
            Numerical IDs of the boundary segments where effective pressure
            should have no normal flow
        diagnostic_solver_type : {'hydropack','petsc'}, optional
            Use hand-written optimization solver ('icepack') or PETSc SNES
            ('petsc') , defalts to hydropack
        prognostic_solver_type : {'lax-wendroff', 'implicit-euler'}, optional
            Timestepping scheme to use for prognostic equations, defaults
            to Lax-Wendroff
        prognostic_solver_parameters : Dict, optional
            Options for prognostic solve routine; defaults to direct
            factorization of the flux matrix using MUMPS


        """

        self._model = model
        self._fields = {}

        # Prepare the diagnostic solver
        diagnostic_parameters = kwargs.get(
            "diagnostic_solver_parameters", default_solver_parameters
        )

        if "diagnostic_solver_type" in kwargs.keys():
            solver_type = kwargs["diagnostic_solver_type"]
            if isinstance(solver_type, str):
                solvers_dict = {"icepack": IcepackSolver, "petsc": PETScSolver}
                solver_type = solvers_dict[solver_type]
        else:
            solver_type = IcepackSolver

        self._diagnostic_solver = solver_type(
            self.model,
            self._fields,
            diagnostic_parameters,
            dirichlet_ids=kwargs.pop("dirichlet_ids", []),
            side_wall_ids=kwargs.pop("side_wall_ids", []),
        )

        # Prepare the prognostic solver
        prognostic_parameters = kwargs.get(
            "prognostic_solver_parameters", default_solver_parameters
        )

        if "prognostic_solver_type" in kwargs.keys():
            solver_type = kwargs["prognostic_solver_type"]
            if isinstance(solver_type, str):
                solvers_dict = {
                    "implicit-euler": ImplicitEuler,
                    "lax-wendroff": LaxWendroff,
                }
                solver_type = solvers_dict[solver_type]
        else:
            solver_type = LaxWendroff

        self._prognostic_solver = solver_type(
            self.model.continuity, self._fields, prognostic_parameters
        )
    @property
    def model(self):
        r"""The physics model that this object solves"""
        return self._model

    @property
    def fields(self):
        r"""Dictionary of all fields that are part of the simulation"""
        return self._fields

    def diagnostic_solve(self, **kwargs):
        r"""Solve the diagnostic model physics for the effective pressure"""
        return self._diagnostic_solver.solve(**kwargs)

    def prognostic_solve(self, dt, **kwargs):
        r"""Solve the prognostic model physics for the new conduit geometry"""
        return self._prognostic_solver.solve(dt, **kwargs)




class PETScSolver:
    def __init__(
        self, model, fields, solver_parameters, dirichlet_ids=[], side_wall_ids=[]
    ):
        r"""Diagnostic solver implementation using PETSc SNES"""
        self._model = model
        self._fields = fields
        self._solver_parameters = solver_parameters
        self._dirichlet_ids = dirichlet_ids
        self._side_wall_ids = side_wall_ids

    def setup(self, **kwargs):
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                if isinstance(field, firedrake.Constant):
                    self._fields[name] = firedrake.Constant(field)
                elif isinstance(field, firedrake.Function):
                    self._fields[name] = field.copy(deepcopy=True)
                else:
                    raise TypeError("Input fields must be Constant or Function!")

        # Create homogeneous BCs for the Dirichlet part of the boundary
        u = self._fields["velocity"]
        V = u.function_space()
        bcs = firedrake.DirichletBC(V, u, self._dirichlet_ids)
        if not self._dirichlet_ids:
            bcs = None

        # Find the numeric IDs for the ice front
        boundary_ids = u.ufl_domain().exterior_facets.unique_markers
        ice_front_ids_comp = set(self._dirichlet_ids + self._side_wall_ids)
        ice_front_ids = list(set(boundary_ids) - ice_front_ids_comp)

        # Create the action and scale functionals
        _kwargs = {"side_wall_ids": self._side_wall_ids, "ice_front_ids": ice_front_ids}
        action = self._model.action(**self._fields, **_kwargs)
        F = firedrake.derivative(action, u)

        degree = self._model.quadrature_degree(**self._fields)
        params = {"form_compiler_parameters": {"quadrature_degree": degree}}
        problem = firedrake.NonlinearVariationalProblem(F, u, bcs, **params)
        self._solver = firedrake.NonlinearVariationalSolver(
            problem, solver_parameters=self._solver_parameters
        )

    def solve(self, **kwargs):
        r"""Solve the diagnostic model physics for the ice velocity"""
        if not hasattr(self, "_solver"):
            self.setup(**kwargs)
        else:
            for name, field in kwargs.items():
                if isinstance(field, firedrake.Function):
                    self._fields[name].assign(field)

        # Solve the minimization problem and return the velocity field
        self._solver.solve()
        u = self._fields["velocity"]
        return u.copy(deepcopy=True)


class ImplicitEuler:
    def __init__(self, continuity, fields, solver_parameters):
        r"""Prognostic solver implementation using the 1st-order, backward
        Euler timestepping scheme
        This solver is included for backward compatibility only.
        """
        self._continuity = continuity
        self._fields = fields
        self._solver_parameters = solver_parameters

    def setup(self, **kwargs):
        r"""Create the internal data structures that help reuse information
        from past prognostic solves"""
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                if isinstance(field, firedrake.Constant):
                    self._fields[name] = firedrake.Constant(field)
                elif isinstance(field, firedrake.Function):
                    self._fields[name] = field.copy(deepcopy=True)
                else:
                    raise TypeError("Input fields must be Constant or Function!")

        dt = firedrake.Constant(1.0)
        dh_dt = self._continuity(dt, **self._fields)
        h = self._fields["thickness"]
        h_0 = h.copy(deepcopy=True)
        q = firedrake.TestFunction(h.function_space())
        F = (h - h_0) * q * dx - dt * dh_dt

        problem = firedrake.NonlinearVariationalProblem(F, h)
        self._solver = firedrake.NonlinearVariationalSolver(
            problem, solver_parameters=self._solver_parameters
        )

        self._thickness_old = h_0
        self._timestep = dt

    def solve(self, dt, **kwargs):
        r"""Compute the thickness evolution after time `dt`"""
        if not hasattr(self, "_solver"):
            self.setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self._fields[name].assign(field)

        h = self._fields["thickness"]
        self._thickness_old.assign(h)
        self._timestep.assign(dt)
        self._solver.solve()
        return h.copy(deepcopy=True)





    def solve(self, **kwargs):
        r"""Solve the diagnostic model physics for the ice velocity"""
        if not hasattr(self, "_solver"):
            self.setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self._fields[name].assign(field)

        # Solve the minimization problem and return the velocity field
        self._solver.solve()
        u = self._fields["velocity"]
        return u.copy(deepcopy=True)

        # melt rate
        m = model.m
        # Sheet height
        h = model.h
        # Channel area
        S = model.S
        # This function stores the value of S**alpha. 
        S_alpha = model.S_alpha
        # hydropotential at zero bed elvation
        phi_m = model.phi_m
        # Basal sliding speed
        u_b = model.u_b
        # Potential
        phi = model.phi
        # Potential at previous time step
        phi_prev = model.phi_prev
        # Potential at overburden pressure
        phi_0 = model.phi_0
        # Representative width of water system
        width = model.width
        # Density of ice
        rho_ice = model.pcs["rho_ice"]
        # Density of water
        rho_water = model.pcs["rho_water"]
        # Rate factor
        A = model.pcs["A"]
        # Sheet conductivity
        k = model.pcs["k"]
        # Channel conductivity
        k_c = model.pcs["k_c"]
        # Bump height
        h_r = model.pcs["h_r"]
        # Distance between bumps
        l_r = model.pcs["l_r"]
        # Sheet width under channel
        l_c = model.pcs["l_c"]
        # Clapeyron slope
        c_t = model.pcs["c_t"]
        # Specific heat capacity of water
        c_w = model.pcs["c_w"]
        # Latent heat
        L = model.pcs["L"]
        # Void storage ratio
        e_v = model.pcs["e_v"]
        # Gravitational acceleration
        g = model.pcs["g"]
        V_cg=e_v.function_space()
        # Exponents
        alpha = model.pcs["alpha"]
        delta = model.pcs["delta"]
        # pcs in front of storage term
        c1 = e_v / (rho_water * g)
        # Regularization parameter
        phi_reg = firedrake.Constant(1e-15)

        ### Set up the sheet model

        # Expression for effective pressure in terms of potential
        N = phi_0 - phi


        # Flux vector for sheet
        q_s = (
            -firedrake.Constant(k)
            * h ** alpha
            * (phi.dx(0)*phi.dx(0) + phi_reg) ** (delta/2.0)
            * phi.dx(0)
        )


        # Opening term
        w = firedrake.conditional(
            firedrake.gt(firedrake.Constant(h_r) - h, firedrake.Constant(0.0)), u_b * (firedrake.Constant(h_r) - h) / firedrake.Constant(l_r), firedrake.Constant(0.0)
        )


        # Closing term
        v = firedrake.Constant(A) * h * N ** 3.0
        # Time step
        dt = firedrake.Constant(1.0)


        ### Set up the channel model


        # Discharge through channel
        Q_c = (
            -firedrake.Constant(k_c)
            * S_alpha
            * abs(phi.dx(0) + firedrake.Constant(phi_reg)) ** delta
            * phi.dx(0)
        )

        # Discharge of sheet in direction of channel
        q_c = (
            -firedrake.Constant(k)
            * h ** alpha
            * abs(phi.dx(0) + firedrake.Constant(phi_reg)) ** delta
            * phi.dx(0)
        )

        # Energy dissipation
        Xi = abs(Q_c * phi.dx(0)) + abs(firedrake.Constant(l_c) * q_c * phi.dx(0))

        # pressure melting
        pw = phi - phi_m
        pw_s = pw.dx(0)
        pw_s = firedrake.interpolate(pw_s, model.V_cg)
        f = firedrake.conditional(firedrake.Or(S>0.0,q_c*pw.dx(0)> 0.0),1.0,0.0)
        II_n = -c_t * c_w * rho_water * 0.3 * (Q_c + f * l_c * q_c) * pw_s
        # Total opening rate (dissapation of potential energy and pressure melting)
        w_c = ((Xi - II_n) / firedrake.Constant(L)) * firedrake.Constant(
            (1.0 / rho_ice) - (1.0 / rho_water))

        # closing term assocaited with creep closure
        v_c = firedrake.Constant(A) * S * N ** firedrake.Constant(3.0)
        
        ### Set up the PDE for the potential ###
        theta = firedrake.TestFunction(model.V_cg)

        # Constant in front of storage term
        C1 = firedrake.interpolate(c1 * width, V_cg)
        # Storage term
        F_s = C1 * (phi - phi_prev) * theta * firedrake.dx
        
        # Sheet contribution to PDE
        F_s += (
            dt
            * (-theta.dx(0) * q_s * width + (w - v - m) * width * theta)
            * firedrake.dx
        )
        # Add any non-zero Neumann boundary conditions
        for (m, c) in model.n_bcs:
            F_s += dt * firedrake.Constant(c) * theta * m

        # Channel contribution to PDE
        F_c = dt * ((-theta.dx(0)) * Q_c + (w_c - v_c) * theta("+")) * firedrake.dx

        # Variational form
        F = F_s + F_c
        # Get the Jacobian
        dphi = firedrake.TrialFunction(model.V_cg)
        J = firedrake.derivative(F, phi, dphi)

        ### Assign local variables

        self.F = F
        self.J = J
        self.model = model
        self.dt = dt

    # Steps the potential forward by dt. Returns true if the  converged or false if it
    # had to use a smaller relaxation parameter.
    def solve(self, dt, **kwargs):
        r"""Compute the effective pressure
        """

        self.dt.assign(dt)

        try:

            # Solve for potential
            firedrake.solve(
                self.F == 0,
                self.model.phi,
                self.model.d_bcs,
                J=self.J,
                solver_parameters={
                    "snes_monitor": None,
                    "snes_view": None,
                    "ksp_monitor_true_residual": None,
                    "snes_converged_reason": None,
                    "ksp_converged_reason": None,
                },
            )  # , solver_parameters = self.model.newton_params)

            # Derive values from the new potential
            self.model.update_phi()
        except:

            # Try the solve again with a lower relaxation param
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
            )  # , solver_parameters = self.model.newton_params)

            # Derive values from potential
            self.model.update_phi()

            # Didn't converge with standard params
            return False

        # Did converge with standard params
        return True
