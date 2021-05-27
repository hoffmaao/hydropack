"""Solves ODEs for the sheet height h and channel area S."""

import firedrake
from scipy.integrate import ode
import numpy as np


class HSSolver:
    def __init__(self, model):

        ### Get a few fields and parameters from the model

        # Effective pressure
        N = firedrake.interpolate(model.N, model.V_cg)
        # hydropotential
        phi = firedrake.interpolate(model.phi, model.V_cg)
        # hydropotential at zero bed elvation
        phi_m = firedrake.interpolate(model.phi_m, model.V_cg)
        # Cavity gap height
        h = firedrake.interpolate(model.h, model.V_cg)
        # Channel height
        S = firedrake.interpolate(model.S, model.V_cg)
        # Initial model time
        t0 = model.t
        # Rate factor
        A = model.pcs["A"]
        # Distance between bumps
        l_r = model.pcs["l_r"]
        # Bump height
        h_r = model.pcs["h_r"]
        # Density of ice
        rho_ice = model.pcs["rho_ice"]
        # Density of water
        rho_water = model.pcs["rho_water"]
        # Latent heat
        L = model.pcs["L"]
        # Sheet conductivity
        k = model.pcs["k"]
        # Channel conductivity
        k_c = model.pcs["k_c"]
        # Sheet width under channel
        l_c = model.pcs["l_c"]
        # Clapeyron slope
        c_t = model.pcs["c_t"]
        # Specific heat capacity of water
        c_w = model.pcs["c_w"]
        # Exponent
        alpha = model.pcs["alpha"]
        delta = model.pcs["delta"]
        # Regularization parameter
        phi_reg = 1e-15

        # Vector for sliding speed
        u_b_n = firedrake.interpolate(model.u_b, model.V_cg)
        u_b_n = u_b_n.vector().array()
        h0 = firedrake.interpolate(
            model.h, model.V_cg
        )  # model.h.vector().array()#firedrake.interpolate(model.h,model.V_cg)
        h0 = h0.vector().array()
        # Initial channel areas
        S0 = firedrake.interpolate(model.S, model.V_cg)
        S0 = S0.vector().array()
        # Length of h vector
        h_len = len(h0)

        # Right hand side for the gap height ODE
        def h_rhs(t, h_n):
            # Ensure that the sheet height is positive
            h_n[h_n < 0.0] = 0.0
            # Get effective pressures
            N_n = N.dat.data_ro
            # Sheet opening term
            w_n = u_b_n * (h_r - h_n) / l_r
            # Ensure that the opening term is non-negative
            w_n[w_n < 0.0] = 0.0
            # Sheet closure term
            v_n = A * h_n * N_n ** 3.0

            # Return the time rate of change of the sheet
            dhdt = w_n - v_n

            return dhdt

        # Right hand side for the channel area ODE
        def S_rhs(t, S_n):
            # Channel area is positive
            S_n[S_n < 0.0] = 0.0
            # Effective pressures
            N_n = N.dat.data_ro
            # Sheet thickness
            h_n = h.dat.data_ro

            phi_grad = model.phi.dx(0)
            phi_s = firedrake.interpolate(phi_grad, model.V_cg)

            # Along channel flux
            Q_n = (
                -k_c
                * S_n ** alpha
                * np.abs(phi_s.dat.data_ro + phi_reg) ** delta
                * phi_s.dat.data_ro
            )
            # Flux of sheet under channel
            q_n = (
                -k
                * h_n ** alpha
                * np.abs(phi_s.dat.data_ro + phi_reg) ** delta
                * phi_s.dat.data_ro
            )
            # Dissipation melting due to turbulent flux
            Xi_n = abs(Q_n * phi_s.dat.data_ro) + np.abs(l_c * q_n * phi_s.dat.data_ro)

            # Creep closure
            v_c_n = A * S_n * N_n ** 3
            # pressure melting term
            pw = phi - phi_m
            pw_s = pw.dx(0)
            pw_s = firedrake.interpolate(pw_s, model.V_cg)
            f=np.zeros(np.size(q_n))
            f[(S_n > 0) & (phi_s.dat.data_ro*q_n>0.0) ]=1.0
            II_n = (
                -c_t * c_w * rho_water * 0.3 * (Q_n + f * l_c * q_n) * pw_s.dat.data_ro
            )
            # Total opening rate (dissapation of potential energy and pressure melting)
            v_o_n = (Xi_n - II_n) / (rho_ice * L)
            # Disallow negative opening rate where the channel area is 0
            v_o_n[(S_n==0.0) & (v_o_n<0.0)] = 0.0

            dsdt = v_o_n - v_c_n
            return dsdt

        # Combined right hand side for h and S
        def rhs(t, Y):
            Ys = np.split(Y, [h_len])
            h_n = Ys[0]
            S_n = Ys[1]
            dsdt = S_rhs(t, S_n)
            dhdt = h_rhs(t, h_n)
            return np.hstack((dhdt, dsdt))

        # ODE solver initial condition
        Y0 = np.hstack((h0, S0))
        # Set up ODE solver
        ode_solver = ode(rhs).set_integrator("vode",method="adams", max_step=60.0 * 5)
        ode_solver.set_initial_value(Y0, t0)

        # Set local variables
        self.ode_solver = ode_solver
        self.model = model
        self.h_len = h_len

    def step(self, dt):
        # Step h and S forward
        self.ode_solver.integrate(self.model.t + dt)

        # Retrieve values from the ODE solver
        Y = np.split(self.ode_solver.y, [self.h_len])
        self.model.h.vector().set_local(Y[0])
        self.model.h.vector().apply("insert")
        self.model.S.vector().set_local(Y[1])
        self.model.S.vector().apply("insert")
        # Update S**alpha
        self.model.update_S_alpha()

        # Update the model time
        self.model.t = self.ode_solver.t
