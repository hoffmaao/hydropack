import firedrake


""" Solves phi with h and S fixed."""


class PhiSolver(object):
    def __init__(self, model):

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
        # Derivative of phi

        # Flux vector for sheet
        q_s = (
            -firedrake.Constant(k)
            * h ** alpha
            * (firedrake.dot(firedrake.grad(phi),firedrake.grad(phi)) + phi_reg) ** (delta/2.0)
            * firedrake.grad(phi)
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

        # Discharge through channels
        # Normal and tangent vectors
        n = firedrake.FacetNormal(model.mesh)
        t = firedrake.as_vector([n[1],-n[0]])

        dphi_ds = firedrake.dot(firedrake.grad(phi),t)

        Q_c = (
            -firedrake.Constant(k_c)
            * S_alpha
            * abs(dphi_ds + firedrake.Constant(phi_reg)) ** delta
            * dphi_ds
        )
        # Approximate discharge of sheet in direction of channel
        q_c = (
            -firedrake.Constant(k)
            * h ** alpha
            * abs(dphi_ds + firedrake.Constant(phi_reg)) ** delta
            * dphi_ds
        )

        # Energy dissipation
        Xi = abs(Q_c * dphi_ds) + abs(firedrake.Constant(l_c) * q_c * dphi_ds)
        # f switch that rurn on or of the sheet flow's contribution to refreezing 

        f = firedrake.conditional(firedrake.gt(S,0),1.0,0.0)

        Pi = (firedrake.Constant(c_t * c_w + rho_water) 
            * (Q_c + f* l_c * q_c)
            * firedrake.dot(firedrake.grad(phi - phi_m),t)
        )


        # Another channel source term
        w_c = ((Xi - Pi) / firedrake.Constant(L) 
            * firedrake.Constant((1. / rho_ice) - (1. / rho_water))
        )


        # closing term assocaited with creep closure
        v_c = firedrake.Constant(A) * S * N ** firedrake.Constant(3.0)
        
        ### Set up the PDE for the potential ###


        theta = firedrake.TestFunction(model.U)
        #dphi = firedrake.TrialFunction(model.U)


        # Constant in front of storage term
        C1 = firedrake.Constant(c1)
        # Storage term
        F_s = C1 * (phi - phi_prev) * theta * firedrake.dx
        
        # Sheet contribution to PDE
        F_s += ((-firedrake.dot(firedrake.grad(theta), q_s)  + (w - v - m) * theta)
            * firedrake.dx
            )

        # Channel contribution to PDE
        F_c = dt * (-firedrake.dot(firedrake.grad(theta),t) * Q_c + (w_c - v_c) * theta("+")) * firedrake.dS

        # Variational form
        F = F_s + F_c
        # Get the Jacobian
        dphi = firedrake.TrialFunction(model.U)
        J = firedrake.derivative(F, phi, dphi)

        ### Assign local variables

        self.F = F
        self.J = J
        self.model = model
        self.dt = dt

    # Steps the potential forward by dt. Returns true if the  converged or false if it
    # had to use a smaller relaxation parameter.
    def step(self, dt):

        self.dt.assign(dt)

        try:

            # Solve for potential
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
