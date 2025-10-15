# hydropack/solvers/hs_solver.py
#
# Implicit RK (Irksome) solver for the GLADS sheet/channel ODEs.
# - h on U:    dh/dt = u_b * (h_r - h)/l_r  -  A * h * N^3
# - S on CR:   dS/dt = [ |Q*phi_s| + |l_c*q*phi_s| ]/(rho_i*L)  -  A * S * N_cr^3
#              where Q = -k_c S^alpha |phi_s + eps|^delta phi_s,
#                    q = -k   h_cr^alpha |phi_s + eps|^delta phi_s.
#
# This mirrors the FEniCS (Downs) GLADS rhs with the same symbols and ordering.
# We advance h first, update h_cr, then advance S.
#
# Requires: pip install irksome

import firedrake as fd
from irksome import Dt, TimeStepper, RadauIIA, GaussLegendre

class HSSolver:
    def __init__(self, model):
        """
        Parameters
        ----------
        model : Glads2DModel
            Must provide:
              - mesh, U (CG1), CR (CR1)
              - Functions: h (U), S (CR), N (U), N_cr (CR),
                           dphi_ds_cr (CR), u_b (U)
              - cr_tools.midpoint(h, h_cr)
              - update_h_cr(), update_S_alpha()
              - pcs dict with keys: A, l_r, h_r, rho_ice, L, k, k_c, l_c, alpha, delta
                (aliases below are supported)
        """
        self.m = model

        # --- helpers for constants (accept floats or fd.Constant) ---
        def C(x):
            return x if isinstance(x, fd.Constant) else fd.Constant(x)

        pcs = model.pcs
        A       = C(pcs.get("A",        pcs.get("rate_factor")))
        l_r     = C(pcs.get("l_r",      pcs.get("bump_spacing", 1.0)))
        h_r     = C(pcs.get("h_r",      pcs.get("bump_height", 0.0)))
        rho_i   = C(pcs.get("rho_ice",  pcs.get("rho_i", 910.0)))
        L       = C(pcs.get("L",        3.34e5))
        k       = C(pcs.get("k",        pcs.get("sheet_k", 1.0)))
        k_c     = C(pcs.get("k_c",      pcs.get("channel_k", 1.0)))
        l_c     = C(pcs.get("l_c",      1.0))
        alpha   = C(pcs.get("alpha",    5.0/4.0))
        delta   = C(pcs.get("delta",    1.0/2.0))
        phi_reg = C(pcs.get("phi_reg",  1e-15))

        self._A, self._l_r, self._h_r   = A, l_r, h_r
        self._rho_i, self._L           = rho_i, L
        self._k, self._k_c, self._l_c  = k, k_c, l_c
        self._alpha, self._delta       = alpha, delta
        self._phi_reg                  = phi_reg

        # Optional CR mask (1 on interior/channel edges, 0 on no-opening edges)
        if hasattr(model, "mask") and isinstance(model.mask, fd.Function) and model.mask.function_space() == model.CR:
            self._mask_cr = model.mask
        else:
            self._mask_cr = fd.Function(model.CR, name="mask_cr")
            self._mask_cr.assign(1.0)

        # Time objects for Irksome
        self._t  = fd.Constant(model.t)
        self._dt = fd.Constant(0.0)

        # Unknowns and tests
        self.h   = model.h                  # in U
        self.S   = model.S                  # in CR
        z        = fd.TestFunction(model.U)
        w        = fd.TestFunction(model.CR)

        # Positivity lower bounds per-space (for SNESVI)
        # Positivity lower bounds per-space (for SNESVI)
        self._lbU  = fd.Function(model.U,  name="lbU");  self._lbU.assign(0.0)
        self._lbCR = fd.Function(model.CR, name="lbCR"); self._lbCR.assign(0.0)

        # NEW: add very-large upper bounds so bounds=(lower, upper) has both Functions
        self._ubU  = fd.Function(model.U,  name="ubU");  self._ubU.assign(1.0e30)
        self._ubCR = fd.Function(model.CR, name="ubCR"); self._ubCR.assign(1.0e30)


        # Scratch functions for CR→U projections used in _rhs_S
        self._phi_s_u = fd.Function(model.U, name="phi_s_u")
        self._N_cr_u  = fd.Function(model.U, name="N_cr_u")
        self._h_cr_u  = fd.Function(model.U, name="h_cr_u")


        # Coefficients that change each global step but are held fixed *within* a substep:
        u_b   = model.u_b                  # U
        N     = model.N                    # U
        N_cr  = model.N_cr                 # CR
        phi_s = model.dphi_ds_cr           # CR
        h_cr  = model.h_cr                 # CR (updated after we advance h)

        # ---- Sheet height equation on U (matches Downs GLADS rhs) ----
        # F_h = (Dt(h), z) - (u_b*(h_r - h)/l_r - A*h*N^3, z) = 0
        F_h = fd.inner(Dt(self.h), z) * fd.dx \
              - fd.inner(u_b * (self._h_r - self.h) / self._l_r
                         - self._A * self.h * N**3, z) * fd.dx

        # Robust, simple (implicit Euler) to start
        tableau_h = RadauIIA(1)
        self._stepper_h = TimeStepper(
            F_h, tableau_h, self._t, self._dt, self.h,
            solver_parameters={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-10,
                "snes_atol": 1e-12,
                "snes_max_it": 50,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "mat_type": "aij",
            },
        )

        # ---- Channel area equation on CR (matches Downs GLADS rhs) ----
        # Q = -k_c S^alpha |phi_s + eps|^delta phi_s
        # q = -k   h_cr^alpha |phi_s + eps|^delta phi_s
        # Xi = |Q*phi_s| + |l_c*q*phi_s|
        # dS/dt = mask * Xi/(rho_i L) - A S N_cr^3
        Q   = - self._k_c * self.S**self._alpha \
              * fd.sqrt(phi_s*phi_s + self._phi_reg)**self._delta * phi_s
        q   = - self._k   * h_cr**self._alpha \
              * fd.sqrt(fd.inner(phi_s + self._phi_reg,phi_s + self._phi_reg))**self._delta * phi_s
        Xi  = fd.sqrt(Q * phi_s*Q * phi_s) + fd.sqrt(self._l_c * q * phi_s*self._l_c * q * phi_s)

        v_open  = self._mask_cr * Xi / (self._rho_i * self._L)

        v_close = self._A * self.S * N_cr**3

        F_S = fd.inner(Dt(self.S), w) * fd.dx - fd.inner(v_open - v_close, w) * fd.dx

        tableau_S = RadauIIA(1)
        self._stepper_S = TimeStepper(
            F_S, tableau_S, self._t, self._dt, self.S,
            solver_parameters={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-9,
                "snes_atol": 1e-12,
                "snes_max_it": 50,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "mat_type": "aij",
            },
        )
    def _refresh_edge_mids_to_U(self):
        """Keep all drivers of S-ODE in the same space (U) as S."""
        m = self.m
        # make sure model-side edge data are up to date
        m.update_N_cr()
        m.update_h_cr()
        m.update_dphi_ds_cr()
        # Cheap and robust: L2 project CR → U (OK for now; optimise later if needed)
        self._phi_s_u = fd.project(m.dphi_ds_cr, m.U)
        self._N_cr_u = fd.project(m.N_cr,       m.U)
        self._h_cr_u = fd.project(m.h_cr,       m.U)

    def _abs_smooth(self, x,reg=1.0e-12):
        return fd.sqrt(x**2+fd.Constant(reg))

    def _rhs_h(self, h):
        """Pointwise RHS for cavity height (U space)."""
        m = self.m
        A   = m.pcs["A"]
        l_r = m.pcs["l_r"]
        h_r = m.pcs["h_r"]
        # sliding speed in U
        u_b = m.u_b

        # Opening and closure
        w_open = u_b * (h_r - h) / l_r
        v_close = A * h * (m.N**3)

        # No post-assign clamps here; we’ll enforce bounds via SNESVI
        return w_open - v_close


    def _rhs_S(self, S):
        """Pointwise RHS for channel area (U space) with CR drivers mapped into U."""
        m = self.m
        alpha = m.pcs["alpha"]
        delta = m.pcs["delta"]
        k     = m.pcs["k"]
        k_c   = m.pcs["k_c"]
        l_c   = m.pcs["l_c"]
        rho_i = m.pcs["rho_ice"]
        L     = m.pcs["L"]

        # CR → U copies (kept current by _refresh_edge_mids_to_U)
        phi_s = self._phi_s_u
        Nmid  = self._N_cr_u
        hmid  = self._h_cr_u

        # Regularised |phi_s| for turbulent flux laws
        abs_phi = self._abs_smooth(phi_s)

        # Along-channel and sheet flux terms (all U)
        Q = -k_c * (S**alpha) * (abs_phi**delta) * phi_s
        q = -k   * (hmid**alpha) * (abs_phi**delta) * phi_s

        # Dissipation (smooth |.|)
        Xi = self._abs_smooth(Q*phi_s) + self._abs_smooth(l_c * q * phi_s)

        # Creep closure on channels uses N at edges → mapped to U as Nmid
        v_close = m.pcs["A"] * S * (Nmid**3)

        # (Optional) Clapeyron-pressure term; uncomment if you want it active
        # c_t, c_w, rho_w = m.pcs["c_t"], m.pcs["c_w"], m.pcs["rho_water"]
        # pw_s = fd.grad(m.phi - m.phi_m)[0]
        # theta = fd.Constant(0.3)
        # f = fd.conditional(fd.gt(phi_s*q, 0.0), 1.0, 0.0)
        # II = -c_t*c_w*rho_w*theta * (Q + f*l_c*q) * pw_s
        # v_open = (Xi - II) / (rho_i * L)

        v_open = Xi / (rho_i * L)

        return v_open - v_close

    def _make_stage_bounds(self, stepper, base_lb, base_ub):
        # Grab the actual mixed stage space from the stepper’s solver unknown
        prob = getattr(stepper.solver, "problem", getattr(stepper.solver, "_problem"))
        W = prob.u.function_space()              # MixedFunctionSpace: V^s
        s = W.num_sub_spaces()

        lb_big = fd.Function(W); ub_big = fd.Function(W)
        # Fill each stage slot with the base-space bounds
        for i in range(s):
            lb_big.subfunctions[i].assign(base_lb)
            ub_big.subfunctions[i].assign(base_ub)
        return lb_big, ub_big



    def _build_h_stepper(self, dt):
        m  = self.m
        Vh = m.h.function_space()         # space of h (likely CG1)
        v  = fd.TestFunction(Vh)
        h  = m.h  # use the model’s function directly so the stepper updates it
        Fh = fd.inner(Dt(h), v)*fd.dx - fd.inner(self._rhs_h(h), v)*fd.dx

        GL2 = GaussLegendre(2)  # 2-stage, A-stable

        stepper = TimeStepper(
            Fh, GL2, fd.Constant(m.t), dt, m.h,
            solver_parameters={
                "snes_type": "vinewtonrsls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1.0e-9,
                "snes_atol": 1.0e-12,
                "snes_max_it": 100,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "mat_type": "aij",
            },
        )
        lb_big, ub_big = self._make_stage_bounds(stepper, self._lbU, self._ubU)
        stepper.stage_bounds = (lb_big, ub_big)
        return stepper


    def _build_S_stepper(self, dt):
        m  = self.m
        VS = m.S.function_space()
        v  = fd.TestFunction(VS)
        S  = m.S

        phi_s = m.dphi_ds_cr     # CR
        N_cr  = m.N_cr           # CR
        h_cr  = m.h_cr           # CR

        abs_phi = self._abs_smooth(phi_s)
        Q = -self._k_c * (S**self._alpha)   * (abs_phi**self._delta) * phi_s
        q = -self._k   * (h_cr**self._alpha)* (abs_phi**self._delta) * phi_s

        Xi = self._abs_smooth(Q*phi_s) + self._abs_smooth(self._l_c*q*phi_s)

        v_open  = self._mask_cr * Xi / (self._rho_i * self._L)
        v_close = self._A * S * (N_cr**3)

        Fs = fd.inner(Dt(S), v)*fd.dx - fd.inner(v_open - v_close, v)*fd.dx


        GL2 = GaussLegendre(2)

        stepper = TimeStepper(
            Fs, GL2, fd.Constant(m.t), dt, m.S,
            solver_parameters={
                "snes_type": "vinewtonrsls",
                "snes_linesearch_type": "bt",
                "snes_linesearch_damping": 0.2,
                "snes_rtol": 1.0e-7,
                "snes_atol": 1.0e-12,
                "snes_max_it": 100,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "mat_type": "aij"
            }
        )
        lb_big, ub_big = self._make_stage_bounds(stepper, self._lbCR, self._ubCR)
        stepper.stage_bounds = (lb_big, ub_big)
        return stepper

    def step(self, dt):
        m = self.m

        # Rebuild steppers if dt changed (Irksome keeps dt internally)
        if abs(float(self._stepper_h.dt) - float(dt)) > 0.0:
            self._stepper_h = self._build_h_stepper(dt)
        if self._stepper_S is None or abs(float(self._dt) - float(dt)) > 0:
            self._dt = fd.Constant(float(dt))
            self._stepper_S = self._build_S_stepper(self._dt)

        # Make sure all drivers that live on CR are available on U
        self._refresh_edge_mids_to_U()

        # Advance h then S; both with positivity constraints through SNESVI
        try:
            self._stepper_h.advance()
            self._refresh_edge_mids_to_U()   # hmid changed → update S drivers
            self._stepper_S.advance()
        except fd.ConvergenceError:
            # Simple backtracking on failure: halve dt and retry once
            new_dt = 0.5*float(self._dt.values()[0] if hasattr(self._dt, "values") else float(self._dt))
            if new_dt < 1.0:  # guard against vanishing dt
                raise
            self._dt.assign(new_dt)
            self._stepper_h = self._build_h_stepper(new_dt)
            self._stepper_S = self._build_S_stepper(new_dt)
            self._refresh_edge_mids_to_U()
            self._stepper_h.advance()
            self._refresh_edge_mids_to_U()
            self._stepper_S.advance()

        # Keep derived fields consistent
        m.update_S_alpha()  # uses m.S
        m.t += float(self._stepper_h.dt)

