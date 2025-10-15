# hydropack/solvers/hs_solver.py
import numpy as np
import firedrake as fd


class HSSolver:
    """
    NumPy-only ODE integrator for the distributed sheet height h (on U)
    and the channel area S (on CR).  Per the GLADS split, phi is held
    fixed during an HS step; we therefore cache phi-derived fields once
    per call to step().
    """

    def __init__(self, model):
        self.model = model
        pcs = model.pcs

        # ---- constants / params
        self.A         = pcs["A"]
        self.l_r       = pcs["l_r"]
        self.h_r       = pcs["h_r"]
        self.rho_ice   = pcs["rho_ice"]
        self.rho_water = pcs["rho_water"]
        self.L         = pcs["L"]
        self.k         = pcs["k"]
        self.k_c       = pcs["k_c"]
        self.l_c       = pcs["l_c"]
        self.c_t       = pcs["c_t"]
        self.c_w       = pcs["c_w"]
        self.alpha     = pcs["alpha"]
        self.delta     = pcs["delta"]

        # small regularization for |grad phi|
        self.phi_reg   = 1e-15

        # cap each substep (seconds) to mirror your old SciPy max_step
        self.max_substep = 60.0 * 5.0

        # ---- immutable arrays we use repeatedly
        # sliding speed on U dofs (matches h)
        self.u_b_u = model.u_b.dat.data_ro.copy()  # |U|

        # an optional CR mask (1 on interior/channel edges; 0 where opening forbidden)
        try:
            mask = model.mask.dat.data_ro
            self.mask_cr = mask.copy() if mask is not None else None
        except Exception:
            self.mask_cr = None

        # lengths
        self.h_len = model.h.dat.data_ro.shape[0]     # |U|
        self.S_len = model.S.dat.data_ro.shape[0]     # |CR|

    # ---------- helpers ----------

    def _cache_for_step(self):
        """
        Cache fields that depend on phi (which is frozen during HS stepping)
        and that live on CR so shapes match |CR|.
        """
        m = self.model

        # Effective pressure N on U and on CR
        N_u  = m.N.dat.data_ro.copy()      # |U|
        N_cr = m.N_cr.dat.data_ro.copy()   # |CR|  (should already be in sync with phi)

        # Sheet height on CR (midpoint of h) — this depends on h from the LAST outer step.
        # It stays fixed during the HS solve to mimic your original Fenics setup.
        h_cr = m.h_cr.dat.data_ro.copy()   # |CR|

        # Along-edge gradient of phi on CR
        phi_s = m.dphi_ds_cr.dat.data_ro.copy()  # |CR|

        # Pressure-melting gradient term: (phi - phi_m)_x on CR (compute once)
        # NOTE: If your mesh x-direction is not index 0, adjust .dx(0) accordingly.
        pw_s_cr = fd.project((m.phi - m.phi_m).dx(0), m.CR).dat.data_ro.copy()  # |CR|

        return {
            "N_u": N_u,
            "N_cr": N_cr,
            "h_cr": h_cr,
            "phi_s": phi_s,
            "pw_s_cr": pw_s_cr,
            "mask_cr": self.mask_cr if self.mask_cr is not None else 1.0,
            "u_b_u": self.u_b_u,
        }

    # rhs for h on U dofs
    def _rhs_h(self, h_u, cache):
        A, l_r, h_r = self.A, self.l_r, self.h_r
        u_b_u, N_u  = cache["u_b_u"], cache["N_u"]

        # positivity of h in the model state
        h_u = np.maximum(h_u, 0.0)

        # opening by sliding over roughness
        w_u = u_b_u * (h_r - h_u) / l_r
        w_u = np.maximum(w_u, 0.0)

        # creep closure: A * h * N^3
        v_u = A * h_u * (N_u ** 3.0)

        return w_u - v_u

    # rhs for S on CR dofs
    def _rhs_S(self, S_cr, cache):
        k, k_c, l_c  = self.k, self.k_c, self.l_c
        A, alpha, delta = self.A, self.alpha, self.delta
        rho_i, L = self.rho_ice, self.L
        c_t, c_w, rho_w = self.c_t, self.c_w, self.rho_water
        phi_reg = self.phi_reg

        N_cr    = cache["N_cr"]      # |CR|
        h_cr    = cache["h_cr"]      # |CR|
        phi_s   = cache["phi_s"]     # |CR|
        pw_s_cr = cache["pw_s_cr"]   # |CR|
        mask    = cache["mask_cr"]   # |CR| or scalar 1.0

        S_cr = np.maximum(S_cr, 0.0)

        # along-channel flux and sub-sheet flux beneath channel
        abs_phi = np.abs(phi_s + phi_reg) ** delta
        Q_n = -k_c * (S_cr ** alpha) * abs_phi * phi_s
        q_n = -k   * (h_cr ** alpha) * abs_phi * phi_s

        # dissipation-driven opening
        Xi_n = np.abs(Q_n * phi_s) + np.abs(l_c * q_n * phi_s)

        # creep closure
        v_c_n = A * S_cr * (N_cr ** 3.0)

        # pressure-melting contribution (Downs et al. style; keep your 0.3 factor)
        f = np.zeros_like(S_cr)
        f[(S_cr > 0.0) & (phi_s * q_n > 0.0)] = 1.0
        II_n = -c_t * c_w * rho_w * 0.3 * (Q_n + f * l_c * q_n) * pw_s_cr

        # total opening
        v_o_n = (Xi_n - II_n) / (rho_i * L)
        v_o_n[(S_cr == 0.0) & (v_o_n < 0.0)] = 0.0

        return (v_o_n - v_c_n) * mask

    # one RK4 substep for both h and S (they are decoupled when phi is fixed)
    def _rk4_substep(self, h_u, S_cr, dt, cache):
        # h
        k1h = self._rhs_h(h_u, cache)
        k2h = self._rhs_h(h_u + 0.5*dt*k1h, cache)
        k3h = self._rhs_h(h_u + 0.5*dt*k2h, cache)
        k4h = self._rhs_h(h_u + dt*k3h, cache)
        h_u_new = h_u + (dt/6.0)*(k1h + 2*k2h + 2*k3h + k4h)

        # S
        k1s = self._rhs_S(S_cr, cache)
        k2s = self._rhs_S(S_cr + 0.5*dt*k1s, cache)
        k3s = self._rhs_S(S_cr + 0.5*dt*k2s, cache)
        k4s = self._rhs_S(S_cr + dt*k3s, cache)
        S_cr_new = S_cr + (dt/6.0)*(k1s + 2*k2s + 2*k3s + k4s)

        # enforce positivity after the update
        h_u_new = np.maximum(h_u_new, 0.0)
        S_cr_new = np.maximum(S_cr_new, 0.0)
        return h_u_new, S_cr_new

    # ---------- public API ----------

    def step(self, dt):
        """
        Advance h (U) and S (CR) by dt using substepped RK4.
        Mirrors the old semantics: phi is held fixed during this call.
        """
        m = self.model

        # read current state as numpy views
        h_u = m.h.dat.data_ro.copy()   # |U|
        S_cr = m.S.dat.data_ro.copy()  # |CR|

        # cache phi-dependent pieces once per HS step
        cache = self._cache_for_step()

        # substepping to respect max_substep (like SciPy's max_step)
        nsub = int(np.ceil(dt / self.max_substep))
        subdt = dt / nsub

        for _ in range(nsub):
            h_u, S_cr = self._rk4_substep(h_u, S_cr, subdt, cache)

        # write back to Firedrake functions
        m.h.dat.data[:] = h_u
        m.S.dat.data[:] = S_cr

        # update derived fields that phi solver will need next time
        m.update_S_alpha()
        m.update_h_cr()     # keep h_cr in sync for the next phi solve

        # advance model clock
        m.t += dt
