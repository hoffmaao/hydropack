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

r"""Bounded-pressure sheet-channel drainage model on a staggered grid

This module contains :class:`StaggeredGrid` and
:class:`BoundedHydrologyModel`, a finite-difference implementation of
GlaDS-2 (Wells et al., 2026).  The model solves the mass-conservation
equation (Eq. 17) for the total water thickness :math:`h`, which is the
conserved quantity, and recovers water pressure from the closure relation
of Eq. (4).  Regions of ice uplift and free-surface flow are therefore
captured without tracking their boundaries.

Following Sect. 2.4 of the paper, cell-centred variables are
:math:`h, h_g, m, B, H, u_b` (and derived :math:`\phi, p_w, h_w, h_e,
\Delta h`).  Normal sheet fluxes, channel areas :math:`S, S_w` and
discharges :math:`Q` live on cell faces.  A channel on an x face runs
between the centres of the two cells it separates, so it carries water in
the x direction.  Each step

1. computes :math:`h_w, p_w, \phi, S_w, q, Q` (Eqs. 1, 4, 7, 15, 6, 16),
2. picks a stable time step (Eqs. 19-23),
3. advances :math:`h` with forward Euler and central differences, and
4. advances :math:`h_g` and :math:`S` with semi-implicit Euler (Eqs. 5, 10).
"""

import numpy as np

from hydropack.glads2 import physics as _physics
from hydropack.glads2.constants import (
    gravity,
    water_density,
    default_parameters,
)

_SIDES = ("west", "east", "south", "north")


class StaggeredGrid:
    r"""Uniform structured grid with cell-centred and face-centred variables.

    Parameters
    ----------
    nx, ny : int
        Number of cells in the x and y directions
    dx, dy : float
        Grid spacing (m)
    x0, y0 : float
        Coordinates of the lower-left domain corner (m)
    """

    def __init__(self, nx, ny, dx, dy, x0=0.0, y0=0.0):
        self.nx, self.ny = int(nx), int(ny)
        self.dx, self.dy = float(dx), float(dy)
        self.x0, self.y0 = float(x0), float(y0)
        self.x = x0 + (np.arange(self.nx) + 0.5) * self.dx
        self.y = y0 + (np.arange(self.ny) + 0.5) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.x_faces = x0 + np.arange(self.nx + 1) * self.dx
        self.y_faces = y0 + np.arange(self.ny + 1) * self.dy
        self.cell_area = self.dx * self.dy

    @property
    def shape(self):
        return (self.nx, self.ny)

    @property
    def x_face_shape(self):
        return (self.nx + 1, self.ny)

    @property
    def y_face_shape(self):
        return (self.nx, self.ny + 1)

    @property
    def length(self):
        return self.nx * self.dx

    @property
    def width(self):
        return self.ny * self.dy

    def cell_index(self, x, y):
        r"""Return the index ``(i, j)`` of the cell containing the point ``(x, y)``"""
        i = int(np.clip(np.floor((x - self.x0) / self.dx), 0, self.nx - 1))
        j = int(np.clip(np.floor((y - self.y0) / self.dy), 0, self.ny - 1))
        return i, j


class PointSource:
    r"""Volumetric water input into a single grid cell.

    Parameters
    ----------
    x, y : float
        Location of the source (m)
    rate : callable
        Function of time returning the input rate (m^3 / s)
    """

    def __init__(self, x, y, rate):
        self.x, self.y = x, y
        self.rate = rate


def _parse_boundary_condition(bc):
    if bc is None or bc == "no_flux":
        return ("no_flux", 0.0)
    if bc == "atmospheric":
        return ("atmospheric", 0.0)
    if isinstance(bc, (tuple, list)) and len(bc) == 2 and bc[0] == "flux":
        return ("flux", float(bc[1]))
    raise ValueError(
        f"Unknown boundary condition {bc!r}; expected 'no_flux', "
        "'atmospheric' or ('flux', inflow_per_unit_width)"
    )


def _as_field(value, shape, name):
    array = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


class BoundedHydrologyModel:
    r"""GlaDS-2 sheet-channel model with ice uplift and free-surface flow.

    Parameters
    ----------
    grid : StaggeredGrid
        The computational grid.
    bed : array_like
        Bed elevation :math:`B` at cell centres (m).
    thickness : array_like
        Ice thickness :math:`H` at cell centres (m); zero in the proglacial zone.
    sliding_speed : array_like, optional
        Basal sliding speed :math:`u_b` (m/s); defaults to Table 2 value.
    melt_rate : array_like or callable, optional
        Distributed water input :math:`m` (m/s), or a function of time
        returning it.
    point_sources : sequence of PointSource, optional
        Concentrated inputs such as moulins or supraglacial lake drainage.
    h_init, h_g_init : array_like, optional
        Initial total water thickness and cavity height (m).  ``h_g_init``
        defaults to :math:`h_r/2`; ``h_init`` defaults to ``h_g_init``
        (atmospheric pressure).
    S_x_init, S_y_init : array_like, optional
        Initial channel areas on x and y faces (m^2); default zero.
    boundary_conditions : dict, optional
        Maps ``"west"``, ``"east"``, ``"south"``, ``"north"`` to ``"no_flux"``
        (default), ``"atmospheric"`` (:math:`p_w = 0` imposed with a ghost
        cell) or ``("flux", q_in)`` for an inflow ``q_in`` per unit width
        (m^2/s).
    channels : bool
        Whether channels are allowed to form on grid faces.
    cfl : float
        Safety factor :math:`C` in Eq. (23).
    **kwargs
        Parameter overrides (``k``, ``k_c``, ``A``, ``A_c``, ``e_v``, ``l_r``,
        ``h_r``, ``l_c``, ``phi_reg``) and replacements for the constitutive
        relations (``water_pressure``, ``sheet_flux``, ``water_filled_area``,
        ``channel_discharge``, ``channel_dissipation``,
        ``channel_pressure_melting``, ``update_cavity_height``,
        ``update_channel_area``, ``effective_diffusivity``,
        ``effective_velocity``).
    """

    _physics_names = (
        "water_pressure",
        "sheet_flux",
        "water_filled_area",
        "channel_discharge",
        "channel_dissipation",
        "channel_pressure_melting",
        "update_cavity_height",
        "update_channel_area",
        "effective_diffusivity",
        "effective_velocity",
    )

    def __init__(
        self,
        grid,
        *,
        bed,
        thickness,
        sliding_speed=None,
        melt_rate=0.0,
        point_sources=(),
        h_init=None,
        h_g_init=None,
        S_x_init=None,
        S_y_init=None,
        boundary_conditions=None,
        channels=True,
        cfl=0.5,
        **kwargs,
    ):
        self.grid = grid
        shape = grid.shape

        self.params = dict(default_parameters)
        for name in list(kwargs):
            if name in self.params:
                self.params[name] = float(kwargs.pop(name))
        self.physics = {name: getattr(_physics, name) for name in self._physics_names}
        for name in list(kwargs):
            if name in self.physics:
                self.physics[name] = kwargs.pop(name)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        if self.params["e_v"] <= 0.0:
            raise ValueError("The englacial void ratio e_v must be positive (Sect. 4.1.1)")

        self.B = _as_field(bed, shape, "bed")
        self.H = _as_field(thickness, shape, "thickness")
        if np.any(self.H < 0.0):
            raise ValueError("thickness must be non-negative")
        u_b = self.params["u_b"] if sliding_speed is None else sliding_speed
        self.u_b = _as_field(u_b, shape, "sliding_speed")
        self.melt_rate = melt_rate
        self.point_sources = list(point_sources)
        self._source_cells = [grid.cell_index(s.x, s.y) for s in self.point_sources]

        h_g = 0.5 * self.params["h_r"] if h_g_init is None else h_g_init
        self.h_g = _as_field(h_g, shape, "h_g_init")
        self.h = _as_field(self.h_g if h_init is None else h_init, shape, "h_init")
        self.S_x = _as_field(0.0 if S_x_init is None else S_x_init, grid.x_face_shape, "S_x_init")
        self.S_y = _as_field(0.0 if S_y_init is None else S_y_init, grid.y_face_shape, "S_y_init")

        bcs = boundary_conditions or {}
        unknown = set(bcs) - set(_SIDES)
        if unknown:
            raise ValueError(f"Unknown boundary sides {sorted(unknown)}")
        self.boundary_conditions = {s: _parse_boundary_condition(bcs.get(s)) for s in _SIDES}

        self.channels = bool(channels)
        self.cfl = float(cfl)
        self._build_channel_masks()

        self.p_i = _physics.overburden_pressure(self.H)
        self.t = 0.0
        self.n_steps = 0
        self.boundary_inflow_volume = 0.0
        self.source_volume = 0.0
        self.clipped_volume = 0.0
        self.diagnose()

    # ------------------------------------------------------------------
    # set-up helpers
    # ------------------------------------------------------------------
    @property
    def e_v(self):
        return self.params["e_v"]

    @property
    def delta_h(self):
        r"""Englacial storage capacity :math:`\Delta h` (Eq. 3)"""
        return _physics.storage_capacity(self.H, self.e_v)

    def _kind(self, side):
        return self.boundary_conditions[side][0]

    def _build_channel_masks(self):
        nx, ny = self.grid.shape
        self.channel_mask_x = np.zeros((nx + 1, ny), dtype=bool)
        self.channel_mask_y = np.zeros((nx, ny + 1), dtype=bool)
        if not self.channels:
            return
        self.channel_mask_x[1:-1, :] = nx > 1
        self.channel_mask_y[:, 1:-1] = ny > 1
        self.channel_mask_x[0, :] = self._kind("west") == "atmospheric"
        self.channel_mask_x[-1, :] = self._kind("east") == "atmospheric"
        self.channel_mask_y[:, 0] = self._kind("south") == "atmospheric"
        self.channel_mask_y[:, -1] = self._kind("north") == "atmospheric"
        self.S_x[~self.channel_mask_x] = 0.0
        self.S_y[~self.channel_mask_y] = 0.0

    def set_englacial_void_ratio(self, e_v, preserve_pressure=True):
        r"""Change :math:`e_v`, optionally keeping the water pressure fixed.

        Steady states do not depend on :math:`e_v` (Sect. 4.1.1), so a large
        value can be used to spin up a steady state quickly before switching
        to the value of interest.  With ``preserve_pressure`` the stored
        water is rescaled so that :math:`p_w` and :math:`h_w` are unchanged.
        """
        if e_v <= 0.0:
            raise ValueError("e_v must be positive")
        p_w, h_w = self.p_w.copy(), self.h_w.copy()
        self.params["e_v"] = float(e_v)
        if preserve_pressure:
            self.h = h_w + e_v * p_w / (water_density * gravity)
        self.diagnose()

    def melt(self, t):
        r"""Return the distributed water input :math:`m` at time ``t``"""
        m = self.melt_rate(t) if callable(self.melt_rate) else self.melt_rate
        return np.broadcast_to(np.asarray(m, dtype=float), self.grid.shape)

    def point_source_rate(self, t):
        r"""Return the point-source input as a rate per unit area (m/s)"""
        rate = np.zeros(self.grid.shape)
        for (i, j), source in zip(self._source_cells, self.point_sources):
            rate[i, j] += source.rate(t) / self.grid.cell_area
        return rate

    # ------------------------------------------------------------------
    # discretisation
    # ------------------------------------------------------------------
    def _pad(self, field, ghost=None):
        r"""Pad a cell field with one layer of ghost cells.

        Ghost cells replicate the adjacent interior value, unless ``ghost``
        supplies an override for an atmospheric side.
        """
        padded = np.pad(field, 1, mode="edge")
        if ghost is not None:
            for side, values in ghost.items():
                if side == "west":
                    padded[0, 1:-1] = values
                elif side == "east":
                    padded[-1, 1:-1] = values
                elif side == "south":
                    padded[1:-1, 0] = values
                elif side == "north":
                    padded[1:-1, -1] = values
        return padded

    def _boundary_slices(self, side):
        return {
            "west": ((0, slice(None)), (1, slice(None))),
            "east": ((-1, slice(None)), (-2, slice(None))),
            "south": ((slice(None), 0), (slice(None), 1)),
            "north": ((slice(None), -1), (slice(None), -2)),
        }[side]

    def _atmospheric_ghosts(self, h_w):
        r"""Return ghost values of :math:`\phi` and :math:`h_w` with :math:`p_w = 0`.

        The ghost cell copies the cavity height of its neighbour and holds
        water at atmospheric pressure: where the boundary cell is
        ice-covered, :math:`h_w^\mathrm{ghost} = \min(h, h_g)`; in the
        proglacial zone :math:`h_w^\mathrm{ghost} = h_w`.  The bed is
        extrapolated linearly so that water drains down the bed slope.
        """
        phi_ghost, h_w_ghost = {}, {}
        for side in _SIDES:
            if self._kind(side) != "atmospheric":
                continue
            edge, inner = self._boundary_slices(side)
            n_normal = self.grid.nx if side in ("west", "east") else self.grid.ny
            B_edge = self.B[edge]
            B_ghost = 2.0 * B_edge - self.B[inner] if n_normal > 1 else B_edge
            hw = np.where(self.H[edge] > 0.0, np.minimum(self.h[edge], self.h_g[edge]), h_w[edge])
            h_w_ghost[side] = hw
            phi_ghost[side] = water_density * gravity * (B_ghost + hw)
        return phi_ghost, h_w_ghost

    def diagnose(self):
        r"""Compute all derived fields from the current state (step 1)."""
        g = self.grid
        p = self.params
        ph = self.physics
        rho_g = water_density * gravity

        delta_h = self.delta_h
        self.p_w = ph["water_pressure"](self.h, self.h_g, self.p_i, delta_h)
        self.h_e = self.e_v * self.p_w / rho_g
        self.h_w = self.h - self.h_e
        self.N = self.p_i - self.p_w
        self.phi = _physics.hydraulic_potential(self.B, self.h_w, self.p_w)

        phi_ghost, h_w_ghost = self._atmospheric_ghosts(self.h_w)
        p_w_ghost = {side: 0.0 for side in phi_ghost}
        N_ghost = {side: self.p_i[self._boundary_slices(side)[0]] for side in phi_ghost}
        phi_p = self._pad(self.phi, phi_ghost)
        h_w_p = self._pad(self.h_w, h_w_ghost)
        p_w_p = self._pad(self.p_w, p_w_ghost)
        N_p = self._pad(self.N, N_ghost)

        # normal gradients on faces; zero across no-flux and flux boundaries
        gx = (phi_p[1:, 1:-1] - phi_p[:-1, 1:-1]) / g.dx
        gy = (phi_p[1:-1, 1:] - phi_p[1:-1, :-1]) / g.dy
        # tangential gradients interpolated from cell centres
        gx_c = 0.5 * (gx[1:, :] + gx[:-1, :])
        gy_c = 0.5 * (gy[:, 1:] + gy[:, :-1])
        gy_p = np.pad(gy_c, ((1, 1), (0, 0)), mode="edge")
        gx_p = np.pad(gx_c, ((0, 0), (1, 1)), mode="edge")
        grad2_x = gx**2 + (0.5 * (gy_p[1:, :] + gy_p[:-1, :])) ** 2
        grad2_y = gy**2 + (0.5 * (gx_p[:, 1:] + gx_p[:, :-1])) ** 2

        # upwind flowing thickness on faces
        h_wx = np.where(gx > 0.0, h_w_p[1:, 1:-1], h_w_p[:-1, 1:-1])
        h_wy = np.where(gy > 0.0, h_w_p[1:-1, 1:], h_w_p[1:-1, :-1])

        flux_kw = dict(k=p["k"], phi_reg=p["phi_reg"])
        q_x = ph["sheet_flux"](h_wx, gx, grad2_x, **flux_kw)
        q_y = ph["sheet_flux"](h_wy, gy, grad2_y, **flux_kw)
        self._apply_flux_boundary_conditions(q_x, q_y)

        # channels
        p_wx = 0.5 * (p_w_p[1:, 1:-1] + p_w_p[:-1, 1:-1])
        p_wy = 0.5 * (p_w_p[1:-1, 1:] + p_w_p[1:-1, :-1])
        dpw_x = (p_w_p[1:, 1:-1] - p_w_p[:-1, 1:-1]) / g.dx
        dpw_y = (p_w_p[1:-1, 1:] - p_w_p[1:-1, :-1]) / g.dy
        self.N_x = 0.5 * (N_p[1:, 1:-1] + N_p[:-1, 1:-1])
        self.N_y = 0.5 * (N_p[1:-1, 1:] + N_p[1:-1, :-1])

        ch_kw = dict(k_c=p["k_c"], phi_reg=p["phi_reg"])
        mx, my = self.channel_mask_x, self.channel_mask_y
        self.S_wx = np.where(mx, ph["water_filled_area"](self.S_x, p_wx), 0.0)
        self.S_wy = np.where(my, ph["water_filled_area"](self.S_y, p_wy), 0.0)
        self.Q_x = np.where(mx, ph["channel_discharge"](self.S_wx, gx, **ch_kw), 0.0)
        self.Q_y = np.where(my, ph["channel_discharge"](self.S_wy, gy, **ch_kw), 0.0)
        l_c = dict(l_c=p["l_c"])
        self.Xi_x = np.where(mx, ph["channel_dissipation"](self.Q_x, q_x, gx, **l_c), 0.0)
        self.Xi_y = np.where(my, ph["channel_dissipation"](self.Q_y, q_y, gy, **l_c), 0.0)
        self.Pi_x = np.where(
            mx, ph["channel_pressure_melting"](self.Q_x, q_x, dpw_x, self.S_x, **l_c), 0.0
        )
        self.Pi_y = np.where(
            my, ph["channel_pressure_melting"](self.Q_y, q_y, dpw_y, self.S_y, **l_c), 0.0
        )

        self.q_x, self.q_y = q_x, q_y
        self._faces = dict(
            gx=gx, gy=gy, grad2_x=grad2_x, grad2_y=grad2_y, h_wx=h_wx, h_wy=h_wy
        )

    def _apply_flux_boundary_conditions(self, q_x, q_y):
        for side in _SIDES:
            kind, value = self.boundary_conditions[side]
            if kind == "atmospheric":
                continue
            if side == "west":
                q_x[0, :] = value
            elif side == "east":
                q_x[-1, :] = -value
            elif side == "south":
                q_y[:, 0] = value
            elif side == "north":
                q_y[:, -1] = -value

    def total_fluxes(self):
        r"""Return the face fluxes :math:`q + Q\,\delta(x_c)` per unit width"""
        g = self.grid
        return self.q_x + self.Q_x / g.dy, self.q_y + self.Q_y / g.dx

    def channel_melt_source(self):
        r"""Return the channel melt water source per unit cell area (m/s)

        Each channel segment spans half of each adjacent cell, so half of
        its melt :math:`(\Xi+\Pi)/(\rho_w L)` goes to each neighbour.
        """
        g = self.grid
        M_x = _physics.channel_melt_source(self.Xi_x, self.Pi_x)
        M_y = _physics.channel_melt_source(self.Xi_y, self.Pi_y)
        return (M_x[1:, :] + M_x[:-1, :]) / (2.0 * g.dy) + (M_y[:, 1:] + M_y[:, :-1]) / (2.0 * g.dx)

    def dhdt(self, t=None):
        r"""Return :math:`\partial h/\partial t` from the mass balance (Eq. 17)"""
        g = self.grid
        t = self.t if t is None else t
        F_x, F_y = self.total_fluxes()
        divergence = (F_x[1:, :] - F_x[:-1, :]) / g.dx + (F_y[:, 1:] - F_y[:, :-1]) / g.dy
        sources = self.melt(t) + self.point_source_rate(t) + self.channel_melt_source()
        return sources - divergence

    def timestep(self):
        r"""Return the stable explicit time step (Eqs. 19-23)"""
        g, p, ph, f = self.grid, self.params, self.physics, self._faces
        delta_h = self.delta_h
        pressurised = (self.h >= self.h_g) & (self.h <= self.h_g + delta_h) & (delta_h > 0.0)
        pres_p = np.pad(pressurised, 1, mode="edge")
        pres_x = pres_p[1:, 1:-1] | pres_p[:-1, 1:-1]
        pres_y = pres_p[1:-1, 1:] | pres_p[1:-1, :-1]
        kw = dict(k=p["k"], k_c=p["k_c"], e_v=p["e_v"], phi_reg=p["phi_reg"])

        D_x = ph["effective_diffusivity"](f["h_wx"], self.S_wx, f["grad2_x"], pres_x, 1.0 / g.dy, **kw)
        D_y = ph["effective_diffusivity"](f["h_wy"], self.S_wy, f["grad2_y"], pres_y, 1.0 / g.dx, **kw)
        V_x = ph["effective_velocity"](
            f["h_wx"], self.S_wx, self.q_x, self.Q_x, f["gx"], f["grad2_x"], pres_x, 1.0 / g.dy, **kw
        )
        V_y = ph["effective_velocity"](
            f["h_wy"], self.S_wy, self.q_y, self.Q_y, f["gy"], f["grad2_y"], pres_y, 1.0 / g.dx, **kw
        )

        # faces closed by no-flux or flux conditions transmit nothing
        open_x = np.ones(g.x_face_shape, dtype=bool)
        open_y = np.ones(g.y_face_shape, dtype=bool)
        open_x[0, :] = self._kind("west") == "atmospheric"
        open_x[-1, :] = self._kind("east") == "atmospheric"
        open_y[:, 0] = self._kind("south") == "atmospheric"
        open_y[:, -1] = self._kind("north") == "atmospheric"

        D_max = max(_masked_max(D_x, open_x), _masked_max(D_y, open_y))
        V_max = max(_masked_max(V_x, open_x), _masked_max(V_y, open_y))
        # a direction with a single cell carries no flow, so it does not
        # constrain the time step
        dx = g.dx if g.nx > 1 else np.inf
        dy = g.dy if g.ny > 1 else np.inf
        return _physics.stable_timestep(D_max, V_max, dx, dy, C=self.cfl)

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def step(self, dt_max=np.inf):
        r"""Advance the model by one stable time step and return its size."""
        g, p, ph = self.grid, self.params, self.physics
        dt = min(self.timestep(), dt_max)
        if not np.isfinite(dt) or dt <= 0.0:
            raise RuntimeError(f"Invalid time step {dt}")

        F_x, F_y = self.total_fluxes()
        sources = self.melt(self.t) + self.point_source_rate(self.t) + self.channel_melt_source()
        divergence = (F_x[1:, :] - F_x[:-1, :]) / g.dx + (F_y[:, 1:] - F_y[:, :-1]) / g.dy
        h_new = self.h + dt * (sources - divergence)

        self.source_volume += dt * sources.sum() * g.cell_area
        self.boundary_inflow_volume += dt * (
            (F_x[0, :].sum() - F_x[-1, :].sum()) * g.dy + (F_y[:, 0].sum() - F_y[:, -1].sum()) * g.dx
        )
        negative = h_new < 0.0
        if np.any(negative):
            self.clipped_volume -= h_new[negative].sum() * g.cell_area
            h_new[negative] = 0.0

        cav_kw = dict(A=p["A"], h_r=p["h_r"], l_r=p["l_r"])
        self.h_g = ph["update_cavity_height"](self.h_g, self.N, self.u_b, dt, **cav_kw)
        if self.channels:
            ch_kw = dict(A_c=p["A_c"])
            S_x = ph["update_channel_area"](self.S_x, self.Xi_x, self.Pi_x, self.N_x, dt, **ch_kw)
            S_y = ph["update_channel_area"](self.S_y, self.Xi_y, self.Pi_y, self.N_y, dt, **ch_kw)
            self.S_x = np.where(self.channel_mask_x, S_x, 0.0)
            self.S_y = np.where(self.channel_mask_y, S_y, 0.0)
        self.h = h_new

        self.t += dt
        self.n_steps += 1
        self.diagnose()
        return dt

    def run(self, t_end, dt_max=np.inf, callback=None, output_times=()):
        r"""Integrate until time ``t_end`` (s).

        Parameters
        ----------
        t_end : float
            Final time
        dt_max : float
            Upper bound on the time step
        callback : callable, optional
            Called as ``callback(model)`` at each of ``output_times``
        output_times : sequence of float
            Times at which to stop exactly and call ``callback``
        """
        stops = sorted(t for t in output_times if self.t <= t <= t_end)
        if callback is not None and stops and np.isclose(stops[0], self.t):
            callback(self)
            stops = stops[1:]
        for stop in stops + [t_end]:
            while self.t < stop - 1e-9 * max(1.0, abs(stop)):
                self.step(min(dt_max, stop - self.t))
            if callback is not None and stop in stops:
                callback(self)
        return self

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def total_water_volume(self):
        r"""Return :math:`\int h\,\mathrm{d}A` (m^3)"""
        return self.h.sum() * self.grid.cell_area

    def flotation_fraction(self):
        r"""Return :math:`p_w/p_i`, defined as 0 in the proglacial zone"""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.p_i > 0.0, self.p_w / self.p_i, 0.0)

    def sheet_regime(self):
        r"""Return the sheet regime per cell, see :func:`physics.sheet_regime`"""
        return _physics.sheet_regime(self.h, self.h_g, self.delta_h)

    def channel_regimes(self):
        r"""Return channel regimes on x and y faces"""
        return (
            _physics.channel_regime(self.S_x, self.S_wx),
            _physics.channel_regime(self.S_y, self.S_wy),
        )

    def blister_mask(self):
        r"""Return the ice-covered cells undergoing ice uplift"""
        return (self.sheet_regime() == _physics.UPLIFT) & (self.H > 0.0)

    def blister_volume(self):
        r"""Return the subglacial blister volume :math:`\int (h_w - h_g)\,\mathrm{d}A`

        The integral is over the blister area (Sect. 4.1.1).
        """
        mask = self.blister_mask()
        return ((self.h_w - self.h_g)[mask]).sum() * self.grid.cell_area

    def mass_balance_residual(self, initial_volume):
        r"""Return the volume error relative to the inputs since ``initial_volume``"""
        expected = initial_volume + self.source_volume + self.boundary_inflow_volume + self.clipped_volume
        return self.total_water_volume() - expected

    def width_integrated_discharge(self):
        r"""Return sheet and channel discharge through each x face, summed over y"""
        g = self.grid
        return self.q_x.sum(axis=1) * g.dy, self.Q_x.sum(axis=1)

    def state(self):
        r"""Return a dictionary snapshot of the prognostic and derived fields"""
        names = ("h", "h_g", "h_w", "h_e", "p_w", "phi", "N", "S_x", "S_y", "S_wx", "S_wy", "Q_x", "Q_y")
        snapshot = {name: getattr(self, name).copy() for name in names}
        snapshot["t"] = self.t
        snapshot["delta_h"] = self.delta_h
        snapshot["flotation_fraction"] = self.flotation_fraction()
        return snapshot


def _masked_max(values, mask):
    selected = values[mask]
    return float(selected.max()) if selected.size else 0.0
