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

r"""Reproduce Figs. 4-6 of Wells et al. (2026): synthetic lake drainage

* Fig. 4: evolution of the subglacial system after a 44e6 m^3 lake drains
  at (10, 7.5) km over 1.4 h, with m = 1 mm/d and e_v = 1e-4
* Fig. 5: blister outlines and volume for e_v from 1e-6 to 1e-2
* Fig. 6: blister outlines and volume for m = 1 and 20 mm/d, with a
  20e6 m^3 lake

The initial steady states are spun up with e_v = 1e-3, which does not change
them, and cached in the output directory.

    python examples/glads2/fig4_6_synthetic_lake_drainage.py --figure 4
    python examples/glads2/fig4_6_synthetic_lake_drainage.py --figure 5 --ev 1e-5 1e-4 1e-3 1e-2
    python examples/glads2/fig4_6_synthetic_lake_drainage.py --figure 6
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydropack.glads2 import experiments
from hydropack.glads2.constants import seconds_per_day, seconds_per_hour

SNAPSHOT_HOURS = [0, 2, 10, 40, 90, 300]
OUTLINE_HOURS = [2, 10, 20, 40, 60, 80]


def steady_state(out, melt_mm, spin_up_days):
    r"""Return (and cache) the pre-drainage steady state for a given melt rate"""
    cache = os.path.join(out, f"steady_m{melt_mm:g}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    model = experiments.synthetic_lake_drainage(
        lake_volume=0.0, melt_rate=melt_mm * 1e-3 / seconds_per_day, e_v=1e-3, h_g_init=0.02
    )
    model.S_x[model.channel_mask_x] = 1e-4
    model.S_y[model.channel_mask_y] = 1e-4
    model.h = model.h_g + 0.7 * model.delta_h
    model.diagnose()
    model.run(spin_up_days * seconds_per_day, dt_max=seconds_per_hour)
    state = dict(h_w=model.h_w, p_w=model.p_w, h_g=model.h_g, S_x=model.S_x, S_y=model.S_y)
    print(f"steady state m = {melt_mm} mm/d: max|dh/dt| = {np.abs(model.dhdt()).max() * seconds_per_day:.2e} m/d")
    np.savez(cache, **state)
    return state


def lake_drainage(steady, melt_mm, e_v, lake_volume, hours, snapshot_hours=(), series_dt=0.5):
    r"""Run the drainage from a steady state; return snapshots and a blister-volume series"""
    model = experiments.synthetic_lake_drainage(
        lake_volume=lake_volume,
        melt_rate=melt_mm * 1e-3 / seconds_per_day,
        e_v=e_v,
        h_g_init=steady["h_g"],
        S_x_init=steady["S_x"],
        S_y_init=steady["S_y"],
    )
    # restore the steady pressure for this value of e_v
    model.h = steady["h_w"] + e_v * steady["p_w"] / (1000.0 * 9.81)
    model.diagnose()
    snapshots, series = {}, []
    stops = sorted(set(np.arange(0.0, hours + 1e-9, series_dt)) | set(snapshot_hours))
    for hour in stops:
        # short steps while the lake drains, then the stability limit governs
        while model.t < hour * seconds_per_hour - 1e-6:
            dt_max = 60.0 if model.t < 2 * seconds_per_hour else seconds_per_hour
            model.step(min(dt_max, hour * seconds_per_hour - model.t))
        series.append((hour, model.blister_volume()))
        if hour in snapshot_hours:
            snap = model.state()
            snap["blister"] = model.blister_mask()
            snapshots[hour] = snap
    return model, snapshots, np.array(series)


def outline(ax, field, grid, level, color, **kw):
    if np.nanmin(field) < level < np.nanmax(field) or (field == level).any():
        ax.contour(grid.x / 1e3, grid.y / 1e3, field.T, levels=[level], colors=color, linewidths=1, **kw)


def figure4(out, steady, e_v=1e-4):
    model, snaps, series = lake_drainage(steady, 1.0, e_v, 44e6, 300.0, SNAPSHOT_HOURS)
    grid = model.grid
    j = grid.cell_index(0.0, 7.5e3)[1]
    x = grid.x / 1e3
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1.2, 1.2, 1.2])
    for n, hour in enumerate(SNAPSHOT_HOURS):
        ax = fig.add_subplot(gs[n // 3, n % 3])
        ff = snaps[hour]["flotation_fraction"]
        im = ax.pcolormesh(grid.x_faces / 1e3, grid.y_faces / 1e3, ff.T, vmin=0, vmax=1, cmap="viridis")
        outline(ax, (ff >= 1 - 1e-9).astype(float) * (model.H > 0), grid, 0.5, "k")
        outline(ax, ((ff <= 1e-9) & (model.H > 0)).astype(float), grid, 0.5, "w")
        ax.plot(10, 7.5, "m^")
        ax.axhline(grid.y[j] / 1e3, color="gray", ls="--", lw=0.8)
        ax.set_title(f"t = {hour} h")
        ax.set_aspect("equal")
    fig.colorbar(im, ax=fig.axes, shrink=0.3, label=r"$p_w / p_i$")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SNAPSHOT_HOURS)))
    ax_phi, ax_h, ax_S = (fig.add_subplot(gs[k, :]) for k in (2, 3, 4))
    phi_m = 1000.0 * 9.81 * model.B[:, j]
    ax_phi.fill_between(x, 0, (phi_m + model.p_i[:, j]) / 1e6, color="lightblue", label=r"$\phi_0$")
    ax_phi.fill_between(x, 0, phi_m / 1e6, color="brown", label=r"$\phi_m$")
    for c, hour in zip(colors, SNAPSHOT_HOURS):
        s = snaps[hour]
        ax_phi.plot(x, s["phi"][:, j] / 1e6, color=c, label=f"t = {hour} h")
        ax_h.plot(x, s["h"][:, j], color=c)
        ax_h.plot(x, s["h_g"][:, j], ":", color=c)
        ax_S.plot(grid.x_faces / 1e3, s["S_wx"][:, j], color=c)
        ax_S.plot(grid.x_faces / 1e3, s["S_x"][:, j], ":", color=c)
    ax_h.fill_between(x, 0, model.delta_h[:, j], color="gray", alpha=0.4, label=r"$\Delta h$")
    ax_phi.set_ylabel(r"$\phi$ (MPa)")
    ax_h.set_ylabel(r"$h$, $h_g$, $\Delta h$ (m)")
    ax_S.set_ylabel(r"$S_w$ (solid), $S$ (dotted) (m$^2$)")
    ax_S.set_xlabel("x (km)")
    ax_phi.legend(fontsize=7, ncol=2)
    inset = ax_h.inset_axes([0.6, 0.45, 0.35, 0.5])
    sel = x > 41
    for c, hour in zip(colors, SNAPSHOT_HOURS):
        inset.plot(x[sel], snaps[hour]["h"][sel, j], color=c)
        inset.plot(x[sel], snaps[hour]["h_g"][sel, j], ":", color=c)
    inset.set_ylim(0, 0.2)
    fig.savefig(os.path.join(out, "fig4_synthetic_lake_drainage.png"), dpi=130)
    np.savez(os.path.join(out, "fig4_series.npz"), series=series)
    return snaps, series


def outline_figure(out, runs, name, hours=OUTLINE_HOURS, channel_threshold=None):
    fig, axes = plt.subplots(len(runs) + 1, 1, figsize=(8, 2.6 * (len(runs) + 1)))
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(hours)))
    for ax, (label, (model, snaps, series)) in zip(axes, runs.items()):
        grid = model.grid
        for c, hour in zip(colors, hours):
            outline(ax, snaps[hour]["blister"].astype(float), grid, 0.5, [c])
        if channel_threshold is not None:
            Q = snaps[0]["Q_x"] if 0 in snaps else snaps[hours[0]]["Q_x"]
            for i, j in zip(*np.nonzero(np.abs(Q) > channel_threshold)):
                ax.plot(grid.x_faces[i] / 1e3 + np.array([-0.25, 0.25]), [grid.y[j] / 1e3] * 2, "k-", lw=1)
        ax.set_xlim(0, grid.length / 1e3)
        ax.set_ylim(0, grid.width / 1e3)
        ax.set_aspect("equal")
        ax.set_title(label)
        axes[-1].plot(series[:, 0], series[:, 1] / 1e6, label=label)
    for c, hour in zip(colors, hours):
        axes[0].plot([], [], color=c, label=f"{hour} h")
    axes[0].legend(fontsize=7, ncol=len(hours))
    axes[-1].set_xlabel("t (h)")
    axes[-1].set_ylabel(r"blister volume (10$^6$ m$^3$)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", type=int, choices=[4, 5, 6], default=4)
    parser.add_argument("--ev", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--spin-up-days", type=float, default=365.0)
    parser.add_argument("--out", default="outputs/glads2")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.figure == 4:
        figure4(args.out, steady_state(args.out, 1.0, args.spin_up_days))
    elif args.figure == 5:
        steady = steady_state(args.out, 1.0, args.spin_up_days)
        hours = args.hours or 100.0
        runs = {}
        for e_v in args.ev:
            runs[f"$e_v$ = {e_v:g}"] = lake_drainage(steady, 1.0, e_v, 44e6, hours, OUTLINE_HOURS)
            print(f"e_v = {e_v:g} done", flush=True)
        outline_figure(args.out, runs, "fig5_englacial_void_ratio.png")
    else:
        hours = args.hours or 80.0
        runs = {}
        for melt_mm in (20.0, 1.0):
            steady = steady_state(args.out, melt_mm, args.spin_up_days)
            runs[f"m = {melt_mm:g} mm/d"] = lake_drainage(steady, melt_mm, 1e-4, 20e6, hours, [0] + OUTLINE_HOURS)
        outline_figure(args.out, runs, "fig6_channelization.png", channel_threshold=1.0)


if __name__ == "__main__":
    main()
