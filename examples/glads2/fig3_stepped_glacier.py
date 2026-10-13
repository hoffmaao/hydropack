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

r"""Reproduce Fig. 3 of Wells et al. (2026): stepped-glacier steady states

Runs the three steady states of Sect. 3: m = 20 mm/d sheet only,
m = 20 mm/d with channels, and m = 5 mm/d with channels.  Each one is
spun up with a large englacial void ratio, which does not change the steady
state but allows long time steps, and then relaxed with e_v = 1e-5 as in
the paper.

    python examples/glads2/fig3_stepped_glacier.py [--spin-up-days 400] [--out outputs/glads2]
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydropack.glads2 import experiments, physics
from hydropack.glads2.constants import seconds_per_day

CASES = [
    ("m20_sheet", 20.0, False, "m = 20 mm d$^{-1}$, sheet only"),
    ("m20_channels", 20.0, True, "m = 20 mm d$^{-1}$"),
    ("m5_channels", 5.0, True, "m = 5 mm d$^{-1}$"),
]


def run_case(melt_mm, channels, spin_up_days, relax_days):
    model = experiments.stepped_glacier(melt_rate=melt_mm * 1e-3 / seconds_per_day, channels=channels)
    if channels:
        # small seed so that channels can grow where melt opening wins
        model.S_x[model.channel_mask_x] = 1e-3
    model.h = model.h_g + 0.8 * model.delta_h
    model.diagnose()
    experiments.spin_up(
        model,
        spin_up_days * seconds_per_day,
        e_v_spin_up=1e-2,
        final_duration=relax_days * seconds_per_day,
    )
    q_sheet, Q_channel = model.width_integrated_discharge()
    return dict(
        x=model.grid.x,
        x_faces=model.grid.x_faces,
        phi=model.phi[:, 0],
        phi_m=(1000.0 * 9.81 * model.B)[:, 0],
        phi_0=(1000.0 * 9.81 * model.B + model.p_i)[:, 0],
        h=model.h[:, 0],
        h_w=model.h_w[:, 0],
        h_g=model.h_g[:, 0],
        S=model.S_x[:, 0],
        S_w=model.S_wx[:, 0],
        q_sheet=q_sheet,
        Q_channel=Q_channel,
        regime=model.sheet_regime()[:, 0],
        max_dhdt=np.abs(model.dhdt()).max(),
    )


def plot(results, path):
    ref = experiments.load_stepped_glacier()
    fig, axes = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
    for col, (name, _, channels, title) in enumerate(CASES):
        r = results[name]
        x = r["x"] / 1e3
        xf = r["x_faces"] / 1e3
        ax = axes[0, col]
        ax.fill_between(x, 0, r["phi_0"] / 1e6, color="lightblue", label=r"$\phi_0$")
        ax.fill_between(x, 0, r["phi_m"] / 1e6, color="gray", label=r"$\phi_m$")
        ax.plot(x, r["phi"] / 1e6, ":", color="C0", lw=2, label=r"$\phi$ (hydropack)")
        ax.set_title(title)
        ax = axes[1, col]
        # Fig. 3b plots the flowing sheet height, which exceeds h_g under uplift
        ax.plot(x, 100 * r["h_w"], ":", color="C0", lw=2, label="$h_w$")
        ax.plot(x, 100 * r["h_g"], ":", color="k", lw=2, label="$h_g$")
        if col == 0:
            axes[0, 0].plot(ref["x"] / 1e3, ref["phi_ref"] / 1e6, "r-", lw=1, label="Schoof/Hewitt 2012 (digitised)")
            ax.plot(ref["x"] / 1e3, 100 * ref["h_ref"], "r-", lw=1, label="Schoof/Hewitt 2012 (digitised)")
        ax = axes[2, col]
        if channels:
            ax.plot(xf, r["S"], ":", color="k", lw=2, label="$S$")
            ax.plot(xf, r["S_w"], ":", color="C0", lw=2, label="$S_w$")
        ax = axes[3, col]
        ax.plot(xf, r["q_sheet"], ":", color="k", lw=2, label="sheet")
        if channels:
            ax.plot(xf, r["Q_channel"], ":", color="C0", lw=2, label="channel")
    axes[0, 0].set_ylabel(r"$\phi$ (MPa)")
    axes[1, 0].set_ylabel(r"$h_w$, $h_g$ (cm)")
    axes[2, 1].set_ylabel(r"$S$, $S_w$ (m$^2$)")
    axes[3, 0].set_ylabel(r"Q (m$^3$ s$^{-1}$)")
    for ax in axes[-1]:
        ax.set_xlabel("x (km)")
    axes[2, 0].axis("off")
    for ax in (axes[0, 0], axes[1, 0], axes[2, 1], axes[3, 1]):
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spin-up-days", type=float, default=400.0)
    parser.add_argument("--relax-days", type=float, default=1.0)
    parser.add_argument("--out", default="outputs/glads2")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for name, melt_mm, channels, _ in CASES:
        cache = os.path.join(args.out, f"fig3_{name}.npz")
        if os.path.exists(cache):
            results[name] = dict(np.load(cache))
            continue
        results[name] = run_case(melt_mm, channels, args.spin_up_days, args.relax_days)
        np.savez(cache, **results[name])
        print(f"{name}: max|dh/dt| = {results[name]['max_dhdt'] * seconds_per_day:.2e} m/d", flush=True)
    plot(results, os.path.join(args.out, "fig3_stepped_glacier.png"))


if __name__ == "__main__":
    main()
