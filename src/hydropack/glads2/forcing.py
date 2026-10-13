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

r"""Water-input forcings used in Wells et al. (2026)"""

import numpy as np


def parabolic_pulse(volume, duration, t_start=0.0):
    r"""Return a function giving a parabolic-in-time input rate (m^3/s)

    The rate is zero outside :math:`[t_0, t_0 + T]` and peaks at the
    half-time, so that its integral is the total ``volume``:

    .. math::
        r(t) = \frac{6V}{T^3}\,(t - t_0)(t_0 + T - t)

    This is how the supraglacial lake volume is injected in Sect. 4.
    """
    T = float(duration)
    V = float(volume)

    def rate(t):
        s = t - t_start
        if s <= 0.0 or s >= T:
            return 0.0
        return 6.0 * V / T**3 * s * (T - s)

    return rate


def parabolic_pulse_volume(volume, duration, t, t_start=0.0):
    r"""Return the volume injected by :func:`parabolic_pulse` up to time ``t``"""
    T = float(duration)
    s = np.clip(t - t_start, 0.0, T)
    return volume * (3.0 * s**2 / T**2 - 2.0 * s**3 / T**3)


def temperature_index_melt(surface_elevation, c1=2e-8, T0=11.0, c2=-0.006, E0=50.0):
    r"""Return the temperature-index melt input (Eqs. 24-25)

    .. math::
        m = c_1 \max(0, T), \qquad T = T_0 + c_2 (E - E_0)

    Defaults are the early-summer values used for North Lake (Sect. 4.2):
    :math:`c_1 = 2\times10^{-8}` m s^-1 K^-1, :math:`T_0 = 11` C,
    :math:`c_2 = -0.006` K m^-1 and :math:`E_0 = 50` m.
    """
    temperature = T0 + c2 * (np.asarray(surface_elevation, dtype=float) - E0)
    return c1 * np.maximum(0.0, temperature)
