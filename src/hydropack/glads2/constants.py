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

r"""Physical constants and default parameters for GlaDS-2

Values follow Table 2 of Wells et al. (2026).  These differ slightly from
the SHMIP defaults in :mod:`hydropack.constants` (e.g. :math:`g = 9.81`,
:math:`k = 0.01`, :math:`A = 6.75\times10^{-24}`), so they are kept separate.
"""

#: seconds per hour
seconds_per_hour = 3600.0

#: seconds per day
seconds_per_day = 86400.0

#: seconds per year (365-day year)
seconds_per_year = 365.0 * 86400.0

#: acceleration due to gravity (m / s^2)
gravity = 9.81

#: density of ice (kg / m^3)
ice_density = 910.0

#: density of water (kg / m^3)
water_density = 1000.0

#: latent heat of fusion (J / kg)
latent_heat = 3.34e5

#: specific heat capacity of water (J / (kg K))
heat_capacity_water = 4220.0

#: Clapeyron slope (K / Pa); negative in the sign convention of Wells et al. (2026)
clapeyron_slope = -7.5e-8

#: Glen flow-law exponent
glen_exponent = 3.0

#: flow-law area exponent alpha
flux_alpha = 5.0 / 4.0

#: flow-law gradient exponent beta
flux_beta = 3.0 / 2.0

#: default model parameters (Table 2 of Wells et al., 2026)
default_parameters = {
    "k": 0.01,  # sheet conductivity (m^(7/4) kg^(-1/2))
    "k_c": 0.1,  # channel conductivity (m^(3/2) kg^(-1/2))
    "A": 6.75e-24,  # rheological constant for cavities (Pa^-n s^-1)
    "A_c": 6.75e-24,  # rheological constant for channels (Pa^-n s^-1)
    "e_v": 1e-4,  # englacial void ratio
    "l_r": 2.0,  # cavity spacing (m)
    "h_r": 0.1,  # bump height (m)
    "l_c": 2.0,  # sheet melt width (m)
    "u_b": 3.2e-6,  # basal sliding speed (m / s), 100 m / a
    "phi_reg": 1.0,  # regularisation of |grad phi| (Pa / m)
}
