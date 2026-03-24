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

r"""Physical constants and default parameter values for subglacial hydrology

This module contains physical constants and default model parameters used
throughout the library.  We use SI units (Pa, m, s, kg, K) throughout.
Default values follow the SHMIP specification (de Fleurian et al., 2018).
"""

#: seconds per day
seconds_per_day = 86400.0

#: seconds per year (365-day year)
seconds_per_year = 365.0 * 86400.0

#: density of ice (kg / m^3)
ice_density = 910.0

#: density of fresh water (kg / m^3)
water_density = 1000.0

#: acceleration due to gravity (m / s^2)
gravity = 9.8

#: latent heat of fusion (J / kg)
latent_heat = 3.34e5

#: specific heat capacity of water (J / (kg K))
heat_capacity_water = 4.22e3

#: Clausius-Clapeyron slope (K / Pa)
clapeyron_slope = 7.5e-8

#: Glen flow-law exponent
glen_exponent = 3.0

#: sheet flux-law exponent alpha (Werder et al., 2013)
sheet_flux_alpha = 5.0 / 4.0

#: sheet flux-law exponent beta
sheet_flux_beta = 3.0 / 2.0

#: derived exponent delta = beta - 2
sheet_flux_delta = sheet_flux_beta - 2.0

#: Nye (1953) closure prefactor 2 / n^n for n = 3
nye_closure_factor = 2.0 / glen_exponent ** glen_exponent

#: sheet conductivity (m^(7/4) / kg^(1/2))
default_sheet_conductivity = 0.005

#: channel conductivity (m^(3/2) / kg^(1/2))
default_channel_conductivity = 0.1

#: Glen's flow-rate factor A (Pa^-n s^-1)
default_flow_rate_factor = 3.375e-24

#: average bump height (m)
default_bump_height = 0.1

#: typical spacing between bumps (m)
default_bump_spacing = 2.0

#: sheet width beneath a channel (m)
default_channel_sheet_width = 2.0

#: englacial void ratio (dimensionless)
default_englacial_void_ratio = 0.0
