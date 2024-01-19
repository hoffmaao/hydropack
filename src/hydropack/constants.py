# Copyright (C) 2019 by Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of Andrew's 1D hydrology model
#
# The full text of the license can be found in the file LICENSE in the
# source directory or at <http://www.gnu.org/licenses/>.

r"""Physical constants
This module constains physical constants used throughout the library such
as the acceleration due to gravity, the density, of ice and water, etc.
"""



year = 60.0 * 60.0 * 24.0 * 365.0
# Density of water 
water_density = 1000.0 # kg / m^3
# Density of ice 
ice_density = 910.0 # kg / m^3
# Gravitational acceleration 
gravity = 9.81 # m / s^2
# Flow rate factor of ice
f = 2.25e-25 # 1 / Pa^3 * s
# Glen exponent
glen_flow_law = 3
# Gravitational constant
gravity = 9.8
# Specific heat capacity of ice 
specific_heat = 4.22e3 # J / (kg * K)
# Pressure melting coefficient 
pressure_melting = 7.5e-8 # J / (kg * K)
# Latent heat 
latent_heat = 3.34e5 # J / kg
# Exponents
alpha = 5.0 / 4.0
beta = 3.0 / 2.0
delta = beta - 2.0