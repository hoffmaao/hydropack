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

r"""GlaDS-2: subglacial drainage with ice uplift and free-surface flow

A NumPy implementation of the bounded-pressure formulation of Wells et al.
(2026), "GlaDS-2: modeling subglacial drainage including ice uplift and
free-surface flow in two dimensions", EGUsphere,
https://doi.org/10.5194/egusphere-2026-4310.  It does not require Firedrake.
"""

from hydropack.glads2 import constants, physics, forcing, experiments, north_lake
from hydropack.glads2.model import BoundedHydrologyModel, StaggeredGrid, PointSource

__all__ = [
    "constants",
    "physics",
    "forcing",
    "experiments",
    "north_lake",
    "BoundedHydrologyModel",
    "StaggeredGrid",
    "PointSource",
]
