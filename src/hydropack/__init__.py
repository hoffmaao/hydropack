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

import importlib.util

import hydropack.constants
import hydropack.glads2

__all__ = [
    "constants",
    "glads2",
]

# The finite element GlaDS model requires Firedrake; the finite difference
# GlaDS-2 model in :mod:`hydropack.glads2` only requires NumPy.
if importlib.util.find_spec("firedrake") is not None:
    import hydropack.physics
    import hydropack.models
    import hydropack.solvers
    import hydropack.plot

    __all__ += ["physics", "models", "solvers", "plot"]
