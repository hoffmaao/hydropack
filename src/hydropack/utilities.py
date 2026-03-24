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

r"""Crouzeix-Raviart facet utilities

Parallel-safe utilities for computing facet lengths, CG-to-CR midpoint
transfers, and tangential derivatives on Crouzeix-Raviart spaces.  All
operations use facet integrals and are MPI-safe.
"""

import numpy as np
import firedrake as fd


class CRTools:
    r"""Facet-level operations for Crouzeix-Raviart function spaces.

    Parameters
    ----------
    mesh : firedrake.Mesh
        The computational mesh.
    U : firedrake.FunctionSpace or firedrake.Function
        A CG1 function space (or a function on one).
    CR : firedrake.FunctionSpace or firedrake.Function
        A CR1 function space (or a function on one).
    """

    def __init__(self, mesh, U, CR):
        self.mesh = mesh
        self.U = U.function_space() if isinstance(U, fd.Function) else U
        self.CR = CR.function_space() if isinstance(CR, fd.Function) else CR

        n = fd.FacetNormal(mesh)
        self.t = fd.as_vector([n[1], -n[0]])

        self._w = fd.TestFunction(self.CR)
        self._tmpU = fd.Function(self.U, name="tmpU")
        self._tmpCR = fd.Function(self.CR, name="tmpCR")

        mass_int = fd.assemble(self._w('+') * fd.dS)
        mass_bnd = fd.assemble(self._w * fd.ds)
        self._facet_mass = fd.Function(self.CR, name="facet_mass")
        self._facet_mass.assign(0.0)
        self._facet_mass.dat.data[:] = mass_int.dat.data_ro + mass_bnd.dat.data_ro

        len_int = fd.assemble(fd.FacetArea(mesh)('+') * self._w('+') * fd.dS)
        len_bnd = fd.assemble(fd.FacetArea(mesh) * self._w * fd.ds)
        self.e_lens = fd.Function(self.CR, name="edge_length")
        self.e_lens.assign(0.0)
        self.e_lens.dat.data[:] = len_int.dat.data_ro + len_bnd.dat.data_ro

    def edge_lengths(self):
        """Return CR function holding edge (facet) lengths."""
        return self.e_lens

    def midpoint(self, cg_func, cr_out=None):
        """Evaluate a CG function at CR facet midpoints.

        Parameters
        ----------
        cg_func : firedrake.Function
            A CG1 function to evaluate.
        cr_out : firedrake.Function or None
            Output CR1 function (allocated if ``None``).

        Returns
        -------
        firedrake.Function
        """
        if cr_out is None:
            cr_out = self._tmpCR
        fd.Interpolator(cg_func, cr_out).interpolate()
        return cr_out

    def midpoint_array(self, cg_func):
        """Return NumPy view of facet midpoint values."""
        return self.midpoint(cg_func).dat.data_ro

    def ds(self, cg_func, cr_out=None):
        r"""Compute :math:`\partial u/\partial s` on each facet.

        The sign is consistent with the (+)-side tangent vector,
        matching the orientation used by the phi solver weak form.

        Parameters
        ----------
        cg_func : firedrake.Function
            A CG1 function to differentiate.
        cr_out : firedrake.Function or None
            Output CR1 function (allocated if ``None``).

        Returns
        -------
        firedrake.Function
        """
        if cr_out is None:
            cr_out = self._tmpCR

        self._tmpU.assign(cg_func)

        num_int = fd.assemble(fd.dot(fd.grad(self._tmpU), self.t)('+') * self._w('+') * fd.dS)
        num_bnd = fd.assemble(fd.dot(fd.grad(self._tmpU), self.t) * self._w * fd.ds)

        numerator = num_int.dat.data_ro + num_bnd.dat.data_ro
        denom = self._facet_mass.dat.data_ro

        with np.errstate(divide="ignore", invalid="ignore"):
            vals = numerator / np.where(denom > 0.0, denom, 1.0)

        cr_out.dat.data[:] = vals
        return cr_out

    def ds_array(self, cg_func):
        r"""Return NumPy view of signed :math:`\partial u/\partial s` on facets."""
        return self.ds(cg_func).dat.data_ro

    def ds_magnitude(self, cg_func, cr_out=None):
        r"""Compute :math:`|\partial u/\partial s|` on each facet."""
        result = self.ds(cg_func, cr_out)
        result.dat.data[:] = np.abs(result.dat.data_ro)
        return result

    def ds_assemble(self, cg_func, cr_out):
        r"""Write signed :math:`\partial u/\partial s` into *cr_out*."""
        self.ds(cg_func, cr_out)
