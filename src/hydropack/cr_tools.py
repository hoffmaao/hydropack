import numpy as np
import firedrake as fd


class CRTools:
    """
    Parallel-safe Crouzeix–Raviart (CR) facet utilities:
      - facet (edge) lengths on a CR space
      - CG -> CR facet midpoint transfer
      - |∂u/∂s| on each facet (magnitude of tangential derivative)
    All operations use facet integrals (assembly) and avoid global numpy→local
    vector writes or par_loops, so they are MPI-safe out of the box.
    """

    def __init__(self, mesh, U, CR):
        # Accept either FunctionSpace or Function for U/CR
        self.mesh = mesh
        self.U = U.function_space() if isinstance(U, fd.Function) else U
        self.CR = CR.function_space() if isinstance(CR, fd.Function) else CR

        # Geometry bits on facets
        n = fd.FacetNormal(mesh)          # unit normal on facets
        self.t = fd.as_vector([n[1], -n[0]])  # rotate n to get a unit tangent (2-D)

        # Convenience test function on CR
        self._w = fd.TestFunction(self.CR)

        # Scratch fields we reuse
        self._tmpU = fd.Function(self.U, name="tmpU")
        self._tmpCR = fd.Function(self.CR, name="tmpCR")

        # -------------------------------
        # Precompute facet "mass" and edge lengths on CR DOFs
        # -------------------------------
        # Facet mass for each CR basis function (integral of test over the facet).
        # We assemble separately on interior and boundary facets.
        mass_int = fd.assemble(self._w('+') * fd.dS)   # interior facets
        mass_bnd = fd.assemble(self._w * fd.ds)        # boundary facets
        self._facet_mass = fd.Function(self.CR, name="facet_mass")
        self._facet_mass.assign(0.0)
        self._facet_mass.dat.data[:] = mass_int.dat.data_ro + mass_bnd.dat.data_ro

        # Edge lengths (FacetArea) per CR DOF via facet integrals.
        # (FacetArea is valid on facet integrals; the earlier crash was only when
        # trying to point-evaluate it with Interpolator.)
        len_int = fd.assemble(fd.FacetArea(mesh)('+') * self._w('+') * fd.dS)
        len_bnd = fd.assemble(fd.FacetArea(mesh) * self._w * fd.ds)
        self.e_lens = fd.Function(self.CR, name="edge_length")
        self.e_lens.assign(0.0)
        self.e_lens.dat.data[:] = len_int.dat.data_ro + len_bnd.dat.data_ro

    # ---------------- Public API ----------------

    def edge_lengths(self):
        """Return CR Function holding edge (facet) lengths."""
        return self.e_lens

    def midpoint(self, cg_func, cr_out=None):
        """
        Evaluate a CG function at CR facet DOFs (midpoints) and write into CR.
        Uses Interpolator(cg -> CR), which is valid for Functions (point eval).
        """
        if cr_out is None:
            cr_out = self._tmpCR
        fd.Interpolator(cg_func, cr_out).interpolate()
        return cr_out

    def midpoint_array(self, cg_func):
        """Local (this-rank) NumPy view of facet midpoint values."""
        return self.midpoint(cg_func).dat.data_ro

    def ds(self, cg_func, cr_out=None):
        """
        Compute |∂u/∂s| on each facet:
            numerator = ∫_facet (grad u · t) * w_CR ds   (assembled on dS and ds)
            denominator = ∫_facet w_CR ds  (facet mass, precomputed)
            result = |numerator| / denominator
        This yields the average magnitude of the tangential derivative over
        each facet, aligned with the CR basis' dual.
        """
        if cr_out is None:
            cr_out = self._tmpCR

        # Ensure tmpU contains cg_func
        self._tmpU.assign(cg_func)

        # Assemble numerator on interior and boundary facets
        num_int = fd.assemble(fd.dot(fd.grad(self._tmpU), self.t)('+') * self._w('+') * fd.dS)
        num_bnd = fd.assemble(fd.dot(fd.grad(self._tmpU), self.t) * self._w * fd.ds)

        # Combine and normalize by facet mass
        numerator = num_int.dat.data_ro + num_bnd.dat.data_ro
        denom = self._facet_mass.dat.data_ro

        # Protect against divide-by-zero (degenerate facets not expected, but be safe)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.abs(numerator) / np.where(denom > 0.0, denom, 1.0)

        cr_out.dat.data[:] = vals
        return cr_out

    def ds_array(self, cg_func):
        """Local (this-rank) NumPy view of |∂u/∂s| on facets."""
        return self.ds(cg_func).dat.data_ro

    # glads.py expects this exact name/signature:
    def ds_assemble(self, cg_func, cr_out):
        """
        Compatibility wrapper for GLADS: writes |∂u/∂s| on facets into cr_out.
        """
        self.ds(cg_func, cr_out)
