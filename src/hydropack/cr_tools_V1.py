import numpy as np
import firedrake as fd


class CRTools:
    def __init__(self, mesh, U, CR):
        self.mesh = mesh
        self.U = U.function_space() if isinstance(U, fd.Function) else U
        self.CR = CR.function_space() if isinstance(CR, fd.Function) else CR

        # Coordinate fields
        self.x = fd.SpatialCoordinate(mesh)

        # Facet normals and tangent vectors
        self.n = fd.FacetNormal(mesh)
        self.t = fd.as_vector([self.n[1], -self.n[0]])

        # Precompute forms for directional derivatives
        self._tmp_cg = fd.Function(self.U)
        self._cr_test = fd.TestFunction(self.CR)
        self._ds_form = fd.dot(fd.grad(self._tmp_cg), self.t)('+') * self._cr_test('+') * fd.dS

        # Edge-to-vertex mappings
        self._compute_edge_vertex_maps()
        self._compute_edge_lengths()

    def _compute_edge_vertex_maps(self):
        """Compute mappings from each CR edge to two closest CG dofs."""
        # Use vector-valued CG1 space for coordinates
        V_cg_vec = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        cg_vec_coords = fd.interpolate(fd.SpatialCoordinate(self.mesh), V_cg_vec).dat.data_ro.copy()

        V_cr_vec = fd.VectorFunctionSpace(self.mesh, "CR", 1)
        cr_vec_coords = fd.interpolate(fd.SpatialCoordinate(self.mesh), V_cr_vec).dat.data_ro.copy()

        num_edges = self.CR.dim()
        self.le_gv0 = np.zeros(num_edges, dtype=int)
        self.le_gv1 = np.zeros(num_edges, dtype=int)

        for i, edge_coord in enumerate(cr_vec_coords):
            dists = np.linalg.norm(cg_vec_coords - edge_coord, axis=1)
            closest_two = np.argsort(dists)[:2]
            self.le_gv0[i], self.le_gv1[i] = closest_two



    def _compute_edge_lengths(self):
        """Compute lengths of edges associated with CR dofs."""
        V_cg_vec = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        cg_vec_coords = fd.interpolate(fd.SpatialCoordinate(self.mesh), V_cg_vec).dat.data_ro.copy()

        edge_start = cg_vec_coords[self.le_gv0]
        edge_end = cg_vec_coords[self.le_gv1]

        edge_lens = np.linalg.norm(edge_end - edge_start, axis=1)
        self.e_lens = fd.Function(self.CR)
        self.e_lens.dat.data[:] = edge_lens

    def midpoint_array(self, cg):
        """Return midpoint values of CG function array for each CR edge"""
        cg_array = cg.vector().array()
        mid_vals = 0.5 * (cg_array[self.le_gv0] + cg_array[self.le_gv1])
        return mid_vals

    def midpoint(self, cg, cr):
        """Assign midpoint values from CG to CR"""
        mid_vals = self.midpoint_array(cg)
        cr.vector().set_local(mid_vals)
        cr.vector().apply("insert")

    def ds_array(self, cg):
        """Compute directional derivative of CG along edges"""
        cg_array = cg.vector().array()
        delta_vals = cg_array[self.le_gv1] - cg_array[self.le_gv0]
        return np.abs(delta_vals) / self.e_lens.dat.data_ro

    def ds(self, cg, cr):
        """Assign directional derivative values from CG to CR"""
        ds_vals = self.ds_array(cg)
        cr.vector().set_local(ds_vals)
        cr.vector().apply("insert")

    def ds_assemble(self, cg, cr):
        """Use Firedrake assembly for directional derivatives"""
        self._tmp_cg.assign(cg)
        assembled = fd.assemble(self._ds_form)
        ds_vals = np.abs(assembled.dat.data_ro) / self.e_lens.dat.data_ro

        cr.vector().set_local(ds_vals)
        cr.vector().apply("insert")
