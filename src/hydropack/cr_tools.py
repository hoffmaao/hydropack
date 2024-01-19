import firedrake
import numpy as np





# Computes directional of CG functions along edges as well as midpoints of CG
# functions along edges in parallel
class CRTools(object):
  
  def __init__(self, mesh, U, CR) :
    self.mesh = mesh
    # DG function Space
    self.U = U
    # CR function space
    self.CR = CR
    #self.facet_length = firedrake.FacetFunction('double',mesh)

    # We'll set up a form that allows us to take the derivative of CG functions
    # over edges 
    self.U = firedrake.Function(U)
    # CR test function
    V_CR = firedrake.TestFunction(CR)
    # Facet and tangent normals
    self.n = firedrake.FacetNormal(mesh)
    self.t = firedrake.as_vector([self.n[1], -self.n[0]])
    # Directional derivative form
    self.F = (firedrake.dot(firedrake.grad(self.U), self.t) * V_CR)('+') * dS
    
    # Facet function for plotting 
    #self.ff_plot = firedrake.FacetFunctionDouble(mesh)
    
  # Copies a CR function to a facet function
  def copy_cr_to_facet(self, cr, ff) :
    # Gather all edge values from each of the local arrays on each process
    cr_vals = firedrake.Vector()
    cr.vector().gather(cr_vals, np.array(range(self.CR.dim()), dtype = 'intc'))
    # Get the edge values corresponding to each local facet
    local_vals = cr_vals[self.lf_ge]    
    ff.array()[:] = local_vals
  
  
  # Computes the directional derivatives of a CG function along each edge and
  def ds(self, cg, cr):
    cr.vector().set_local(self.ds_array(cg))
    cr.vector().apply("insert")

  # Computes the directional derivatives of a CG function along each edge and
  # returns an array
  def ds_array(self, cg):
    # Gather all edge values from each of the local arrays on each process
    cg_vals = firedrake.Vector()
    cg.vector().gather(cg_vals, np.array(range(self.U.dim()), dtype = 'intc'))  
    
    # Get the two vertex values on each local edge
    local_vals0 = cg_vals.array()[self.le_gv0] 
    local_vals1 = cg_vals.array()[self.le_gv1]
    
    return abs(local_vals0 - local_vals1) / self.e_lens.vector().array()

  def ds_assemble(self, cg, cr):
    self.U.assign(cg)
    
    # Get the height difference of two vertexes on each edge
    A = firedrake.abs(firedrake.assemble(self.F).array())
    # Now divide by the edge lens
    dcg_ds = A / self.e_lens.vector().array()
    
    cr.vector().set_local(dcg_ds)
    cr.vector().apply("insert")
    
  def test(self, cg):
    cg_vals = firedrake.Vector()
    cg.vector().gather(cg_vals, np.array(range(self.U.dim()), dtype = 'intc'))  
    
    local_vals0 = cg_vals.array()[self.le_gv0] 
    local_vals1 = cg_vals.array()[self.le_gv1]
    
    edge_numbers = np.array(self.f_cr.vector().array(), dtype = 'int')
    indexes = edge_numbers.argsort()
    
    v0 = np.array(local_vals0[indexes], dtype = 'int')
    v1 = np.array(local_vals1[indexes], dtype = 'int')
    
    A = np.transpose([edge_numbers[indexes], v0, v1]) 
    
    out = "P" + str(self.MPI_rank) 
    np.savetxt(out, A, fmt = "%i")

    
    #print(self.MPI_rank, "edges", edge_numbers[indexes][:100])
    #print(self.MPI_rank, "vals0", local_vals0[indexes][:100])
    #print(self.MPI_rank, "vals1", local_vals1[indexes][:100])   
  
  # Computes the value of a CG functions at the midpoint of edges and copies
  # the result to a CR function
  def midpoint(self, cg, cr):
    cr.vector().set_local(self.midpoint_array(cg))
    cr.vector().apply("insert")
  
  # Computes the value of a CG functions at the midpoint of edges and returns
  # an array
  def midpoint_array(self, cg):
    cg_vals = firedrake.Vector()
    cg.vector().gather(cg_vals, np.array(range(self.U.dim()), dtype = 'intc'))
    
    # Get the two vertex values on each local edge
    local_vals0 = cg_vals.array()[self.le_gv0] 
    local_vals1 = cg_vals.array()[self.le_gv1]
    
    return (local_vals0 + local_vals1) / 2.0
  
  # Plots a CR function
  def plot_cr(self, cr):
    self.copy_cr_to_facet(cr, self.ff_plot)
    plot(self.ff_plot, interactive = True)
  

  def calculate_edge_to_facet_map(self, V):
    mesh = V.mesh()
    n_V = V.dim()

    # Find coordinates of dofs and put into array with index
    coords_V = np.hstack((np.reshape(V.dofmap().tabulate_all_coordinates(mesh),(n_V,2)), np.zeros((n_V,1))))
    coords_V[:,2] = range(n_V)

    # Find coordinates of facets and put into array with index
    coords_f = np.zeros((n_V,3))
    for f in dolfin.facets(mesh):
        coords_f[f.index(),0] = f.midpoint().x()
        coords_f[f.index(),1] = f.midpoint().y()
        coords_f[f.index(),2] = f.index() 

    # Sort these the same way
    coords_V = np.array(sorted(coords_V,key=tuple))
    coords_f = np.array(sorted(coords_f,key=tuple))

    # the order of the indices becomes the map
    V2fmapping = np.zeros((n_V,2))
    V2fmapping[:,0] = coords_V[:,2]
    V2fmapping[:,1] = coords_f[:,2]

    return (V2fmapping[V2fmapping[:,0].argsort()][:,1]).astype('int')


  def copy_to_facet(self, f, f_out) :
    f_out.array()[self.e2f] = f.vector()
  
  def copy_vector_to_facet(self, v, f_out) :
    f_out.array()[self.e2f] = v