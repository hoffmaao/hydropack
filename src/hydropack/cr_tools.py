import firedrake
import numpy as np

# Computes directional of CG functions along edges as well as midpoints of CG
# functions along edges in parallel
class CRTools(object):
  
  def __init__(self, mesh, U, CR, maps_dir) :
    self.mesh = mesh
    # DG function Space
    self.U = U
    # CR function space
    self.CR = CR
    # Parallel maps input directory
    self.maps_dir = maps_dir
    # Process
    self.MPI_rank = MPI.rank(mpi_comm_world())
    
    # Load in some functions that help do stuff in parallel
    self.load_maps()
    # Create a map from local facets to global edges    
    self.compute_lf_ge_map()
    # Create maps from local edges to global vertex dofs
    self.compute_le_gv_maps()
    
    # We'll set up a form that allows us to take the derivative of CG functions
    # over edges 
    self.U = firedrake.Function(U)
    # CR test function
    CR = firedrake.TestFunction(CR)
    # Facet and tangent normals
    n = firedrake.FacetNormal(mesh)
    t = firedrake.as_vector([n[1], -n[0]])
    # Directional derivative form
    self.F = (firedrake.dot(firedrake.grad(self.U), t) * CR)('+') * dS
    
    # Facet function for plotting 
    self.ff_plot = firedrake.FacetFunctionDouble(mesh)
    
  # Copies a CR function to a facet function
  def copy_cr_to_facet(self, cr, ff) :
    # Gather all edge values from each of the local arrays on each process
    cr_vals = firedrake.Vector()
    cr.vector().gather(cr_vals, np.array(range(self.CR.dim()), dtype = 'intc'))
    # Get the edge values corresponding to each local facet
    local_vals = cr_vals[self.lf_ge]    
    ff.array()[:] = local_vals
  
  # Compute a map from local facets to indexes in a global array of edge values
  def compute_lf_ge_map(self):
    #Gather an array of global edge values
    x = firedrake.Vector()
    self.f_cr.vector().gather(x, np.array(range(self.CR.dim()), dtype = 'intc'))
    x = x.array()
    
    # Sort the array
    indexes = x.argsort()

    # Create the map    
    self.lf_ge = indexes[self.f_e.array()]
  
  # Create  maps from local edges to indexes in a global array of vertex values
  def compute_le_gv_maps(self):
    # Gather an array of global vertex values
    x = firedrake.Vector()
    self.f_cg.vector().gather(x, np.array(range(self.U.dim()), dtype = 'intc'))
    x = x.array()
    
    # Sort the array
    indexes = x.argsort()

    # Create the maps
    self.le_gv0 = indexes[np.array(self.e_v0.vector().array(), dtype = 'int')]
    self.le_gv1 = indexes[np.array(self.e_v1.vector().array(), dtype = 'int')]

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
  
  # Loads all parallel maps from the directory self.maps_dir
  def load_maps(self):
    # Facets to edges
    self.f_e = firedrake.FacetFunction('size_t', self.mesh)
    File(self.maps_dir + "/f_e.xml") >> self.f_e
    
    # Global edge order                                          
    self.f_cr = firedrake.Function(self.CR)
    File(self.maps_dir + "/f_cr.xml") >> self.f_cr
    
    # Edges to first global vertex dof
    self.e_v0 = firedrake.Function(self.CR)
    File(self.maps_dir + "/e_v0.xml") >> self.e_v0
    
    # Edges to second global vertex dof
    self.e_v1 = firedrake.Function(self.CR)
    File(self.maps_dir + "/e_v1.xml") >> self.e_v1
    
    # Global vertex order
    self.f_cg = firedrake.Function(self.U)
    File(self.maps_dir + "/f_cg.xml") >> self.f_cg
    
    # Edge lengths
    self.e_lens = firedrake.Function(self.CR)
    File(self.maps_dir + "/e_lens.xml") >> self.e_lens
  
  # Plots a CR function
  def plot_cr(self, cr):
    self.copy_cr_to_facet(cr, self.ff_plot)
    plot(self.ff_plot, interactive = True)