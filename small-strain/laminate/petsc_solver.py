# this file is used to interface the petsc solver
# It will require: current sigma, projection operator and projection operator with consistent tangent functions
# It will return: depsm

import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# class to handle solving the system
class FourEq(object):

    def __init__(self, da, G, G_K_deps):
        assert da.getDim() == 1
        self.da = da
        self.localX = da.createLocalVec()
        # Functions used in Newton solve
        self.G = G
        self.G_K_deps = G_K_deps

    def formRHS(self, _B, rhs):
        # function called by PETSc to create right hand side
        b = self.da.getVecArray(_B)
        # indices might need to be taken into account for parallelisation
        b[:] = rhs

    def mult(self, mat, X, Y):
        # function called by PETSc to evaluate the matrix vector product
        # possibly already parallel ready
        self.da.globalToLocal(X, self.localX)
        x = self.da.getVecArray(self.localX)
        y = self.da.getVecArray(Y)
        #
        y[:] = self.G_K_deps(x[:])

def cg(ndof, tol, rhs, G, G_K_deps, i_print=False):
    #############################################
    # Petsc setup
    #############################################
    da = PETSc.DMDA().create([ndof], stencil_width=1)
    OptDB = PETSc.Options()
    pycontext = FourEq(da, G, G_K_deps)
    # vectors used for CG solve
    depsm = da.createGlobalVec()
    b = da.createGlobalVec()
    A = PETSc.Mat().createPython(
        [depsm.getSizes(), b.getSizes()], comm=da.comm)
    A.setPythonContext(pycontext)
    A.setUp()

    # OptDB.setValue('-ksp_monitor_singular_value', '')
    if i_print:
        OptDB.setValue('-ksp_monitor', '')
    OptDB.setValue('-ksp_atol', tol)

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('cg')
    pc = ksp.getPC()
    pc.setType('none')
    ksp.setFromOptions()

    # not the most pretty way
    pycontext.formRHS(b, rhs)
    ksp.setConvergenceHistory()
    ksp.solve(b, depsm)                   # solve linear system using CG
    hist = ksp.getConvergenceHistory()    # convergence history to return

    return depsm.array, hist