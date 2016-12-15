import numpy as np
import scipy.sparse.linalg as sp
import itertools

# ----------------------------------- GRID ------------------------------------

ndim   = 3            # number of dimensions
N      = 31           # number of voxels (assumed equal for all directions)
ndof   = ndim**2*N**3 # number of degrees-of-freedom

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2   : np.einsum('ijxyz          ->jixyz  ',A2   )
ddot42 = lambda A4,B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijklxyz,lkmnxyz->ijmnxyz',A4,B4)
dot22  = lambda A2,B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ijxyz  ,jkmnxyz->ikmnxyz',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijklxyz,lmxyz  ->ijkmxyz',A4,B2)
dyad22 = lambda A2,B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz',A2,B2)

# identity tensor                                               [single tensor]
i      = np.eye(ndim)
# identity tensors                                            [grid of tensors]
I      = np.einsum('ij,xyz'           ,                  i   ,np.ones([N,N,N]))
I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones([N,N,N]))
I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones([N,N,N]))
I4s    = (I4+I4rt)/2.
II     = dyad22(I,I)

# projection operator                                         [grid of tensors]
# NB can be vectorized (faster, less readable), see: "elasto-plasticity.py"
# - support function / look-up list / zero initialize
delta  = lambda i,j: np.float(i==j)            # Dirac delta function
freq   = np.arange(-(N-1)/2.,+(N+1)/2.)        # coordinate axis -> freq. axis
Ghat4  = np.zeros([ndim,ndim,ndim,ndim,N,N,N]) # zero initialize
# - compute
for i,j,l,m in itertools.product(range(ndim),repeat=4):
    for x,y,z    in itertools.product(range(N),   repeat=3):
        q = np.array([freq[x], freq[y], freq[z]])  # frequency vector
        if not q.dot(q) == 0:                      # zero freq. -> mean
            Ghat4[i,j,l,m,x,y,z] = -(q[i]*q[j]*q[l]*q[m])/(q.dot(q))**2+\
             (delta(j,l)*q[i]*q[m]+delta(j,m)*q[i]*q[l]+\
              delta(i,l)*q[j]*q[m]+delta(i,m)*q[j]*q[l])/(2.*q.dot(q))

# (inverse) Fourier transform (for each tensor component in each direction)
fft  = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[N,N,N]))
ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[N,N,N]))

# functions for the projection 'G', and the product 'G : K : eps'
G        = lambda A2   : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
K_deps   = lambda depsm: ddot42(K4,depsm.reshape(ndim,ndim,N,N,N))
G_K_deps = lambda depsm: G(K_deps(depsm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# phase indicator: cubical inclusion of volume fraction (9**3)/(31**3)
phase  = np.zeros([N,N,N]); phase[-9:,:9,-9:] = 1.
# material parameters + function to convert to grid of scalars
param  = lambda M0,M1: M0*np.ones([N,N,N])*(1.-phase)+M1*np.ones([N,N,N])*phase
K      = param(0.833,8.33)  # bulk  modulus                   [grid of scalars]
mu     = param(0.386,3.86)  # shear modulus                   [grid of scalars]
# stiffness tensor                                            [grid of tensors]
K4     = K*II+2.*mu*(I4s-1./3.*II)

# ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize stress and strain tensor                         [grid of tensors]
sig      = np.zeros([ndim,ndim,N,N,N])
eps      = np.zeros([ndim,ndim,N,N,N])

# set macroscopic loading
DE       = np.zeros([ndim,ndim,N,N,N])
DE[0,1] += 0.01
DE[1,0] += 0.01

# initial residual: distribute "DE" over grid using "K4"
b        = -G_K_deps(DE)
eps     +=           DE
En       = np.linalg.norm(eps)
iiter    = 0

# iterate as long as the iterative update does not vanish
while True:
    depsm,_ = sp.cg(tol=1.e-8,
      A = sp.LinearOperator(shape=(ndof,ndof),matvec=G_K_deps,dtype='float'),
      b = b,
    )                                     # solve linear system using CG
    eps += depsm.reshape(ndim,ndim,N,N,N) # update DOFs (array -> tens.grid)
    sig  = ddot42(K4,eps)                 # new residual stress
    b     = -G(sig)                       # convert residual stress to residual
    print('%10.2e'%(np.max(depsm)/En))    # print residual to the screen
    if np.linalg.norm(depsm)/En<1.e-5 and iiter>0: break # check convergence
    iiter += 1
