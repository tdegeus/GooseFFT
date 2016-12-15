import numpy as np
import scipy.sparse.linalg as sp
import itertools

# turn of warning for zero division (occurs due to vectorization)
np.seterr(divide='ignore', invalid='ignore')

# ----------------------------------- GRID ------------------------------------

Nx     = 31             # number of voxels in x-direction
Ny     = 31             # number of voxels in y-direction
Nz     =  1             # number of voxels in z-direction
shape  = [Nx,Ny,Nz]     # number of voxels as list: [Nx,Ny,Nz]
ndof   = 3**2*Nx*Ny*Nz  # number of degrees-of-freedom

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2   : np.einsum('ijxyz          ->jixyz  ',A2   )
ddot22 = lambda A2,B2: np.einsum('ijxyz  ,jixyz  ->xyz    ',A2,B2)
ddot42 = lambda A4,B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijklxyz,lkmnxyz->ijmnxyz',A4,B4)
dot11  = lambda A1,B1: np.einsum('ixyz   ,ixyz   ->xyz    ',A1,B1)
dot22  = lambda A2,B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ijxyz  ,jkmnxyz->ikmnxyz',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijklxyz,lmxyz  ->ijkmxyz',A4,B2)
dyad22 = lambda A2,B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz',A2,B2)

# identity tensor                                               [single tensor]
i      = np.eye(3)
# identity tensors                                            [grid of tensors]
I      = np.einsum('ij,xyz'           ,                  i   ,np.ones([Nx,Ny,Nz]))
I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones([Nx,Ny,Nz]))
I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones([Nx,Ny,Nz]))
II     = dyad22(I,I)
I4s    = (I4+I4rt)/2.
I4d    = (I4s-II/3.)

# projection operator (zero for zero frequency, associated with the mean)
# NB: vectorized version of "../linear-elasticity.py"
# - allocate / define support function
Ghat4  = np.zeros([3,3,3,3,Nx,Ny,Nz])                # projection operator
x      = np.zeros([3      ,Nx,Ny,Nz],dtype='int64')  # position vectors
q      = np.zeros([3      ,Nx,Ny,Nz],dtype='int64')  # frequency vectors
delta  = lambda i,j: np.float(i==j)                  # Dirac delta function
# - set "x" as position vector of all grid-points   [grid of vector-components]
x[0],x[1],x[2] = np.mgrid[:Nx,:Ny,:Nz]
# - convert positions "x" to frequencies "q"        [grid of vector-components]
for i in range(3):
    freq = np.arange(-(shape[i]-1)/2,+(shape[i]+1)/2,dtype='int64')
    q[i] = freq[x[i]]
# - compute "Q = ||q||", and "norm = 1/Q" being zero for the mean (Q==0)
#   NB: avoid zero division
q       = q.astype(np.float64)
Q       = dot11(q,q)
Z       = Q==0
Q[Z]    = 1.
norm    = 1./Q
norm[Z] = 0.
# - set projection operator                                   [grid of tensors]
for i, j, l, m in itertools.product(range(3), repeat=4):
    Ghat4[i,j,l,m] =   -(norm**2.)*(q[i]*q[j]*q[l]*q[m])+\
    .5*norm*( delta(j,l)*q[i]*q[m]+delta(j,m)*q[i]*q[l] +\
              delta(i,l)*q[j]*q[m]+delta(i,m)*q[j]*q[l] )

# (inverse) Fourier transform (for each tensor component in each direction)
fft  = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[Nx,Ny,Nz]))
ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[Nx,Ny,Nz]))

# functions for the projection 'G', and the product 'G : K : eps'
G        = lambda A2   : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
K_deps   = lambda depsm: ddot42(K4,depsm.reshape(3,3,Nx,Ny,Nz))
G_K_deps = lambda depsm: G(K_deps(depsm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# constitutive response to a certain loading (and history)
# NB: completely uncoupled from the FFT-solver, but implemented as a regular
#     grid of quadrature points, to have an efficient code;
#     each point is completely independent, just evaluated at the same time
# NB: all points for both models, but selectively ignored per materials
#     this avoids loops or a problem specific constitutive implementation

# linear elasticity
# -----------------

def elastic(eps):

    # parameters
    K       = 2.   # bulk  modulus
    mu      = 1.   # shear modulus

    # elastic stiffness tensor, and stress response
    C4      = K*II+2.*mu*I4d
    sig     = ddot42(C4,eps)

    return sig,C4

# elasto-plasticity
# -----------------

def elastoplastic(eps,eps_t,epse_t,ep_t):

    # parameters
    K          = 2.   # bulk  modulus
    mu         = 1.   # shear modulus
    sigy0      = 0.01 # initial yield stress
    H          = 0.05 # hardening modulus
    n          = 1.   # hardening exponent

    # elastic stiffness tensor
    C4e        = K*II+2.*mu*I4d

    # trial state
    epse_s     = epse_t+(eps-eps_t)
    sig_s      = ddot42(C4e,epse_s)
    sigm_s     = ddot22(sig_s,I)/3.
    sigd_s     = sig_s-sigm_s*I
    sigeq_s    = np.sqrt(3./2.*ddot22(sigd_s,sigd_s))
    # avoid zero division below ("phi_s" is corrected below)
    Z          = sigeq_s==0.
    sigeq_s[Z] = 1.

    # evaluate yield surface, set to zero if elastic (or stress-free)
    phi_s      = sigeq_s-(sigy0+H*ep_t**n)
    phi_s      = 1./2.*(phi_s+np.abs(phi_s))
    phi_s[Z]   = 0.
    el         = phi_s<=0.

    # plastic multiplier, based on non-linear hardening
    # - initialize
    dgamma     = np.zeros([Nx,Ny,Nz])
    res        = np.array(phi_s,copy=True)
    dH         = n*H*(ep_t)**(n-1.); dH[np.abs(ep_t)<=1.e-6] = 0.
    # - incrementally solve scalar non-linear return-map equation
    while np.max(np.abs(res))/sigy0>1.e-6:
        dgamma   -= res/(-3.*mu-dH)
        dH        = n*H*(ep_t+dgamma)**(n-1.); dH[np.abs(ep_t+dgamma)<=1.e-6] = 0.
        res       = sigeq_s-3.*mu*dgamma-(sigy0+H*(ep_t+dgamma)**n)
        res[el]   = 0.
    # - enforce elastic quadrature points to stay elastic
    dgamma[el] = 0.
    dH    [el] = 0.

    # return map
    N    = 3./2.*sigd_s/sigeq_s
    ep   = ep_t  +dgamma
    sig  = sig_s -dgamma*N*2.*mu
    epse = epse_s-dgamma*N

    # plastic tangent stiffness
    C4ep = C4e-\
           6.*(mu**2.)* dgamma/sigeq_s               *I4d+\
           4.*(mu**2.)*(dgamma/sigeq_s-1./(3.*mu+dH))*dyad22(N,N)
    # consistent tangent operator: elastic/plastic switch
    el   = el.astype(np.float)
    K4   = C4e*el+C4ep*(1.-el)

    return sig,K4,epse,ep


# laminate of two materials
# -------------------------

def constitutive(eps,eps_t,epse_t,ep_t):

    phase = np.zeros([Nx,Ny,Nz]); phase[:26,:,:] = 1.

    sig_P1,K4_P1,epse,ep = elastoplastic(eps,eps_t,epse_t,ep_t)
    sig_P2,K4_P2         = elastic      (eps                  )

    sig = phase*sig_P1+(1.-phase)*sig_P2
    K4  = phase*K4_P1 +(1.-phase)*K4_P2

    return sig,K4,epse,ep

# ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize: stress and strain tensor, history
sig    = np.zeros([3,3,Nx,Ny,Nz])
eps    = np.zeros([3,3,Nx,Ny,Nz])
eps_t  = np.zeros([3,3,Nx,Ny,Nz])
epse_t = np.zeros([3,3,Nx,Ny,Nz])
ep_t   = np.zeros([    Nx,Ny,Nz])

# initial constitutive response / tangent
sig,K4,epse,ep = constitutive(eps,eps_t,epse_t,ep_t)

# set macroscopic loading
DE       = np.zeros([3,3,Nx,Ny,Nz])
DE[0,1] += 0.05
DE[1,0] += 0.05

# initial residual: distribute "DE" over grid using "K4"
b      = -G_K_deps(DE)
eps   +=           DE

# compute DOF-normalization, set Newton iteration counter
En     = np.linalg.norm(eps)
iiter  = 0

# iterate as long as the iterative update does not vanish
while True:

    # solve linear system using the Conjugate Gradient iterative solver
    depsm,_ = sp.cg(tol=1.e-14,
      A = sp.LinearOperator(shape=(ndof,ndof),matvec=G_K_deps,dtype='float'),
      b = b,
    )

    # add solution of linear system to DOFs
    eps += depsm.reshape(3,3,Nx,Ny,Nz)

    # new residual
    sig,K4,epse,ep = constitutive(eps,eps_t,epse_t,ep_t)
    b              = -G(sig)

    # check for convergence
    print('{0:10.2e}'.format(np.linalg.norm(depsm)/En))
    if np.linalg.norm(depsm)/En<1.e-6 and iiter>0: break

    # update Newton iteration counter
    iiter += 1

# store history
ep_t   = np.array(ep  ,copy=True)
epse_t = np.array(epse,copy=True)
eps_t  = np.array(eps ,copy=True)

