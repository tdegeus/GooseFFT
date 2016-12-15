import numpy as np
import scipy.sparse.linalg as sp
import itertools

# turn of warning for zero division
np.seterr(divide='ignore', invalid='ignore')

# ----------------------------------- GRID ------------------------------------

Nx     = 31             # number of voxels in x-direction
Ny     = 31             # number of voxels in y-direction
Nz     =  1             # number of voxels in z-direction
shape  = [Nx,Ny,Nz]     # number of voxels in all directions
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

# projection operator (only for non-zero frequency, associated with the mean)
# NB: vectorized version of "linear-elasticity.py"
# - allocate / support function
Ghat4  = np.zeros([3,3,3,3,Nx,Ny,Nz])                # projection operator
x      = np.zeros([3      ,Nx,Ny,Nz],dtype='int32')  # position vectors
q      = np.zeros([3      ,Nx,Ny,Nz],dtype='int32')  # frequency vectors
delta  = lambda i,j: np.float(i==j)                  # Dirac delta function
# - set "x" as position vector of all grid-points (grid)
x[0],x[1],x[2] = np.mgrid[:Nx,:Ny,:Nz]
# - convert positions "x" to frequencies "q" (grid)
for i in range(3):
    freq = np.arange(-(shape[i]-1)/2,+(shape[i]+1)/2)
    q[i] = freq[x[i]]
# - compute "Q = ||q||", and "norm = 1/Q" being zero for the mean (Q==0)
Q          = dot11(q,q)
norm       = 1./Q
norm[Q==0] = 0.
# - set projection operator (grid)
for i, j, l, m in itertools.product(range(3), repeat=4):
    Ghat4[i,j,l,m] =   -(norm**2.)*(q[i]*q[j]*q[l]*q[m])+\
    .5*norm*( delta(j,l)*q[i]*q[m]+delta(j,m)*q[i]*q[l] +\
              delta(i,l)*q[j]*q[m]+delta(i,m)*q[j]*q[l] )

# (inverse) Fourier transform
fft  = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[Nx,Ny,Nz]))
ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[Nx,Ny,Nz]))

# projection 'G', and product 'G : K : eps'
G        = lambda A2   : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
K_deps   = lambda depsm: ddot42(K4,depsm.reshape(3,3,Nx,Ny,Nz))
G_K_deps = lambda depsm: G(K_deps(depsm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# linear elasticity
# -----------------

def elastic(eps):

    # parameters
    K       = 2.                  # bulk  modulus
    mu      = 1.                  # shear modulus

    # elastic stiffness tensor, and stress response
    C4      = K*II+2.*mu*I4d
    sig     = ddot42(C4,eps)

    return sig,C4

# visco-plasticity
# ----------------

def viscoplastic(eps,eps_t,epse_t,ep_t,dt):

    # parameters
    K          = 2.               # bulk  modulus
    mu         = 1.               # shear modulus
    gamma0     = 0.1/np.sqrt(3.)  # reference shear strain
    sig0       = 0.1              # reference shear stress
    n          = 1.               # hardening exponent

    # elastic stiffness tensor
    C4e        = K*II+2.*mu*I4d

    # trial state
    epse_s     = epse_t+(eps-eps_t)
    epsem_s    = ddot22(epse_s,I)/3.
    epsed_s    = epse_s-epsem_s*I
    sigm_s     = 3.*K *epsem_s
    sigd_s     = 2.*mu*epsed_s
    sigeq_s    = np.sqrt(3./2.*ddot22(sigd_s,sigd_s))

    # artificially modify "sigeq_s" to avoid zero-division below
    Z          =  sigeq_s<=0.
    Zf         = (sigeq_s<=0.).astype(np.float)
    sigeq_s[Z] = 1.

    # plastic multiplier
    dgamma     = np.zeros([Nx,Ny,Nz])
    res        = -gamma0*dt*(sigeq_s/sig0)**(1./n)

    while np.linalg.norm(np.abs(res))/mu>1.e-6:
        dres      = 1.+3.*mu*gamma0*dt/n*((sigeq_s-3.*mu*dgamma)/sig0)**(1./n-1.)
        dgamma   -= res/dres
        dgamma[Z] = 0.
        res       = dgamma-gamma0*dt*((sigeq_s-3.*mu*dgamma)/sig0)**(1./n);
        res[Z]    = 0.

    # return map
    N    = 3./2.*sigd_s/sigeq_s
    ep   = ep_t+dgamma
    sigd = (1.-3.*mu*dgamma/sigeq_s)*sigd_s
    sig  = sigm_s *I+sigd
    epse = epsem_s*I+sigd/(2.*mu)

    # consistent tangent operator
    C4ep = C4e-\
           6.*(mu**2.)* dgamma/sigeq_s*I4d+\
           4.*(mu**2.)*(dgamma/sigeq_s-\
           1./(3.*mu+n*sig0/(gamma0*dt)*(dgamma/(gamma0*dt))**(n-1.)))*dyad22(N,N)
    K4   = C4e*Zf+C4ep*(1.-Zf) # use elastic tangent if "sigeq_s==0"

    return sig,K4,epse,ep


# laminate of two materials
# -------------------------

def constitutive(eps,eps_t,epse_t,ep_t,dt):

    phase = np.zeros([Nx,Ny,Nz]); phase[:26,:,:] = 1.

    sig_P1,K4_P1,epse,ep = viscoplastic(eps,eps_t,epse_t,ep_t,dt)
    sig_P2,K4_P2         = elastic     (eps                     )

    sig = phase*sig_P1+(1.-phase)*sig_P2
    K4  = phase*K4_P1 +(1.-phase)*K4_P2

    return sig,K4,epse,ep

# ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize: stress and strain tensor, history
sig    = np.zeros([3,3,Nx,Ny,Nz])                           # [grid of tensors]
eps    = np.zeros([3,3,Nx,Ny,Nz])                           # [grid of tensors]
eps_t  = np.zeros([3,3,Nx,Ny,Nz])                           # [grid of tensors]
epse_t = np.zeros([3,3,Nx,Ny,Nz])                           # [grid of tensors]
ep_t   = np.zeros([    Nx,Ny,Nz])                           # [grid of scalars]

# set macroscopic loading increment
ninc     = 200
DE       = np.zeros([3,3,Nx,Ny,Nz])
DE[0,1] += 0.05/float(ninc)
DE[1,0] += 0.05/float(ninc)
dt       = 1.  /float(ninc)

# initial constitutive response / tangent
sig,K4,epse,ep = constitutive(eps,eps_t,epse_t,ep_t,dt)

# incremental loading
for inc in range(ninc):

    print('=============================')
    print('inc: {0:d}'.format(inc))

    # initial residual: distribute "DE" over grid using "K4"
    b      = -G_K_deps(DE)
    eps   +=           DE

    # compute DOF-normalization, set Newton iteration counter
    En     = np.linalg.norm(eps)
    iiter  = 0

    # iterate as long as the iterative update does not vanish
    while True:

        # solve linear system
        depsm,_ = sp.cg(tol=1.e-14,
          A = sp.LinearOperator(shape=(ndof,ndof),matvec=G_K_deps,dtype='float'),
          b = b,
        )

        # add solution of linear system to DOFs
        eps += depsm.reshape(3,3,Nx,Ny,Nz)

        # new residual
        sig,K4,epse,ep = constitutive(eps,eps_t,epse_t,ep_t,dt)
        b              = -G(sig)

        # check for convergence
        print('%10.2e'%(np.linalg.norm(depsm)/En))
        if np.linalg.norm(depsm)/En<1.e-8 and iiter>0: break
        iiter += 1

    # store history
    ep_t   = np.array(ep  ,copy=True)
    epse_t = np.array(epse,copy=True)
    eps_t  = np.array(eps ,copy=True)
