import numpy as np
import scipy.sparse.linalg as sp
import itertools

# ----------------------------------- GRID ------------------------------------

# load image: phase indicator for each pixel (0=soft, 1=hard)
phase  = np.load('image.npz')['bw'].astype(np.float64)
Nx     = phase.shape[0]  # number of pixels in x-direction
Ny     = phase.shape[1]  # number of pixels in y-direction
shape  = phase.shape     # number of pixels as list: [Nx,Ny]
ndof   = 2**2*Nx*Ny      # number of degrees-of-freedom

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2   : np.einsum('ijxy         ->jixy  ',A2   )
ddot22 = lambda A2,B2: np.einsum('ijxy  ,jixy  ->xy    ',A2,B2)
ddot42 = lambda A4,B2: np.einsum('ijklxy,lkxy  ->ijxy  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijklxy,lkmnxy->ijmnxy',A4,B4)
dot11  = lambda A1,B1: np.einsum('ixy   ,ixy   ->xy    ',A1,B1)
dot22  = lambda A2,B2: np.einsum('ijxy  ,jkxy  ->ikxy  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ijxy  ,jkmnxy->ikmnxy',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijklxy,lmxy  ->ijkmxy',A4,B2)
dyad22 = lambda A2,B2: np.einsum('ijxy  ,klxy  ->ijklxy',A2,B2)

# identity tensor                                               [single tensor]
i      = np.eye(3).astype(np.float64)
# identity tensors                                            [grid of tensors]
I      = np.einsum('ij,xy'           ,                 i   ,np.ones([Nx,Ny])).astype(np.float64)
I4     = np.einsum('ijkl,xy->ijklxy',np.einsum('il,jk',i,i),np.ones([Nx,Ny])).astype(np.float64)
I4rt   = np.einsum('ijkl,xy->ijklxy',np.einsum('ik,jl',i,i),np.ones([Nx,Ny])).astype(np.float64)
II     = dyad22(I,I)
I4s    = (I4+I4rt)/2.
I4d    = (I4s-II/3.)

# projection operator (zero for zero frequency, associated with the mean)
# NB: vectorized version of "../linear-elasticity.py"
# - allocate / define support function
Ghat4_2 = np.zeros([2,2,2,2,Nx,Ny],dtype='float64')   # projection operator
x_2     = np.zeros([2      ,Nx,Ny],dtype='int64'  )   # position vectors
q_2     = np.zeros([2      ,Nx,Ny],dtype='int64'  )   # frequency vectors
delta   = lambda i,j: np.float64(i==j)                # Dirac delta function
# - set "x_2" as position vector of all grid-points [grid of vector-components]
x_2[0],x_2[1] = np.mgrid[:Nx,:Ny]
# - convert positions "x_2" to frequencies "q_2"    [grid of vector-components]
for i in range(2):
    freq   = np.arange(-(shape[i]-1)/2,+(shape[i]+1)/2,dtype='int64')
    q_2[i] = freq[x_2[i]]
# - compute "Q = ||q_2||", and "norm = 1/Q" being zero for the mean (Q==0)
#   NB: avoid zero division
q_2     = q_2.astype(np.float64)
Q       = dot11(q_2,q_2)
Z       = Q==0
Q[Z]    = 1.
norm    = 1./Q
norm[Z] = 0.
# - set projection operator                                   [grid of tensors]
for i, j, l, m in itertools.product(range(2), repeat=4):
    Ghat4_2[i,j,l,m] = -(norm**2.)*(q_2[i]*q_2[j]*q_2[l]*q_2[m])+\
    .5*norm*( delta(j,l)*q_2[i]*q_2[m]+delta(j,m)*q_2[i]*q_2[l] +\
              delta(i,l)*q_2[j]*q_2[m]+delta(i,m)*q_2[j]*q_2[l] )

# (inverse) Fourier transform (for each tensor component in each direction)
fft  = lambda x_2: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x_2),[Nx,Ny]))
ifft = lambda x_2: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x_2),[Nx,Ny]))

# functions for the projection 'G', and the product 'G : K : eps'
G        = lambda A2_2   : np.real(ifft(ddot42(Ghat4_2,fft(A2_2)))).reshape(-1)
K_deps   = lambda depsm_2: ddot42(K4_2,depsm_2.reshape(2,2,Nx,Ny))
G_K_deps = lambda depsm_2: G(K_deps(depsm_2))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# constitutive response to a certain loading and history
# NB: completely uncoupled from the FFT-solver, but implemented as a regular
#     grid of quadrature points, to have an efficient code;
#     each point is completely independent, just evaluated at the same time
def constitutive(eps,eps_t,epse_t,ep_t):

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
    sigy,dH    = yield_function(ep_t)
    phi_s      = sigeq_s-sigy
    phi_s      = 1./2.*(phi_s+np.abs(phi_s))
    phi_s[Z]   = 0.
    el         = phi_s<=0.

    # plastic multiplier, based on non-linear hardening
    # - initialize
    dgamma     = np.zeros(ep_t.shape,dtype='float64')
    res        = np.array(phi_s     ,copy =True     )
    # - incrementally solve scalar non-linear return-map equation
    while np.max(np.abs(res)/sigy0)>1.e-6:
        dgamma   -= res/(-3.*mu-dH)
        sigy,dH   = yield_function(ep_t+dgamma)
        res       = sigeq_s-3.*mu*dgamma-sigy
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
    el   = el.astype(np.float64)
    K4   = C4e*el+C4ep*(1.-el)

    # return 3-D stress, 2-D stress/tangent, and history
    return sig,sig[:2,:2,:,:],K4[:2,:2,:2,:2,:,:],epse,ep

# function to convert material parameters to grid of scalars
param  = lambda soft,hard: soft*np.ones([Nx,Ny],dtype='float64')*(1.-phase)+\
                           hard*np.ones([Nx,Ny],dtype='float64')*    phase

# material parameters
K      = param( 0.833 , 0.833   )  # bulk  modulus
mu     = param( 0.386 , 0.386   )  # shear modulus
sigy0  = param( 0.005 , 0.005*2.)  # initial yield stress
H      = param( 0.005 , 0.005*2.)  # hardening modulus
n      = param( 0.2   , 0.2     )  # hardening exponent

# yield function: return yield stress and incremental hardening modulus
# NB: all integration points are independent, but treated at the same time
def yield_function(ep):
    # - distinguish very low plastic strains -> linear hardening for "ep<=h"
    h           = 0.0001
    low         = ep<=h
    ep_hgh      = np.array(ep,copy=True)
    ep_hgh[low] = h
    # - normal non-linear hardening
    Sy_hgh      = sigy0+H*ep_hgh**n
    dH_hgh      = n*H*ep_hgh**(n-1.)
    # - linearized hardening for "ep<=h": ensure continuity at "ep==h"
    dH_low      = n*H*h**(n-1.)
    Sy_low      = (sigy0+H*h**n-dH_low*h)+dH_low*ep
    # - combine initial linear hardening with non-linear hardening
    low         = low.astype(np.float64)
    sigy        = (1.-low)*Sy_hgh+low*Sy_low
    dH          = (1.-low)*dH_hgh+low*dH_low
    # - return yield stress and linearized hardening modulus
    return sigy,dH

# ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize: stress and strain tensor, and history
eps      = np.zeros([3,3,Nx,Ny],dtype='float64')
eps_t    = np.zeros([3,3,Nx,Ny],dtype='float64')
epse_t   = np.zeros([3,3,Nx,Ny],dtype='float64')
ep_t     = np.zeros([    Nx,Ny],dtype='float64')

# initial tangent operator: the elastic tangent
K4_2     = (K*II+2.*mu*I4d)[:2,:2,:2,:2]

# define incremental macroscopic strain
ninc     =  100
epsbar   =  0.1
DE       =  np.zeros([3,3,Nx,Ny],dtype='float64')
DE[0,0]  = +np.sqrt(3.)/2.*epsbar/float(ninc)
DE[1,1]  = -np.sqrt(3.)/2.*epsbar/float(ninc)

# incremental deformation
for inc in range(1,ninc+1):

    print('=============================')
    print('inc: {0:d}'.format(inc))

    # initial residual: distribute "DE" over grid using "K4_2"
    b      = -G_K_deps(DE[:2,:2])
    eps   +=           DE

    # compute DOF-normalization, set Newton iteration counter
    En     = np.linalg.norm(eps)
    iiter  = 0

    # iterate as long as the iterative does not vanish
    while True:

        # solve linear system using the Conjugate Gradient iterative solver
        depsm_2,i = sp.cg(tol=1e-8,
          A       = sp.LinearOperator(shape=(ndof,ndof),matvec=G_K_deps,dtype='float64'),
          b       = b,
          maxiter = 2000,
        )
        if i: raise RuntimeError('CG solver failed')

        # add solution of linear system to DOFs
        eps[:2,:2] += depsm_2.reshape(2,2,Nx,Ny)

        # new residual
        sig,sig_2,K4_2,epse,ep =  constitutive(eps,eps_t,epse_t,ep_t)
        b                      = -G(sig_2)

        # check for convergence
        print('{0:10.2e}'.format(np.linalg.norm(depsm_2)/En))
        if np.linalg.norm(depsm_2)/En<1.e-5 and iiter>0: break

        # update Newton iteration counter
        iiter += 1

    # end-of-increment: update history
    ep_t   = np.array(ep  ,copy=True)
    epse_t = np.array(epse,copy=True)
    eps_t  = np.array(eps ,copy=True)
