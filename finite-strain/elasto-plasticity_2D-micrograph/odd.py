import numpy as np
import scipy.sparse.linalg as sp
import itertools

# turn of warning for zero division
# (which occurs in the linearization of the logarithmic strain)
np.seterr(divide='ignore', invalid='ignore')

# ----------------------------------- GRID ------------------------------------

# read phase indicator from micrograph: 0=soft, 1=hard
phase  = np.load('odd_image.npz')['phase']

# grid dimensions
Nx     = phase.shape[0]  # number of pixels in x-direction
Ny     = phase.shape[1]  # number of pixels in y-direction
shape  = [Nx,Ny]         # number of voxels in all directions

# ----------------------------- TENSOR OPERATIONS -----------------------------

# tensor operations / products: np.einsum enables index notation, avoiding loops
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
dyad11 = lambda A1,B1: np.einsum('ixy   ,jxy   ->ijxy  ',A1,B1)

# eigenvalue decomposition of 2nd-order tensor: return in convention i,j,x,y
# NB requires to swap default order of NumPy (in in/output)
def eig2(A2):
    swap1i    = lambda A1: np.einsum('xyi ->ixy ',A1)
    swap2     = lambda A2: np.einsum('ijxy->xyij',A2)
    swap2i    = lambda A2: np.einsum('xyij->ijxy',A2)
    vals,vecs = np.linalg.eig(swap2(A2))
    vals      = swap1i(vals)
    vecs      = swap2i(vecs)
    return vals,vecs

# logarithm of grid of 2nd-order tensors
def ln2(A2):
    vals,vecs = eig2(A2)
    return sum([np.log(vals[i])*dyad11(vecs[:,i],vecs[:,i]) for i in range(3)])

# exponent of grid of 2nd-order tensors
def exp2(A2):
    vals,vecs = eig2(A2)
    return sum([np.exp(vals[i])*dyad11(vecs[:,i],vecs[:,i]) for i in range(3)])

# determinant of grid of 2nd-order tensors
def det2(A2):
    return (A2[0,0]*A2[1,1]*A2[2,2]+A2[0,1]*A2[1,2]*A2[2,0]+A2[0,2]*A2[1,0]*A2[2,1])-\
           (A2[0,2]*A2[1,1]*A2[2,0]+A2[0,1]*A2[1,0]*A2[2,2]+A2[0,0]*A2[1,2]*A2[2,1])

# inverse of grid of 2nd-order tensors
def inv2(A2):
    A2det = det2(A2)
    A2inv = np.empty([3,3,Nx,Ny])
    A2inv[0,0] = (A2[1,1]*A2[2,2]-A2[1,2]*A2[2,1])/A2det
    A2inv[0,1] = (A2[0,2]*A2[2,1]-A2[0,1]*A2[2,2])/A2det
    A2inv[0,2] = (A2[0,1]*A2[1,2]-A2[0,2]*A2[1,1])/A2det
    A2inv[1,0] = (A2[1,2]*A2[2,0]-A2[1,0]*A2[2,2])/A2det
    A2inv[1,1] = (A2[0,0]*A2[2,2]-A2[0,2]*A2[2,0])/A2det
    A2inv[1,2] = (A2[0,2]*A2[1,0]-A2[0,0]*A2[1,2])/A2det
    A2inv[2,0] = (A2[1,0]*A2[2,1]-A2[1,1]*A2[2,0])/A2det
    A2inv[2,1] = (A2[0,1]*A2[2,0]-A2[0,0]*A2[2,1])/A2det
    A2inv[2,2] = (A2[0,0]*A2[1,1]-A2[0,1]*A2[1,0])/A2det
    return A2inv

# ------------------------ INITIATE (IDENTITY) TENSORS ------------------------

# identity tensor (single tensor)
i    = np.eye(3)
# identity tensors (grid)
I    = np.einsum('ij,xy'          ,                  i   ,np.ones([Nx,Ny]))
I4   = np.einsum('ijkl,xy->ijklxy',np.einsum('il,jk',i,i),np.ones([Nx,Ny]))
I4rt = np.einsum('ijkl,xy->ijklxy',np.einsum('ik,jl',i,i),np.ones([Nx,Ny]))
I4s  = (I4+I4rt)/2.
II   = dyad22(I,I)

# ------------------------------------ FFT ------------------------------------

# 2-D projection operator (only for non-zero frequency, associated with the mean)
# - allocate / support function
Ghat4_2 = np.zeros([2,2,2,2,Nx,Ny])                # projection operator
x_2     = np.zeros([2      ,Nx,Ny],dtype='int64')  # position vectors
q_2     = np.zeros([2      ,Nx,Ny],dtype='int64')  # frequency vectors
delta   = lambda i,j: np.float(i==j)               # Dirac delta function
# - set "x_2" as position vector of all grid-points
x_2[0],x_2[1] = np.mgrid[:Nx,:Ny]
# - convert positions "x_2" for frequencies "q_2"
for i in range(2):
    freq   = np.arange(-(shape[i]-1)/2,+(shape[i]+1)/2,dtype='int64')
    q_2[i] = freq[x_2[i]]
# - compute "Q = ||q_2||", and "norm = 1/Q" being zero for the mean (Q==0)
#   NB: avoid zero division
q_2     = q_2.astype(np.float)
Q       = dot11(q_2,q_2)
Z       = Q==0
Q[Z]    = 1.
norm    = 1./Q
norm[Z] = 0.
# - set projection operator                                   [grid of tensors]
for i, j, l, m in itertools.product(range(2), repeat=4):
    Ghat4_2[i,j,l,m] = norm*delta(i,m)*q_2[j]*q_2[l]

# 2-D (inverse) Fourier transform (for each tensor component in each direction)
fft    = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[Nx,Ny]))
ifft   = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[Nx,Ny]))

# functions for the 2-D projection 'G', and the product 'G : K^LT : (delta F)^T'
G      = lambda A2_2 : np.real( ifft( ddot42(Ghat4_2,fft(A2_2)) ) ).reshape(-1)
K_dF   = lambda dFm_2: trans2(ddot42(K4_2,trans2(dFm_2.reshape(2,2,Nx,Ny))))
G_K_dF = lambda dFm_2: G(K_dF(dFm_2))

# --------------------------- CONSTITUTIVE RESPONSE ---------------------------

# constitutive response to a certain loading and history
# NB: completely uncoupled from the FFT-solver, but implemented as a regular
#     grid of quadrature points, to have an efficient code;
#     each point is completely independent, just evaluated at the same time
def constitutive(F,F_t,be_t,ep_t):

    # function to compute linearization of the logarithmic Finger tensor
    def dln2_d2(A2):
        vals,vecs = eig2(A2)
        K4        = np.zeros([3,3,3,3,Nx,Ny])
        for m, n in itertools.product(range(3),repeat=2):
            gc  = (np.log(vals[n])-np.log(vals[m]))/(vals[n]-vals[m])
            gc[vals[n]==vals[m]] = (1.0/vals[m])[vals[n]==vals[m]]
            K4 += gc*dyad22(dyad11(vecs[:,m],vecs[:,n]),dyad11(vecs[:,m],vecs[:,n]))
        return K4

    # elastic stiffness tensor
    C4e      = K*II+2.*mu*(I4s-1./3.*II)

    # trial state
    Fdelta   = dot22(F,inv2(F_t))
    be_s     = dot22(Fdelta,dot22(be_t,trans2(Fdelta)))
    lnbe_s   = ln2(be_s)
    tau_s    = ddot42(C4e,lnbe_s)/2.
    taum_s   = ddot22(tau_s,I)/3.
    taud_s   = tau_s-taum_s*I
    taueq_s  = np.sqrt(3./2.*ddot22(taud_s,taud_s))
    N_s      = 3./2.*taud_s/taueq_s
    phi_s    = taueq_s-(tauy0+H*ep_t)
    phi_s    = 1./2.*(phi_s+np.abs(phi_s))

    # return map
    dgamma   = phi_s/(H+3.*mu)
    ep       = ep_t  +   dgamma
    tau      = tau_s -2.*dgamma*N_s*mu
    lnbe     = lnbe_s-2.*dgamma*N_s
    be       = exp2(lnbe)
    P        = dot22(tau,trans2(inv2(F)))

    # consistent tangent operator
    a0       = dgamma*mu/taueq_s
    a1       = mu/(H+3.*mu)
    C4ep     = ((K-2./3.*mu)/2.+a0*mu)*II+(1.-3.*a0)*mu*I4s+2.*mu*(a0-a1)*dyad22(N_s,N_s)
    dlnbe4_s = dln2_d2(be_s)
    dbe4_s   = 2.*dot42(I4s,be_s)
    K4       = (C4e/2.)*(phi_s<=0.).astype(np.float)+C4ep*(phi_s>0.).astype(np.float)
    K4       = ddot44(K4,ddot44(dlnbe4_s,dbe4_s))
    K4       = dot42(-I4rt,tau)+K4
    K4       = dot42(dot24(inv2(F),K4),trans2(inv2(F)))

    return P,P[:2,:2],K4[:2,:2,:2,:2],be,ep

# function to convert material parameters to grid of scalars
param  = lambda soft,hard: soft*np.ones([Nx,Ny])*(1.-phase)+\
                           hard*np.ones([Nx,Ny])*    phase

# material parameters
K      = param(   0.833        ,      0.833        )  # bulk      modulus
mu     = param(   0.386        ,      0.386        )  # shear     modulus
H      = param(2000.0e6/200.0e9,2.*2000.0e6/200.0e9)  # hardening modulus
tauy0  = param( 600.0e6/200.0e9,2.* 600.0e6/200.0e9)  # initial yield stress

# ---------------------------------- LOADING ----------------------------------

# stress, deformation gradient, plastic strain, elastic Finger tensor
# NB "_t" signifies that it concerns the value at the previous increment
ep_t   = np.zeros([    Nx,Ny])
P      = np.zeros([3,3,Nx,Ny])
F      = np.array(I,copy=True)
F_t    = np.array(I,copy=True)
be_t   = np.array(I,copy=True)

# initialize macroscopic incremental loading
ninc    = 250
epsbar  = np.linspace(0.0,0.2,ninc+1)[1:]
stretch = np.exp(np.sqrt(3.0)/2.0*epsbar)
barF    = np.array(I,copy=True)
barF_t  = np.array(I,copy=True)

# initial tangent operator: the elastic tangent
K4_2    = (K*II+2.*mu*(I4s-1./3.*II))[:2,:2,:2,:2]

# incremental deformation
for inc,lam in zip(range(1,ninc+1),stretch):

    print('=============================')
    print('inc: {0:d}'.format(inc))

    # set macroscopic deformation gradient (pure-shear)
    barF      = np.array(I,copy=True)
    barF[0,0] =    (lam)
    barF[1,1] = 1./(lam)

    # first iteration residual: distribute "barF" over grid using "K4"
    b     = -G_K_dF((barF-barF_t)[:2,:2])
    F    +=          barF-barF_t

    # parameters for Newton iterations: normalization and iteration counter
    Fn    = np.linalg.norm(F)
    iiter = 0

    # iterate as long as the iterative update does not vanish
    while True:

        # solve linear system using the Conjugate Gradient iterative solver
        dFm,i = sp.cg(tol=1.e-8,
          A       = sp.LinearOperator(shape=(2*2*Nx*Ny,2*2*Nx*Ny),matvec=G_K_dF,dtype='float'),
          b       = b,
          maxiter = 1000,
        )
        if i: raise IOError('CG-solver failed')

        # apply iterative update to (3-D) DOFs
        F[:2,:2] += dFm.reshape(2,2,Nx,Ny)

        # compute residual stress and tangent, convert to residual
        P,P_2,K4_2,be,ep = constitutive(F,F_t,be_t,ep_t)
        b                = -G(P_2)

        # check for convergence, print convergence info to screen
        print('{0:10.2e}'.format(np.linalg.norm(dFm)/Fn))
        if np.linalg.norm(dFm)/Fn<1.e-5 and iiter>0: break
        if iiter>9: raise IOError('Maximum number of Newton iterations have been exceeded')
        iiter += 1

    # end-of-increment: update history
    barF_t = np.array(barF,copy=True)
    F_t    = np.array(F   ,copy=True)
    be_t   = np.array(be  ,copy=True)
    ep_t   = np.array(ep  ,copy=True)

    np.savez('odd_output_inc%03d.npz'%inc,ep=ep)

