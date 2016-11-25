import numpy as np
import scipy.sparse.linalg as sp
import itertools

# turn of warning for zero division
# (which occurs in the linearization of the logarithmic strain)
np.seterr(divide='ignore', invalid='ignore')

# ----------------------------------- GRID ------------------------------------

Nx     = 10          # number of voxels in x-direction
Ny     = 12          # number of voxels in y-direction
Nz     = 14          # number of voxels in z-direction
shape  = [Nx,Ny,Nz]  # number of voxels in all directions

# ----------------------------- TENSOR OPERATIONS -----------------------------

# tensor operations / products: np.einsum enables index notation, avoiding loops
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
dyad11 = lambda A1,B1: np.einsum('ixyz   ,jxyz   ->ijxyz  ',A1,B1)

# eigenvalue decomposition of 2nd-order tensor: return in convention i,j,x,y,z
# NB requires to swap default order of NumPy (in in/output)
def eig2(A2):
    swap1i    = lambda A1: np.einsum('xyzi ->ixyz ',A1)
    swap2     = lambda A2: np.einsum('ijxyz->xyzij',A2)
    swap2i    = lambda A2: np.einsum('xyzij->ijxyz',A2)
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
    A2inv = np.empty([3,3,Nx,Ny,Nz])
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

# identity tensor
i      = np.eye(3)
# identity tensors
I      = np.einsum('ij,xyz'           ,                  i   ,np.ones([Nx,Ny,Nz]))
I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones([Nx,Ny,Nz]))
I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones([Nx,Ny,Nz]))
I4s    = (I4+I4rt)/2.
II     = dyad22(I,I)
# stress, deformation gradient, plastic strain, elastic Finger tensor
# NB "_t" signifies that it concerns the value at the previous increment
ep_t   = np.zeros([    Nx,Ny,Nz])
P      = np.zeros([3,3,Nx,Ny,Nz])
F      = np.array(I,copy=True)
F_t    = np.array(I,copy=True)
be_t   = np.array(I,copy=True)

# ------------------------------------ FFT ------------------------------------

# projection operator (only for non-zero frequency, associated with the mean)
# NB: vectorized version of "hyper-elasticity_even-grid.py"
# - allocate / support function
Ghat4  = np.zeros([3,3,3,3,Nx,Ny,Nz])                # projection operator
x      = np.zeros([3      ,Nx,Ny,Nz],dtype='int32')  # position vectors
q      = np.zeros([3      ,Nx,Ny,Nz],dtype='int32')  # frequency vectors
delta  = lambda i,j: np.float(i==j)                  # Dirac delta function
# - set "x" as position vector of all grid-points (grid)
x[0],x[1],x[2] = np.mgrid[:Nx,:Ny,:Nz]
# - convert positions "x" to frequencies "q" (grid)
for i in range(3):
    freq = np.arange(-shape[i]/2,+shape[i]/2)
    q[i] = freq[x[i]]
# - length "Q = ||q||",
#   and "norm = 1/Q" being zero for Q==0 and Nyquist frequencies
Q           = dot11(q,q)
norm        = 1./Q
norm[Q==0]  = 0.
norm[0,:,:] = 0.
norm[:,0,:] = 0.
norm[:,:,0] = 0.
# - set projection operator (grid)
for i, j, k, l in itertools.product(range(3), repeat=4):
    Ghat4[i,j,k,l] = norm*delta(i,l)*q[j]*q[k]

# (inverse) Fourier transform
fft    = lambda x  : np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[Nx,Ny,Nz]))
ifft   = lambda x  : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[Nx,Ny,Nz]))

# projection 'G', and product 'G : K^LT : (delta F)^T'
G      = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
K_dF   = lambda dFm: trans2(ddot42(K4,trans2(dFm.reshape(3,3,Nx,Ny,Nz))))
G_K_dF = lambda dFm: G(K_dF(dFm))

# ---------------------------- MATERIAL PARAMETERS ----------------------------

# phase indicator: square inclusion of volume fraction (3*3*14)/(10*12*14)
phase  = np.zeros([Nx,Ny,Nz]); phase[:3,:3,:] = 1.
# material parameters + function to convert to grid of scalars
param  = lambda M0,M1: M0*np.ones([Nx,Ny,Nz])*(1.-phase)+M1*np.ones([Nx,Ny,Nz])*phase
K      = param(0.833,0.833)  # bulk      modulus
mu     = param(0.386,0.386)  # shear     modulus
H      = param(0.004,0.008)  # hardening modulus
tauy0  = param(0.003,0.006)  # initial yield stress

# --------------------------- CONSTITUTIVE RESPONSE ---------------------------

def constitutive(F,F_t,be_t,ep_t):

    # function to compute linearization of the logarithmic Finger tensor
    def dln2_d2(A2):
        vals,vecs = eig2(A2)
        K4        = np.zeros([3,3,3,3,Nx,Ny,Nz])
        for m, n in itertools.product(range(3),repeat=2):
            gc  = (np.log(vals[n])-np.log(vals[m]))/(vals[n]-vals[m])
            gc[vals[n]==vals[m]] = (1.0/vals[m])[vals[n]==vals[m]]
            K4 += gc*dyad22(dyad11(vecs[:,m],vecs[:,n]),dyad11(vecs[:,m],vecs[:,n]))
        return K4

    # elastic stiffness
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
    ep       = ep_t+dgamma
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

    return P,K4,be,ep

# ---------------------------------- LOADING ----------------------------------

# initialize macroscopic incremental loading
ninc   = 50
lam    = 0.0
barF   = np.array(I,copy=True)
barF_t = np.array(I,copy=True)

# initial tangent operator: the elastic tangent
K4     = K*II+2.*mu*(I4s-1./3.*II)

# incremental deformation
for inc in range(1,ninc):

    print('=============================')
    print('inc: {0:d}'.format(inc))

    # set macroscopic deformation gradient (pure-shear)
    lam      += 0.2/float(ninc)
    barF      = np.array(I,copy=True)
    barF[0,0] =    (1.+lam)
    barF[1,1] = 1./(1.+lam)

    # store normalization
    Fn = np.linalg.norm(F)

    # first iteration residual: distribute "barF" over grid using "K4"
    b  = -G_K_dF(barF-barF_t)
    F +=         barF-barF_t

    # iterate as long as the iterative update does not vanish
    while True:

        # solve linear system; update DOFs
        dFm,_ = sp.cg(tol=1.e-8,
          A   = sp.LinearOperator(shape=(F.size,F.size),matvec=G_K_dF,dtype='float'),
          b   = b,
        )
        F    += dFm.reshape(3,3,Nx,Ny,Nz)

        # compute residual stress and tangent, convert to residual
        P,K4,be,ep = constitutive(F,F_t,be_t,ep_t)
        b          = -G(P)

        # check for convergence, print convergence info to screen
        print('{0:10.2e}'.format(np.linalg.norm(dFm)/Fn))
        if np.linalg.norm(dFm)/Fn<1.e-5: break

    # end-of-increment: update history
    barF_t = np.array(barF,copy=True)
    F_t    = np.array(F   ,copy=True)
    be_t   = np.array(be  ,copy=True)
    ep_t   = np.array(ep  ,copy=True)

# ---------------------------- PLOT TYPICAL RESULT ----------------------------

import matplotlib.pylab as plt

plt.figure()
plt.imshow(ep[:,:,1],interpolation='nearest',cmap='YlOrRd',clim=[0,0.4])
plt.show()
