import numpy as np
import scipy.sparse.linalg as sp
import itertools

# True: use GPU computations. False: regular CPU computations
GPU = False
if GPU:
    import cupy as cp


def create_bin_sphere(matrix_size, center, radius):
    # function to create a sphere (defined by "center" and "radius")
    # as ones in a grid (defined by "shape") of zeros
    coords = np.ogrid[:matrix_size[0], :matrix_size[1], :matrix_size[2]]
    distance = np.sqrt(
        (coords[0] - center[0]) ** 2 + (coords[1] - center[1]) ** 2 + (
                coords[2] - center[2]) ** 2)
    return 1 * (distance <= radius)


def create_microstructure_spheres(cf):
    # phase indicator: "nspheres" random spherical inclusions,
    # with total volume fraction "cf" and maximum/minimum radius r_max / r_min.
    # Algorithm will fail for large fiber volume fractions "cf" (>>0.4)
    r_max = 1
    r_min = 3

    micro_phase = np.zeros((N, N, N), dtype=np.int)
    while (micro_phase[micro_phase == 1].size / micro_phase.size) <= cf:
        r = np.random.uniform(low=r_min, high=r_max, size=None)
        sphere_center = np.random.randint(low=-1, high=N - 1, size=(3, 1))
        sphere = create_bin_sphere(np.shape(micro_phase), sphere_center, r)
        if np.any(micro_phase[sphere == 1]) == 1:
            pass
        else:
            micro_phase[sphere == 1] = 1
    return micro_phase


# ----------------------------------- GRID ------------------------------------

ndim = 3  # number of dimensions
N = 32  # number of voxels (assumed equal for all directions)
ndof = ndim ** 2 * N ** 3  # number of degrees-of-freedom
shape = [N, N, N]  # number of voxels as list: [Nx,Ny,Nz]

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2: np.einsum('ijxyz          ->jixyz  ', A2)
ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
ddot44 = lambda A4, B4: np.einsum('ijklxyz,lkmnxyz->ijmnxyz', A4, B4)
dot22 = lambda A2, B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ', A2, B2)
dot24 = lambda A2, B4: np.einsum('ijxyz  ,jkmnxyz->ikmnxyz', A2, B4)
dot42 = lambda A4, B2: np.einsum('ijklxyz,lmxyz  ->ijkmxyz', A4, B2)
dyad22 = lambda A2, B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz', A2, B2)
dot11 = lambda A1, B1: np.einsum('ixyz   ,ixyz   ->xyz    ', A1, B1)

# identity tensor                                               [single tensor]
i = np.eye(ndim)
# identity tensors                                            [grid of tensors]
I = np.einsum('ij,xyz', i, np.ones([N, N, N]))
I4 = np.einsum('ijkl,xyz->ijklxyz', np.einsum('il,jk', i, i),
               np.ones([N, N, N]))
I4rt = np.einsum('ijkl,xyz->ijklxyz', np.einsum('ik,jl', i, i),
                 np.ones([N, N, N]))
I4s = (I4 + I4rt) / 2.
II = dyad22(I, I)

# projection operator (only for non-zero frequency, associated with the mean)
# NB: vectorized version of "hyper-elasticity_even.py"
# - allocate / support function
Ghat4 = np.zeros([3, 3, 3, 3, N, N, N])  # projection operator
x = np.zeros([3, N, N, N], dtype='int64')  # position vectors
q = np.zeros([3, N, N, N], dtype='int64')  # frequency vectors
delta = lambda i, j: np.float(i == j)  # Dirac delta function
# - set "x" as position vector of all grid-points   [grid of vector-components]
x[0], x[1], x[2] = np.mgrid[:N, :N, :N]
# - convert positions "x" to frequencies "q"        [grid of vector-components]
for i in range(3):
    freq = np.arange(-shape[i] / 2, +shape[i] / 2, dtype='int64')
    q[i] = freq[x[i]]
# - compute "Q = ||q||",
#   and "norm = 1/Q" being zero for Q==0 and Nyquist frequencies
q = q.astype(np.float)
Q = dot11(q, q)
Z = Q == 0
Q[Z] = 1.
norm = 1. / Q
norm[Z] = 0.
norm[0, :, :] = 0.
norm[:, 0, :] = 0.
norm[:, :, 0] = 0.
# - set projection operator                                   [grid of tensors]
for i, j, l, m in itertools.product(range(3), repeat=4):
    Ghat4[i, j, l, m] = norm * delta(i, m) * q[j] * q[l]

if GPU:
    # (inverse) Fourier transform (for each tensor component in each direction)
    fft = lambda x: cp.asnumpy(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(
        cp.asarray(x, x.dtype)), [N, N, N])))
    ifft = lambda x: cp.asnumpy(np.fft.fftshift(np.fft.ifftn(
        np.fft.ifftshift(cp.asarray(x, x.dtype)), [N, N, N])))
else:
    # (inverse) Fourier transform (for each tensor component in each direction)
    fft = lambda x: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), [N, N, N]))
    ifft = lambda x: np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(x), [N, N, N]))

# functions for the projection 'G', and the product 'G : K : eps'
G = lambda A2: np.real(ifft(ddot42(Ghat4, fft(A2)))).reshape(-1)
K_deps = lambda depsm: ddot42(K4, depsm.reshape(ndim, ndim, N, N, N))
G_K_deps = lambda depsm: G(K_deps(depsm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------
phase = create_microstructure_spheres(cf=0.2)

# material parameters + function to convert to grid of scalars
param = lambda M0, M1: M0 * np.ones([N, N, N]) * (1. - phase) + M1 * np.ones(
    [N, N, N]) * phase
K = param(0.833, 8.33)  # bulk  modulus                   [grid of scalars]
mu = param(0.386, 3.86)  # shear modulus                   [grid of scalars]
# stiffness tensor                                            [grid of tensors]
K4 = K * II + 2. * mu * (I4s - 1. / 3. * II)

# ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize stress and strain tensor                         [grid of tensors]
sig = np.zeros([ndim, ndim, N, N, N])
eps = np.zeros([ndim, ndim, N, N, N])

# set macroscopic loading
DE = np.zeros([ndim, ndim, N, N, N])
DE[0, 1] += 0.01
DE[1, 0] += 0.01

# initial residual: distribute "DE" over grid using "K4"
b = -G_K_deps(DE)
eps += DE
En = np.linalg.norm(eps)
iiter = 0

# iterate as long as the iterative update does not vanish
while True:
    depsm, _ = sp.cg(tol=1.e-8,
                     A=sp.LinearOperator(shape=(ndof, ndof), matvec=G_K_deps,
                                         dtype='float'),
                     b=b,
                     )  # solve linear system using CG
    eps += depsm.reshape(ndim, ndim, N, N,
                         N)  # update DOFs (array -> tens.grid)
    sig = ddot42(K4, eps)  # new residual stress
    b = -G(sig)  # convert residual stress to residual
    print('%10.2e' % (np.max(depsm) / En))  # print residual to the screen
    if np.linalg.norm(
            depsm) / En < 1.e-5 and iiter > 0: break  # check convergence
    iiter += 1
