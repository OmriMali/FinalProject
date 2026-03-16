from src import util
from src import visuals
from src.recovery_algorithms import kronecker_omp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def generate_gaussian_matrix(p, n):
    """
    Generates a (p, n) Gaussian random matrix.
    """
    matrix = np.random.randn(p, n)

    return matrix

def generate_subsampling_matrix(p, n):
    """
    Generates a (p, n) matrix where each row has exactly one '1', 
    and the index of the '1' is unique for every row.
    """
    if p > n:
        raise ValueError(f"Cannot have unique indices: rows (p={p}) > columns (n={n})")

    # 1. Initialize a zero matrix
    matrix = np.zeros((p, n))
    
    # 2. Generate a list of all possible column indices and shuffle them
    # This ensures that once an index is picked, it won't be picked again
    col_indices = np.arange(n)
    np.random.shuffle(col_indices)
    
    # 3. Take the first p shuffled indices (one for each row)
    selected_indices = col_indices[:p]
    
    # 4. Use advanced indexing to set the chosen positions to 1
    # np.arange(p) targets each row, selected_indices targets the unique columns
    matrix[np.arange(p), selected_indices] = 1.0
    
    return matrix

def generate_dct_matrix(n):
    return sp.fft.dct(np.eye(n), axis=0, norm='ortho')

def generate_idct_matrix(n):
    return sp.fft.idct(np.eye(n), axis=0, norm='ortho')


orig = util.load_hsi("raw\\Indian_pines_corrected.mat")
orig = orig[:64, :64, :64]
minval, maxval, depth = util.get_hsi_statistics(orig)
norm_orig = util.normalize_zero_mean(orig, minval, maxval)
shape = norm_orig.shape


sr = 0.5
Phi_0 = generate_subsampling_matrix(int(sr*shape[0]), shape[0])
Phi_1 = generate_subsampling_matrix(int(sr*shape[1]), shape[1])
Phi_2 = np.eye(shape[2])

# Psi_0 = util.DCTBasis().inverse(np.eye(shape[0]), 0)
# Psi_1 = util.DCTBasis().inverse(np.eye(shape[1]), 0)
# Psi_2 = util.DCTBasis().inverse(np.eye(shape[2]), 0)
Psi_0 = generate_idct_matrix(shape[0])
Psi_1 = generate_idct_matrix(shape[1])
Psi_2 = generate_idct_matrix(shape[2])


Phis = [Phi_0, Phi_1, Phi_2]
Psis = [Psi_0, Psi_1, Psi_2]
Ds = []

for n in range(len(Phis)):
    D = Phis[n] @ Psis[n]

    # Optionally normalize columns to exactly 1
    col_norms = np.linalg.norm(D, axis=0)
    print(f"D_{n} column norms min/max: {col_norms.min():.6f}/{col_norms.max():.6f}")
    D /= col_norms

    Ds.append(D)

Y = norm_orig.copy()
for n in range(len(Phis)):
    Y = util.mode_n_product(Y, Phis[n], n)

mu, k_bound = util.analyze_dictionary_coherence(Ds[0])
print(f'Coherence: {mu}  |  K < {k_bound}')


Is, a = kronecker_omp(Ds, Y, 2000)

X = np.zeros_like(norm_orig)
for j in range(len(a)):
    coord = tuple(Is[n][j] for n in range(len(Is)))
    X[coord] = a[j]

Z = X.copy()
for n in range(len(Psis)):
    Z = util.mode_n_product(Z, Psis[n], n)


rec = util.denormalize_zero_mean(Z, minval, maxval)
print(f'RMSE: {util.calc_rmse(rec, orig)}')
print(f'SAM: {util.calc_sam(rec, orig)}')
print(f'PSNR: {util.calc_psnr(rec, orig, depth)}')

plt.figure()
origrgb = visuals.to_false_color(orig)
recrgb = visuals.to_false_color(rec)
plt.subplot(1, 2, 1)
plt.imshow(origrgb)
plt.subplot(1, 2, 2)
plt.imshow(recrgb)
plt.show()

