from src import util, visuals, recovery_algorithms, transforms, measurement_matrices, dictionary_learning
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Load data
hsi = util.load_hsi("raw\\Indian_pines_corrected.mat")
minval, maxval, depth = util.get_hsi_statistics(hsi)
hsi = util.normalize_zero_mean(hsi, minval, maxval)

# Reshape so columns contain spectra
Y = util.mode_n_unfold(hsi, n=2)

# # Subsample Y
# N_train = 1000
# idx = np.random.choice(Y.shape[1], N_train, replace=False)
# Y_train = Y[:, idx]

# # Progress bar
# pbar = tqdm(total=100)
# def progress_bar(fraction):
#     pbar.n = int(100 * fraction)
#     pbar.refresh()

# # K-SVD
# K = 300
# T_0 = 3
# max_iter = 30
# D, X = dictionary_learning.k_svd(Y_train, K, T_0, max_iter=30, progress_callback=progress_bar)

# # Save dictionary
# util.save_array_to_path(D, "results\\learned_dicts\\D_ksvd.npz", metadata={'K': K, 'T_0': T_0})

# # Reconstruction error
# err = np.linalg.norm(Y_train - D @ X) / np.linalg.norm(Y_train)
# print(f"Reconstruction Error: {err}")

# # Sparsity
# mean_sparsity = np.mean(np.count_nonzero(X, axis=0))
# print(f"Mean Sparsity: {mean_sparsity}")


D, metadata = util.load_array_from_path("results\\learned_dicts\\D_ksvd.npz")
T_0 = metadata['T_0']
K = metadata['K']

N = Y.shape[1]
X_full = np.zeros((K, N))
for i in range(N):
    X_full[:, i] = recovery_algorithms.omp(D, Y[:, i], T_0)
    
Y_hat = D @ X_full
hsi_hat = util.mode_n_fold(Y_hat, n=2, original_shape=hsi.shape)
print(f"RMSE: {util.calc_rmse(hsi, hsi_hat)}")
orig = visuals.to_false_color(hsi)
rec = visuals.to_false_color(hsi_hat)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(orig)
plt.subplot(1, 2, 2)
plt.imshow(rec)
plt.show()