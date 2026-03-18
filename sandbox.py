from src import util, visuals, recovery_algorithms, transforms, measurement_matrices
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


Z = util.load_hsi("raw\\Indian_pines_corrected.mat")
Z = Z[:100, :100, :100]
minval, maxval, depth = util.get_hsi_statistics(Z)
Z = util.normalize_zero_mean(Z, minval, maxval)

Phi = measurement_matrices.get_measurement_matrix("SUBSAMPLING", 60, 100)
Psi = transforms.get_inverse_transform("DCT", 100)
N = 3

D = Phi @ Psi
col_norms = np.linalg.norm(D, axis=0)
col_norms[col_norms == 0] = 1.0
S_inv = np.diag(1.0 / col_norms)
D_norm = D @ S_inv

Y = Z.copy()
for n in range(N):
    Y = util.mode_n_product(Y, Phi, n)

Ds = []
for n in range(N):
    Ds.append(D_norm)

X_hat = recovery_algorithms.n_bomp(Ds, Y, 25000)

Z_hat = X_hat.copy()
Psi_norm = Psi @ S_inv
for n in range(N):
    Z_hat = util.mode_n_product(Z_hat, Psi_norm , n)

Z = util.denormalize_zero_mean(Z, minval, maxval)
Z_hat = util.denormalize_zero_mean(Z_hat, minval, maxval)

print(f"RMSE: {util.calc_rmse(Z_hat, Z)}")
print(f"SAM: {util.calc_sam(Z_hat, Z)}")

orig = visuals.to_false_color(Z)
rec = visuals.to_false_color(Z_hat)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(orig)
plt.subplot(1,2,2)
plt.imshow(rec)
plt.show()


