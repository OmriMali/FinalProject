from src import util, visuals, recovery_algorithms, transforms, measurement_matrices
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


Phi = measurement_matrices.get_measurement_matrix("SUBSAMPLING", 50, 100)
Psi = transforms.get_transform("DCT", 100)
D = Phi @ Psi
D /= np.linalg.norm(D, axis=0, keepdims=True)
N = 3

S = 40
Y, X_true = util.generate_block_sparse_signal(D, S, N)

Ds = []
for n in range(N):
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    Ds.append(Dn)

X_hat = recovery_algorithms.n_bomp(Ds, Y, 2 * S**N)
Y_hat = X_hat.copy()
for n in range(N):
    Y_hat = util.mode_n_product(Y_hat, D, n)


print(f"RMSE: {util.calc_rmse(Y_hat, Y)}")
print(f"SAM: {util.calc_sam(Y_hat, Y)}")


