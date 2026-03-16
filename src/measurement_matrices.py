import numpy as np
import scipy as sp

def identity_matrix(m, n, rng=None):
    return np.eye(m, n)

def gaussian_matrix(m, n, rng=None):
    M = rng.standard_normal((m, n))
    return M / np.linalg.norm(M, axis=0, keepdims=True)

def subsampling_matrix(m, n, rng=None):
    if m > n:
        raise ValueError(f"Cannot have unique indices: rows (p={m}) > columns (n={n})")
    matrix = np.zeros((m, n))
    col_indices = np.arange(n)
    rng.shuffle(col_indices)
    selected_indices = col_indices[:m]
    matrix[np.arange(m), selected_indices] = 1.0
    return matrix

MEASUREMENT_MATRICES = {
    "IDENTITY": identity_matrix,
    "GAUSSIAN": gaussian_matrix,
    "SUBSAMPLING": subsampling_matrix
}

def get_measurement_matrix(name, m, n, seed=None):
    try:
        rng = np.random.default_rng(seed)
        return MEASUREMENT_MATRICES[name](m, n)
    except KeyError:
        raise ValueError(f"Unknown transform: {name}")
