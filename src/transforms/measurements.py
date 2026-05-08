import numpy as np
from src.registry.measurement_matrices import register_measurement

@register_measurement("IDENTITY")
def identity_matrix(m, n, rng=None, **kwargs):
    return np.eye(m, n)

@register_measurement("GAUSSIAN")
def gaussian_matrix(m, n, rng=42, **kwargs):
    M = rng.standard_normal((m, n))
    return M / np.linalg.norm(M, axis=0, keepdims=True)

@register_measurement("SUBSAMPLING")
def subsampling(m, n, rng=None, **kwargs):
    if m > n:
        raise ValueError("m cannot exceed n")

    mat = np.zeros((m, n))
    idx = np.arange(n)
    rng.shuffle(idx)

    chosen = idx[:m]
    mat[np.arange(m), chosen] = 1.0

    return mat