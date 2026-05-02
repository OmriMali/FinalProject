import numpy as np
from src.core.registry import make_registry

# ===== Registry ===== #

MEASUREMENTS, register_measurement, get_measurement_matrix, list_measurements = make_registry(configurable=True)

# ===== Implementations ===== #

@register_measurement("IDENTITY")
def identity_matrix(m, n, seed=None, **kwargs):
    return np.eye(m, n)

@register_measurement("GAUSSIAN")
def gaussian_matrix(m, n, seed=42, **kwargs):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((m, n))
    return M / np.linalg.norm(M, axis=0, keepdims=True)

@register_measurement("SUBSAMPLING")
def subsampling_matrix(m, n, seed=42, **kwargs):
    
    rng = np.random.default_rng(seed)
    if m > n:
        raise ValueError(f"Cannot have unique indices: rows (p={m}) > columns (n={n})")
    
    matrix = np.zeros((m, n))
    col_indices = np.arange(n)
    rng.shuffle(col_indices)
    selected_indices = col_indices[:m]
    
    matrix[np.arange(m), selected_indices] = 1.0
    return matrix