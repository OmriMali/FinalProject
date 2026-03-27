import numpy as np
from src import util

# ===== Registry ===== #

MEASUREMENT_MATRICES, register_measurement = util.make_registry()

def list_measurements():
    return list(MEASUREMENT_MATRICES.keys())

# ===== Implementations ===== #

@register_measurement("IDENTITY")
def identity_matrix(m, n, rng=None, **kwargs):
    return np.eye(m, n)

@register_measurement("GAUSSIAN")
def gaussian_matrix(m, n, rng=None, **kwargs):
    if rng is None:
        rng = np.random.default_rng()

    M = rng.standard_normal((m, n))
    return M / np.linalg.norm(M, axis=0, keepdims=True)

@register_measurement("SUBSAMPLING")
def subsampling_matrix(m, n, rng=None, **kwargs):
    if rng is None:
        rng = np.random.default_rng()

    if m > n:
        raise ValueError(f"Cannot have unique indices: rows (p={m}) > columns (n={n})")
    
    matrix = np.zeros((m, n))
    col_indices = np.arange(n)
    rng.shuffle(col_indices)
    selected_indices = col_indices[:m]
    
    matrix[np.arange(m), selected_indices] = 1.0
    return matrix


# ===== Public API ===== #

def get_measurement_matrix(name, m, n, seed=None):
    base_name, params = util.parse_config_string(name)
    try:
        fn = MEASUREMENT_MATRICES[base_name]
    except KeyError:
        raise ValueError(f"Unknown measurement matrix: {name}")

    rng = np.random.default_rng(seed)

    return fn(m, n, rng=rng, **params)