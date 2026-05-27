import numpy as np
from typing import Callable, Dict



_MEASUREMENTS: Dict[str, Callable] = {}

def register_measurement(name: str):
    def decorator(fn):
        key = name.upper()

        if key in _MEASUREMENTS:
            raise ValueError(f"Measurement '{name}' already registered")

        _MEASUREMENTS[key] = fn
        return fn

    return decorator

def get_measurement(name: str, m: int, n: int, seed: int | None = None, **kwargs):
    key = name.upper()

    if key not in _MEASUREMENTS:
        raise ValueError(f"Unknown measurement: {name}")

    rng = np.random.default_rng(seed)

    fn = _MEASUREMENTS[key]
    return fn(m, n, rng=rng, **kwargs)

def list_measurements():
    return list(_MEASUREMENTS.keys())



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

@register_measurement("BERNOULLI")
def bernoulli(m, n, rng=None, **kwargs):
    M = rng.binomial(1, 0.5, size=(m, n))
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms