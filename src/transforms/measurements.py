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
    """
    Build a measurement matrix.

    Supports parameters such as:

    ``BERNOULLI:p=...``
    """
    if ":" in name:
        meas, spec = name.split(":", 1)
        key = meas.upper()

        for item in spec.split(","):
            if "=" not in item:
                raise ValueError(
                    f"Invalid parameter: {item}"
                )

            k, v = item.split("=", 1)
            kwargs[k.strip()] = v.strip()

    else:
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
def bernoulli(m, n, rng=None, p: str | None = None, **kwargs):
    if p is None:
        p = 0.1
    else:
        p = float(p)
        if p >= 1 or p <= 0:
            raise ValueError("p must be between 0 and 1")
    
    M = rng.binomial(1, p, size=(m, n))
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms