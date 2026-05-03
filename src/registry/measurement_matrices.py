from typing import Callable, Dict
import numpy as np

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