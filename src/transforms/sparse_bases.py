import numpy as np
import scipy as sp
from typing import Callable, Dict

from src.io.loaders import load_array



_SPARSE_BASES: Dict[str, Callable] = {}

def register_sparse_base(name: str):
    def decorator(fn):
        key = name.upper()

        if key in _SPARSE_BASES:
            raise ValueError(f"Transform '{name}' already registered")

        _SPARSE_BASES[key] = fn
        return fn

    return decorator

def get_sparse_base(name: str, n: int, **kwargs):
    if name.upper().startswith("LEARNED"):
        base = "LEARNED"

        if ":" in name:
            _, spec = name.split(":", 1)

            if spec.startswith("path="):
                kwargs["path"] = spec.split("=", 1)[1]

        key = base
    else:
        key = name.upper()

    if key not in _SPARSE_BASES:
        raise ValueError(f"Unknown transform: {name}")

    fn = _SPARSE_BASES[key]

    return fn(n, **kwargs)

def list_sparse_base():
    return list(_SPARSE_BASES.keys())



@register_sparse_base("IDENTITY")
def identity_basis(n, **kwargs):
    return np.eye(n)

@register_sparse_base("DCT")
def dct_basis(n, **kwargs):
    return sp.fft.dct(np.eye(n), axis=0, norm='ortho')

@register_sparse_base("IDCT")
def idct_basis(n, **kwargs):
    return sp.fft.idct(np.eye(n), axis=0, norm='ortho')

@register_sparse_base("LEARNED")
def learned_basis(n, path=None, **kwargs):
    
    if path is None:
        raise ValueError("LEARNED transform requires path")

    D, _ = load_array(path)

    if n != D.shape[0]:
        raise ValueError(f"Signal length {n} does not match dictionary signals of length {D.shape[0]}")
    
    return D
