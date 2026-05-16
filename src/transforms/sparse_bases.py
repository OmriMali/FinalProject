import numpy as np
import scipy as sp
from typing import Callable, Dict

from src.io.dictionary import load_dictionary



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
    """
    Build a sparse basis matrix.

    Supports parameterized transforms such as:

    ``LEARNED:directory=...,name=...``
    """

    if ":" in name:
        base, spec = name.split(":", 1)
        key = base.upper()

        for item in spec.split(","):
            if "=" not in item:
                raise ValueError(
                    f"Invalid transform parameter: {item}"
                )

            k, v = item.split("=", 1)
            kwargs[k.strip()] = v.strip()

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
def learned_basis(n, directory: str | None = None, name: str | None = None, **kwargs):
    
    if directory is None:
        raise ValueError("LEARNED transform requires directory")

    if name is None:
        raise ValueError("LEARNED transform requires name")

    dictionary = load_dictionary(directory, name)
    D = dictionary.data

    if n != D.shape[0]:
        raise ValueError(
            f"Signal length {n} does not match dictionary "
            f"signal length {D.shape[0]}"
        )

    return D
