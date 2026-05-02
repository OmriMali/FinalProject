import numpy as np
import scipy as sp
from src.core.registry import make_registry


# ===== Registry ===== #

TRANSFORMS, register_transform, get_transform, list_transforms = make_registry(configurable=True)

# ===== Implementations ===== #

@register_transform("IDENTITY")
def identity_basis(n, **kwargs):
    return np.eye(n)

@register_transform("DCT")
def dct_basis(n, **kwargs):
    return sp.fft.dct(np.eye(n), axis=0, norm='ortho')

@register_transform("IDCT")
def idct_basis(n, **kwargs):
    return sp.fft.idct(np.eye(n), axis=0, norm='ortho')

@register_transform("LEARNED")
def learned_basis(n, path=None, **kwargs):
    if path is None:
        raise ValueError("LEARNED transform requires 'path'")

    D, _ = util.load_array_from_path(path)

    if n != D.shape[0]:
        raise ValueError(f"Signal length {n} does not match dictionary signals of length {D.shape[0]}")
    
    return D
