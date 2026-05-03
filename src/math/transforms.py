import numpy as np
import scipy as sp

from src.registry.transforms import register_transform
from src.io.loaders import load_array


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
        raise ValueError("LEARNED transform requires path")

    D, _ = load_array(path)

    if n != D.shape[0]:
        raise ValueError(f"Signal length {n} does not match dictionary signals of length {D.shape[0]}")
    
    return D
