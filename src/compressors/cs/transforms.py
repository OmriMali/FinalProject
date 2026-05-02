import numpy as np
import scipy as sp
from src import util


# ===== Registry ===== #

TRANSFORMS, register_transform = util.make_registry()

def list_transforms():
    return list(TRANSFORMS.keys())

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

# ===== Public API ===== #

def get_transform(name, n):
    base_name, params = util.parse_config_string(name)

    try:
        fn = TRANSFORMS[base_name]
    except KeyError:
        raise ValueError(f"Unknown transform: {name}")
    return fn(n, **params)


