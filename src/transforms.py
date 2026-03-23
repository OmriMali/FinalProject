import numpy as np
import scipy as sp
from src import util

def identity_basis(n):
    return np.eye(n)

def dct_basis(n):
    return sp.fft.dct(np.eye(n), axis=0, norm='ortho')

def idct_basis(n):
    return sp.fft.idct(np.eye(n), axis=0, norm='ortho')

def learned_basis(n, path):
    D, _ = util.load_array_from_path(path)
    if n != D.shape[0]:
        raise ValueError(f"Signal length {n} does not match dictionary signals of length {D.shape[0]}")
    return D

TRANSFORMS = {
    "IDENTITY": identity_basis,
    "DCT": dct_basis,
    "IDCT": idct_basis,
}

def get_transform(name, n):
    if name.startswith("LEARNED:"):
        path = name.split(":", 1)[1]
        name = "LEARNED"
        return learned_basis(n, path)
    try:
        return TRANSFORMS[name](n)
    except KeyError:
        raise ValueError(f"Unknown transform: {name}")