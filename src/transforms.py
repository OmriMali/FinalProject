import numpy as np
import scipy as sp


def identity_basis(n, inverse=False):
    if inverse:
        return np.eye(n)
    return np.eye(n)

def dct_basis(n, inverse=False):
    if inverse:
        return sp.fft.idct(np.eye(n), axis=0, norm='ortho')
    return sp.fft.dct(np.eye(n), axis=0, norm='ortho')

TRANSFORMS = {
    "IDENTITIY": identity_basis,
    "DCT": dct_basis
}

def get_transform(name, n):
    try:
        return TRANSFORMS[name](n)
    except KeyError:
        raise ValueError(f"Unknown transform: {name}")

def get_inverse_transform(name, n):
    try:
        return TRANSFORMS[name](n, True)
    except KeyError:
        raise ValueError(f"Unknown transform: {name}")