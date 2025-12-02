import numpy as np

def calc_RMSE(original, reconstruction):
    
    shape = original.shape
    factor = 1
    for dim in shape:
        factor *= dim
    
    return np.sqrt(np.sum(np.pow(original - reconstruction, 2)) / factor)


def calc_SAM(original, reconstruction):

    # dot product for each pixel (vector over Nz)
    dots = np.sum(original * reconstruction, axis=-1)

    # norms
    norms = np.linalg.norm(original, axis=-1)
    norms_hat = np.linalg.norm(reconstruction, axis=-1)

    # cosine
    cosines = dots / (norms * norms_hat)
    cosines = np.clip(cosines, -1.0, 1.0)

    # SAM per pixel
    angles = np.arccos(cosines)

    # average SAM
    return np.mean(angles)