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


def calc_compression_ratio(original, bitstring):
    
    original_bits_per_element = np.floor(np.log2(original + 1)).astype(int) + 1
    original_total_bits = np.sum(original_bits_per_element)
    bitstream_total_bits = len(bitstring)

    ratio = original_total_bits / bitstream_total_bits

    return ratio