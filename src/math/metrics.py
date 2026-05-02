import numpy as np
from src.core.hsi import HSI


def calc_rmse(reference, target):
    """
    Calculate the Root Mean Square Error (RMSE) between two arrays.

    Parameters
    ----------
    reference : numpy.ndarray
        The original reference hyperspectral cube.
    target : numpy.ndarray
        The reconstructed or processed hyperspectral cube.

    Returns
    -------
    float
        The RMSE value in the original units of the data.
    """
    mse = np.mean((reference.astype(np.float64) - target.astype(np.float64)) ** 2)
    return np.sqrt(mse)

def calc_psnr(reference, target, bit_depth):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    This implementation uses the formula: 20 * log10(MAX_I / RMSE).

    Parameters
    ----------
    reference : numpy.ndarray
        The original ground truth HSI cube.
    target : numpy.ndarray
        The reconstructed HSI cube.
    bit_depth : int
        The bit depth (D) used to calculate the dynamic range (2^D - 1).

    Returns
    -------
    float
        The PSNR value in dB. Returns infinity if the images are identical.
    """
    max_i = float((1 << bit_depth) - 1)
    rmse_val = calc_rmse(reference, target)
    
    if rmse_val == 0:
        return float('inf')
        
    return 20 * np.log10(max_i / rmse_val)

def calc_sam(reference, target):
    """
    Calculate the Mean Spectral Angle Mapper (SAM) in degrees.
    
    SAM measures spectral similarity by calculating the angle between 
    spectral vectors. It is inherently scale-invariant.

    Parameters
    ----------
    reference : numpy.ndarray
        The original HSI cube, where the last dimension is the spectral axis.
    target : numpy.ndarray
        The reconstructed HSI cube.

    Returns
    -------
    float
        The mean spectral angle in degrees.
    """
    ref = reference.astype(np.float64)
    tgt = target.astype(np.float64)

    dot_product = np.sum(ref * tgt, axis=-1)
    norm_ref = np.linalg.norm(ref, axis=-1)
    norm_tgt = np.linalg.norm(tgt, axis=-1)

    # Calculate cosine similarity with epsilon to avoid division by zero
    cos_theta = dot_product / (norm_ref * norm_tgt + 1e-15)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angles_rad = np.arccos(cos_theta)
    return np.degrees(np.mean(angles_rad))

def calc_compression_ratio(original_cube, bitstream, bit_depth):
    """
    Calculate the Compression Ratio (CR) of the bitstream.

    Parameters
    ----------
    original_cube : numpy.ndarray
        The original HSI data.
    bitstream : bytes or list
        The compressed representation (bytes or bit list).
    bit_depth : int
        The number of bits used to represent the original pixel values.

    Returns
    -------
    float
        The compression ratio (Original Size / Compressed Size).
    """
    total_pixels = original_cube.size
    original_bits = total_pixels * bit_depth
    
    if isinstance(bitstream, (bytes, bytearray)):
        compressed_bits = len(bitstream) * 8
    else:
        compressed_bits = len(bitstream)

    return original_bits / compressed_bits

def compute_all_metrics(reference: HSI, target: HSI, bitstream):
    """
    Compute a comprehensive set of HSI performance metrics.

    Parameters
    ----------
    reference : HSI
        The ground truth HSI.
    target : HSI
        The reconstructed HSI.
    bitstream : bytes or list
        The resulting compressed bitstream.

    Returns
    -------
    dict
        A dictionary containing RMSE, PSNR, SAM, and CR.
    """
    ref_arr = reference.data
    target_arr = target.data
    bitdepth = reference.bitdepth

    return {
        "rmse": calc_rmse(ref_arr, target_arr),
        "psnr": calc_psnr(ref_arr, target_arr, bitdepth),
        "sam": calc_sam(ref_arr, target_arr),
        "cr": calc_compression_ratio(ref_arr, bitstream, bitdepth),
    }
