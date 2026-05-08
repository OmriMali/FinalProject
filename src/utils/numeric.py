import numpy as np



def normalize(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Normalize an array to the range [0, 1].

    The returned array is converted to ``float32``.
    Constant-valued arrays are mapped to zeros.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    tuple[np.ndarray, float, float]
        Tuple containing:

        - Normalized array with dtype ``float32``
        - Minimum value used for normalization
        - Maximum value used for normalization
    """

    a = a.astype(np.float32, copy=True)

    amin = float(a.min())
    amax = float(a.max())

    denom = amax - amin

    if denom == 0:
        return np.zeros_like(a), amin, amax

    return (a - amin) / denom, amin, amax

def denormalize(a: np.ndarray, amin: float, amax: float) -> np.ndarray:
    """
    Restore a normalized array from the range [0, 1]
    back to its original value range.

    Parameters
    ----------
    a : np.ndarray
        Normalized input array.

    amin : float
        Minimum value used during normalization.

    amax : float
        Maximum value used during normalization.

    Returns
    -------
    np.ndarray
        Denormalized array with dtype ``float32``.
    """
    a = a.astype(np.float32, copy=True)

    return a * (amax - amin) + amin


def quantize_symmetric(a: np.ndarray, bit_depth: int) -> tuple[np.ndarray, float]
    """
    Quantize an array using a symmetric range around zero.

    Values are mapped from ``[-amax, amax]`` to integer values in
    ``[0, 2**bit_depth - 1]``.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    bit_depth : int
        Number of bits used for quantization.

    Returns
    -------
    tuple[np.ndarray, float]
        Tuple containing:

        - Quantized array with dtype ``uint64``
        - Maximum absolute value used for quantization
    """
    if bit_depth <= 0:
        raise ValueError("Bit depth must be positive")

    amax = float(np.max(np.abs(a)))
    max_int = (1 << bit_depth) - 1

    if amax == 0:
        return np.zeros_like(a, dtype=np.uint64), amax

    q = np.round((a + amax) / (2 * amax) * max_int)
    q = np.clip(q, 0, max_int).astype(np.uint64)

    return q, amax

def dequantize_symmetric(q: np.ndarray, bit_depth: int, amax: float) -> np.ndarray:
    """
    Dequantize an array quantized with symmetric zero-centered scaling.

    Parameters
    ----------
    q : np.ndarray
        Quantized integer array.

    bit_depth : int
        Number of bits used for quantization.

    amax : float
        Maximum absolute value used during quantization.

    Returns
    -------
    np.ndarray
        Dequantized array with dtype ``float32``.
    """

    if bit_depth <= 0:
        raise ValueError("Bit depth must be positive")

    max_int = (1 << bit_depth) - 1
    q = q.astype(np.float32, copy=True)

    if amax == 0:
        return np.zeros_like(q, dtype=np.float32)

    return (q / max_int) * (2 * amax) - amax

