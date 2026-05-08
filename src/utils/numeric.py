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

    a = a.astype(np.float32)

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

    a = a.astype(np.float32)

    return a * (amax - amin) + amin