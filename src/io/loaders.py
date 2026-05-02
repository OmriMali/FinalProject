import numpy as np
import scipy as sp
from src.core.hsi import HSI


def load_array(path):
    """
    Load a NumPy array and optional metadata saved with `save_array_to_path`.

    Parameters
    ----------
    path : str
        Path to load.

    Returns
    -------
    arr : ndarray
        The main array.
    metadata : dict
        Metadata dictionary (empty if none saved).
    """
    with np.load(path, allow_pickle=True) as data:
        # Main array
        arr = data["array"]
        
        # Extract metadata
        metadata = {}
        for key in data.files:
            if key.startswith("meta_"):
                metadata[key[5:]] = data[key].item() if data[key].shape == () else data[key]
    
    return arr, metadata


def load_hsi(path: str) -> HSI:
    """
    Load an HSI object previously saved with `save_hsi`.
    """
    obj = np.load(path, allow_pickle=True).item()
    return HSI(
        data=obj["data"],
        wavelengths=obj["wavelengths"],
        dtype=obj["dtype"],
        metadata=obj["metadata"]
    )


def load_hsi_from_mat(mat_path: str, name: str, site: str, sensor: str) -> HSI:
    """
    Load an HSI object from a .mat file.
    """

    mat = sp.io.loadmat(mat_path)

    # Find the actual data key
    key = [k for k in mat.keys() if not k.startswith("__")][0]
    cube = mat[key]

    # ===== Wavelengths ===== #
    bands = cube.shape[2]

    # Option 1: simple approximation
    wavelengths = np.linspace(400, 2500, bands)

    # ===== Metadata ===== #
    metadata = {
        "name": name,
        "site": site,
        "sensor": sensor,
    }

    return HSI(
        data=cube,
        wavelengths=wavelengths,
        dtype=cube.dtype,
        metadata=metadata
    )


