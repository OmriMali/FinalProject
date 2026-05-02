import os
import numpy as np
from src.core.hsi import HSI


def save_array_to_path(arr, path, metadata=None):
    """
    Save a NumPy array to a file, optionally with metadata.

    Parameters
    ----------
    arr : ndarray
        Array to save.
    path : str
        File path to save the array to.
    metadata : dict, optional
        Dictionary of metadata to save alongside the array.
    """
    # Prepare save dictionary
    save_dict = {"array": arr}
    if metadata is not None:
        for key, value in metadata.items():
            save_dict[f"meta_{key}"] = value
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save as npz
    np.savez_compressed(path, **save_dict)


def save_hsi(hsi: HSI, path: str):
    """
    Save an HSI object to a file.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image object to save.
    path : str
        Path to save the file. Should end with `.npy`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    np.save(path, {
        "data": hsi.data,
        "wavelengths": hsi.wavelengths,
        "metadata": hsi.metadata,
        "dtype": hsi.dtype
    })


def save_bitstream(bitstream, path: str):
    """
    Save bitstream in appropriate format.
    """
    if isinstance(bitstream, bytes):
        with open(path + ".bin", "wb") as f:
            f.write(bitstream)

    elif isinstance(bitstream, np.ndarray):
        np.save(path + ".npy", bitstream)

    else:
        # fallback
        with open(path + ".pkl", "wb") as f:
            pickle.dump(bitstream, f)