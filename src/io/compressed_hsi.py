from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.core.hsi import CompressedHSI, HSIMetadata
from src.io.common import make_npz_path, resolve_npz_path


OBJECT_TYPE = "COMPRESSED_HSI"


def save_compressed_hsi(compressed: CompressedHSI, directory: str | Path, name: str) -> Path:
    """
    Save a compressed HSI to ``directory / f"{name}.npz"``.

    Parameters
    ----------
    compressed : CompressedHSI
        Compressed HSI object to save.

    directory : str | Path
        Output directory.

    name : str
        Output file name without the ``.npz`` suffix.
    """
    path = make_npz_path(directory, name)

    payload = asdict(compressed)

    np.savez(
        path,
        object_type=OBJECT_TYPE,
        payload=payload,
    )

    return path


def load_compressed_hsi(directory: str | Path, name: str) -> CompressedHSI:
    """
    Load a compressed HSI from ``directory / f"{name}.npz"``.

    Parameters
    ----------
    directory : str | Path
        Directory containing the compressed HSI file.

    name : str
        Compressed HSI file name without the ``.npz`` suffix.

    Returns
    -------
    CompressedHSI
        Loaded compressed HSI object.
    """
    path = resolve_npz_path(directory, name)

    return _load_single_compressed_hsi(path)


def _load_single_compressed_hsi(path: str | Path) -> CompressedHSI:
    """
    Load a single compressed HSI object from an ``.npz`` file.

    Parameters
    ----------
    path : str | Path
        Path to the compressed HSI file.

    Returns
    -------
    CompressedHSI
        Loaded compressed HSI object.

    Raises
    ------
    ValueError
        If the file does not contain a compressed HSI object.
    """
    file = np.load(path, allow_pickle=True)

    object_type = file["object_type"].item()

    if object_type != OBJECT_TYPE:
        raise ValueError(
            f"Expected object_type {OBJECT_TYPE}, got {object_type}"
        )

    payload = file["payload"].item()

    payload["metadata"] = HSIMetadata(**payload["metadata"])

    return CompressedHSI(**payload)