from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.core.hsi import HSI, HSIMetadata
from src.io.common import make_npz_path, resolve_npz_path, list_npz_files


OBJECT_TYPE = "HSI"


def save_hsi(hsi: HSI, directory: str | Path, name: str) -> None:
    """
    Save an HSI object to ``directory / f"{name}.npz"``.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image to save.

    directory : str | Path
        Output directory.

    name : str
        Output file name without the ``.npz`` suffix.
    """
    path = make_npz_path(directory, name)

    np.savez(
        path,
        object_type=OBJECT_TYPE,
        data=hsi.data,
        metadata=asdict(hsi.metadata),
    )


def load_hsi(directory: str | Path, name: str) -> HSI:
    """
    Load a single HSI object from ``directory / f"{name}.npz"``.

    Parameters
    ----------
    directory : str | Path
        Directory containing the HSI file.

    name : str
        HSI file name without the ``.npz`` suffix.

    Returns
    -------
    HSI
        Loaded hyperspectral image.
    """
    path = resolve_npz_path(directory, name)

    return _load_single_hsi(path)


def load_many_hsi(directory: str | Path) -> list[HSI]:
    """
    Load all HSI objects from a directory.

    Only files whose ``object_type`` is ``"HSI"`` are loaded. Other
    ``.npz`` files are ignored.

    Parameters
    ----------
    directory : str | Path
        Directory containing HSI files.

    Returns
    -------
    list[HSI]
        Loaded hyperspectral images.
    """
    hsis = []

    for path in list_npz_files(directory):
        try:
            hsis.append(_load_single_hsi(path))
        except ValueError:
            continue

    return hsis


def _load_single_hsi(path: str | Path) -> HSI:
    """
    Load a single HSI object from an ``.npz`` file.

    Parameters
    ----------
    path : str | Path
        Path to the HSI file.

    Returns
    -------
    HSI
        Loaded hyperspectral image.

    Raises
    ------
    ValueError
        If the file does not contain an HSI object.
    """
    file = np.load(path, allow_pickle=True)

    object_type = file["object_type"].item()

    if object_type != OBJECT_TYPE:
        raise ValueError(
            f"Expected object_type {OBJECT_TYPE}, got {object_type}"
        )

    metadata_dict = file["metadata"].item()
    metadata = HSIMetadata(**metadata_dict)

    return HSI(
        data=file["data"],
        metadata=metadata,
    )