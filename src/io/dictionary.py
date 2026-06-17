from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.core.dictionary import Dictionary, Axis
from src.io.common import make_npz_path, resolve_npz_path


OBJECT_TYPE = "DICTIONARY"


def save_dictionary(dictionary: Dictionary, directory: str | Path, name: str) -> Path:
    """
    Save a dictionary object to ``directory / f"{name}.npz"``.

    Parameters
    ----------
    dictionary : Dictionary
        Dictionary object to save.

    directory : str | Path
        Output directory.

    name : str
        Output file name without the ``.npz`` suffix.
    """

    path = make_npz_path(directory, name)

    dictionary_dict = asdict(dictionary)

    # Store enums as serializable values
    dictionary_dict["axis"] = dictionary.axis.value

    np.savez(
        path,
        object_type=OBJECT_TYPE,
        dictionary=dictionary_dict,
    )

    return path


def load_dictionary(directory: str | Path, name: str) -> Dictionary:
    """
    Load a single dictionary object from
    ``directory / f"{name}.npz"``.

    Parameters
    ----------
    directory : str | Path
        Directory containing the dictionary file.

    name : str
        Dictionary file name without the ``.npz`` suffix.

    Returns
    -------
    Dictionary
        Loaded dictionary object.
    """

    path = resolve_npz_path(directory, name)

    return _load_single_dictionary(path)


def _load_single_dictionary(path: str | Path) -> Dictionary:
    """
    Load a single dictionary object from an ``.npz`` file.

    Parameters
    ----------
    path : str | Path
        Path to the dictionary file.

    Returns
    -------
    Dictionary
        Loaded dictionary object.

    Raises
    ------
    ValueError
        If the file does not contain a dictionary object.
    """

    file = np.load(path, allow_pickle=True)

    object_type = file["object_type"].item()

    if object_type != OBJECT_TYPE:
        raise ValueError(
            f"Expected object_type {OBJECT_TYPE}, "
            f"got {object_type}"
        )

    dictionary_dict = file["dictionary"].item()

    # Restore enums
    dictionary_dict["axis"] = Axis(dictionary_dict["axis"])

    return Dictionary(**dictionary_dict)