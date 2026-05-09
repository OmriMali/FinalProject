import numpy as np

from src.core.dictionary import Dictionary, Axis


def save_dictionary(dictionary: Dictionary, path: str) -> None:
    np.savez(
        path,
        data=dictionary.data,
        axis=dictionary.axis.value,
        name=dictionary.name,
    )


def load_dictionary(path: str) -> Dictionary:
    file = np.load(path, allow_pickle=True)

    return Dictionary(
        data=file["data"],
        axis=Axis(int(file["axis"])),
        name=file["name"].item(),
    )


