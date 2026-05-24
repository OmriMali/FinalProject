from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.core.dictionary import Axis
from src.core.hsi import HSIMetadata
from src.core.training_signals import TrainingSignals
from src.io.common import make_npz_path, resolve_npz_path, list_npz_files


OBJECT_TYPE = "TRAINING_SIGNALS"


def save_training_signals(signals: TrainingSignals, directory: str | Path, name: str) -> None:
    """
    Save training signals to ``directory / f"{name}.npz"``.
    """
    path = make_npz_path(directory, name)

    payload = asdict(signals)
    payload["axis"] = signals.axis.value

    np.savez(
        path,
        object_type=OBJECT_TYPE,
        payload=payload,
    )


def load_training_signals(directory: str | Path, name: str) -> TrainingSignals:
    """
    Load a single training signals object from
    ``directory / f"{name}.npz"``.
    """
    path = resolve_npz_path(directory, name)

    return _load_single_training_signals(path)


def load_many_training_signals(directory: str | Path) -> list[TrainingSignals]:
    """
    Load all training signals objects from a directory.
    """
    signals_list = []

    for path in list_npz_files(directory):
        try:
            signals_list.append(_load_single_training_signals(path))
        except ValueError:
            continue

    return signals_list


def _load_single_training_signals(path: str | Path) -> TrainingSignals:
    """
    Load a single training signals object from an ``.npz`` file.
    """
    file = np.load(path, allow_pickle=True)

    object_type = file["object_type"].item()

    if object_type != OBJECT_TYPE:
        raise ValueError(
            f"Expected object_type {OBJECT_TYPE}, got {object_type}"
        )

    payload = file["payload"].item()

    payload["axis"] = Axis(payload["axis"])

    payload["sources"] = [
        HSIMetadata(**source)
        for source in payload["sources"]
    ]

    