import numpy as np

from src.core.training_signals import TrainingSignals
from src.core.dictionary import Axis
from src.core.hsi import HSIMetadata


def save_training_signals(signals: TrainingSignals, path: str) -> None:
    np.savez(
        path,
        data=signals.data,
        axis=signals.axis.value,
        sources=np.array(signals.sources, dtype=object),
        metadata=signals.metadata,
    )


def load_training_signals(path: str) -> TrainingSignals:
    file = np.load(path, allow_pickle=True)

    return TrainingSignals(
        data=file["data"],
        axis=Axis(int(file["axis"])),
        sources=list(file["sources"]),
        metadata=file["metadata"].item(),
    )
