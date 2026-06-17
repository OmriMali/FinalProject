import numpy as np
from dataclasses import dataclass, field

from src.core.hsi import HSIMetadata
from src.core.dictionary import Axis



@dataclass(frozen=True)
class TrainingSignals:
    """
    Signals used for dictionary training.

    Parameters
    ----------
    data : np.ndarray
        Signal matrix with shape (signal_length, num_signals).

    axis : Axis
        HSI axis the signals were extracted from.

    sources : list[HSIMetadata]
        Metadata of the hyperspectral images the signals were extracted from.

    metadata : dict, optional
        Additional extraction-specific metadata.
    """
    data: np.ndarray
    axis: Axis
    sources: list[HSIMetadata]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Validate object consistency after initialization.
        """
        if self.data.ndim != 2:
            raise ValueError(
                "Training signals must have shape "
                "(signal_length, num_signals)"
            )

    @property
    def signal_length(self) -> int:
        """
        int : Length of each training signal.
        """
        return self.data.shape[0]

    @property
    def num_signals(self) -> int:
        """
        int : Number of training signals.
        """
        return self.data.shape[1]