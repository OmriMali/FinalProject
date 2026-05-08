import numpy as np
from dataclasses import dataclass

from src.core.hsi import HSI, CompressedHSI
from src.core.dictionary import Dictionary
from src.core.training_signals import TrainingSignals
from src.metrics.base import MetricResult

@dataclass(frozen=True)
class CompressionRunResult:
    """
    Complete result of a compression-decompression run.

    Stores the original hyperspectral image, its compressed
    representation, the reconstructed hyperspectral image,
    and all evaluated metrics for the run.

    Parameters
    ----------
    original : HSI
        Original reference hyperspectral image.

    compressed : CompressedHSI
        Compressed hyperspectral image representation.

    reconstructed : HSI
        Reconstructed hyperspectral image obtained after
        decompression.

    metrics : dict[str, MetricResult]
        Mapping of metric identifiers to computed metric
        results.
    """
    original: HSI
    compressed: CompressedHSI
    reconstructed: HSI

    metrics: dict[str, MetricResult]



@dataclass(frozen=True)
class DictionaryTrainingResult:
    """
    Complete result of a dictionary training run.

    Stores the training signals and their coefficients,
    the trained dictionary and all evaluated metrics for the run.

    Parameters
    ----------
    signals : TrainingSignals
        Signals used for dictionary training.

    coefficients : np.ndarray
        Signals representation in the dictionary domain.

    dictionary : Dictionary
        Trained dictionary.

    metrics : dict[str, MetricResult]
        Mapping of metric identifiers to computed metric results.
    """
    signals: TrainingSignals
    coefficients: np.ndarray
    dictionary: Dictionary

    metrics: dict[str, MetricResult]