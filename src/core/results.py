import numpy as np
from dataclasses import dataclass, field

from src.core.hsi import HSI, CompressedHSI
from src.core.dictionary import Dictionary
from src.core.training_signals import TrainingSignals
from src.metrics.base import MetricResult


@dataclass(frozen=True)
class RunMetadata:
    """
    Metadata describing an algorithm run.

    Parameters
    ----------
    timestamp : str
        Time when the run was created.
    
    experiment : str
        Identifier for the experiment.

    machine : str | None
        Identifier for the computer used for the run.

    algorithm_name : str | None
        Name of the algorithm used in the run.

    algorithm_config : dict
        Flat dictionary of algorithm configuration parameters.

    artifact_dir : str | None
        Location of run artifacts, such as the reconstructed HSI.    
    
    tags: dict | None
        Additional run specific info.
    """
    timestamp: str
    experiment: str
    machine: str | None = None
    algorithm_name: str | None = None
    algorithm_config: dict = field(default_factory=dict)
    artifact_dir: str | None = None
    tags: dict = field(default_factory=dict)


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

    run_metadata : RunMetadata
        Metadata describing the algorithm run.

    metrics : dict[str, MetricResult]
        Mapping of metric identifiers to computed metric
        results.
    """
    original: HSI
    compressed: CompressedHSI
    reconstructed: HSI

    run_metadata: RunMetadata

    metrics: dict[str, MetricResult] = field(default_factory=dict)


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

    run_metadata : RunMetadata
        Metadata describing the algorithm run.    
    
    metrics : dict[str, MetricResult]
        Mapping of metric identifiers to computed metric results.    
    """
    signals: TrainingSignals
    coefficients: np.ndarray
    dictionary: Dictionary

    run_metadata: RunMetadata

    metrics: dict[str, MetricResult] = field(default_factory=dict)

