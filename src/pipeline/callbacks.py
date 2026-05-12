from src.core.hsi import HSI
from src.core.training_signals import TrainingSignals
from src.core.results import CompressionRunResult, DictionaryTrainingResult
from src.compressors.base import Compressor
from src.dictionary_trainers.base import DictionaryTrainer
from src.pipeline.progress import RunProgress


class RunnerCallback:
    """
    Base class for runner callbacks.

    Subclasses may override any method.
    """

    def on_compression_start(self, hsi: HSI, compressor: Compressor) -> None:
        pass

    def on_compression_end(self, result: CompressionRunResult) -> None:
        pass

    def on_dictionary_training_start(self, signals: TrainingSignals, trainer: DictionaryTrainer) -> None:
        pass

    def on_dictionary_training_end(self, result: DictionaryTrainingResult) -> None:
        pass

    def on_progress(self, progress: RunProgress) -> None:
        pass

    def on_error(self, error: Exception) -> None:
        pass