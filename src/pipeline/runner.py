from datetime import datetime
from time import perf_counter
from dataclasses import replace
from socket import gethostname

from src.core.hsi import HSI
from src.core.training_signals import TrainingSignals
from src.core.results import CompressionRunResult, DictionaryTrainingResult, RunMetadata

from src.metrics.base import Metric, MetricResult
from src.metrics.compression import DEFAULT_COMPRESSION_METRICS
from src.metrics.dictionary import DEFAULT_DICTIONARY_METRICS

from src.compressors.base import Compressor
from src.dictionary_trainers.base import DictionaryTrainer

from src.pipeline.progress import RunProgress
from src.pipeline.callbacks import RunnerCallback
from src.pipeline.serialization import config_to_row



class Runner:
    """
    Orchestrates algorithm execution, timing, and metric evaluation.
    """
    def __init__(self, callbacks: list[RunnerCallback] | None = None):
        self.callbacks = callbacks or []
    
    def _notify_compression_start(self, hsi: HSI, compressor: Compressor) -> None:
        for callback in self.callbacks:
            callback.on_compression_start(hsi, compressor)
    
    def _notify_compression_end(self, result: CompressionRunResult) -> None:
        for callback in self.callbacks:
            callback.on_compression_end(result)

    def _notify_dictionary_training_start(self, signals: TrainingSignals, trainer: DictionaryTrainer) -> None:
        for callback in self.callbacks:
            callback.on_dictionary_training_start(signals, trainer)

    def _notify_dictionary_training_end(self, result: DictionaryTrainingResult) -> None:
        for callback in self.callbacks:
            callback.on_dictionary_training_end(result)

    def _notify_progress(self, stage: str, value: float, message: str | None = None) -> None:
        
        value = max(0.0, min(1.0, value))
        progress = RunProgress(stage, value, message)

        for callback in self.callbacks:
            callback.on_progress(progress)

    def _notify_error(self, error: Exception) -> None:
        for callback in self.callbacks:
            callback.on_error(error)

    def _make_progress_callback(self, stage: str, message: str | None = None):
        def callback(value: float) -> None:
            self._notify_progress(stage, value, message)

        return callback


    def run_compression(self,
                        hsi: HSI, compressor: Compressor,
                        metrics: list[Metric] = DEFAULT_COMPRESSION_METRICS,
                        tags: dict | None = None,
                        ) -> CompressionRunResult:
        """
        Run a complete compression-decompression experiment.

        Parameters
        ----------
        hsi : HSI
            Hyperspectral image to compress.

        compressor : Compressor
            Compressor instance used for compression and decompression.

        metrics : list[Metric]
            Metrics to compute after reconstruction.

        tags : dict | None, optional
            Additional user-defined run tags.

        Returns
        -------
        CompressionRunResult
            Complete compression run result.
        """
        run_metadata = RunMetadata(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            machine=gethostname(),
            algorithm_name=compressor.name,
            algorithm_config=config_to_row(compressor.config),
            tags=tags or {}
        )

        self._notify_compression_start(hsi, compressor)

        compressor._progress_callback = self._make_progress_callback("compression")
        start = perf_counter()
        compressed = compressor.compress(hsi)
        compression_time = perf_counter() - start

        compressor._progress_callback = self._make_progress_callback("decompression")
        start = perf_counter()
        reconstructed = compressor.decompress(compressed)
        decompression_time = perf_counter() - start

        partial = CompressionRunResult(
            original=hsi,
            compressed=compressed,
            reconstructed=reconstructed,
            run_metadata=run_metadata,
        )

        computed_metrics = {
            metric.short_name: metric.compute(partial)
            for metric in metrics
        }

        computed_metrics["COMP_TIME"] = MetricResult(name="Compression Time",
                                                     short_name="COMP_TIME",
                                                     value=float(compression_time),
                                                     unit="s")
        
        computed_metrics["DECOMP_TIME"] = MetricResult(name="Decompression Time",
                                                        short_name="DECOMP_TIME",
                                                        value=float(decompression_time),
                                                        unit="s")
        
        result = replace(partial, metrics=computed_metrics)

        self._notify_compression_end(result)

        return result


    def run_dictionary_training(self,
                                signals: TrainingSignals,
                                trainer: DictionaryTrainer,
                                metrics: list[Metric] = DEFAULT_DICTIONARY_METRICS,
                                tags: dict | None = None,
                                ) -> DictionaryTrainingResult:
        """
        Run a dictionary training experiment.

        Parameters
        ----------
        signals : TrainingSignals
            Training signals for the dictionary.

        trainer : DictionaryTrainer
            Trainer used to the experiment.

        metrics : list[Metric]
            Metrics to compute after dictionary created.

        tags : dict | None, optional
            Additional user-defined run tags.

        Returns
        -------
        DictionaryTrainingResult
            Complete dictionary training result.
        """
        run_metadata = RunMetadata(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        machine=gethostname(),
        algorithm_name=trainer.name,
        algorithm_config=config_to_row(trainer.config),
        tags=tags or {}
        )
        
        self._notify_dictionary_training_start(signals, trainer)

        trainer._progress_callback = self._make_progress_callback("training")
        start = perf_counter()
        dictionary, coefficients = trainer.fit(signals)
        training_time = perf_counter() - start

        partial = DictionaryTrainingResult(signals,
                                           coefficients,
                                           dictionary,
                                           run_metadata,)
        
        computed_metrics = {
            metric.short_name: metric.compute(partial)
            for metric in metrics
        }

        computed_metrics["TRAIN_TIME"] = MetricResult(name="Training Time",
                                                     short_name="TRAIN_TIME",
                                                     value=float(training_time),
                                                     unit="s")
        
        result = replace(partial, metrics=computed_metrics)

        self._notify_dictionary_training_end(result)

        return result
        

