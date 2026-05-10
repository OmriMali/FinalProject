from datetime import datetime
from time import perf_counter
from dataclasses import replace
from socket import gethostname

from src.core.hsi import HSI
from src.core.training_signals import TrainingSignals
from src.core.results import CompressionRunResult, DictionaryTrainingResult, RunMetadata
from src.metrics.base import Metric, MetricResult
from src.metrics.compression import DEFAULT_COMRESSION_METRICS
from src.metrics.dictionary import DEFAULT_DICTIONARY_METRICS
from src.compressors.base import Compressor
from src.dictionary_trainers.base import DictionaryTrainer
from src.pipeline.status import RunStatus, RunEventType, StatusCallback

class Runner:
    """
    Orchestrates algorithm execution, timing, and metric evaluation.
    """
    def __init__(self, status_callback: StatusCallback | None = None):
        self.status_callback = status_callback
    
    def _emit_status(self, status: RunStatus) -> None:
        """
        Emit a status event to the configured status callback.
        """
        if self.status_callback is not None:
            self.status_callback(status)
    
    def _emit_message(self, stage: str, message: str) -> None:
        """
        Emit a message event.
        """
        self._emit_status(
            RunStatus(
                event_type=RunEventType.MESSAGE,
                stage=stage,
                message=message,
            )
        )

    def _start_progress(self, stage: str, message: str | None = None) -> None:
        """
        Emit a progress-start event.
        """
        self._emit_status(
            RunStatus(
                event_type=RunEventType.PROGRESS_START,
                stage=stage,
                message=message,
                progress=0.0,
            )
        )

    def _update_progress(self, stage: str, progress: float) -> None:
        """
        Emit a progress-update event.
        """
        self._emit_status(
            RunStatus(
                event_type=RunEventType.PROGRESS_UPDATE,
                stage=stage,
                progress=progress,
            )
        )

    def _end_progress(self, stage: str, message: str | None = None) -> None:
        """
        Emit a progress-end event.
        """
        self._emit_status(
            RunStatus(
                event_type=RunEventType.PROGRESS_END,
                stage=stage,
                message=message,
                progress=1.0,
            )
        )

    def _emit_done(self, message: str = "Run complete") -> None:
        """
        Emit a run-complete event.
        """
        self._emit_status(
            RunStatus(
                event_type=RunEventType.DONE,
                stage="done",
                message=message,
            )
        )

    def _emit_error(self, stage: str, message: str) -> None:
        """
        Emit an error event.
        """
        self._emit_status(
            RunStatus(
                event_type=RunEventType.ERROR,
                stage=stage,
                message=message,
            )
        )

    def _make_progress_callback(self, stage: str):
        """
        Create a float progress callback for an algorithm stage.
        """
        def callback(progress: float) -> None:
            self._update_progress(stage, progress)

        return callback


    def run_compression(self,
                        hsi: HSI, compressor: Compressor,
                        metrics: list[Metric] = DEFAULT_COMRESSION_METRICS,
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
            tags=tags or {}
        )

        self._start_progress("compression")
        compressor._progress_callback = self._make_progress_callback("compression")
        start = perf_counter()
        compressed = compressor.compress(hsi)
        compression_time = perf_counter() - start
        self._end_progress("compression")

        self._start_progress("decompression")
        compressor._progress_callback = self._make_progress_callback("decompression")
        start = perf_counter()
        reconstructed = compressor.decompress(compressed)
        decompression_time = perf_counter() - start
        self._end_progress("decompression")

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
        tags=tags or {}
        )

        self._start_progress("training")
        trainer._progress_callback = self._make_progress_callback("training")
        start = perf_counter()
        dictionary, coefficients = trainer.fit(signals)
        training_time = perf_counter() - start
        self._end_progress("training")

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

        return result
        

