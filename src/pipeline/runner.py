from datetime import datetime
from time import perf_counter
from dataclasses import replace
from socket import gethostname

from src.core.hsi import HSI
from src.core.training_signals import TrainingSignals
from src.core.results import CompressionRunResult, DictionaryTrainingResult, RunMetadata
from src.metrics.base import Metric, MetricResult
from src.compressors.base import Compressor
from src.dictionary_trainers.base import DictionaryTrainer

class Runner:

    def run_compression(self,
                        hsi: HSI, compressor: Compressor,
                        metrics: list[Metric],
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


        start = perf_counter()
        compressed = compressor.compress(hsi)
        compression_time = perf_counter() - start

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

        return result


    def run_dictionary_training(self,
                                signals: TrainingSignals,
                                trainer: DictionaryTrainer,
                                metrics: list[Metric],
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

        return result
        

