from tqdm import tqdm

from src.pipeline.callbacks import RunnerCallback
from src.pipeline.progress import RunProgress
from src.compressors.base import Compressor
from src.core.hsi import HSI
from src.core.results import CompressionRunResult, DictionaryTrainingResult
from src.core.training_signals import TrainingSignals
from src.dictionary_trainers.base import DictionaryTrainer


class ConsoleCallback(RunnerCallback):
    def __init__(self):
        self._current_stage: str | None = None
        self._bar: tqdm | None = None

    def on_compression_start(self, hsi: HSI, compressor: Compressor):
        metadata = hsi.metadata

        print()
        print("=" * 60)
        print("COMPRESSION RUN")
        print("=" * 60)

        if metadata.scene_name is not None:
            print(f"Scene:\t\t{metadata.scene_name}")

        if metadata.section_row is not None:
            print(f"Section:\trow {metadata.section_row}, col {metadata.section_col}")

        if metadata.sensor is not None:
            print(f"Sensor:\t\t{metadata.sensor}")

        print(f"Shape:\t\t({metadata.shape[0]},{metadata.shape[1]},{metadata.shape[2]})")
        print(f"Compressor:\t{compressor.name}")
        print("=" * 60)
        print()

    def on_compression_end(self, result: CompressionRunResult):
        """
        Print compression run metrics.
        """
        print()
        print("=" * 60)
        print("COMPRESSION RESULTS")
        print("=" * 60)

        for metric in result.metrics.values():
            value = f"{metric.value:.4f}"

            if metric.unit is not None:
                value += f"\t\t[{metric.unit}]"

            print(f"{metric.name:<25}\t{value}")

        print("=" * 60)
        print()

    def on_dictionary_training_start(self, signals: TrainingSignals, trainer: DictionaryTrainer):
        """
        Print a short summary before a dictionary training.
        """
        print()
        print("=" * 60)
        print("DICTIONARY TRAINING")
        print("=" * 60)

        print(f"Num of Signals:\t{signals.num_signals}")
        print(f"Num of Atoms:\t{trainer.config.K}")
        print(f"Aimed Sparsity:\t{trainer.config.T_0}")
        print(f"Axis:\t\t{signals.axis.name}")
        print("=" * 60)
        print()

    def on_dictionary_training_end(self, result: DictionaryTrainingResult):
        """
        Print Dictionary training results.
        """
        print()
        print("=" * 60)
        print("DICTIONARY TRAINING RESULTS")
        print("=" * 60)

        for metric in result.metrics.values():
            value = f"{metric.value:.4f}"

            if metric.unit is not None:
                value += f"\t\t[{metric.unit}]"

            print(f"{metric.name:<25}\t{value}")

        print("=" * 60)
        print()

    def on_progress(self, progress: RunProgress):
        """
        Update console progress display.

        A new progress bar is created whenever the runner switches
        to a new stage, such as compression, decompression, or training.
        """
        if progress.stage != self._current_stage:
            self._close_progress_bar()

            self._current_stage = progress.stage
            self._bar = tqdm(
                total=100,
                desc=progress.message or progress.stage,
                unit="%",
            )

        if self._bar is None:
            return

        target = int(progress.value * 100)
        delta = target - self._bar.n

        if delta > 0:
            self._bar.update(delta)

        if progress.value >= 1.0:
            self._close_progress_bar()

    def on_error(self, error: Exception):
        self._close_progress_bar()
        print(f"ERROR: {error}")

    def _close_progress_bar(self):
        """
        Close the active progress bar.
        """
        if self._bar is not None:
            remaining = 100 - self._bar.n

            if remaining > 0:
                self._bar.update(remaining)

            self._bar.close()
            self._bar = None

        self._current_stage = None