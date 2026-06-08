from typing import Callable

from src.pipeline.callbacks import RunnerCallback
from src.pipeline.progress import RunProgress
from src.compressors.base import Compressor
from src.core.hsi import HSI
from src.core.results import CompressionRunResult


class GuiRunnerCallback(RunnerCallback):
    """
    Runner callback that forwards runner events to the Qt GUI.

    This class must not directly access GUI widgets. It only calls
    thread-safe emit functions supplied by the worker.
    """

    def __init__(
        self,
        progress_callback: Callable[[float], None],
        status_callback: Callable[[str], None] | None = None,
    ):
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def on_compression_start(self, hsi: HSI, compressor: Compressor):
        if self.status_callback is not None:
            self.status_callback(
                f"Running {compressor.name} on {hsi.metadata.scene_name or 'HSI'}"
            )

        self.progress_callback(0.0)

    def on_compression_end(self, result: CompressionRunResult):
        self.progress_callback(1.0)

        if self.status_callback is not None:
            self.status_callback("Finished")

    def on_progress(self, progress: RunProgress):
        value = max(0.0, min(1.0, progress.value))
        self.progress_callback(value)

        if self.status_callback is not None:
            message = progress.message or progress.stage
            self.status_callback(f"{message}: {value * 100:.0f}%")

    def on_error(self, error: Exception):
        if self.status_callback is not None:
            self.status_callback(f"Error: {error}")