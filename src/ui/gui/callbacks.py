from __future__ import annotations

from typing import Callable

from src.compressors.base import Compressor
from src.core.hsi import HSI
from src.core.results import CompressionRunResult
from src.pipeline.callbacks import RunnerCallback
from src.pipeline.progress import RunProgress


class GuiRunnerCallback(RunnerCallback):
    """
    Forward runner progress events to the GUI.
    """

    def __init__(
        self,
        progress_callback: Callable[[float], None] | None = None,
        message_callback: Callable[[str], None] | None = None,
    ):
        self.progress_callback = progress_callback
        self.message_callback = message_callback

    def on_compression_start(
        self,
        hsi: HSI,
        compressor: Compressor,
    ) -> None:
        self._emit_message(f"Running {compressor.name}")
        self._emit_progress(0.0)

    def on_progress(self, progress: RunProgress) -> None:
        self._emit_progress(progress.value)
        self._emit_message(self._progress_message(progress))

    def on_compression_end(
        self,
        result: CompressionRunResult,
    ) -> None:
        self._emit_progress(1.0)
        self._emit_message("Finished")

    def on_error(self, error: Exception) -> None:
        self._emit_message(f"Error: {error}")

    def _emit_progress(self, value: float) -> None:
        if self.progress_callback is None:
            return

        value = max(0.0, min(1.0, value))
        self.progress_callback(value)

    def _emit_message(self, message: str) -> None:
        if self.message_callback is None:
            return

        self.message_callback(message)

    def _progress_message(self, progress: RunProgress) -> str:
        message = getattr(progress, "message", None)
        stage = getattr(progress, "stage", None)

        if message:
            return str(message)

        if stage:
            return str(stage)

        return "Running"