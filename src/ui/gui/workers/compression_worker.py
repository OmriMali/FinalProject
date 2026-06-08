from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from src.ui.gui.controllers.compression_controller import CompressionController


class CompressionWorker(QObject):
    """
    Background worker for running one compression experiment.
    """

    progress_changed = Signal(float)
    finished = Signal(dict)
    failed = Signal(str)
    status_changed = Signal(str)

    def __init__(
        self,
        hsi_path: Path,
        compressor_name: str,
        config_values: dict[str, Any],
        experiment_settings: dict[str, Any],
    ):
        super().__init__()

        self.hsi_path = hsi_path
        self.compressor_name = compressor_name
        self.config_values = config_values
        self.experiment_settings = experiment_settings

        # Create controller inside the worker.
        # This avoids sharing runner/compressor state between threads.
        self.controller = CompressionController()

    @Slot()
    def run(self):
        """
        Run compression in the worker thread.
        """
        try:
            result = self.controller.run_compression(
                hsi_path=self.hsi_path,
                compressor_name=self.compressor_name,
                config_values=self.config_values,
                experiment_settings=self.experiment_settings,
                progress_callback=self.progress_changed.emit,
                status_callback=self.status_changed.emit,
            )

            self.finished.emit(result)

        except Exception as exc:
            self.failed.emit(str(exc))