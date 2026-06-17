from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from src.core.hsi import HSI
from src.ui.gui.models.workspace_item import WorkspaceItem
from src.ui.gui.services.compression_service import CompressionService
from src.ui.gui.services.workspace_loader import (
    WorkspaceLoader,
    WorkspaceLoadError,
)


class CompressionWorker(QObject):
    """
    Background worker for compression workflows.
    """

    progress_changed = Signal(float)
    progress_message_changed = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        source_item: WorkspaceItem,
        compressor_name: str,
        config_values: dict[str, Any],
        experiment_settings: dict[str, Any],
    ):
        super().__init__()

        self.source_item = source_item
        self.compressor_name = compressor_name
        self.config_values = config_values
        self.experiment_settings = experiment_settings

        self.workspace_loader = WorkspaceLoader()
        self.compression_service = CompressionService()

    @Slot()
    def run(self):
        try:
            self.progress_message_changed.emit("Loading HSI")
            self.progress_changed.emit(0.0)

            obj = self.workspace_loader.load_object(self.source_item)

            self.progress_changed.emit(0.02)

            if not isinstance(obj, HSI):
                raise WorkspaceLoadError(
                    "Compress + Decompress requires an HSI item"
                )

            self.progress_message_changed.emit("Starting compression")

            gui_result = self.compression_service.compress_and_decompress(
                hsi=obj,
                source_item=self.source_item,
                compressor_name=self.compressor_name,
                config_values=self.config_values,
                experiment_settings=self.experiment_settings,
                progress_callback=self.progress_changed.emit,
                message_callback=self.progress_message_changed.emit,
            )

            self.progress_changed.emit(1.0)
            self.progress_message_changed.emit("Finished")
            self.finished.emit(gui_result)

        except Exception as exc:
            self.progress_message_changed.emit("Failed")
            self.failed.emit(str(exc))