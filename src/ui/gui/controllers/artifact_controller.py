from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from src.ui.gui.models import WorkspaceItem, WorkspaceItemType
from src.ui.gui.services import WorkspaceLoader, WorkspaceLoadError
from src.ui.gui.services.metrics_extractor import LoadedMetric


class ArtifactController(QObject):
    """
    Converts compression process output payloads into WorkspaceItems.

    Owns:
    - reading compressed/reconstructed artifact paths
    - loading artifacts through WorkspaceLoader
    - assigning type/method/metrics
    - deserializing process metrics
    """

    item_ready = Signal(object)
    metrics_item_ready = Signal(object)
    warning = Signal(str, str)

    def __init__(self, workspace_loader: WorkspaceLoader, parent=None):
        super().__init__(parent)
        self.workspace_loader = workspace_loader

    def handle_compression_finished(self, payload: dict):
        method = payload.get("compressor_name", "unknown")
        metrics = payload.get("metrics", {})

        compressed_path = payload.get("compressed_path")
        reconstructed_path = payload.get("reconstructed_path")

        if compressed_path:
            self._load_compressed_output(
                path=Path(compressed_path),
                method=method,
            )

        if reconstructed_path:
            reconstructed_item = self._load_reconstructed_output(
                path=Path(reconstructed_path),
                method=method,
                metrics=metrics,
            )

            if reconstructed_item is not None:
                self.metrics_item_ready.emit(reconstructed_item)

    def _load_reconstructed_output(
        self,
        path: Path,
        method: str,
        metrics: dict,
    ) -> WorkspaceItem | None:
        try:
            item = self.workspace_loader.inspect_hsi(path)
        except WorkspaceLoadError as exc:
            self.warning.emit(
                "Could not load reconstructed output",
                str(exc),
            )
            return None

        item.type = WorkspaceItemType.RECONSTRUCTION
        item.method = method
        item.metrics = self._deserialize_metrics(metrics)

        self.item_ready.emit(item)

        return item

    def _load_compressed_output(
        self,
        path: Path,
        method: str,
    ) -> WorkspaceItem | None:
        try:
            item = self.workspace_loader.inspect_compressed_hsi(path)
        except WorkspaceLoadError as exc:
            self.warning.emit(
                "Could not load compressed output",
                str(exc),
            )
            return None

        item.method = method
        item.type = WorkspaceItemType.COMPRESSED

        self.item_ready.emit(item)

        return item

    def _deserialize_metrics(self, metrics: dict) -> dict:
        return {
            name: LoadedMetric(
                value=metric.get("value"),
                unit=metric.get("unit", ""),
            )
            for name, metric in metrics.items()
        }