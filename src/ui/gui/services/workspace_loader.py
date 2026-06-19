from __future__ import annotations

from pathlib import Path

from src import io
from src.core.hsi import HSI, CompressedHSI
from src.ui.gui.models.workspace_item import WorkspaceItem, WorkspaceItemType
from src.ui.gui.services.metrics_extractor import MetricsExtractor


class WorkspaceLoadError(RuntimeError):
    """
    Raised when a workspace item cannot be loaded.
    """

    pass


class WorkspaceLoader:
    """
    Load project objects and convert them to workspace items.

    The loader keeps heavy object loading outside MainWindow.
    """

    def __init__(self):
        self.metrics_extractor = MetricsExtractor() 

    def inspect_hsi(self, path: Path) -> WorkspaceItem:
        hsi = self._load_hsi_from_path(path)

        metrics = self.metrics_extractor.extract_for_hsi(
            hsi=hsi,
            path=path,
        )

        run_info = hsi.metadata.attributes.get("run")
        if run_info:
            item_type = WorkspaceItemType.RECONSTRUCTION
        else:
            item_type = WorkspaceItemType.ORIGINAL

        return WorkspaceItem.from_hsi(
            hsi=hsi,
            type=item_type,
            path=path,
            metrics=metrics,
            keep_cached=False,
        )

    def inspect_compressed_hsi(self, path: Path) -> WorkspaceItem:
        """
        Inspect a CompressedHSI file and return a lightweight workspace item.
        """
        compressed = self._load_compressed_hsi_from_path(path)

        return WorkspaceItem.from_compressed_hsi(
            compressed=compressed,
            path=path,
            keep_cached=False,
        )

    def load_object(self, item: WorkspaceItem) -> HSI | CompressedHSI:
        """
        Load the actual object represented by a workspace item.
        """
        if item.cached_object is not None:
            return item.cached_object

        if item.path is None:
            raise WorkspaceLoadError(
                "Workspace item has no path and no cached object"
            )

        if item.is_hsi:
            return self._load_hsi_from_path(item.path)

        if item.is_compressed:
            return self._load_compressed_hsi_from_path(item.path)

        raise WorkspaceLoadError(
            f"Unsupported workspace item type: {item.type}"
        )

    def _load_hsi_from_path(self, path: Path) -> HSI:
        try:
            hsi = io.load_hsi(path.parent, path.stem)
        except Exception as exc:
            raise WorkspaceLoadError(
                f"Could not load HSI from '{path}'"
            ) from exc

        if not isinstance(hsi, HSI):
            raise WorkspaceLoadError(
                f"Expected HSI object, got {type(hsi).__name__}"
            )

        return hsi

    def _load_compressed_hsi_from_path(self, path: Path) -> CompressedHSI:
        if not hasattr(io, "load_compressed_hsi"):
            raise WorkspaceLoadError(
                "io.load_compressed_hsi(...) is not implemented yet"
            )

        try:
            compressed = io.load_compressed_hsi(path.parent, path.stem)
        except Exception as exc:
            raise WorkspaceLoadError(
                f"Could not load CompressedHSI from '{path}'"
            ) from exc

        if not isinstance(compressed, CompressedHSI):
            raise WorkspaceLoadError(
                f"Expected CompressedHSI object, got {type(compressed).__name__}"
            )

        return compressed