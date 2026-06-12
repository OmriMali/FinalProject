from __future__ import annotations

from pathlib import Path

from src import io
from src.core.hsi import HSI, CompressedHSI
from src.ui.gui.models.workspace_item import (
    WorkspaceItem,
    WorkspaceItemKind,
)


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

    def inspect_hsi(self, path: Path) -> WorkspaceItem:
        """
        Inspect an HSI file and return a lightweight workspace item.

        Currently this loads the HSI once to obtain metadata, then discards
        the heavy object. Later, this can be optimized with metadata-only IO.
        """
        hsi = self._load_hsi_from_path(path)

        return WorkspaceItem.from_hsi(
            hsi=hsi,
            name=path.stem,
            path=path,
            keep_cached=False,
        )

    def inspect_compressed_hsi(self, path: Path) -> WorkspaceItem:
        """
        Inspect a CompressedHSI file and return a lightweight workspace item.
        """
        compressed = self._load_compressed_hsi_from_path(path)

        return WorkspaceItem.from_compressed_hsi(
            compressed=compressed,
            name=path.stem,
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
                f"Workspace item '{item.name}' has no path and no cached object"
            )

        if item.kind == WorkspaceItemKind.HSI:
            return self._load_hsi_from_path(item.path)

        if item.kind == WorkspaceItemKind.COMPRESSED_HSI:
            return self._load_compressed_hsi_from_path(item.path)

        raise WorkspaceLoadError(
            f"Unsupported workspace item kind: {item.kind}"
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