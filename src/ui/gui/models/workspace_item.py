from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from src.core.hsi import HSI, CompressedHSI, HSIMetadata


class WorkspaceItemKind(Enum):
    """
    Type of object represented in the GUI workspace.
    """

    HSI = "HSI"
    COMPRESSED_HSI = "CompressedHSI"

class WorkspaceItemRole(Enum):
    """
    Logical role of an item in the workspace.
    """

    ORIGINAL = "Original"
    RECONSTRUCTION = "Reconstruction"
    COMPRESSED = "Compressed"
    UNKNOWN = "Unknown"

@dataclass
class WorkspaceItem:
    """
    Lightweight GUI-side description of a workspace item.

    The actual HSI / CompressedHSI object is not kept in memory by default.
    It can be loaded later from `path` when needed.

    Parameters
    ----------
    item_id : str
        Internal unique ID used by the GUI table.

    name : str
        Display name.

    kind : WorkspaceItemKind
        Type of represented object.

    role : WorkspaceItemRole
        Logical role, such as original, reconstruction, or compressed.

    metadata : HSIMetadata
        Metadata used for display and action decisions.

    path : Path or None, optional
        File path used to reload the object.

    cached_object : HSI or CompressedHSI or None, optional
        Optional in-memory object. Used mainly for newly generated results.
    """

    DISPLAY_COLUMNS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("#", "number_text"),
        ("Name", "name"),
        ("Kind", "kind_text"),
        ("Role", "role_text"),
        ("Scene", "scene_name"),
        ("Section", "section_text"),
        ("Shape", "shape_text"),
        ("Method", "method_text"),
        ("Directory", "directory_text"),
    )

    item_id: str
    name: str
    kind: WorkspaceItemKind
    role: WorkspaceItemRole
    metadata: HSIMetadata
    path: Path | None = None
    method: str | None = None
    directory: Path | None = None
    number: int | None = None
    metrics: dict | None = None
    cached_object: HSI | CompressedHSI | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_hsi(
        cls,
        hsi: HSI,
        name: str | None = None,
        path: Path | None = None,
        role: WorkspaceItemRole | None = None,
        number: int | None = None,
        method: str | None = None,
        directory: Path | None = None,
        metrics: dict | None = None,
        keep_cached: bool = False,
    ) -> WorkspaceItem:
        metadata = hsi.metadata

        if name is None:
            name = _default_item_name(metadata, path)

        if role is None:
            role = _infer_hsi_role(metadata, path)

        if method is None:
            method = _compression_method(metadata)

        return cls(
            item_id=str(uuid4()),
            name=name,
            kind=WorkspaceItemKind.HSI,
            role=role,
            metadata=metadata,
            path=path,
            number=number,
            method=method,
            metrics=metrics,
            directory=directory,
            cached_object=hsi if keep_cached else None,
        )

    @classmethod
    def from_compressed_hsi(
        cls,
        compressed: CompressedHSI,
        name: str | None = None,
        path: Path | None = None,
        number: int | None = None,
        method: str | None = None,
        directory: Path | None = None,
        metrics: dict | None = None,
        keep_cached: bool = False,
    ) -> WorkspaceItem:
        metadata = compressed.metadata

        if name is None:
            name = _default_item_name(metadata, path)

        if method is None:
            method = _compression_method(metadata)

        return cls(
            item_id=str(uuid4()),
            name=name,
            kind=WorkspaceItemKind.COMPRESSED_HSI,
            role=WorkspaceItemRole.COMPRESSED,
            metadata=metadata,
            path=path,
            number=number,
            metrics=metrics,
            method=method,
            directory=directory,
            cached_object=compressed if keep_cached else None,
        )

    @classmethod
    def table_headers(cls) -> list[str]:
        """
        Return table headers for workspace display.
        """
        return [header for header, _ in cls.DISPLAY_COLUMNS]

    def table_values(self) -> list[str]:
        """
        Return row values for workspace display.
        """
        return [
            str(getattr(self, attr))
            for _, attr in self.DISPLAY_COLUMNS
        ]

    @property
    def plot_label(self) -> str:
        parts = [
            self._scene_slug(),
            self._section_slug(),
            self._method_slug(),
        ]

        return "_".join(part for part in parts if part)

    def _scene_slug(self) -> str:
        scene = self.metadata.scene_name

        if scene is None and self.path is not None:
            scene = self.path.stem

        if scene is None:
            scene = "HSI"

        return _slug(scene)

    def _section_slug(self) -> str:
        row = getattr(self.metadata, "section_row", None)
        col = getattr(self.metadata, "section_col", None)

        if row is not None and col is not None:
            return f"r{row}_c{col}"

        section_idx = getattr(self.metadata, "section_idx", None)

        if section_idx is not None:
            return f"s{section_idx}"

        return ""

    def _method_slug(self) -> str:
        method = self.method_text

        if method == "-":
            if self.role == WorkspaceItemRole.ORIGINAL:
                return "Original"

            if self.role == WorkspaceItemRole.RECONSTRUCTION:
                return "Reconstruction"

            if self.role == WorkspaceItemRole.COMPRESSED:
                return "Compressed"

            return ""

        return _format_method_name(method)

    @property
    def kind_text(self) -> str:
        return self.kind.value

    @property
    def role_text(self) -> str:
        return self.role.value

    @property
    def scene_name(self) -> str:
        return self.metadata.scene_name or "-"

    @property
    def section_text(self) -> str:
        row = getattr(self.metadata, "section_row", None)
        col = getattr(self.metadata, "section_col", None)

        if row is None or col is None:
            return "whole"

        return f"r{row}, c{col}"

    @property
    def shape_text(self) -> str:
        h, w, b = self.metadata.shape
        return f"({h}, {w}, {b})"

    @property
    def method_text(self) -> str:
        if self.method is not None:
            return str(self.method)

        method = _compression_method(self.metadata)
        return "-" if method is None else str(method)

    @property
    def directory_text(self) -> str:
        if self.directory is not None:
            return str(self.directory)

        if self.path is not None:
            return str(self.path.parent)

        return "-"

    @property
    def number_text(self) -> str:
        return "-" if self.number is None else str(self.number)

def _default_item_name(
    metadata: HSIMetadata,
    path: Path | None,
) -> str:
    if path is not None:
        return path.stem

    if metadata.scene_name is not None:
        row = getattr(metadata, "section_row", None)
        col = getattr(metadata, "section_col", None)

        if row is not None and col is not None:
            return f"{metadata.scene_name}_r{row}_c{col}"

        return metadata.scene_name

    return "workspace_item"

def _infer_hsi_role(
    metadata: HSIMetadata,
    path: Path | None,
) -> WorkspaceItemRole:
    if _compression_method(metadata) is not None:
        return WorkspaceItemRole.RECONSTRUCTION

    if path is not None:
        path_text = f"{path.stem} {path.parent.name}".lower()

        reconstruction_tokens = (
            "reconstructed",
            "reconstruction",
            "recon",
        )

        if any(token in path_text for token in reconstruction_tokens):
            return WorkspaceItemRole.RECONSTRUCTION

    return WorkspaceItemRole.ORIGINAL

def _compression_method(metadata: HSIMetadata) -> Any:
    for key in (
        "method",
        "algorithm_name",
        "compressor",
        "compressor_name",
    ):
        value = _metadata_value(metadata, key)

        if value is not None:
            return value

    return None

def _metadata_value(metadata: HSIMetadata, key: str) -> Any:
    """
    Read metadata values robustly, including nested attributes.
    """
    if hasattr(metadata, key):
        value = getattr(metadata, key)

        if value is not None:
            return value

    return _dict_deep_get(metadata.attributes, key)

def _dict_deep_get(data: dict, key: str) -> Any:
    if key in data:
        return data[key]

    for value in data.values():
        if isinstance(value, dict):
            result = _dict_deep_get(value, key)

            if result is not None:
                return result

    return None

def _slug(value: str) -> str:
    return str(value).strip().replace(" ", "_")

def _format_method_name(method: str) -> str:
    method_map = {
        "hcs1d": "HCS1D",
        "hcs3d": "HCS3D",
        "hybrid": "Hybrid",
        "ccsds123": "CCSDS123",
    }

    key = str(method).lower()

    if key in method_map:
        return method_map[key]

    return _slug(str(method))