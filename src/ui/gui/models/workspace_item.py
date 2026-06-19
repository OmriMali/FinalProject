from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from src.core.hsi import HSI, CompressedHSI, HSIMetadata


class WorkspaceItemType(Enum):
    ORIGINAL = "Original"
    RECONSTRUCTION = "Reconstruction"
    COMPRESSED = "Compressed"
    UNKNOWN = "Unknown"


@dataclass
class WorkspaceItem:
    item_id: str
    type: WorkspaceItemType
    metadata: HSIMetadata
    path: Path | None = None
    number: int | None = None
    method: str | None = None
    metrics: dict | None = None
    cached_object: HSI | CompressedHSI | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    DISPLAY_COLUMNS: ClassVar[tuple[str, ...]] = (
        "#",
        "Scene",
        "Row",
        "Col",
        "Shape",
        "Type",
        "Method",
        "Path",
    )

    @classmethod
    def from_hsi(
        cls,
        hsi: HSI,
        path: Path | None = None,
        type: WorkspaceItemType | None = None,
        number: int | None = None,
        method: str | None = None,
        metrics: dict | None = None,
        keep_cached: bool = False,
    ) -> WorkspaceItem:
        metadata = hsi.metadata

        if type is None:
            type = _infer_hsi_type(metadata, path)

        if method is None:
            method = _compression_method(metadata)

        return cls(
            item_id=str(uuid4()),
            type=type,
            metadata=metadata,
            path=path,
            number=number,
            method=method,
            metrics=metrics,
            cached_object=hsi if keep_cached else None,
        )

    @classmethod
    def from_compressed_hsi(
        cls,
        compressed: CompressedHSI,
        path: Path | None = None,
        number: int | None = None,
        method: str | None = None,
        metrics: dict | None = None,
        keep_cached: bool = False,
    ) -> WorkspaceItem:
        metadata = compressed.metadata

        if method is None:
            method = _compression_method(metadata)

        return cls(
            item_id=str(uuid4()),
            type=WorkspaceItemType.COMPRESSED,
            metadata=metadata,
            path=path,
            number=number,
            method=method,
            metrics=metrics,
            cached_object=compressed if keep_cached else None,
        )

    @property
    def is_hsi(self) -> bool:
        return self.type in {
            WorkspaceItemType.ORIGINAL,
            WorkspaceItemType.RECONSTRUCTION,
        }

    @property
    def is_compressed(self) -> bool:
        return self.type == WorkspaceItemType.COMPRESSED

    @classmethod
    def table_headers(cls) -> list[str]:
        return list(cls.DISPLAY_COLUMNS)

    def table_values(self) -> list[str]:
        return [
            self.number_text,
            self.scene_text,
            self.row_text,
            self.col_text,
            self.shape_text,
            self.type_text,
            self.method_text,
            self.path_text,
        ]

    @property
    def number_text(self) -> str:
        return "-" if self.number is None else str(self.number)

    @property
    def scene_text(self) -> str:
        return self.metadata.scene_name or "-"

    @property
    def row_text(self) -> str:
        row = getattr(self.metadata, "section_row", None)
        return "-" if row is None else str(row)

    @property
    def col_text(self) -> str:
        col = getattr(self.metadata, "section_col", None)
        return "-" if col is None else str(col)

    @property
    def shape_text(self) -> str:
        shape = getattr(self.metadata, "shape", None)

        if shape is None:
            return "-"

        return " × ".join(str(value) for value in shape)

    @property
    def type_text(self) -> str:
        return self.type.value

    @property
    def method_text(self) -> str:
        if self.method is not None:
            return str(self.method)

        method = _compression_method(self.metadata)
        return "-" if method is None else str(method)

    @property
    def path_text(self) -> str:
        return "-" if self.path is None else str(self.path)

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

        if method != "-":
            return _format_method_name(method)

        if self.type == WorkspaceItemType.ORIGINAL:
            return "Original"

        if self.type == WorkspaceItemType.RECONSTRUCTION:
            return "Reconstruction"

        if self.type == WorkspaceItemType.COMPRESSED:
            return "Compressed"

        return ""


def _infer_hsi_type(
    metadata: HSIMetadata,
    path: Path | None,
) -> WorkspaceItemType:
    if _compression_method(metadata) is not None:
        return WorkspaceItemType.RECONSTRUCTION

    if path is not None:
        path_text = f"{path.stem} {path.parent.name}".lower()

        reconstruction_tokens = (
            "reconstructed",
            "reconstruction",
            "recon",
        )

        if any(token in path_text for token in reconstruction_tokens):
            return WorkspaceItemType.RECONSTRUCTION

    return WorkspaceItemType.ORIGINAL

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

    attributes = getattr(metadata, "attributes", None)

    if not isinstance(attributes, dict):
        return None

    return _dict_deep_get(attributes, key)

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