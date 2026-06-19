from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompressionRunSpec:
    """
    Fully resolved GUI compression run request.
    """

    source_path: Path
    compressor_name: str
    config_values: dict[str, Any]
    experiment_settings: dict[str, Any]
    label: str = ""
