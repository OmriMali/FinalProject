from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from src.core.hsi import HSI
from src.core.dictionary import Axis
from src.ui.gui.services.workspace_loader import WorkspaceLoader
from src.ui.gui.services.compression_service import CompressionService


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)

def emit_progress(value: float) -> None:
    emit({
        "type": "progress",
        "value": value,
    })

def emit_message(message: str) -> None:
    emit({
        "type": "message",
        "message": message,
    })

def serialize_metrics(metrics: dict) -> dict:
    serialized = {}

    for name, metric in metrics.items():
        serialized[name] = {
            "value": getattr(metric, "value", metric),
            "unit": getattr(metric, "unit", ""),
        }

    return serialized

def decode_config_values(values: dict) -> dict:
    decoded = {}

    for key, value in values.items():
        decoded[key] = decode_value(value)

    return decoded

def decode_value(value):
    if isinstance(value, dict) and value.get("__enum__") == "Axis":
        return Axis[value["name"]]

    if isinstance(value, list):
        return tuple(decode_value(item) for item in value)

    if isinstance(value, dict):
        return {
            key: decode_value(item)
            for key, item in value.items()
        }

    return value

def _path_to_str(path: Path | None) -> str | None:
    if path is None:
        return None

    return str(path)


def main() -> int:
    try:
        job_path = Path(sys.argv[1])

        with job_path.open("r", encoding="utf-8") as f:
            job = json.load(f)

        source_path = Path(job["source_path"])
        compressor_name = job["compressor_name"]
        config_values = decode_config_values(job["config_values"])
        experiment_settings = job["experiment_settings"]

        emit_message("Loading HSI")
        emit_progress(0.0)

        loader = WorkspaceLoader()
        source_item = loader.inspect_hsi(source_path)
        obj = loader.load_object(source_item)

        if not isinstance(obj, HSI):
            raise TypeError("Compression job source must be an HSI")

        emit_message("Starting compression")

        service = CompressionService()

        gui_result = service.compress_and_decompress(
            hsi=obj,
            source_item=source_item,
            compressor_name=compressor_name,
            config_values=config_values,
            experiment_settings=experiment_settings,
            progress_callback=emit_progress,
            message_callback=emit_message,
        )

        result = gui_result.result
        artifact_paths = gui_result.artifact_paths or {}

        emit({
            "type": "finished",
            "compressor_name": compressor_name,
            "artifact_dir": (
                str(gui_result.artifact_dir)
                if gui_result.artifact_dir is not None
                else None
            ),
            "reconstructed_path": _path_to_str(
                artifact_paths.get("reconstructed")
            ),
            "compressed_path": _path_to_str(
                artifact_paths.get("compressed")
            ),
            "metrics": serialize_metrics(result.metrics),
        })

        emit_progress(1.0)
        emit_message("Finished")

        return 0

    except Exception as exc:
        emit({
            "type": "error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        return 1
    



if __name__ == "__main__":
    raise SystemExit(main())