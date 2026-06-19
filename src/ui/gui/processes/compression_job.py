from __future__ import annotations

import json
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.core.hsi import HSI, CompressedHSI
from src.core.dictionary import Axis
from src.io import save_hsi, save_compressed_hsi
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

def _run_attributes(result) -> dict:
    return {
        "timestamp": result.run_metadata.timestamp,
        "machine": result.run_metadata.machine,
        "method": result.run_metadata.algorithm_name,
        "experiment": result.run_metadata.experiment,
        "tags": result.run_metadata.tags,
        "artifact_dir": result.run_metadata.artifact_dir,
        "algorithm_config": result.run_metadata.algorithm_config,
    }

def _with_run_info_hsi(hsi: HSI, result) -> HSI:
    attributes = dict(hsi.metadata.attributes)
    attributes["run"] = _run_attributes(result)

    return HSI(
        data=hsi.data,
        metadata=replace(hsi.metadata, attributes=attributes),
    )

def _with_run_info_compressed(
    compressed: CompressedHSI,
    result,
) -> CompressedHSI:
    attributes = dict(compressed.metadata.attributes)
    attributes["run"] = _run_attributes(result)

    return replace(
        compressed,
        metadata=replace(compressed.metadata, attributes=attributes),
    )

def _ensure_workspace_artifacts(
    result,
    artifact_paths: dict,
) -> tuple[dict, str | None, bool]:
    if artifact_paths.get("reconstructed") and artifact_paths.get("compressed"):
        return artifact_paths, None, False

    temporary_artifact_dir = Path(
        tempfile.mkdtemp(prefix="hsi_gui_workspace_")
    )

    artifact_paths = dict(artifact_paths)
    artifact_paths["reconstructed"] = save_hsi(
        _with_run_info_hsi(result.reconstructed, result),
        temporary_artifact_dir,
        "reconstructed",
    )
    artifact_paths["compressed"] = save_compressed_hsi(
        _with_run_info_compressed(result.compressed, result),
        temporary_artifact_dir,
        "compressed",
    )

    return artifact_paths, str(temporary_artifact_dir), True


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
        artifact_paths, temporary_artifact_dir, temporary_artifacts = (
            _ensure_workspace_artifacts(result, artifact_paths)
        )

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
            "temporary_artifacts": temporary_artifacts,
            "temporary_artifact_dir": temporary_artifact_dir,
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
