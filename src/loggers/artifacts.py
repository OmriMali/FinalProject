from dataclasses import replace
from pathlib import Path
import json
import re

import numpy as np

from src.pipeline.callbacks import RunnerCallback
from src.core.results import CompressionRunResult, DictionaryTrainingResult
from src.io import save_hsi, save_compressed_hsi, save_dictionary


class ArtifactLoggerCallback(RunnerCallback):
    """
    Save experiment artifacts to disk.

    Compression artifacts are saved under:

       results/compression/{method}/artifacts/{scene_name}_{section}_{experiment}_{timestamp}/

    Dictionary artifacts are saved under:

        results/dictionary_training/{trainer}/artifacts/{dictionary_name}_{experiment}_{timestamp}/
    """

    def __init__(
        self,
        results_dir: str | Path,
        save_reconstructed: bool = True,
        save_compressed: bool = False,
        save_dictionary: bool = True,
        save_coefficients: bool = False,
        save_config: bool = True,
        save_metadata: bool = False,
    ):
        self.results_dir = Path(results_dir)

        self.save_reconstructed = save_reconstructed
        self.save_compressed = save_compressed
        self.save_dictionary = save_dictionary
        self.save_coefficients = save_coefficients
        self.save_config = save_config
        self.save_metadata = save_metadata

        self.last_artifact_dir: Path | None = None
        self.last_artifact_paths: dict[str, Path] = {}


    def on_compression_end(
        self,
        result: CompressionRunResult,
    ) -> CompressionRunResult:
        artifact_dir = self._make_compression_dir(result)
        
        self.last_artifact_dir = artifact_dir
        self.last_artifact_paths = {}

        if self.save_reconstructed:
            rec_path = save_hsi(
                result.reconstructed,
                artifact_dir,
                "reconstructed",
            )
            self.last_artifact_paths["reconstructed"] = rec_path

        if self.save_compressed:
            compressed_path = save_compressed_hsi(
                result.compressed,
                artifact_dir,
                "compressed",
            )
            self.last_artifact_paths["compressed"] = compressed_path

        if self.save_config:
            config_path = artifact_dir / "config.json",
            self._save_json(
                config_path,
                result.run_metadata.algorithm_config,
            )
            self.last_artifact_paths["config"] = config_path

        if self.save_metadata:
            metadata_path = artifact_dir / "metadata.json",
            self._save_json(
                metadata_path,
                self._compression_metadata(result),
            )
            self.last_artifact_paths["metadata"] = metadata_path

        return self._with_artifact_dir(result, artifact_dir)
    
    def on_dictionary_training_end(
        self,
        result: DictionaryTrainingResult,
    ) -> DictionaryTrainingResult:
        artifact_dir = self._make_dictionary_dir(result)

        self.last_artifact_dir = artifact_dir
        self.last_artifact_paths = {}

        if self.save_dictionary:
            dictionary_path = save_dictionary(
                result.dictionary,
                artifact_dir,
                "dictionary",
            )

            self.last_artifact_paths["dictionary"] = dictionary_path

        if self.save_coefficients:
            coefficients_path = artifact_dir / "coefficients.npy"

            np.save(
                coefficients_path,
                result.coefficients,
            )

            self.last_artifact_paths["coefficients"] = coefficients_path

        if self.save_config:
            config_path = artifact_dir / "config.json"

            self._save_json(
                config_path,
                result.run_metadata.algorithm_config,
            )

            self.last_artifact_paths["config"] = config_path

        if self.save_metadata:
            metadata_path = artifact_dir / "metadata.json"

            self._save_json(
                metadata_path,
                self._dictionary_metadata(result),
            )

            self.last_artifact_paths["metadata"] = metadata_path

        return self._with_artifact_dir(result, artifact_dir)


    def _make_compression_dir(self, result: CompressionRunResult) -> Path:
        metadata = result.original.metadata
        method = self._safe_name(result.run_metadata.algorithm_name)
        experiment = self._safe_name(result.run_metadata.experiment)
        timestamp = self._safe_timestamp(result.run_metadata.timestamp)

        scene_name = metadata.scene_name or metadata.scene_id or "hsi"
        scene_name = self._safe_name(scene_name)

        section = self._format_section(
            metadata.section_row,
            metadata.section_col,
        )

        parts = [scene_name]

        if section:
            parts.append(section)

        if experiment:
            parts.append(experiment)

        parts.append(timestamp)

        folder_name = "_".join(parts)

        path = self.results_dir / "compression" / method / "artifacts" / folder_name
        path.mkdir(parents=True, exist_ok=True)

        return path

    def _make_dictionary_dir(self, result: DictionaryTrainingResult) -> Path:
        trainer = self._safe_name(result.run_metadata.algorithm_name)

        dictionary_name = (
            result.dictionary.name
            or result.signals.axis.name
            or "dictionary"
        )
        dictionary_name = self._safe_name(dictionary_name)
        experiment = self._safe_name(result.run_metadata.experiment)
        timestamp = self._safe_timestamp(result.run_metadata.timestamp)

        parts = [dictionary_name]

        if experiment:
            parts.append(experiment)

        parts.append(timestamp)

        folder_name = "_".join(parts)

        path = self.results_dir / "dictionary_training" / trainer / "artifacts" / trainer / folder_name
        path.mkdir(parents=True, exist_ok=True)

        return path

    def _with_artifact_dir(self, result, artifact_dir: Path):
        run_metadata = replace(
            result.run_metadata,
            artifact_dir=self._as_project_relative_path(artifact_dir),
        )

        return replace(
            result,
            run_metadata=run_metadata,
        )

    def _compression_metadata(self, result: CompressionRunResult) -> dict:
        metadata = result.original.metadata

        return {
            "timestamp": result.run_metadata.timestamp,
            "machine": result.run_metadata.machine,
            "method": result.run_metadata.algorithm_name,
            "experiment": result.run_metadata.experiment,
            "ber": result.run_metadata.tags.get("ber", 0.0),
            "sensor": metadata.sensor,
            "scene_id": metadata.scene_id,
            "scene_name": metadata.scene_name,
            "section_row": metadata.section_row,
            "section_col": metadata.section_col,
            "shape": metadata.shape,
            "bit_depth": metadata.bit_depth,
        }

    def _dictionary_metadata(self, result: DictionaryTrainingResult) -> dict:
        return {
            "timestamp": result.run_metadata.timestamp,
            "machine": result.run_metadata.machine,
            "trainer": result.run_metadata.algorithm_name,
            "experiment": result.run_metadata.experiment,
            "axis": result.signals.axis.name,
            "num_signals": result.signals.num_signals,
            "signal_length": result.signals.signal_length,
            "dictionary_shape": result.dictionary.shape,
            "dictionary_name": result.dictionary.name,
        }

    def _format_section(
        self,
        section_row,
        section_col,
    ) -> str:
        if section_row in ("", None) and section_col in ("", None):
            return ""

        if section_row in ("", None) or section_col in ("", None):
            return ""

        return f"r{section_row}_c{section_col}"

    def _safe_timestamp(self, timestamp: str) -> str:
        """
        Convert ISO-like timestamps to filesystem-safe compact timestamps.
        """
        timestamp = str(timestamp)

        timestamp = timestamp.replace("T", "_")
        timestamp = timestamp.replace(":", "")
        timestamp = timestamp.replace("-", "")

        return self._safe_name(timestamp)

    def _safe_name(self, name: str) -> str:
        """
        Convert strings to lowercase filesystem-safe names.
        """
        name = str(name).strip().lower()
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")

    def _save_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=4,
                default=self._json_default,
            )

    def _json_default(self, obj):
        """
        Convert common non-JSON objects to JSON-safe values.
        """
        if hasattr(obj, "tolist"):
            return obj.tolist()

        if isinstance(obj, Path):
            return str(obj)

        return str(obj)

    def _as_project_relative_path(self, path: Path) -> str:
        """
        Return a path string relative to the current working directory when possible.
        """
        path = path.resolve()

        try:
            return str(path.relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)