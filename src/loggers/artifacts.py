from dataclasses import replace
from pathlib import Path
import re
import numpy as np

from src.pipeline.callbacks import RunnerCallback
from src.core.results import CompressionRunResult, DictionaryTrainingResult
from src.io import save_hsi, save_compressed_hsi, save_dictionary


class ArtifactLoggerCallback(RunnerCallback):
    """
    Save experiment artifacts to disk.
    """
    def __init__(
        self,
        root_dir: str | Path,
        save_reconstructed: bool = True,
        save_compressed: bool = False,
        save_dictionary: bool = True,
        save_coefficients: bool = False,
    ):
        self.root_dir = Path(root_dir)

        self.save_reconstructed = save_reconstructed
        self.save_compressed = save_compressed
        self.save_dictionary = save_dictionary
        self.save_coefficients = save_coefficients

    def on_compression_end(self, result: CompressionRunResult) -> CompressionRunResult:
        artifact_dir = self._make_compression_dir(result)

        if self.save_reconstructed:
            save_hsi(
                result.reconstructed,
                artifact_dir,
                "reconstructed",
            )

        if self.save_compressed:
            save_compressed_hsi(
                result.compressed,
                artifact_dir,
                "compressed",
            )

        return self._with_artifact_dir(result, artifact_dir)

    def on_dictionary_training_end(self, result: DictionaryTrainingResult) -> DictionaryTrainingResult:
        artifact_dir = self._make_dictionary_dir(result)

        if self.save_dictionary:
            save_dictionary(
                result.dictionary,
                artifact_dir,
                "dictionary",
            )

        if self.save_coefficients:
            np.save(
                artifact_dir / "coefficients.npy",
                result.coefficients,
            )

        return self._with_artifact_dir(result, artifact_dir)

    def _make_compression_dir(self, result: CompressionRunResult) -> Path:
        metadata = result.original.metadata

        name = self._safe_name(
            f"{result.run_metadata.timestamp}_"
            f"{result.run_metadata.algorithm_name}_"
            f"{metadata.scene_name or metadata.scene_id or 'hsi'}"
        )

        path = self.root_dir / "compression" / name
        path.mkdir(parents=True, exist_ok=True)

        return path

    def _make_dictionary_dir(self, result: DictionaryTrainingResult) -> Path:
        name = self._safe_name(
            f"{result.run_metadata.timestamp}_"
            f"{result.run_metadata.algorithm_name}_"
            f"{result.dictionary.name or result.signals.axis.name}"
        )

        path = self.root_dir / "dictionary" / name
        path.mkdir(parents=True, exist_ok=True)

        return path

    def _with_artifact_dir(self, result, artifact_dir: Path):
        run_metadata = replace(
            result.run_metadata,
            artifact_dir=str(artifact_dir),
        )

        return replace(
            result,
            run_metadata=run_metadata,
        )

    def _safe_name(self, name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)