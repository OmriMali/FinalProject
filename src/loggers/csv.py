import csv
from pathlib import Path

from src.pipeline.callbacks import RunnerCallback
from src.core.results import CompressionRunResult, DictionaryTrainingResult

from src.pipeline.serialization import format_shape

COMPRESSION_COLUMNS = [
    # Run info
    "timestamp", "machine", "scene_id", "scene_name", "section_row", "section_col", "shape", "bit_depth", "compressor",

    # HCS1D
    "K", "sr", "axis", "Phi", "Psi",

    # HCS3D
    "sr_h", "sr_w", "sr_b", "Phi_h", "Phi_w", "Phi_b", "Psi_h", "Psi_w", "Psi_b",

    # CCSDS123 config
    "local_sum_mode", "P", "Omega", "a", "block_size", "protect_bitstream",

    # Metrics
    "RMSE", "PSNR", "SAM", "CR", "COMP_TIME", "DECOMP_TIME",

    # Tags
    "BER",

    # Directories
    "artifact_dir",
]

DICTIONARY_COLUMNS = [
    # Run info
    "timestamp", "machine", "trainer", "axis", "num_signals", "signal_length", "dictionary_shape", "dictionary_name",

    # K-SVD config
    "K", "T_0", "tol", "max_iter",

    # Metrics
    "REP_ERR", "MEAN_K", "MU", "TRAIN_TIME",

    # Directories
    "artifact_dir",
]

def _normalize_row(row: dict, columns: list[str]) -> dict:
    """
    Return a row containing exactly the requested columns.
    """

    return {
        column: row.get(column, "")
        for column in columns
    }

def _append_csv_row(path: Path, row: dict, columns: list[str]) -> None:
    """
    Append one row to a CSV file, writing the header if needed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(_normalize_row(row, columns))

class CSVLoggerCallback(RunnerCallback):
    """
    CSV logger callback for experiment results.

    Logs one row per completed compression or dictionary training run.
    """
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.compression_path = self.log_dir / "compression_runs.csv"
        self.dictionary_path = self.log_dir / "dictionary_training_runs.csv"

    def on_compression_end(self, result: CompressionRunResult) -> None:
        row = self._compression_row(result)
        _append_csv_row(
            self.compression_path,
            row,
            COMPRESSION_COLUMNS,
        )

    def on_dictionary_training_end(self, result: DictionaryTrainingResult) -> None:
        row = self._dictionary_row(result)
        _append_csv_row(
            self.dictionary_path,
            row,
            DICTIONARY_COLUMNS,
        )

    def _compression_row(self, result: CompressionRunResult) -> dict:
        metadata = result.original.metadata

        row = {
            "timestamp": result.run_metadata.timestamp,
            "machine": result.run_metadata.machine,
            "scene_id": metadata.scene_id,
            "scene_name": metadata.scene_name,
            "section_row": metadata.section_row,
            "section_col": metadata.section_col,
            "shape": format_shape(metadata.shape),
            "bit_depth": metadata.bit_depth,
            "compressor": result.run_metadata.algorithm_name,
        }

        row.update(result.run_metadata.algorithm_config)

        row.update({
            metric.short_name: metric.value
            for metric in result.metrics.values()
        })

        row.update({
            f"tag_{key}": value
            for key, value in result.run_metadata.tags.items()
        })

        row.update({"artifact_dir": result.run_metadata.artifact_dir})

        return row
    
    def _dictionary_row(self, result: DictionaryTrainingResult) -> dict:
        row = {
            "timestamp": result.run_metadata.timestamp,
            "machine": result.run_metadata.machine,
            "trainer": result.run_metadata.algorithm_name,
            "axis": result.signals.axis.name,
            "num_signals": result.signals.num_signals,
            "signal_length": result.signals.signal_length,
            "dictionary_shape": str(result.dictionary.shape),
            "dictionary_name": result.dictionary.name,
        }

        row.update(result.run_metadata.algorithm_config)

        row.update({
            metric.short_name: metric.value
            for metric in result.metrics.values()
        })

        row.update({"artifact_dir": result.run_metadata.artifact_dir})

        return row