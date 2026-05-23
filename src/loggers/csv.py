import csv
from pathlib import Path

from src.pipeline.callbacks import RunnerCallback
from src.core.results import CompressionRunResult, DictionaryTrainingResult
from src.pipeline.serialization import format_shape

IDENTITY_COLUMNS = [
    "run_id",
    "timestamp",
    "method",
    "experiment",
    "machine",
]

DATASET_COLUMNS = [
    "sensor",
    "scene_id",
    "scene_name",
    "section_row",
    "section_col",
    "height",
    "width",
    "bands",
    "bit_depth",
]

CHANNEL_COLUMNS = [
    "ber",
]

COMPRESSION_METRIC_COLUMNS = [
    "rmse",
    "psnr",
    "sam",
    "cr",
    "comp_time",
    "decomp_time",
]

ARTIFACT_COLUMNS = [
    "artifact_dir",
]

COMPRESSION_BASE_COLUMNS = (
    IDENTITY_COLUMNS
    + DATASET_COLUMNS
    + CHANNEL_COLUMNS
    + COMPRESSION_METRIC_COLUMNS
    + ARTIFACT_COLUMNS
)


DICTIONARY_IDENTITY_COLUMNS = [
    "run_id",
    "timestamp",
    "trainer",
    "experiment",
    "machine",
]

DICTIONARY_DATA_COLUMNS = [
    "axis",
    "num_signals",
    "signal_length",
    "dictionary_shape",
    "dictionary_name",
]

DICTIONARY_METRIC_COLUMNS = [
    "rep_err",
    "mean_k",
    "mu",
    "train_time",
]

DICTIONARY_BASE_COLUMNS = (
    DICTIONARY_IDENTITY_COLUMNS
    + DICTIONARY_DATA_COLUMNS
    + DICTIONARY_METRIC_COLUMNS
    + ARTIFACT_COLUMNS
)

def _normalize_key(key: str) -> str:
    """
    Normalize a column name to lowercase snake_case.
    """
    return (
        str(key)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )

def _normalize_dict_keys(data: dict) -> dict:
    """
    Return a dictionary with normalized keys.
    """
    return {
        _normalize_key(key): value
        for key, value in data.items()
    }

def _normalize_row(row: dict, columns: list[str]) -> dict:
    """
    Return a row containing exactly the requested columns.
    """
    return {
        column: row.get(column, "")
        for column in columns
    }

def _read_existing_columns(path: Path) -> list[str] | None:
    """
    Read CSV columns if the file already exists.
    """
    if not path.exists():
        return None

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None

def _read_next_run_id(path: Path) -> int:
    """
    Return the next serial run id for a CSV log file.
    """
    if not path.exists():
        return 1

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        max_run_id = 0
        for row in reader:
            value = row.get("run_id", "")
            if value == "":
                continue

            try:
                max_run_id = max(max_run_id, int(value))
            except ValueError:
                continue

    return max_run_id + 1

def _rewrite_csv_with_columns(path: Path, columns: list[str]) -> None:
    """
    Rewrite an existing CSV file with a new column order.

    Existing values are preserved where possible. New columns are filled
    with empty values.
    """
    if not path.exists():
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            normalized_row = _normalize_dict_keys(row)
            writer.writerow(_normalize_row(normalized_row, columns))

def _append_csv_row(path: Path, row: dict, base_columns: list[str]) -> None:
    """
    Append one row to a CSV file, writing or extending the header if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    row = _normalize_dict_keys(row)
    base_columns = [_normalize_key(column) for column in base_columns]

    existing_columns = _read_existing_columns(path)

    if existing_columns is None:
        columns = list(base_columns)

        for key in row.keys():
            if key not in columns:
                columns.append(key)

    else:
        existing_columns = [_normalize_key(column) for column in existing_columns]
        columns = list(existing_columns)

        for key in row.keys():
            if key not in columns:
                columns.append(key)

        if columns != existing_columns:
            _rewrite_csv_with_columns(path, columns)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
        )

        if existing_columns is None:
            writer.writeheader()

        writer.writerow(_normalize_row(row, columns))

def _get_tag(result, key: str, default=""):
    """
    Safely get a value from run metadata tags.
    """
    return result.run_metadata.tags.get(key, default)

def _get_metric_rows(result) -> dict:
    """
    Convert MetricResult objects to normalized CSV columns.
    """
    return {
        _normalize_key(metric.short_name): metric.value
        for metric in result.metrics.values()
    }

def _get_config_rows(result) -> dict:
    """
    Convert algorithm config to normalized CSV columns.
    """
    return _normalize_dict_keys(result.run_metadata.algorithm_config)


class CSVLoggerCallback(RunnerCallback):
    """
    CSV logger callback for experiment results.

    Logs one row per completed compression or dictionary training run.

    Logs are saved per method:
        results/compression/{method}/log.csv
        results/dictionary_training/{trainer}/log.csv
    """

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)

    def on_compression_end(self, result: CompressionRunResult) -> None:
        method = _normalize_key(result.run_metadata.algorithm_name)
        log_path = self.results_dir / "compression" / method / "log.csv"

        row = self._compression_row(result, log_path)

        _append_csv_row(
            log_path,
            row,
            COMPRESSION_BASE_COLUMNS,
        )

    def on_dictionary_training_end(self, result: DictionaryTrainingResult) -> None:
        trainer = _normalize_key(result.run_metadata.algorithm_name)
        log_path = self.results_dir / "dictionary_training" / trainer / "log.csv"

        row = self._dictionary_row(result, log_path)

        _append_csv_row(
            log_path,
            row,
            DICTIONARY_BASE_COLUMNS,
        )

    def _compression_row(
        self,
        result: CompressionRunResult,
        log_path: Path,
    ) -> dict:
        metadata = result.original.metadata
        method = _normalize_key(result.run_metadata.algorithm_name)
        experiment = _normalize_key(result.run_metadata.experiment)
        row = {
            "run_id": _read_next_run_id(log_path),
            "timestamp": result.run_metadata.timestamp,
            "method": method,
            "experiment": experiment,
            "machine": result.run_metadata.machine,

            "sensor": metadata.sensor,
            "scene_id": metadata.scene_id,
            "scene_name": metadata.scene_name,
            "section_row": metadata.section_row,
            "section_col": metadata.section_col,
            "height": metadata.shape[0],
            "width": metadata.shape[1],
            "bands": metadata.shape[2],
            "bit_depth": metadata.bit_depth,

            "ber": _get_tag(result, "ber", 0.0),

            "artifact_dir": result.run_metadata.artifact_dir,
        }

        row.update(_get_metric_rows(result))
        row.update(_get_config_rows(result))

        return row

    def _dictionary_row(
        self,
        result: DictionaryTrainingResult,
        log_path: Path,
    ) -> dict:
        trainer = _normalize_key(result.run_metadata.algorithm_name)
        experiment = _normalize_key(result.run_metadata.experiment)
        row = {
            "run_id": _read_next_run_id(log_path),
            "timestamp": result.run_metadata.timestamp,
            "trainer": trainer,
            "experiment": experiment,
            "machine": result.run_metadata.machine,

            "axis": result.signals.axis.name.lower(),
            "num_signals": result.signals.num_signals,
            "signal_length": result.signals.signal_length,
            "dictionary_shape": format_shape(result.dictionary.shape),
            "dictionary_name": result.dictionary.name,

            "artifact_dir": result.run_metadata.artifact_dir,
        }

        row.update(_get_metric_rows(result))
        row.update(_get_config_rows(result))

        return row