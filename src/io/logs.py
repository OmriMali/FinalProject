from pathlib import Path
from typing import Any

import pandas as pd

from src.io.hsi import load_hsi


def load_recent_compression(log_path: str | Path) -> dict[str, Any]:
    """
    Load the most recent compression run from a compression log.

    Parameters
    ----------
    log_path : str | Path
        Path to a compression log CSV file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the log row fields and, if available,
        the reconstructed HSI loaded from the artifact folder.
    """
    df, log_path = _load_compression_log(log_path)

    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"],
                                        format="%Y-%m-%dT%H:%M:%S", 
                                        errors="coerce"
                                        )
        df = df.sort_values("timestamp")

    row = df.iloc[-1].to_dict()

    return _load_compression_row_objects(row)


def load_compression_run(
    log_path: str | Path,
    run_id: int | str,
) -> dict[str, Any]:
    """
    Load a specific compression run from a compression log.

    Parameters
    ----------
    log_path : str | Path
        Path to a compression log CSV file.

    run_id : int | str
        Run ID to load from the log.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the log row fields and, if available,
        the reconstructed HSI loaded from the artifact folder.

    Raises
    ------
    ValueError
        If the log does not contain a run_id column or if no matching
        run_id is found.
    """
    df, log_path = _load_compression_log(log_path)

    if "run_id" not in df.columns:
        raise ValueError(f"'run_id' column not found in compression log: {log_path}")

    matches = df[df["run_id"].astype(str) == str(run_id)]

    if matches.empty:
        raise ValueError(
            f"No compression run with run_id={run_id} found in {log_path}"
        )

    row = matches.iloc[-1].to_dict()

    return _load_compression_row_objects(row)


def _load_compression_log(log_path: str | Path) -> tuple[pd.DataFrame, Path]:
    """
    Load a compression log CSV file.
    """
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Compression log not found: {log_path}")

    df = pd.read_csv(log_path)

    if df.empty:
        raise ValueError(f"Compression log is empty: {log_path}")

    return df, log_path


def _load_compression_row_objects(row: dict[str, Any]) -> dict[str, Any]:
    """
    Load objects associated with a compression log row.
    """
    artifact_dir = _get_artifact_dir(row)

    row["artifact_dir"] = artifact_dir
    row["reconstructed"] = None

    if artifact_dir is not None:
        row["reconstructed"] = load_hsi(artifact_dir, "reconstructed")

    return row


def _get_artifact_dir(row: dict[str, Any]) -> Path | None:
    """
    Extract the artifact directory from a log row.
    """
    for key in ("artifact_dir", "artifact_path", "artifacts_dir", "output_dir"):
        value = row.get(key)

        if value is not None and not pd.isna(value):
            return Path(value)

    return None