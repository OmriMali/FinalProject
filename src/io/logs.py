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
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Compression log not found: {log_path}")

    df = pd.read_csv(log_path)

    if df.empty:
        raise ValueError(f"Compression log is empty: {log_path}")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    row = df.iloc[-1].to_dict()

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