from pathlib import Path
from typing import Any

import pandas as pd

from src.core.hsi import HSI
from src.io.hsi import load_hsi


def load_compression_log(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load a compression experiment log CSV.

    Parameters
    ----------
    path : str | Path
        Path to ``log.csv`` or ``compression_runs.csv``.

    Returns
    -------
    pd.DataFrame
        Loaded compression log.
    """

    df = pd.read_csv(path)

    return _clean_log_dataframe(df)


def load_dictionary_log(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load a dictionary training experiment log CSV.

    Parameters
    ----------
    path : str | Path
        Path to dictionary training log CSV.

    Returns
    -------
    pd.DataFrame
        Loaded dictionary training log.
    """

    df = pd.read_csv(path)

    return _clean_log_dataframe(df)


def load_logs(
    paths: list[str | Path],
) -> pd.DataFrame:
    """
    Load and concatenate multiple log CSV files.

    Parameters
    ----------
    paths : list[str | Path]
        Log CSV paths.

    Returns
    -------
    pd.DataFrame
        Concatenated log dataframe.
    """

    if len(paths) == 0:
        raise ValueError("paths must not be empty")

    dfs = [
        _clean_log_dataframe(pd.read_csv(path))
        for path in paths
    ]

    return pd.concat(dfs, ignore_index=True)


def load_reconstructed_hsis(
    df: pd.DataFrame,
    criteria: list[dict[str, Any]],
    artifact_column: str = "artifact_dir",
    base_dir: str | Path | None = None,
) -> list[HSI]:
    """
    Load reconstructed HSIs selected by dataframe column values.

    Each criteria dictionary must match exactly one row. HSIs are returned in
    the same order as the criteria dictionaries.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing compression runs.

    criteria : list[dict[str, Any]]
        Column-value mappings used to select rows.

    artifact_column : str, optional
        Column containing artifact directory paths.

    base_dir : str | Path | None, optional
        Directory against which relative artifact paths are resolved. If
        omitted, paths are resolved relative to the current working directory.

    Returns
    -------
    list[HSI]
        Reconstructed HSIs in criteria order.

    Raises
    ------
    ValueError
        If a criteria dictionary is empty, references an unknown column, or
        does not match exactly one row.
    """

    if artifact_column not in df.columns:
        raise ValueError(f"Unknown column: {artifact_column}")

    root = Path(base_dir) if base_dir is not None else None
    reconstructed_hsis = []

    for selection in criteria:
        if not isinstance(selection, dict) or not selection:
            raise ValueError("Each criteria item must be a non-empty dictionary")

        matches = df

        for column, value in selection.items():
            if column not in df.columns:
                raise ValueError(f"Unknown column: {column}")

            if pd.isna(value):
                matches = matches[matches[column].isna()]
            else:
                matches = matches[matches[column] == value]

        if matches.empty:
            raise ValueError(f"No row matches criteria: {selection}")

        if len(matches) > 1:
            raise ValueError(f"Multiple rows match criteria: {selection}")

        artifact_value = matches.iloc[0][artifact_column]

        if not isinstance(artifact_value, (str, Path)):
            raise ValueError(
                f"Missing artifact directory for criteria: {selection}"
            )

        if isinstance(artifact_value, str) and not artifact_value.strip():
            raise ValueError(
                f"Missing artifact directory for criteria: {selection}"
            )

        artifact_dir = Path(artifact_value)

        if root is not None and not artifact_dir.is_absolute():
            artifact_dir = root / artifact_dir

        reconstructed_hsis.append(
            load_hsi(artifact_dir, "reconstructed")
        )

    return reconstructed_hsis


def _clean_log_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply basic cleanup to a log dataframe.
    """

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="ISO8601",
            errors="coerce",
        )

    return df
