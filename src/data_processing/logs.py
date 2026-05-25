from pathlib import Path

import pandas as pd


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
            errors="coerce",
        )

    return df