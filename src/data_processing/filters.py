import operator
from pathlib import Path
from typing import Literal

import pandas as pd


_OPERATOR_MAP = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def filter_by(
    df: pd.DataFrame,
    **conditions,
) -> pd.DataFrame:
    """
    Filter a dataframe by exact column matches.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    **conditions
        Column=value filters.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Examples
    --------
    >>> filter_by(df, compressor="hcs1d")

    >>> filter_by(
    ...     df,
    ...     compressor="hcs1d",
    ...     scene_name="Jasper Ridge",
    ... )
    """

    filtered = df.copy()

    for column, value in conditions.items():

        if column not in filtered.columns:
            raise ValueError(f"Unknown column: {column}")

        filtered = filtered[
            filtered[column] == value
        ]

    return filtered.copy()


def filter_in(
    df: pd.DataFrame,
    column: str,
    values: list,
) -> pd.DataFrame:
    """
    Filter rows whose values belong to a set.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    column : str
        Column name.

    values : list
        Allowed values.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Examples
    --------
    >>> filter_in(
    ...     df,
    ...     "compressor",
    ...     ["hcs1d", "ccsds123"],
    ... )
    """

    if column not in df.columns:
        raise ValueError(f"Unknown column: {column}")

    return df[
        df[column].isin(values)
    ].copy()


def filter_notna(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Remove rows containing NaN values in selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    columns : list[str]
        Columns that must not contain NaN values.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Examples
    --------
    >>> filter_notna(df, ["k", "rmse"])
    """

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Unknown columns: {missing}"
        )

    return df.dropna(
        subset=columns
    ).copy()


def drop_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Return a dataframe without the selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    columns : list[str]
        Columns to remove.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe without the selected columns.
    """

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Unknown columns: {missing}")

    return df.drop(columns=columns).copy()


def keep_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Return a dataframe containing only the selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    columns : list[str]
        Columns to retain, in the desired output order.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe containing only the selected columns.
    """

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Unknown columns: {missing}")

    return df.loc[:, columns].copy()


def filter_has_reconstructed_hsi(
    df: pd.DataFrame,
    artifact_column: str = "artifact_dir",
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Keep rows whose artifact directory contains a reconstructed HSI.

    A reconstructed HSI is identified by the ``reconstructed.npz`` file
    written by the artifact logger.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    artifact_column : str, optional
        Column containing artifact directory paths.

    base_dir : str | Path | None, optional
        Directory against which relative artifact paths are resolved. If
        omitted, paths are resolved relative to the current working directory.

    Returns
    -------
    pd.DataFrame
        Rows whose artifact directory contains ``reconstructed.npz``.
    """

    if artifact_column not in df.columns:
        raise ValueError(f"Unknown column: {artifact_column}")

    root = Path(base_dir) if base_dir is not None else None

    def has_reconstructed_hsi(value) -> bool:
        if not isinstance(value, (str, Path)):
            return False

        if isinstance(value, str) and not value.strip():
            return False

        artifact_dir = Path(value)

        if root is not None and not artifact_dir.is_absolute():
            artifact_dir = root / artifact_dir

        return (artifact_dir / "reconstructed.npz").is_file()

    mask = df[artifact_column].map(has_reconstructed_hsi)

    return df[mask].copy()


def filter_compare(
    df: pd.DataFrame,
    column: str,
    op: Literal["<", "<=", ">", ">=", "==", "!="],
    value,
) -> pd.DataFrame:
    """
    Filter rows using a comparison operation.

    If the requested column does not exist, the dataframe is returned
    unchanged. If the column exists but a row has NaN in that column,
    the row is kept.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    column : str
        Column to compare.

    op : {"<", "<=", ">", ">=", "==", "!="}
        Comparison operator.

    value
        Value to compare against.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """

    if op not in _OPERATOR_MAP:
        raise ValueError(f"Unsupported operator: {op}")

    if column not in df.columns:
        return df.copy()

    compare_mask = _OPERATOR_MAP[op](df[column], value)

    # Keep rows where the column is NaN, because this likely means the
    # parameter/metric is not relevant for that method.
    mask = df[column].isna() | compare_mask

    return df[mask].copy()

