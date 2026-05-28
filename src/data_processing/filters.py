import pandas as pd
import operator
from typing import Literal


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

