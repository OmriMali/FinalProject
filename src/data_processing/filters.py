import pandas as pd


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

    return filtered


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
    ]


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
    )