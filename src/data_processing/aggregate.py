import pandas as pd


def aggregate_mean_std(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    dropna_groups: bool = False,
) -> pd.DataFrame:
    """
    Aggregate selected value columns by mean and standard deviation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    group_cols : list[str]
        Columns used for grouping.

    value_cols : list[str]
        Numeric columns to aggregate.

    dropna_groups : bool, optional
        If False, keep groups where some grouping columns are NaN.
        This is useful when comparing different algorithms with different
        parameter columns.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with flattened column names.
    """

    _validate_columns(df, group_cols + value_cols)

    grouped = (
        df.groupby(
            group_cols,
            dropna=dropna_groups,
        )[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped.columns = _flatten_columns(grouped.columns)

    return grouped


def _validate_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Validate that all requested columns exist.
    """

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _flatten_columns(columns) -> list[str]:
    """
    Flatten pandas MultiIndex aggregation columns.
    """

    flat = []

    for column in columns:
        if isinstance(column, tuple):
            name = "_".join(
                str(part)
                for part in column
                if part != ""
            )
            flat.append(name)
        else:
            flat.append(column)

    return flat