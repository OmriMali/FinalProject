from pathlib import Path
from typing import Any

import pandas as pd


def save_dataframe_to_csv(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    **to_csv_kwargs: Any,
) -> Path:
    """
    Save a dataframe to a CSV file.

    Missing parent directories are created automatically.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.

    path : str | Path
        Destination CSV path.

    index : bool, optional
        Whether to include the dataframe index in the CSV.

    **to_csv_kwargs
        Additional keyword arguments passed to ``pandas.DataFrame.to_csv``.

    Returns
    -------
    Path
        Resolved path to the saved CSV file.
    """

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_path,
        index=index,
        **to_csv_kwargs,
    )

    return output_path
