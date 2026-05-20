import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.visuals.style import apply_axis_style, apply_figure_style, apply_legend_style


def plot_compression_time_comparison(
    df: pd.DataFrame,
    compressor_col: str = "compressor",
    compression_time_col: str = "COMP_TIME",
    decompression_time_col: str = "DECOMP_TIME",
    show_errorbars: bool = True,
    log_y : bool = False,
    ax=None,
    title: str | None = None,
    style: dict | None = None,
):
    """
    Plot compression and decompression time comparison between compressors.

    Parameters
    ----------
    df : pd.DataFrame
        Compression log dataframe.

    compressor_col : str, optional
        Column containing compressor names.

    compression_time_col : str, optional
        Column containing compression time values.

    decompression_time_col : str, optional
        Column containing decompression time values.

    show_errorbars : bool, optional
        Display the standard deviation.

    log_y : bool, optional
        Display time axis in logarithmic units.

    ax : matplotlib.axes.Axes | None, optional
        Existing axis to draw on.

    title : str | None, optional
        Plot title.

    style : dict | None, optional
        Optional style dictionary.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the bar plot.
    """

    required = [
        compressor_col,
        compression_time_col,
        decompression_time_col,
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    stats = (df.groupby(compressor_col)[
            [compression_time_col, decompression_time_col]
        ].agg(["mean", "std"])
    )

    compressors = stats.index.astype(str).to_numpy()

    if ax is None:
        fig, ax = plt.subplots()

    style = style or {}
    apply_figure_style(fig, style)
    apply_axis_style(ax, style)

    stages = [
        style.get("compression_label", "Compression"),
        style.get("decompression_label", "Recovery"),
    ]

    x = np.arange(len(stages))

    n_compressors = len(compressors)
    group_width = style.get("group_width", 0.8)
    bar_width = group_width / n_compressors

    colors = style.get("colors", {})

    for i, compressor in enumerate(compressors):

        offset = (i - (n_compressors - 1) / 2) * bar_width

        comp_mean = stats.loc[compressor, (compression_time_col, "mean")]
        comp_std = stats.loc[compressor, (compression_time_col, "std")]

        decomp_mean = stats.loc[compressor, (decompression_time_col, "mean")]
        decomp_std = stats.loc[compressor, (decompression_time_col, "std")]

        means = [comp_mean, decomp_mean]
        stds = [comp_std, decomp_std]
        color = (colors.get(compressor) or colors.get(compressor.upper()))

        ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=stds if show_errorbars else None,
            capsize=4 if show_errorbars else None,
            ecolor=style.get("errorbar_color", "black"),
            label=compressor,
            color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stages)

    if log_y:
        ax.set_yscale("log")

    ax.set_ylabel(style.get("ylabel", "Time [s]"))

    if title is not None:
        ax.set_title(title)

    ax.legend()

    apply_legend_style(ax, style)

    return ax