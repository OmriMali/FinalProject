"""
Metric visualization utilities.

This module contains plotting functions for experiment metrics and
runtime comparisons. It expects already-prepared pandas DataFrames and
does not load CSV files or compute metrics directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.visuals.style import (
    apply_plot_style,
    get_style_color,
    get_style_value,
)


def plot_metric_vs_metric(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_by: str = "compressor",
    yerr: str | None = None,
    xerr: str | None = None,
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_legend: bool = True,
    marker: str = "o",
    linestyle: str = "-",
):
    """
    Plot one metric against another, optionally grouped by method.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metric values to plot.

    x : str
        Column name used for the x-axis.

    y : str
        Column name used for the y-axis.

    group_by : str, optional
        Column used to split the data into separate traces.

    yerr : str | None, optional
        Column containing y-axis error values.

    xerr : str | None, optional
        Column containing x-axis error values.

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are created.

    title : str | None, optional
        Plot title.

    xlabel : str | None, optional
        X-axis label. If None, `x` is used.

    ylabel : str | None, optional
        Y-axis label. If None, `y` is used.

    show_legend : bool, optional
        Whether to show a legend.

    marker : str, optional
        Marker style.

    linestyle : str, optional
        Line style.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    required_columns = [x, y]

    if group_by is not None:
        required_columns.append(group_by)

    if yerr is not None:
        required_columns.append(yerr)

    if xerr is not None:
        required_columns.append(xerr)

    _validate_columns(df, required_columns)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    line_width = get_style_value(style, "line_width", 1.8)
    marker_size = get_style_value(style, "marker_size", 6)
    errorbar_capsize = get_style_value(style, "errorbar_capsize", 3)
    errorbar_color = get_style_value(style, "errorbar_color", None)

    if group_by is None:
        groups = [(None, df)]
    else:
        groups = df.groupby(group_by)

    for label, group in groups:
        group = group.sort_values(x)

        label_str = None if label is None else str(label)
        color = get_style_color(label_str, style) if label_str is not None else None

        x_error = group[xerr] if xerr is not None else None
        y_error = group[yerr] if yerr is not None else None

        if xerr is not None or yerr is not None:
            ax.errorbar(
                group[x],
                group[y],
                xerr=x_error,
                yerr=y_error,
                label=label_str,
                marker=marker,
                linestyle=linestyle,
                linewidth=line_width,
                markersize=marker_size,
                color=color,
                ecolor=errorbar_color,
                capsize=errorbar_capsize,
            )
        else:
            ax.plot(
                group[x],
                group[y],
                label=label_str,
                marker=marker,
                linestyle=linestyle,
                linewidth=line_width,
                markersize=marker_size,
                color=color,
            )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)

    if title is not None:
        ax.set_title(title)

    if show_legend and group_by is not None:
        ax.legend()

    apply_plot_style(fig, ax, style)

    return fig, ax


def plot_runtime_comparison(
    df: pd.DataFrame,
    method_col: str = "compressor",
    compression_time_col: str = "COMP_TIME",
    decompression_time_col: str = "DECOMP_TIME",
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    title: str | None = "Runtime Comparison",
    ylabel: str = "Time [s]",
    show_legend: bool = True,
):
    """
    Plot runtime comparison grouped by action.

    The x-axis contains runtime actions, such as compression and
    decompression. Within each action group, one bar is shown for each
    method/compressor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing runtime values.

    method_col : str, optional
        Column containing method or compressor names.

    compression_time_col : str, optional
        Column containing compression time values.

    decompression_time_col : str, optional
        Column containing decompression time values.

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are created.

    title : str | None, optional
        Plot title.

    ylabel : str, optional
        Y-axis label.

    show_legend : bool, optional
        Whether to show a legend.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    _validate_columns(
        df,
        [
            method_col,
            compression_time_col,
            decompression_time_col,
        ],
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    methods = df[method_col].astype(str).to_list()
    actions = ["Compression", "Decompression"]
    action_positions = np.arange(len(actions))

    n_methods = len(methods)
    total_group_width = 0.75
    bar_width = total_group_width / n_methods

    offsets = (
        np.arange(n_methods) - (n_methods - 1) / 2
    ) * bar_width

    runtime_columns = [
        compression_time_col,
        decompression_time_col,
    ]

    for method_idx, method in enumerate(methods):
        values = [
            df.iloc[method_idx][column]
            for column in runtime_columns
        ]

        color = get_style_color(method, style, default=None)

        ax.bar(
            action_positions + offsets[method_idx],
            values,
            width=bar_width,
            label=method,
            color=color,
        )

    ax.set_xticks(action_positions)
    ax.set_xticklabels(actions)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        ax.legend()

    apply_plot_style(fig, ax, style)

    return fig, ax


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Validate that required columns exist in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    columns : list[str]
        Required column names.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required DataFrame columns: {missing}"
        )