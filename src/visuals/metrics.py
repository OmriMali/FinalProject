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
    DEFAULT_STYLE
)

def plot_metric_vs_metric(
    df: pd.DataFrame,
    x: str,
    y: str,
    method_col: str = "method",
    yerr: str | None = None,
    xerr: str | None = None,
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_legend: bool = True,
    plot_type: str = "line",
):
    """
    Plot one metric against another for multiple methods.

    This function expects an already-prepared dataframe. If averaging or
    error bars are needed, they should be computed before calling this
    function, typically using the data_processing module.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing metric values.

    x : str
        Column name used for the x-axis.

    y : str
        Column name used for the y-axis.

    method_col : str, optional
        Column containing method names. Each method is plotted as a
        separate trace.

    yerr : str | None, optional
        Column containing y-axis error values.

    xerr : str | None, optional
        Column containing x-axis error values.

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are
        created.

    title : str | None, optional
        Plot title.

    xlabel : str | None, optional
        X-axis label. If None, ``x`` is used.

    ylabel : str | None, optional
        Y-axis label. If None, ``y`` is used.

    show_legend : bool, optional
        Whether to show a legend.
    
    plot_type : str, optional
        Type of plot to generate. Options are "line" or "bar". Default is "line".

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    style = style or DEFAULT_STYLE

    required_columns = [method_col, x, y]

    if yerr is not None:
        required_columns.append(yerr)

    if xerr is not None and plot_type == "line":
        required_columns.append(xerr)

    _validate_columns(df, required_columns)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    line_width = get_style_value(style, "line_width", 1.8)
    marker_size = get_style_value(style, "marker_size", 6)
    marker = get_style_value(style, "marker", "o")
    linestyle = get_style_value(style, "linestyle", "-")
    errorbar_capsize = get_style_value(style, "errorbar_capsize", 3)
    errorbar_color = get_style_value(style, "line_errorbar", None)

    methods = df[method_col].unique()

    if plot_type == "bar":
        # Setup for grouped bar chart
        categories = df[x].unique()
        x_indices = np.arange(len(categories))
        
        n_methods = len(methods)
        group_width = get_style_value(style, "group_width", 0.75)
        bar_width = group_width / n_methods
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_width

        for method_idx, method in enumerate(methods):
            method_str = str(method)
            group = df[df[method_col] == method].set_index(x).reindex(categories)
            
            color = get_style_color(method_str, style, default=None)
            y_values = group[y].values
            y_errors = group[yerr].values if yerr is not None else None

            ax.bar(
                x_indices + offsets[method_idx],
                y_values,
                width=bar_width,
                yerr=y_errors,
                capsize=errorbar_capsize if y_errors is not None else 0,
                ecolor=errorbar_color or color,
                label=method_str,
                color=color,
            )
            
        ax.set_xticks(x_indices)
        ax.set_xticklabels(categories)

    else:
        # Line
        for method, group in df.groupby(method_col):
            method_str = str(method)
            group = group.sort_values(x)

            color = get_style_color(method_str, style, default=None)

            x_error = group[xerr] if xerr is not None else None
            y_error = group[yerr] if yerr is not None else None

            if xerr is not None or yerr is not None:
                ax.errorbar(
                    group[x],
                    group[y],
                    xerr=x_error,
                    yerr=y_error,
                    label=method_str,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=line_width,
                    markersize=marker_size,
                    color=color,
                    ecolor=errorbar_color or color,
                    capsize=errorbar_capsize,
                )
            else:
                ax.plot(
                    group[x],
                    group[y],
                    label=method_str,
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

    if show_legend:
        # Pushes it just below the x-axis label
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        fig.tight_layout()

    apply_plot_style(fig, ax, style)

    return fig, ax

def plot_runtime_comparison(
    df: pd.DataFrame,
    method_col: str = "method",
    compression_time_col: str = "comp_time_mean",
    decompression_time_col: str = "decomp_time_mean",
    compression_error_col: str | None = None,
    decompression_error_col: str | None = None,
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str = "Time [s]",
    show_legend: bool = True,
):
    """
    Plot compression and decompression runtime comparison.

    This function expects an already-prepared dataframe. Each row should
    represent one method/compressor. Runtime means and standard
    deviations should be computed before calling this function.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing runtime values.

    method_col : str, optional
        Column containing method names.

    compression_time_col : str, optional
        Column containing compression runtime values.

    decompression_time_col : str, optional
        Column containing decompression runtime values.

    compression_error_col : str | None, optional
        Column containing compression runtime error values.

    decompression_error_col : str | None, optional
        Column containing decompression runtime error values.

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are
        created.

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

    style = style or DEFAULT_STYLE

    required_columns = [
        method_col,
        compression_time_col,
        decompression_time_col,
    ]

    if compression_error_col is not None:
        required_columns.append(compression_error_col)

    if decompression_error_col is not None:
        required_columns.append(decompression_error_col)

    _validate_columns(df, required_columns)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    methods = df[method_col].astype(str).to_list()

    actions = [
        get_style_value(style, "compression_label", "Compression"),
        get_style_value(style, "decompression_label", "Decompression"),
    ]

    action_positions = np.arange(len(actions))

    n_methods = len(methods)
    group_width = get_style_value(style, "group_width", 0.75)
    bar_width = group_width / n_methods

    offsets = (
        np.arange(n_methods) - (n_methods - 1) / 2
    ) * bar_width

    errorbar_capsize = get_style_value(style, "errorbar_capsize", 4)
    errorbar_color = get_style_value(style, "bar_errorbar", None)

    for method_idx, method in enumerate(methods):
        values = [
            df.iloc[method_idx][compression_time_col],
            df.iloc[method_idx][decompression_time_col],
        ]

        errors = None

        if (
            compression_error_col is not None
            or decompression_error_col is not None
        ):
            errors = [
                (
                    df.iloc[method_idx][compression_error_col]
                    if compression_error_col is not None
                    else 0.0
                ),
                (
                    df.iloc[method_idx][decompression_error_col]
                    if decompression_error_col is not None
                    else 0.0
                ),
            ]

        color = get_style_color(method, style, default=None)

        ax.bar(
            action_positions + offsets[method_idx],
            values,
            width=bar_width,
            yerr=errors,
            capsize=errorbar_capsize if errors is not None else 0,
            ecolor=errorbar_color or color,
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