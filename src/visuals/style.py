"""
Reusable plotting styles and style application utilities.

This module defines visual styles for matplotlib figures and provides
small helper functions that apply them consistently across plots.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure


DEFAULT_STYLE: dict[str, Any] = {
    # Figure / axes appearance
    "figure_facecolor": "white",
    "axis_facecolor": "white",

    # Text appearance
    "text_color": "black",
    "title_color": "black",
    "label_color": "black",
    "tick_color": "black",

    # Grid appearance
    "grid": True,
    "grid_color": "0.85",
    "grid_alpha": 0.7,
    "grid_linestyle": "-",

    # Axis border / spines
    "spine_color": "black",

    # Legend appearance
    "legend_facecolor": "white",
    "legend_edgecolor": "black",
    "legend_text_color": "black",
    "legend_frame_alpha": 1.0,

    # Plot defaults
    "line_width": 1.8,
    "marker_size": 6,
    "line_errorbar": None,
    "bar_errorbar": "black",
    "errorbar_capsize": 3,

    # Image defaults
    "cmap": "gray",

    # Named method colors
    "colors": {
        "hcs1d": "#7F00FF",
        "ccsds123": "#CC6600",
        "hcs3d": "#00AA55",
        "original": "#000000",
    },
}

DARK_STYLE: dict[str, Any] = {
    # Figure / axes appearance
    "figure_facecolor": "#808080",
    "axis_facecolor": "#808080",

    # Text appearance
    "text_color": "white",
    "title_color": "white",
    "label_color": "white",
    "tick_color": "white",

    # Grid appearance
    "grid": True,
    "grid_color": "white",
    "grid_alpha": 0.45,
    "grid_linestyle": "-",

    # Axis border / spines
    "spine_color": "white",

    # Legend appearance
    "legend_facecolor": "#404040",
    "legend_edgecolor": "white",
    "legend_text_color": "white",
    "legend_frame_alpha": 1.0,

    # Plot defaults
    "line_width": 1.8,
    "marker_size": 6,
    "line_errorbar": None,
    "bar_errorbar": "white",
    "errorbar_capsize": 3,

    # Image defaults
    "cmap": "gray",

    # Named method colors
    "colors": {
        "hcs1d": "#7F00FF",
        "ccsds123": "#CC6600",
        "hcs3d": "#00AA55",
        "original": "#000000",
    },
}



def get_style_value(style: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    """
    Return a value from a style dictionary.

    Parameters
    ----------
    style : dict or None
        Style dictionary.

    key : str
        Style key.

    default : Any, optional
        Value returned if the key is missing.

    Returns
    -------
    Any
        Requested style value.
    """
    if style is None:
        return default

    return style.get(key, default)


def get_style_color(label: str, style: dict[str, Any] | None = None, default: str | None = None) -> str | None:
    """
    Return a named color from a style dictionary.

    Parameters
    ----------
    label : str
        Label name, usually a compressor or method name.

    style : dict or None, optional
        Style dictionary.

    default : str or None, optional
        Fallback color.

    Returns
    -------
    str or None
        Color associated with the label.
    """
    if style is None:
        return default

    colors = style.get("colors", {})
    return colors.get(label.lower(), default)


def apply_figure_style(fig: Figure, style: dict[str, Any] | None = None) -> None:
    """
    Apply figure-level style settings.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.

    style : dict or None, optional
        Style dictionary.
    """
    if style is None:
        return

    figure_facecolor = style.get("figure_facecolor")
    if figure_facecolor is not None:
        fig.set_facecolor(figure_facecolor)


def apply_axis_style(ax: Axes, style: dict[str, Any] | None = None) -> None:
    """
    Apply axis-level style settings.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis.

    style : dict or None, optional
        Style dictionary.
    """
    if style is None:
        return

    axis_facecolor = style.get("axis_facecolor")
    if axis_facecolor is not None:
        ax.set_facecolor(axis_facecolor)

    if style.get("grid", False):
        ax.grid(
            True,
            color=style.get("grid_color"),
            alpha=style.get("grid_alpha", 0.7),
            linestyle=style.get("grid_linestyle", "-"),
        )
    else:
        ax.grid(False)

    text_color = style.get("text_color")
    title_color = style.get("title_color", text_color)
    label_color = style.get("label_color", text_color)
    tick_color = style.get("tick_color", text_color)

    if title_color is not None:
        ax.title.set_color(title_color)

    if label_color is not None:
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)

    if tick_color is not None:
        ax.tick_params(axis="both", colors=tick_color, which="both")

        for tick in ax.get_xticklabels():
            tick.set_color(tick_color)

        for tick in ax.get_yticklabels():
            tick.set_color(tick_color)

        ax.xaxis.get_offset_text().set_color(tick_color)
        ax.yaxis.get_offset_text().set_color(tick_color)

    spine_color = style.get("spine_color")
    if spine_color is not None:
        for spine in ax.spines.values():
            spine.set_color(spine_color)


def apply_legend_style(ax: Axes, style: dict[str, Any] | None = None) -> None:
    """
    Apply legend style settings.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis.

    style : dict or None, optional
        Style dictionary.
    """
    if style is None:
        return

    legend = ax.get_legend()
    if legend is None:
        return

    frame = legend.get_frame()

    legend_facecolor = style.get("legend_facecolor")
    if legend_facecolor is not None:
        frame.set_facecolor(legend_facecolor)

    legend_edgecolor = style.get("legend_edgecolor")
    if legend_edgecolor is not None:
        frame.set_edgecolor(legend_edgecolor)

    legend_frame_alpha = style.get("legend_frame_alpha")
    if legend_frame_alpha is not None:
        frame.set_alpha(legend_frame_alpha)

    legend_text_color = style.get(
        "legend_text_color",
        style.get("text_color"),
    )
    if legend_text_color is not None:
        for text in legend.get_texts():
            text.set_color(legend_text_color)


def apply_plot_style(fig: Figure, axes: Axes | Iterable[Axes], style: dict[str, Any] | None = None) -> None:
    """
    Apply figure, axis, and legend styling.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.

    axes : Axes or iterable of Axes
        Axis or axes to style.

    style : dict or None, optional
        Style dictionary.
    """
    if style is None:
        return

    apply_figure_style(fig, style)

    if isinstance(axes, Axes):
        axes = [axes]

    for ax in axes:
        apply_axis_style(ax, style)
        apply_legend_style(ax, style)