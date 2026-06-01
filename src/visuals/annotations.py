from typing import Any
from matplotlib.axes import Axes

from src.visuals.style import get_style_value


def format_metrics_text(
    values: dict,
    fields: tuple[str, ...],
    precision: int = 2,
    units: dict[str, str] | None = None,
) -> str:
    """
    Format selected metric values as multiline text.

    Parameters
    ----------
    values : dict
        Dictionary containing metric names and values.

    fields : tuple[str, ...]
        Metric fields to include in the formatted text.

    precision : int, optional
        Number of digits after the decimal point for floating-point values.

    units : dict[str, str] or None, optional
        Mapping from metric field names to unit strings. If a field appears
        in this dictionary, the unit is appended after the formatted value.

    Returns
    -------
    str
        Multiline string containing the selected metrics.
    """
    units = units or {}
    lines = []

    for field in fields:
        value = values.get(field)

        if value is None:
            continue

        label = field.upper()
        unit = units.get(field, "")

        if isinstance(value, float):
            value_text = f"{value:.{precision}f}"
        else:
            value_text = str(value)

        if unit:
            value_text = f"{value_text}{unit}"

        lines.append(f"{label}: {value_text}")

    return "\n".join(lines)


def add_panel_text(
    ax: Axes,
    text: str,
    style: dict[str, Any] | None = None,
    x: float = 0.02,
    y: float = 0.98,
    fontsize: int | None = None,
    color: str | None = None,
    bbox: bool = True,
) -> None:
    """
    Add text inside a plot panel using the visualization style.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis.

    text : str
        Text to display.

    style : dict or None, optional
        Visualization style dictionary.

    x, y : float, optional
        Text position in axis coordinates.

    fontsize : int or None, optional
        Font size. If None, taken from style.

    color : str or None, optional
        Text color. If None, taken from style.

    bbox : bool, optional
        Whether to draw a background box behind the text.
    """
    if not text:
        return

    fontsize = fontsize or get_style_value(
        style,
        "panel_text_fontsize",
        9,
    )

    color = color or get_style_value(
        style,
        "panel_text_color",
        get_style_value(style, "text_color", "black"),
    )

    box = None
    if bbox:
        box = dict(
            facecolor=get_style_value(style, "panel_text_facecolor", "white"),
            edgecolor=get_style_value(style, "panel_text_edgecolor", "none"),
            alpha=get_style_value(style, "panel_text_alpha", 0.7),
        )

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        color=color,
        bbox=box,
    )