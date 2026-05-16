import numpy as np
import matplotlib.pyplot as plt

from src.core.hsi import HSI


def get_spectrum(hsi: HSI, row: int, col: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a pixel spectrum from an HSI.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image.

    row : int
        Pixel row index.

    col : int
        Pixel column index.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Wavelengths and spectral values.
    """
    h, w, _ = hsi.shape

    if not 0 <= row < h:
        raise ValueError(f"row {row} out of range for height {h}")

    if not 0 <= col < w:
        raise ValueError(f"col {col} out of range for width {w}")

    wavelengths = hsi.metadata.wavelengths
    spectrum = hsi.data[row, col, :]

    return wavelengths, spectrum


def plot_spectrum(hsi: HSI, row: int, col: int,
    label: str | None = None,
    title: str | None = None,
    ax=None,
    style: dict | None = None,
):
    """
    Plot a pixel spectrum from an HSI.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image.

    row : int
        Pixel row index.

    col : int
        Pixel column index.

    label : str | None, optional
        Line label.

    title : str | None, optional
        Plot title.

    ax : matplotlib.axes.Axes | None, optional
        Existing axis to draw on.

    style : dict | None, optional
        Optional styling dictionary.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the spectrum plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    wavelengths, spectrum = get_spectrum(hsi, row, col)

    style = style or {}

    ax.plot(
        wavelengths,
        spectrum,
        label=label,
        linewidth=style.get("linewidth", 1.5),
    )

    ax.set_xlabel(style.get("xlabel", "Wavelength [nm]"))
    ax.set_ylabel(style.get("ylabel", "Intensity"))

    if title is not None:
        ax.set_title(title)

    if label is not None:
        ax.legend()

    _apply_axis_style(ax, style)

    return ax


def compare_spectra(hsis: list[HSI], labels: list[str], row: int, col: int,
    title: str | None = None,
    ax=None,
    style: dict | None = None,
):
    """
    Plot spectra from the same pixel across multiple HSIs.

    Parameters
    ----------
    hsis : list[HSI]
        Hyperspectral images to compare.

    labels : list[str]
        Labels for each HSI.

    row : int
        Pixel row index.

    col : int
        Pixel column index.

    title : str | None, optional
        Plot title.

    ax : matplotlib.axes.Axes | None, optional
        Existing axis to draw on.

    style : dict | None, optional
        Optional styling dictionary.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the comparison plot.
    """
    if len(hsis) != len(labels):
        raise ValueError("hsis and labels must have the same length")

    if ax is None:
        _, ax = plt.subplots()

    style = style or {}

    for hsi, label in zip(hsis, labels):
        wavelengths, spectrum = get_spectrum(hsi, row, col)

        ax.plot(
            wavelengths,
            spectrum,
            label=label,
            linewidth=style.get("linewidth", 1.5),
        )

    ax.set_xlabel(style.get("xlabel", "Wavelength [nm]"))
    ax.set_ylabel(style.get("ylabel", "Intensity"))

    if title is not None:
        ax.set_title(title)

    ax.legend()

    _apply_axis_style(ax, style)

    return ax


def _apply_axis_style(ax, style: dict | None) -> None:
    """
    Apply optional axis styling.
    """

    if style is None:
        style = {}

    if style.get("grid", True):
        grid_color = style.get("grid_color")

        if grid_color is None:
            ax.grid(True)
        else:
            ax.grid(True, color=grid_color)

    if "facecolor" in style:
        ax.set_facecolor(style["facecolor"])

    text_color = style.get("text_color")

    if text_color is not None:
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.tick_params(axis="x", colors=text_color)
        ax.tick_params(axis="y", colors=text_color)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(text_color)

    spine_color = style.get("spine_color")

    if spine_color is not None:
        for spine in ax.spines.values():
            spine.set_color(spine_color)