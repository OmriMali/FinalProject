"""
Hyperspectral image visualization utilities.

This module contains plotting functions for displaying hyperspectral
images, spectral bands, spectra, histograms, and visual comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.core.hsi import HSI
from src.visuals.annotations import add_panel_text, format_metrics_text
from src.visuals.style import (
    apply_plot_style,
    get_style_color,
    get_style_value,
)


from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.core.hsi import CompressedHSI
from src.compressors.base import Compressor
from src.visuals.style import apply_plot_style



def plot_compressed_histogram(
    compressed: CompressedHSI,
    compressor: Compressor,
    bins: int = 256,
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
):
    """
    Plot a histogram of compressed-domain values.

    The compressed-domain values are decoded using the compressor-specific
    ``decode_compressed_values`` method. These values are not necessarily
    reconstructed HSI pixel values; they may be measurements, residuals,
    quantized residuals, coefficients, or other compressor-specific symbols.

    Parameters
    ----------
    compressed : CompressedHSI
        Compressed hyperspectral image object.

    compressor : Compressor
        Compressor instance used to decode the compressed-domain values.

    bins : int, optional
        Number of histogram bins.

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are created.

    title : str | None, optional
        Plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    values = compressor.decode_compressed_values(compressed)
    values = np.asarray(values).ravel()

    if values.size == 0:
        raise ValueError("Decoded compressed values are empty")

    if np.issubdtype(values.dtype, np.floating):
        values = values[np.isfinite(values)]

        if values.size == 0:
            raise ValueError("Decoded compressed values contain no finite values")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.hist(values, bins=bins)

    ax.set_xlabel("Compressed-domain value")
    ax.set_ylabel("Count")

    if title is None:
        title = "Compressed HSI Histogram"

    ax.set_title(title)

    apply_plot_style(fig, ax, style)

    return fig, ax


def plot_rgb(
    hsi: HSI,
    bands: tuple[int, int, int] | None = None,
    targets: tuple[float, float, float] = (650, 550, 450),
    stretch: bool = True,
    percentiles: tuple[float, float] = (2, 98),
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    show_axis: bool = False,
):
    """
    Plot an RGB visualization of a hyperspectral image.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image to display.

    bands : tuple[int, int, int] | None, optional
        Band indices used as RGB channels. If None, nearest bands to
        `targets` are selected using hsi metadata wavelengths.

    targets : tuple[float, float, float], optional
        Target wavelengths for RGB band selection, in nanometers.

    stretch : bool, optional
        Whether to apply percentile contrast stretching.

    percentiles : tuple[float, float], optional
        Lower and upper percentiles used for contrast stretching.

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are created.

    title : str | None, optional
        Plot title.

    show_axis : bool, optional
        Whether to show axis ticks and frame.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    if bands is None:
        bands = select_rgb_bands(hsi, targets=targets)

    _validate_band_indices(hsi, bands)

    rgb = hsi.data[:, :, list(bands)]

    if stretch:
        rgb = percentile_stretch(rgb, percentiles=percentiles)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.imshow(rgb)

    if title is not None:
        ax.set_title(title)

    if not show_axis:
        ax.set_axis_off()

    apply_plot_style(fig, ax, style)

    return fig, ax


def plot_band(
    hsi: HSI,
    band: int,
    stretch: bool = True,
    percentiles: tuple[float, float] = (2, 98),
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    show_axis: bool = False,
    colorbar: bool = False,
):
    """
    Plot a single spectral band.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image to display.

    band : int
        Band index to plot.

    stretch : bool, optional
        Whether to apply percentile contrast stretching.

    percentiles : tuple[float, float], optional
        Lower and upper percentiles used for contrast stretching.

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are created.

    title : str | None, optional
        Plot title.

    show_axis : bool, optional
        Whether to show axis ticks and frame.

    colorbar : bool, optional
        Whether to add a colorbar.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    _validate_band_indices(hsi, [band])

    image = hsi.data[:, :, band]

    if stretch:
        image = percentile_stretch(image, percentiles=percentiles)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    cmap = get_style_value(style, "cmap", "gray")
    im = ax.imshow(image, cmap=cmap)

    if title is None:
        title = f"Band {band}"
    ax.set_title(title)

    if not show_axis:
        ax.set_axis_off()

    if colorbar:
        fig.colorbar(im, ax=ax)

    apply_plot_style(fig, ax, style)

    return fig, ax


def plot_spectrum(
    hsi: HSI,
    pixel: tuple[int, int],
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    label: str | None = None,
    title: str | None = None,
    show_legend: bool = True,
):
    """
    Plot the spectrum of a single pixel.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image.

    pixel : tuple[int, int]
        Pixel coordinate as (x, y).

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are created.

    label : str | None, optional
        Curve label.

    title : str | None, optional
        Plot title.

    show_legend : bool, optional
        Whether to show a legend when label is provided.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    x, y = pixel
    _validate_pixel(hsi, x=x, y=y)

    spectrum = hsi.data[y, x, :]
    x_axis = _spectral_axis(hsi)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    color = get_style_color(label, style, default=None) if label else None
    line_width = get_style_value(style, "line_width", 1.8)

    ax.plot(
        x_axis,
        spectrum,
        label=label,
        color=color,
        linewidth=line_width,
    )

    ax.set_xlabel(_spectral_axis_label(hsi))
    ax.set_ylabel("Intensity")

    if title is not None:
        ax.set_title(title)

    if label is not None and show_legend:
        ax.legend()

    apply_plot_style(fig, ax, style)

    return fig, ax


def compare_rgb(
    hsis: list[HSI],
    labels: list[str],
    bands: tuple[int, int, int] | None = None,
    targets: tuple[float, float, float] = (650, 550, 450),
    stretch: bool = True,
    percentiles: tuple[float, float] = (2, 98),
    metrics: dict[str, dict[str, float]] | None = None,
    style: dict[str, Any] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    axes: list[Axes] | np.ndarray | None = None,
):
    """
    Compare RGB visualizations of multiple hyperspectral images.

    Parameters
    ----------
    hsis : list[HSI]
        Hyperspectral images to display.

    labels : list[str]
        Labels corresponding to each HSI.

    bands : tuple[int, int, int] | None, optional
        Band indices used as RGB channels. If None, bands are selected
        using the first HSI wavelengths.

    targets : tuple[float, float, float], optional
        Target wavelengths for RGB band selection, in nanometers.

    stretch : bool, optional
        Whether to apply percentile contrast stretching.

    percentiles : tuple[float, float], optional
        Lower and upper percentiles used for contrast stretching.

    metrics : dict[str, dict[str, float]] | None, optional
        Optional mapping from each HSI label to its metric names and values.
        Metrics are displayed inside the corresponding image panel.

    style : dict | None, optional
        Plot style dictionary.

    title : str | None, optional
        Figure title.

    figsize : tuple[float, float] | None, optional
        Figure size.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """
    _validate_matching_lengths(hsis, labels)

    if bands is None:
        bands = select_rgb_bands(hsis[0], targets=targets)

    n_images = len(hsis)

    if axes is None:
        fig, axes = plt.subplots(
            1,
            n_images,
            figsize=figsize,
            squeeze=False,
        )
        axes = axes.ravel()
    else:
        axes = np.asarray(axes).ravel()

        if len(axes) != n_images:
            raise ValueError("Number of axes must match number of HSIs")

        fig = axes[0].figure

    for ax, hsi, label in zip(axes, hsis, labels):
        plot_rgb(
            hsi,
            bands=bands,
            stretch=stretch,
            percentiles=percentiles,
            style=style,
            ax=ax,
            title=label,
            show_axis=False,
        )

        if metrics is not None and label in metrics:
            metric_values = metrics[label]
            metric_text = format_metrics_text(
                metric_values,
                fields=tuple(metric_values),
            )
            add_panel_text(
                ax,
                metric_text,
                style=style,
            )

    if title is not None:
        fig.suptitle(title)

    apply_plot_style(fig, axes, style)

    return fig, axes


def compare_spectra(
    hsis: list[HSI],
    labels: list[str],
    pixel: tuple[int, int],
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
):
    """
    Compare spectra from multiple hyperspectral images at one pixel.

    Parameters
    ----------
    hsis : list[HSI]
        Hyperspectral images.

    labels : list[str]
        Labels corresponding to each HSI.

    pixel : tuple[int, int]
        Pixel coordinate as (x, y).

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are created.

    title : str | None, optional
        Plot title.

    show_legend : bool, optional
        Whether to show a legend.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    _validate_matching_lengths(hsis, labels)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for hsi, label in zip(hsis, labels):
        x, y = pixel
        _validate_pixel(hsi, x=x, y=y)

        spectrum = hsi.data[y, x, :]
        x_axis = _spectral_axis(hsi)

        color = get_style_color(label, style, default=None)
        line_width = get_style_value(style, "line_width", 1.8)

        ax.plot(
            x_axis,
            spectrum,
            label=label,
            color=color,
            linewidth=line_width,
        )

    ax.set_xlabel(_spectral_axis_label(hsis[0]))
    ax.set_ylabel("Intensity")

    if title is None:
        title = f"Spectrum at pixel ({pixel[0]}, {pixel[1]})"
    ax.set_title(title)

    if show_legend:
        ax.legend()

    apply_plot_style(fig, ax, style)

    return fig, ax


def plot_histogram(
    hsi: HSI,
    band: int | None = None,
    bins: int = 256,
    style: dict[str, Any] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
):
    """
    Plot a histogram of HSI values.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image.

    band : int | None, optional
        Band index. If None, all values in the cube are used.

    bins : int, optional
        Number of histogram bins.

    style : dict | None, optional
        Plot style dictionary.

    ax : Axes | None, optional
        Existing matplotlib axis. If None, a new figure and axis are created.

    title : str | None, optional
        Plot title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    if band is None:
        values = hsi.data.ravel()
    else:
        _validate_band_indices(hsi, [band])
        values = hsi.data[:, :, band].ravel()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.hist(values, bins=bins)

    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

    if title is None:
        title = "HSI Histogram" if band is None else f"Band {band} Histogram"
    ax.set_title(title)

    apply_plot_style(fig, ax, style)

    return fig, ax


def select_rgb_bands(
    hsi: HSI,
    targets: tuple[float, float, float] = (650, 550, 450),
) -> tuple[int, int, int]:
    """
    Select RGB band indices by nearest wavelength.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image.

    targets : tuple[float, float, float], optional
        Target wavelengths for red, green, and blue, in nanometers.

    Returns
    -------
    tuple[int, int, int]
        Selected band indices.
    """
    wavelengths = hsi.metadata.wavelengths

    return tuple(
        int(np.argmin(np.abs(wavelengths - target)))
        for target in targets
    )


def percentile_stretch(
    image: np.ndarray,
    percentiles: tuple[float, float] = (2, 98),
) -> np.ndarray:
    """
    Apply percentile contrast stretching.

    Parameters
    ----------
    image : np.ndarray
        Input 2D or 3D image.

    percentiles : tuple[float, float], optional
        Lower and upper percentiles.

    Returns
    -------
    np.ndarray
        Stretched image in the range [0, 1].
    """
    image = image.astype(np.float32)

    if image.ndim == 2:
        low, high = np.percentile(image, percentiles)
        return _normalize_to_unit_interval(image, low, high)

    if image.ndim == 3:
        stretched = np.empty_like(image, dtype=np.float32)

        for channel in range(image.shape[2]):
            low, high = np.percentile(image[:, :, channel], percentiles)
            stretched[:, :, channel] = _normalize_to_unit_interval(
                image[:, :, channel],
                low,
                high,
            )

        return stretched

    raise ValueError("Image must be 2D or 3D")


def _normalize_to_unit_interval(
    image: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """
    Normalize image values to [0, 1] using given bounds.
    """
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - low) / (high - low)
    return np.clip(normalized, 0, 1)


def _spectral_axis(hsi: HSI) -> np.ndarray:
    """
    Return spectral axis values.
    """
    wavelengths = hsi.metadata.wavelengths

    if wavelengths is None:
        return np.arange(hsi.bands)

    return wavelengths


def _spectral_axis_label(hsi: HSI) -> str:
    """
    Return spectral axis label.
    """
    if hsi.metadata.wavelengths is None:
        return "Band"

    return "Wavelength [nm]"


def _validate_band_indices(
    hsi: HSI,
    bands,
) -> None:
    """
    Validate that band indices are inside the HSI band range.
    """
    for band in bands:
        if not 0 <= band < hsi.bands:
            raise ValueError(
                f"Band index {band} is out of range for "
                f"{hsi.bands} bands"
            )


def _validate_pixel(
    hsi: HSI,
    x: int,
    y: int,
) -> None:
    """
    Validate that a pixel coordinate is inside image bounds.
    """
    height, width = hsi.spatial_shape

    if not 0 <= x < width:
        raise ValueError(
            f"x={x} is out of range for image width {width}"
        )

    if not 0 <= y < height:
        raise ValueError(
            f"y={y} is out of range for image height {height}"
        )


def _validate_matching_lengths(
    hsis: list[HSI],
    labels: list[str],
) -> None:
    """
    Validate that HSI and label lists have matching lengths.
    """
    if len(hsis) != len(labels):
        raise ValueError("hsis and labels must have the same length")

    if len(hsis) == 0:
        raise ValueError("At least one HSI must be provided")


