import numpy as np
import matplotlib.pyplot as plt

from src.core.hsi import HSI


def get_band_image(hsi: HSI, band: int,
    percentile: tuple[float, float] = (2, 98),
) -> np.ndarray:
    """
    Create a display-ready single-band image from an HSI.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image.

    band : int
        Band index to display.

    percentile : tuple[float, float], optional
        Percentile range used for contrast stretching.

    Returns
    -------
    np.ndarray
        2D image with values in ``[0, 1]``.
    """
    if not 0 <= band < hsi.bands:
        raise ValueError(f"Band index {band} out of range for {hsi.bands} bands")

    image = hsi.data[:, :, band].astype(np.float32)

    return _percentile_stretch(image, percentile)

def get_rgb_image(hsi: HSI,
    bands: tuple[int, int, int] | None = None, 
    target_wavelengths: tuple[float, float, float] = (650, 550, 450),
    percentile: tuple[float, float] = (2, 98)
    ) -> np.ndarray:
    """
    Create an RGB image from an HSI.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image.

    bands : tuple[int, int, int] | None, optional
        Band indices used as ``(R, G, B)``. If None, nearest bands to
        ``target_wavelengths`` are used.

    target_wavelengths : tuple[float, float, float], optional
        Target wavelengths for ``(R, G, B)`` selection.

    percentile : tuple[float, float], optional
        Percentile range used for per-channel contrast stretching.

    Returns
    -------
    np.ndarray
        RGB image with shape ``(H, W, 3)`` and values in ``[0, 1]``.
    """
    if bands is None:
        wavelengths = hsi.metadata.wavelengths

        bands = tuple(
            _nearest_band(wavelengths, target)
            for target in target_wavelengths
        )

    rgb = hsi.data[:, :, bands].astype(np.float32)

    for channel in range(3):
        rgb[:, :, channel] = _percentile_stretch(
            rgb[:, :, channel],
            percentile,
        )

    return rgb


def show_band(hsi: HSI, band: int,
    percentile: tuple[float, float] = (2, 98),
    title: str | None = None,
    ax=None,
    cmap: str = "gray",
    show_axis: bool = False,
):
    """
    Display a single spectral band of an HSI.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image.

    band : int
        Band index to display.

    percentile : tuple[float, float], optional
        Percentile range used for contrast stretching.

    title : str | None, optional
        Plot title.

    ax : matplotlib.axes.Axes | None, optional
        Existing axis to draw on.

    cmap : str, optional
        Matplotlib colormap.

    show_axis : bool, optional
        If False, hide image axes.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the displayed image.
    """

    if ax is None:
        _, ax = plt.subplots()

    image = get_band_image(
        hsi=hsi,
        band=band,
        percentile=percentile,
    )

    ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)

    if title is not None:
        ax.set_title(title)

    if not show_axis:
        ax.axis("off")

    return ax

def show_rgb(hsi: HSI,
    bands: tuple[int, int, int] | None = None,
    target_wavelengths: tuple[float, float, float] = (650, 550, 450),
    percentile: tuple[float, float] = (2, 98),
    title: str | None = None,
    ax=None,
    show_axis: bool = False,
):
    """
    Display an RGB rendering of an HSI.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image.

    bands : tuple[int, int, int] | None, optional
        Band indices used as ``(R, G, B)``.

    target_wavelengths : tuple[float, float, float], optional
        Target wavelengths used when ``bands`` is None.

    percentile : tuple[float, float], optional
        Percentile range used for contrast stretching.

    title : str | None, optional
        Plot title.

    ax : matplotlib.axes.Axes | None, optional
        Existing axis to draw on. If None, a new figure and axis are created.

    show_axis : bool, optional
        If False, hide image axes.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the displayed image.
    """

    if ax is None:
        _, ax = plt.subplots()

    rgb = get_rgb_image(
        hsi=hsi,
        bands=bands,
        target_wavelengths=target_wavelengths,
        percentile=percentile,
    )

    ax.imshow(rgb)

    if title is not None:
        ax.set_title(title)

    if not show_axis:
        ax.axis("off")

    return ax


def compare_rgb(hsis: list[HSI],
    labels: list[str] | None = None,
    bands: tuple[int, int, int] | None = None,
    target_wavelengths: tuple[float, float, float] = (650, 550, 450),
    percentile: tuple[float, float] = (2, 98),
    figsize: tuple[float, float] | None = None,
):
    """
    Display RGB renderings of multiple HSIs side by side.

    Parameters
    ----------
    hsis : list[HSI]
        Hyperspectral images to compare.

    labels : list[str] | None, optional
        Titles for each image.

    bands : tuple[int, int, int] | None, optional
        Band indices used as ``(R, G, B)``.

    target_wavelengths : tuple[float, float, float], optional
        Target wavelengths used when ``bands`` is None.

    percentile : tuple[float, float], optional
        Percentile range used for contrast stretching.

    figsize : tuple[float, float] | None, optional
        Matplotlib figure size.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """

    if labels is not None and len(labels) != len(hsis):
        raise ValueError("labels and hsis must have the same length")

    n = len(hsis)

    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, hsi in enumerate(hsis):
        title = labels[i] if labels is not None else None

        show_rgb(
            hsi=hsi,
            bands=bands,
            target_wavelengths=target_wavelengths,
            percentile=percentile,
            title=title,
            ax=axes[i],
        )

    fig.tight_layout()

    return fig, axes

def compare_band(hsis: list[HSI], band: int,
    labels: list[str] | None = None,
    percentile: tuple[float, float] = (2, 98),
    cmap: str = "gray",
    figsize: tuple[float, float] | None = None,
):
    """
    Display the same spectral band from multiple HSIs side by side.

    Parameters
    ----------
    hsis : list[HSI]
        Hyperspectral images to compare.

    band : int
        Band index to display.

    labels : list[str] | None, optional
        Titles for each image.

    percentile : tuple[float, float], optional
        Percentile range used for contrast stretching.

    cmap : str, optional
        Matplotlib colormap.

    figsize : tuple[float, float] | None, optional
        Matplotlib figure size.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """

    if labels is not None and len(labels) != len(hsis):
        raise ValueError("labels and hsis must have the same length")

    n = len(hsis)

    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, hsi in enumerate(hsis):
        title = labels[i] if labels is not None else None

        show_band(
            hsi=hsi,
            band=band,
            percentile=percentile,
            title=title,
            ax=axes[i],
            cmap=cmap,
        )

    fig.tight_layout()

    return fig, axes


def annotate_text(
    ax,
    text: str,
    loc: str = "upper left",
    fontsize: int = 10,
    alpha: float = 0.75,
):
    """
    Add a text annotation box to an image axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to annotate.

    text : str
        Text to display.

    loc : str, optional
        Location of annotation. Supported: ``"upper left"``,
        ``"upper right"``, ``"lower left"``, ``"lower right"``.

    fontsize : int, optional
        Text font size.

    alpha : float, optional
        Background box opacity.

    Returns
    -------
    matplotlib.text.Text
        Created text object.
    """

    locations = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }

    if loc not in locations:
        raise ValueError(f"Unsupported loc: {loc}")

    x, y, ha, va = locations[loc]

    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={
            "facecolor": "white",
            "alpha": alpha,
            "edgecolor": "none",
        },
    )


def _nearest_band(wavelengths: np.ndarray, target: float) -> int:
    """
    Find the band index nearest to a target wavelength.
    """
    return int(np.argmin(np.abs(wavelengths - target)))

def _percentile_stretch(channel: np.ndarray, percentile: tuple[float, float]) -> np.ndarray:
    """
    Stretch an image channel to ``[0, 1]`` using percentiles.
    """

    low, high = np.percentile(channel, percentile)

    if high <= low:
        return np.zeros_like(channel, dtype=np.float32)

    stretched = (channel - low) / (high - low)

    return np.clip(stretched, 0.0, 1.0)