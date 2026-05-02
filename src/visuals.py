import matplotlib.pyplot as plt
import numpy as np
from src.data_processing import fetch_recent, COLOR_MAP
from src.util import load_hsi


def load_recent_hsi(csv_path):
    path = fetch_recent(csv_path)["rec_path"]
    if path is not None:
        return load_hsi(path)
    else:
        raise ValueError("No available HSI reconstruction.")

def show_hsi_rgb(hsi, bands=None, percentiles=(2, 98), title=None):
    rgb, band_indices, wavelengths = hsi.get_rgb(
        bands=bands,
        percentiles=percentiles
    )

    plt.figure()
    plt.imshow(rgb)
    plt.title(title or f"RGB (bands={band_indices}, λ={wavelengths})")
    plt.axis("off")
    plt.show()

def compare_hsi_list(hsi_list, labels, bands=None, percentiles=(2, 98), figsize=(15, 5)):
    """
    Compare multiple HSI objects side-by-side, normalized to the first in the list.

    Parameters
    ----------
    hsi_list : list
    labels : list of str
    bands : tuple (R,G,B) or None
    percentiles : tuple for contrast stretch (after normalization)
    """

    if len(hsi_list) != len(labels):
        raise ValueError("hsi_list and labels must match")

    # ---- Step 1: choose bands ---- #
    if bands is None:
        target_wavelengths = [650, 550, 450]
        wavelengths = hsi_list[0]._wavelengths
        band_indices = tuple(
            int(np.argmin(np.abs(wavelengths - wl)))
            for wl in target_wavelengths
        )
    else:
        band_indices = bands

    # ---- Step 2: build RGBs ---- #
    rgb_list = []

    for hsi in hsi_list:
        data = hsi.get_norm_data()

        rgb = np.stack(
            [data[:, :, i] for i in band_indices],
            axis=2
        )

        rgb_list.append(rgb)

    # ---- Step 3: compute normalization ONLY from first HSI ---- #
    ref_rgb = rgb_list[0]

    low_p, high_p = percentiles
    lo = np.percentile(ref_rgb.reshape(-1, 3), low_p, axis=0)
    hi = np.percentile(ref_rgb.reshape(-1, 3), high_p, axis=0)

    def stretch(rgb):
        out = np.empty_like(rgb)
        for c in range(3):
            if hi[c] == lo[c]:
                out[:, :, c] = 0
            else:
                ch = (rgb[:, :, c] - lo[c]) / (hi[c] - lo[c])
                out[:, :, c] = np.clip(ch, 0, 1)
        return out

    rgb_list = [stretch(rgb) for rgb in rgb_list]


    # ---- Step 4: plot ---- #
    n = len(rgb_list)
    fig, axs = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axs = [axs]

    for ax, rgb, label in zip(axs, rgb_list, labels):
        ax.imshow(rgb)
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def compare_spectra(hsi_list, labels, pixels):
    """
    Plot spectra at multiple pixels, each pixel in a separate subplot.

    Parameters
    ----------
    hsi_list : list
    labels : list of str
    pixels : list of (x, y) tuples
    """


    if len(hsi_list) != len(labels):
        raise ValueError("hsi_list and labels must match")

    n = len(pixels)

    # --- subplot layout (auto grid) --- #
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = np.array(axs).reshape(-1)  # flatten for easy indexing

    for idx, (x, y) in enumerate(pixels):
        ax = axs[idx]

        for hsi, label in zip(hsi_list, labels):
            spectrum = hsi._data[y, x, :]
            wavelengths = hsi._wavelengths

            color = COLOR_MAP.get(label, None)

            ax.plot(
                wavelengths,
                spectrum,
                label=label,
                color=color
            )

        ax.set_title(f"Pixel ({x}, {y})")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.legend(loc="upper right")
        ax.grid()

    # --- remove empty subplots --- #
    for i in range(n, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
