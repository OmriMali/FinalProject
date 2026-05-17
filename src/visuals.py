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
def compare_hsis(items, bands=None, percentiles=(2, 98), figsize=(15, 5), fontsize=11, independent_norm=False):
    """
    Compare multiple HSI objects side-by-side.

    Parameters
    ----------
    items : list of dict
        Contains "hsi" and "label".
    independent_norm : bool
        If True, each image is stretched based on its own data range. 
        If False (default), all images use the first HSI's range.
    fontsize : int
        The base font size for titles and metadata.
    """
    if len(items) == 0:
        raise ValueError("items list is empty")

    hsi_list = [it["hsi"] for it in items]

    # ---- Step 1: choose bands ---- #
    if bands is None:
        target_wavelengths = [650, 550, 450]
        wavelengths = hsi_list[0]._wavelengths
        band_indices = tuple(int(np.argmin(np.abs(wavelengths - wl))) for wl in target_wavelengths)
    else:
        band_indices = bands

    # ---- Step 2: build RGBs ---- #
    rgb_list = [np.stack([hsi.get_norm_data()[:, :, i] for i in band_indices], axis=2) for hsi in hsi_list]

    # ---- Step 3: normalize ---- #
    low_p, high_p = percentiles
    
    def stretch(rgb, ref_rgb):
        lo = np.percentile(ref_rgb.reshape(-1, 3), low_p, axis=0)
        hi = np.percentile(ref_rgb.reshape(-1, 3), high_p, axis=0)
        out = np.empty_like(rgb)
        for c in range(3):
            if hi[c] == lo[c]:
                out[:, :, c] = 0
            else:
                out[:, :, c] = np.clip((rgb[:, :, c] - lo[c]) / (hi[c] - lo[c]), 0, 1)
        return out

    # Apply normalization (independent vs shared)
    if independent_norm:
        rgb_list = [stretch(rgb, rgb) for rgb in rgb_list]
    else:
        ref_rgb = rgb_list[0]
        rgb_list = [stretch(rgb, ref_rgb) for rgb in rgb_list]

    # ---- Step 4: plot ---- #
    n = len(rgb_list)
    fig, axs = plt.subplots(1, n, figsize=figsize)
    if n == 1: axs = [axs]

    for ax, rgb, item in zip(axs, rgb_list, items):
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(item.get("label", ""), fontsize=fontsize + 2, pad=10)

        extras = [f"{k}: {v}" for k, v in item.items() if k not in ("label", "hsi")]
        if extras:
            ax.text(0.02, 0.98, "\n".join(extras), transform=ax.transAxes,
                    fontsize=fontsize, verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.8))
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
