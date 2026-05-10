import numpy as np

from src.core.hsi import HSI, HSIMetadata


def filter_spectral_bands(
    hsi: HSI,
    remove_ranges: list[tuple[int, int]] | None = None,
    remove_bands: list[int] | None = None,
    one_based: bool = True,
    enforce_strictly_increasing: bool = True,
) -> HSI:
    """
    Filter spectral bands from a hyperspectral image.

    This function can remove known bad band ranges, remove individual
    bands, sort bands by wavelength, and enforce a strictly increasing
    wavelength vector.

    Parameters
    ----------
    hsi : HSI
        Input hyperspectral image.

    remove_ranges : list[tuple[int, int]] | None, optional
        Inclusive band index ranges to remove.

    remove_bands : list[int] | None, optional
        Individual band indices to remove.

    one_based : bool, optional
        If True, band indices in ``remove_ranges`` and ``remove_bands``
        are interpreted as 1-based indices.

    enforce_strictly_increasing : bool, optional
        If True, bands are sorted by wavelength and duplicate or
        non-finite wavelengths are removed.

    Returns
    -------
    HSI
        Hyperspectral image after spectral band filtering.
    """

    data = hsi.data
    wavelengths = hsi.metadata.wavelengths

    if data.shape[2] != len(wavelengths):
        raise ValueError(
            "Number of wavelengths must match number of spectral bands"
        )

    keep = np.ones(len(wavelengths), dtype=bool)

    if remove_ranges is not None:
        for start, end in remove_ranges:
            if one_based:
                start -= 1
                end -= 1

            keep[start:end + 1] = False

    if remove_bands is not None:
        for band in remove_bands:
            if one_based:
                band -= 1

            keep[band] = False

    data = data[:, :, keep]
    wavelengths = wavelengths[keep]

    if enforce_strictly_increasing:
        increasing_mask = _strictly_increasing_mask(wavelengths, min_spacing=1.0)

        data = data[:, :, increasing_mask]
        wavelengths = wavelengths[increasing_mask]

    metadata = HSIMetadata(
        shape=data.shape,
        wavelengths=wavelengths,
        bit_depth=hsi.metadata.bit_depth,
        sensor=hsi.metadata.sensor,
        scene_id=hsi.metadata.scene_id,
        scene_name=hsi.metadata.scene_name,
        section_idx=hsi.metadata.section_idx,
        attributes={
            **hsi.metadata.attributes,
            "spectral_band_filtering": {
                "remove_ranges": remove_ranges,
                "remove_bands": remove_bands,
                "one_based": one_based,
                "enforce_strictly_increasing": enforce_strictly_increasing,
            },
        },
    )

    return HSI(
        data=data.copy(),
        metadata=metadata,
    )

def _strictly_increasing_mask(wavelengths: np.ndarray, min_spacing: float = 0.0) -> np.ndarray:
    """
    Build a mask that keeps wavelengths in their original order while
    removing bands that do not strictly increase.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength vector.

    min_spacing : float
        Minimum spacing between wavelengths.

    Returns
    -------
    np.ndarray
        Boolean mask for strictly increasing wavelengths.
    """

    keep = np.zeros(len(wavelengths), dtype=bool)

    last = -np.inf

    for i, wavelength in enumerate(wavelengths):
        if not np.isfinite(wavelength):
            continue

        if wavelength > last + min_spacing:
            keep[i] = True
            last = wavelength

    return keep