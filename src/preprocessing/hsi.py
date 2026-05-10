import numpy as np

from src.core.hsi import HSI, HSIMetadata


def filter_spectral_bands(hsi: HSI,
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


def trim_borders(hsi: HSI, black_value: int | float = -50) -> HSI:
    """
    Crop an HSI to the largest spatial rectangle without black pixels.

    A pixel is considered black if all spectral bands are equal to
    ``black_value``.

    Parameters
    ----------
    hsi : HSI
        Input hyperspectral image.

    black_value : int | float, optional
        Fill value used for black/no-data pixels.

    Returns
    -------
    HSI
        Cropped hyperspectral image containing the largest valid rectangle.
    """

    black_pixels = np.all(hsi.data == black_value, axis=2)
    valid_pixels = ~black_pixels

    row_start, row_end, col_start, col_end = _largest_true_rectangle(
        valid_pixels
    )

    data = hsi.data[row_start:row_end, col_start:col_end, :].copy()

    metadata = HSIMetadata(
        shape=data.shape,
        wavelengths=hsi.metadata.wavelengths,
        bit_depth=hsi.metadata.bit_depth,
        sensor=hsi.metadata.sensor,
        scene_id=hsi.metadata.scene_id,
        scene_name=hsi.metadata.scene_name,
        section_idx=hsi.metadata.section_idx,
        attributes={
            **hsi.metadata.attributes,
            "black_border_trim": {
                "black_value": black_value,
                "row_start": int(row_start),
                "row_end": int(row_end),
                "col_start": int(col_start),
                "col_end": int(col_end),
            },
        },
    )

    return HSI(data=data, metadata=metadata)

def _largest_true_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find the largest rectangle containing only True values.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean mask where True indicates a valid pixel.

    Returns
    -------
    tuple[int, int, int, int]
        Rectangle bounds as ``(row_start, row_end, col_start, col_end)``.
    """

    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=int)

    best_area = 0
    best_bounds = None

    for row in range(rows):
        heights = heights + mask[row]
        heights[~mask[row]] = 0

        stack = []
        extended_heights = np.append(heights, 0)

        for col, height in enumerate(extended_heights):
            while stack and extended_heights[stack[-1]] > height:
                h = extended_heights[stack.pop()]

                right = col
                left = stack[-1] + 1 if stack else 0

                area = h * (right - left)

                if area > best_area:
                    best_area = area

                    row_end = row + 1
                    row_start = row_end - h

                    best_bounds = (
                        row_start,
                        row_end,
                        left,
                        right,
                    )

            stack.append(col)

    if best_bounds is None:
        raise ValueError("No valid rectangle found in mask")

    return best_bounds


def crop_hsi_sections(hsi: HSI, section_shape: tuple[int, int], drop_incomplete: bool = True) -> list[HSI]:
    """
    Split an HSI into fixed-size spatial sections.

    This function does not trim black borders. If border trimming is needed,
    run it before calling this function.

    Parameters
    ----------
    hsi : HSI
        Input hyperspectral image.

    section_shape : tuple[int, int]
        Spatial section shape as ``(height, width)``.

    drop_incomplete : bool, optional
        If True, discard edge sections smaller than ``section_shape``.
        If False, keep incomplete edge sections.

    Returns
    -------
    list[HSI]
        Cropped HSI sections.
    """

    section_h, section_w = section_shape
    h, w, _ = hsi.shape

    if section_h <= 0 or section_w <= 0:
        raise ValueError("section_shape values must be positive")

    sections = []
    section_idx = 0

    for row_start in range(0, h, section_h):
        for col_start in range(0, w, section_w):
            row_end = min(row_start + section_h, h)
            col_end = min(col_start + section_w, w)

            current_h = row_end - row_start
            current_w = col_end - col_start

            if drop_incomplete and (
                current_h != section_h or current_w != section_w
            ):
                continue

            data = hsi.data[
                row_start:row_end,
                col_start:col_end,
                :
            ].copy()

            metadata = HSIMetadata(
                shape=data.shape,
                wavelengths=hsi.metadata.wavelengths,
                bit_depth=hsi.metadata.bit_depth,
                sensor=hsi.metadata.sensor,
                scene_id=hsi.metadata.scene_id,
                scene_name=hsi.metadata.scene_name,
                section_idx=section_idx,
                attributes={
                    **hsi.metadata.attributes,
                    "section_crop": {
                        "row_start": int(row_start),
                        "row_end": int(row_end),
                        "col_start": int(col_start),
                        "col_end": int(col_end),
                    },
                },
            )

            sections.append(
                HSI(
                    data=data,
                    metadata=metadata,
                )
            )

            section_idx += 1

    return sections




