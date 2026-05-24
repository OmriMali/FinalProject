import numpy as np

from src.core.hsi import HSI
from src.core.dictionary import Axis
from src.core.training_signals import TrainingSignals


def sample_training_signals(hsi: HSI, num_signals: int, axis: Axis, seed: int | None = None,) -> TrainingSignals:
    """
    Randomly sample 1D training signals from a hyperspectral image.

    Parameters
    ----------
    hsi : HSI
        Source hyperspectral image.

    num_signals : int
        Number of signals to sample.

    axis : Axis
        Axis along which each signal is extracted.

    seed : int | None, optional
        Random seed.

    Returns
    -------
    TrainingSignals
        Sampled training signals.
    """

    rng = np.random.default_rng(seed)

    data = hsi.data
    h, w, b = data.shape

    if axis == Axis.VERTICAL:
        signal_length = h
        max_signals = w * b
        indices = rng.choice(max_signals, size=num_signals, replace=num_signals > max_signals)

        signals = np.zeros((signal_length, num_signals), dtype=data.dtype)

        for j, idx in enumerate(indices):
            x = idx % w
            z = idx // w
            signals[:, j] = data[:, x, z]

    elif axis == Axis.HORIZONTAL:
        signal_length = w
        max_signals = h * b
        indices = rng.choice(max_signals, size=num_signals, replace=num_signals > max_signals)

        signals = np.zeros((signal_length, num_signals), dtype=data.dtype)

        for j, idx in enumerate(indices):
            y = idx % h
            z = idx // h
            signals[:, j] = data[y, :, z]

    elif axis == Axis.SPECTRAL:
        signal_length = b
        max_signals = h * w
        indices = rng.choice(max_signals, size=num_signals, replace=num_signals > max_signals)

        signals = np.zeros((signal_length, num_signals), dtype=data.dtype)

        for j, idx in enumerate(indices):
            y = idx // w
            x = idx % w
            signals[:, j] = data[y, x, :]

    else:
        raise ValueError(f"Unsupported axis: {axis}")

    return TrainingSignals(
        data=signals,
        axis=axis,
        sources=[hsi.metadata],
        metadata={
            "sampling": "random",
            "seed": seed,
        },
    )


def sample_diverse_training_signals(
    hsis: list[HSI],
    name: str,
    threshold: float = 0.95,
    max_signals: int = 1000,
    candidates_per_hsi: int = 2000,
    seed: int | None = None,
) -> TrainingSignals:
    """
    Extract diverse spectral training signals from a list of HSIs.

    Signals are sampled from the spectral axis and accepted only if their
    absolute correlation with all previously accepted signals is below
    ``threshold``.

    Parameters
    ----------
    hsis : list[HSI]
        Source hyperspectral images.

    name : str
        Name assigned to the generated training signal set.

    threshold : float, optional
        Maximum allowed absolute correlation with existing selected
        signals.

    max_signals : int, optional
        Maximum number of signals to extract.

    candidates_per_hsi : int, optional
        Maximum number of candidate pixels sampled from each HSI.

    seed : int | None, optional
        Random seed.

    Returns
    -------
    TrainingSignals
        Extracted diverse spectral training signals.
    """

    rng = np.random.default_rng(seed)

    hsis = list(hsis)
    rng.shuffle(hsis)

    library = []
    sources = []

    for hsi in hsis:
        if len(library) >= max_signals:
            break

        sources.append(hsi.metadata)

        data = hsi.data.astype(np.float32).copy()

        data_max = np.max(data)
        if data_max > 0:
            data /= data_max

        # Spectral signals: (B, H * W)
        y = data.reshape(-1, data.shape[2]).T
        num_pixels = y.shape[1]

        sample_size = min(num_pixels, candidates_per_hsi)
        pixel_indices = rng.choice(
            num_pixels,
            size=sample_size,
            replace=False,
        )

        for idx in pixel_indices:
            if len(library) >= max_signals:
                break

            current = y[:, idx]

            current_norm = np.linalg.norm(current)
            if current_norm < 1e-6:
                continue

            current_unit = current / current_norm

            if len(library) == 0:
                library.append(current)
                continue

            library_matrix = np.column_stack(library)
            library_norms = np.linalg.norm(
                library_matrix,
                axis=0,
                keepdims=True,
            )

            valid = library_norms > 1e-12
            library_unit = library_matrix[:, valid.ravel()] / library_norms[:, valid.ravel()]

            correlations = current_unit @ library_unit

            if np.max(np.abs(correlations)) < threshold:
                library.append(current)

    if not library:
        raise ValueError("No valid diverse training signals could be extracted")

    return TrainingSignals(
        data=np.column_stack(library),
        axis=Axis.SPECTRAL,
        sources=sources,
        metadata={
            "name": name,
            "sampling": "diverse_correlation",
            "threshold": threshold,
            "max_signals": max_signals,
            "candidates_per_hsi": candidates_per_hsi,
            "seed": seed,
        },
    )


