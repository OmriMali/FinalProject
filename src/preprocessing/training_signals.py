import numpy as np

from src.math import n_way_ops
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