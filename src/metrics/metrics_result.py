from dataclasses import dataclass
from typing import ClassVar




@dataclass(frozen=True)
class MetricsResult:
    """
    Metrics computed for a compression-decompression run.

    Parameters
    ----------
    compression_rate : float
        Compression ratio defined as:

            original size / compressed size

    rmse : float
        Root mean squared error between the original and
        reconstructed hyperspectral images.

    psnr : float
        Peak signal-to-noise ratio between the original and
        reconstructed hyperspectral images.

    sam : float
        Spectral angle mapper value between the original and
        reconstructed hyperspectral images, in degrees.

    compression_time : float
        Compression execution time, in seconds.

    decompression_time : float
        Decompression execution time, in seconds.
    """

    compression_rate: float

    rmse: float
    psnr: float
    sam: float

    compression_time: float
    decompression_time: float

    DISPLAY_NAMES: ClassVar[dict[str, str]] = {
        "compression_rate": "Compression Rate",
        "rmse": "RMSE",
        "psnr": "PSNR",
        "sam": "SAM",
        "compression_time": "Compression Time",
        "decompression_time": "Decompression Time"
    }

    UNITS: ClassVar[dict[str, str | None]] = {
        "compression_rate": None,
        "rmse": None,
        "psnr": "dB",
        "sam": "°",
        "compression_time": "s",
        "decompression_time": "s"
    }