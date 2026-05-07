from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from src.core.hsi import HSI, CompressedHSI



@dataclass(frozen=True)
class CompressorConfig:
    """
    Base class for compressor configuration objects.
    """
    pass

class Compressor(ABC):
    """
    Base interface for hyperspectral image compressors.
    """
    name: str
    Config = CompressorConfig

    def __init__(self, config: CompressorConfig, progress_callback: Callable[[float], None] | None = None):
        self.config = config
        self._progress_callback = progress_callback

    @abstractmethod
    def compress(self, hsi: HSI) -> CompressedHSI:
        """
        Compress a hyperspectral image.

        Parameters
        ----------
        hsi : HSI
            Hyperspectral image to compress.

        Returns
        -------
        CompressedHSI
            Compressed hyperspectral image.
        """
        raise NotImplementedError

    @abstractmethod
    def decompress(self, compressed: CompressedHSI) -> HSI:
        """
        Reconstruct a hyperspectral image from a compressed representation.

        Parameters
        ----------
        compressed : CompressedHSI
            Compressed hyperspectral image.

        Returns
        -------
        HSI
            Reconstructed hyperspectral image.
        """
        raise NotImplementedError

    def report_progress(self, value: float):
        """
        Report progress of compression or decompression.
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("Progress value must be between 0.0 and 1.0")
        
        if self._progress_callback:
            self._progress_callback(value)