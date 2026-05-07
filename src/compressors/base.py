from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.hsi import HSI, CompressedHSI

@dataclass(frozen=True)
class CompressorConfig:
    pass

class Compressor(ABC):
    def __init__(self, config: CompressorConfig, progress_callback):
        self.config = config
        self.progress_callback = progress_callback

    @abstractmethod
    def compress(self, hsi: HSI) -> CompressedHSI:
        pass

    @abstractmethod
    def decompress(self, compressed: CompressedHSI) -> HSI:
        pass

    @abstractmethod
    def update_progress(self, value):
        if self.progress_callback:
            self.progress_callback(value)