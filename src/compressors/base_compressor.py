from abc import ABC, abstractmethod
from src.core.hsi import HSI

class BaseCompressor(ABC):

    def __init__(self, progress_callback=None, **params):
        self.params = params
        self.progress_callback = progress_callback


    def _update_progress(self, value):
        if self.progress_callback:
            self.progress_callback(value)
    
    @abstractmethod
    def compress(self, hsi: HSI) -> tuple[bytes, dict]:
        """Returns (raw_bitstream, metadata_dict)"""
        pass

    @abstractmethod
    def decompress(self, bitstream: bytes, metadata: dict) -> HSI:
        """Returns reconstructed HSI"""
        pass