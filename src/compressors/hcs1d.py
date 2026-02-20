from src.compressors.base import BaseCompressor
from src import util
import numpy as np


class HCS1D(BaseCompressor):

    @property
    def name(self): return "hcs1d"

    @property
    def compressor_id(self): return 21

    def __init__(self, Phi, Psi, targetCR=2, axis=-1, progress_callback=None):
        super().__init__(progress_callback)
        self.targetCR = targetCR
        self.Phi = Phi
        self.Psi = Psi
        self.axis = axis
    

def compress(self, hsi):
        # 1. Statistics and Normalization to [-1, 1]
        min_val, max_val, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, min_val, max_val)

        # 2. Setup Dimensions
        n = hsi.shape[self.axis]
        m = int(n / self.targetCR)

        # 3. Initialize and Project
        self.Phi.initialize(n, m) # Ensure your Phi.initialize handles n, m
        y = self.Phi.project(hsi_norm)
        
        # 4. Quantization (Mapping [-1, 1] -> Integer Range)
        # We use uint64 as an intermediate container for bit-packing
        max_int = (1 << bit_depth) - 1
        y_quantized = np.round((y + 1) / 2 * max_int).astype(np.uint64)
        y_quantized = np.clip(y_quantized, 0, max_int)

        # 5. Bit-Packing
        bitstream = util.pack_to_bit_depth(y_quantized, bit_depth)

        metadata = {
            "y_shape": y.shape,
            "hsi_shape": hsi.shape,
            "min_val": min_val,
            "max_val": max_val,
            "bit_depth": bit_depth,
            "targetCR": self.targetCR,
            "seed": self.Phi.seed,
        }

        return bitstream, metadata

        

