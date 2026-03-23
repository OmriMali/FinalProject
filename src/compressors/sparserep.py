import numpy as np
from src.compressors.base import BaseCompressor
from src import util


class SparseRep(BaseCompressor):

    BASIS_MAP = {"DCT": util.DCTBasis}

    def __init__(self, transforms, axes, progress_callback=None):
        super().__init__(progress_callback)
        self.transforms_names = transforms
        self.transforms = []
        self.axes = axes
        for t in transforms:
             self.transforms.append(self.BASIS_MAP[t]())

    @property
    def name(self): return "spraserep"
    
    @property
    def compressor_id(self): return 13
    
    def compress(self, hsi):
        """Returns (raw_bitstream, metadata_dict)"""
        
        if self.progress_callback:
                    self.progress_callback(0.0)

        # 1. Statistics and Normalization to [-1, 1]
        min_val, max_val, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, min_val, max_val)
        
        if self.progress_callback:
            self.progress_callback(0.2) 
        
        # 2. Perform Transforms
        trans_hsi = hsi_norm
        for t, a in zip(self.transforms, self.axes):
             trans_hsi = t.forward(trans_hsi, a)
        
        if self.progress_callback:
            self.progress_callback(0.6) 

        # 3. Quantization & Packing
        max_trans = np.max(np.abs(trans_hsi))
        norm_trans_hsi = (trans_hsi + max_trans) / (2 * max_trans)      # normalization to [0, 1]

        max_int = (1 << bit_depth) - 1
        quant_hsi = np.clip(np.round((norm_trans_hsi * max_int)).astype(np.uint64), 0, max_int)

        bitstream = util.pack_to_bit_depth(quant_hsi, bit_depth)

        if self.progress_callback:
            self.progress_callback(1.0) 

        metadata = {
            "hsi_shape": hsi.shape,
            "min_val": min_val,
            "max_val": max_val,
            "bit_depth": bit_depth,
            "max_trans": max_trans,
            "params": {
                "transforms": self.transforms_names,
                "axes": self.axes,
            }
        }
        return bitstream, metadata

    def decompress(self, bitstream, metadata):
        """Returns reconstructed HSI"""

        if self.progress_callback:
            self.progress_callback(0.0) 

        # 1. Unpack & Dequantize
        quant_hsi = util.unpack_from_bit_depth(bitstream, metadata["bit_depth"], metadata["hsi_shape"])
        
        max_int = (1 << metadata["bit_depth"]) - 1
        norm_trans_hsi = (quant_hsi.astype(np.float64) / max_int)

        trans_hsi = norm_trans_hsi * 2 * metadata["max_trans"] - metadata["max_trans"]

        if self.progress_callback:
            self.progress_callback(0.4) 

        # 2. Inverse transforms
        for t, a in reversed(list(zip(self.transforms, self.axes))):
            trans_hsi = t.inverse(trans_hsi, a)

        if self.progress_callback:
            self.progress_callback(0.8) 

        # 3. Denormalization
        rec_hsi = util.denormalize_zero_mean(trans_hsi, metadata["min_val"], metadata["max_val"])
        
        if self.progress_callback:
            self.progress_callback(1.0) 

        return rec_hsi




         
