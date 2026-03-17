from src.compressors.base import BaseCompressor
from src import util, transforms, measurement_matrices, recovery_algorithms
import numpy as np


class HCS3D(BaseCompressor):

    @property
    def name(self): return "HCS 3D"

    @property
    def compressor_id(self): return 22

    def __init__(self, K, sr=[0.5, 0.5, 1],
                 Phi_names=["SUBSAMPLING", "SUBSAMPLING", "IDENTITY"],
                 Psi_names=["DCT", "DCT", "DCT"], progress_callback=None):
        super().__init__(progress_callback)
        self.K = K
        self.sr = sr
        self.Phi_names = Phi_names
        self.Psi_names = Psi_names

    def compress(self, hsi):
        if self.progress_callback:
            self.progress_callback(0.0)

        # 1. Statistics and Normalization to [-1, 1]
        hsi_min, hsi_max, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, hsi_min, hsi_max)
        shape = hsi.shape
        if self.progress_callback:
            self.progress_callback(0.2)
        
        # 2. Generate Phis
        seeds = [np.random.randint(0, 1_000_000) for _ in range(len(shape))]
        Phis = []
        for i in range(len(shape)):
            Phis.append(measurement_matrices.get_measurement_matrix(self.Phi_names[i], int(self.sr[i] * shape[i]), shape[i],seed=seeds[i]))
        if self.progress_callback:
            self.progress_callback(0.4)

        # 3. Get Measurement Array
        Y = hsi_norm.copy()
        for i in range(len(shape)):
            Y = util.mode_n_product(Y, Phis[i], i)
        if self.progress_callback:
            self.progress_callback(0.8)

        # 4. Quantization & Bit Packing
        max_int = (1 << bit_depth) - 1
        Y_max = np.max(np.abs(Y))
        Y_quantized = np.clip(np.round((Y + Y_max) / 2 * max_int).astype(np.uint64), 0, max_int)
        bitstream = util.pack_to_bit_depth(Y_quantized, bit_depth)
        if self.progress_callback:
            self.progress_callback(1.0)

        metadata = {
            "Y_shape": Y.shape,
            "Y_max": Y_max,
            "hsi_shape": shape,
            "hsi_min": hsi_min,
            "hsi_max": hsi_max,
            "bit_depth": bit_depth,
            "seeds": seeds,
            "params": {
                "sparsity": self.K,
                "sampling rates": self.sr,
                "transforms": self.Psi_names,
                "measurement matrices": self.Phi_names,
            }
        }
        return bitstream, metadata
    
    def decompress(self, bitstream, metadata):
        
        # 1. Unpack & Dequantize
        Y_shape = metadata["Y_shape"]
        Y_max = metadata["Y_max"]

        shape = metadata["hsi_shape"]
        hsi_min = metadata["hsi_min"]
        hsi_max = metadata["hsi_max"]
        bit_depth = metadata["bit_depth"]

        Y_quantized = util.unpack_from_bit_depth(bitstream, bit_depth, Y_shape)
        max_int = (1 << metadata["bit_depth"]) - 1
        Y = (Y_quantized.astype(np.float64) / max_int) * 2 - Y_max
        if self.progress_callback:
            self.progress_callback(0.05)

        # 2. Setup recovery
        seeds = metadata["seeds"]
        Psis_norm = []
        Ds = []
        for i in range(len(shape)):
            Phi = measurement_matrices.get_measurement_matrix(self.Phi_names[i], int(self.sr[i] * shape[i]), shape[i],seed=seeds[i])
            Psi = transforms.get_transform(self.Psi_names[i], shape[i])
            
            D_raw = Phi @ Psi
            col_norms = np.linalg.norm(D_raw, axis=0, keepdims=True)
            col_norms[col_norms == 0] = 1.0
            Ds.append(D_raw / col_norms)
            Psis_norm.append(Psi /col_norms)

        if self.progress_callback:
            self.progress_callback(0.1)
        
        # 3. Run recovery algorithm
        omp_callback = None
        if self.progress_callback:
            omp_callback = util.scaled_callback(self.progress_callback, 0.1, 0.95)
       
        X = recovery_algorithms.n_bomp(Ds, Y, self.K, progress_callback=omp_callback)

        # 4. Recover the hsi
        Z = X
        for n in range(len(Psis_norm)):
            Z = util.mode_n_product(Z, Psis_norm[n], n)
        
        hsi_rec = util.denormalize_zero_mean(Z, hsi_min, hsi_max)
        
        if self.progress_callback:
            self.progress_callback(1.0)

        return hsi_rec

        





        

