from src.compressors.base import BaseCompressor
from src import util, measurement_matrices, transforms, recovery_algorithms
import numpy as np


class HCS1D(BaseCompressor):
    
    # Map axis to spatial\spectral
    AXIS_MAP = ["Vertical", "Horizontal", "Spectral"]

    @property
    def name(self): return "hcs1d"

    @property
    def compressor_id(self): return 21

    def __init__(self, K, sr=2, axis=-1, Phi_name="SUBSAMPLING",
                 Psi_name="IDCT", progress_callback=None):
        super().__init__(progress_callback)
        self.sr = sr
        self.K = K
        self.axis = axis
        self.Phi_name = Phi_name
        self.Psi_name = Psi_name

    def compress(self, hsi):
        if self.progress_callback:
                    self.progress_callback(0.0)

        # 1. Statistics and Normalization to [-1, 1]
        min_val, max_val, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, min_val, max_val)
        n = hsi.shape[self.axis]
        if self.progress_callback:
            self.progress_callback(0.2)

        # 2. Generate Phi
        seed = np.random.randint(0, 1_000_000)
        p = int(self.sr * n)
        Phi = measurement_matrices.get_measurement_matrix(
             self.Phi_name, p, n, seed)
        if self.progress_callback:
            self.progress_callback(0.4)

        # 3. Get Measurements
        Y = hsi_norm.copy()
        Y = util.mode_n_product(Y, Phi, self.axis)
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
            "hsi_shape": hsi.shape,
            "hsi_min": min_val,
            "hsi_max": max_val,
            "bit_depth": bit_depth,
            "seed": seed,
            "params": {
                "sparsity": self.K,
                "sampling rate": self.sr,
                "transform": self.Psi_name,
                "measurement matrix": self.Phi_name,
                "compression axis": self.AXIS_MAP[self.axis]
            }
        }
        return bitstream, metadata
    
    def decompress(self, bitstream, metadata):
        if self.progress_callback:
            self.progress_callback(0.0)

        # 1. Unpack & Dequantize
        Y_shape = metadata["Y_shape"]
        Y_max = metadata["Y_max"]

        shape = metadata["hsi_shape"]
        hsi_min = metadata["hsi_min"]
        hsi_max = metadata["hsi_max"]
        bit_depth = metadata["bit_depth"]

        Y_quantized = util.unpack_from_bit_depth(bitstream, bit_depth, Y_shape)
        max_int = (1 << bit_depth) - 1
        Y = (Y_quantized.astype(np.float64) / max_int) * 2 - Y_max
        if self.progress_callback:
            self.progress_callback(0.05)

        # 2. Setup recovery
        seed = metadata["seed"]
        n = shape[self.axis]
        p = Y_shape[self.axis]
        Phi = measurement_matrices.get_measurement_matrix(self.Phi_name, p, n, seed)
        Psi = transforms.get_transform(self.Psi_name, n)
        D = Phi @ Psi
        col_norms = np.linalg.norm(D, axis=0)
        col_norms[col_norms == 0] = 1.0
        S_inv = np.diag(1.0 / col_norms)
        D = D @ S_inv
        Psi_norm = Psi @ S_inv
        if self.progress_callback:
            self.progress_callback(0.1)

         # 3. Run recovery algorithm
        Y_unfolded = util.mode_n_unfold(Y, self.axis)
        num_pixels = Y_unfolded.shape[1]
        X_unfolded = np.zeros((D.shape[1], num_pixels))

        for i in range(num_pixels):
            x_rec = recovery_algorithms.omp(D, Y_unfolded[:, i], self.K, tol=1e-2)
            X_unfolded[:, i] = x_rec
            if self.progress_callback and i % 100 == 0:
                self.progress_callback(i / num_pixels)
        
        # 4. Recover the hsi
        pixel_shape = list(shape)
        pixel_shape[self.axis] = D.shape[1]
        X = util.mode_n_fold(X_unfolded, self.axis, pixel_shape)
        Z = util.mode_n_product(X, Psi_norm, self.axis)
        hsi_rec = util.denormalize_zero_mean(Z, hsi_min, hsi_max)
        if self.progress_callback:
            self.progress_callback(1.0)

        return hsi_rec
    

        


        

