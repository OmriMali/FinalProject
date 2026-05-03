import numpy as np
from dataclasses import dataclass

from src.compressors.base_compressor import BaseCompressor
from src.compressors.registry import register_compressor
from src.core.hsi import HSI
from src.math import regression_algs, n_way_ops, transforms, measurement_matrices
from src import util


@dataclass
class HCS1DConfig:
    """
    Configuration for HCS1D compressor.

    Parameters
    ----------
    K : int
        Sparsity for the provided axis.

    sr : float
        Sampling ratio for the provided axis.

    axis: int
        Axis to compress (H, W, B).
    
    Phi_name : str
        Measurement matrix name.

    Psi_name : str
        Sparse basis name.
    """
    K: int = 30
    sr: float = 0.5
    axis: int = 2
    Phi_name: str = "SUBSAMPLING"
    Psi_name: str = "IDCT"


@register_compressor("hcs1d")
class HCS1D(BaseCompressor):

    Config = HCS1DConfig
    
    AXIS_MAP = ["Vertical", "Horizontal", "Spectral"]

    def __init__(self, config: HCS1DConfig, progress_callback=None):
        super().__init__(progress_callback)
        
        self.config = config
        self.__dict__.update(config.__dict__)


    def compress(self, hsi: HSI):
        if self.progress_callback:
                    self.progress_callback(0.0)

        # 1. Extract data cube
        cube = hsi.get_norm_data()
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
        Y = cube.copy()
        Y = n_way_ops.mode_n_product(Y, Phi, self.axis)
        if self.progress_callback:
            self.progress_callback(0.8)

        # 4. Quantization & Bit Packing
        max_int = (1 << hsi.bitdepth) - 1
        Y_max = np.max(np.abs(Y))
        Y_quantized = np.clip(np.round((Y + Y_max) / 2 * max_int).astype(np.uint64), 0, max_int)
        bitstream = util.pack_to_bit_depth(Y_quantized, hsi.bitdepth)
        if self.progress_callback:
            self.progress_callback(1.0)

        metadata = {
            "Y_shape": Y.shape,
            "Y_max": Y_max,
            "hsi_rec_dict": hsi.to_dict(),
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
        hsi_rec_dict = metadata["hsi_rec_dict"]
        Y_shape = metadata["Y_shape"]
        Y_max = metadata["Y_max"]


        Y_quantized = util.unpack_from_bit_depth(bitstream, hsi_rec_dict["bitdepth"], Y_shape)
        max_int = (1 << hsi_rec_dict["bitdepth"]) - 1
        Y = (Y_quantized.astype(np.float64) / max_int) * 2 - Y_max
        if self.progress_callback:
            self.progress_callback(0.05)

        # 2. Setup recovery
        seed = metadata["seed"]
        n = hsi_rec_dict["shape"][self.axis]
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
        Y_unfolded = n_way_ops.mode_n_unfold(Y, self.axis)
        num_pixels = Y_unfolded.shape[1]
        X_unfolded = np.zeros((D.shape[1], num_pixels))

        for i in range(num_pixels):
            x_rec = regression_algs.omp(D, Y_unfolded[:, i], self.K, tol=1e-2)
            X_unfolded[:, i] = x_rec
            if self.progress_callback and i % 100 == 0:
                self.progress_callback(i / num_pixels)
        
        # 4. Recover the hsi
        pixel_shape = list(hsi_rec_dict["shape"])
        pixel_shape[self.axis] = D.shape[1]
        X = n_way_ops.mode_n_fold(X_unfolded, self.axis, pixel_shape)
        Z = n_way_ops.mode_n_product(X, Psi_norm, self.axis)

        hsi_rec = HSI.from_normalized(Z, hsi_rec_dict)
        if self.progress_callback:
            self.progress_callback(1.0)

        return hsi_rec
    

        


        

