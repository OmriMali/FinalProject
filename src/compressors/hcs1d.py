import numpy as np
from dataclasses import dataclass

from src.compressors.base import Compressor, CompressorConfig
from src.compressors.registry import register_compressor
from src.core.hsi import HSI, CompressedHSI
from src.core.dictionary import Axis

from src.transforms.measurements import get_measurement
from src.transforms.sparse_bases import get_sparse_base
from src.math import regression_algs, n_way_ops, numeric
from src.utils import bitstream



@dataclass(frozen=True)
class HCS1DConfig(CompressorConfig):
    """
    Configuration for HCS1D compressor.

    Parameters
    ----------
    K : int
        Sparsity for the provided axis.

    sr : float
        Sampling ratio for the provided axis.

    axis: Axis
        Axis to compress.
    
    Phi : str
        Measurement matrix name.

    Psi : str
        Sparse basis name.
    """
    K: int = 3
    sr: float = 0.1
    axis: Axis = Axis.SPECTRAL
    Phi: str = "BERNOULLI"
    Psi: str = "LEARNED"

@register_compressor
class HCS1D(Compressor):
    """
    1D compressed sensing based hyperspectral image compressor.
    """
    name = "hcs1d"
    Config = HCS1DConfig
    
    def __init__(self, config: HCS1DConfig, progress_callback=None):
        super().__init__(config, progress_callback)


    def compress(self, hsi: HSI) -> CompressedHSI:
        self.report_progress(0.0)

        # 1. Extract data
        y, hsi_min, hsi_max = numeric.normalize(hsi.data)
        self.report_progress(0.1)

        # 2. Generate measurement matrix
        n = hsi.shape[self.config.axis.value]
        p = int(self.config.sr * n)
        seed = np.random.randint(0, 1_000_000)
        Phi = get_measurement(self.config.Phi, p, n, seed)
        self.report_progress(0.2)

        # 3. Get hsi measurements
        y = n_way_ops.mode_n_product(y, Phi, self.config.axis.value)
        self.report_progress(0.5)

        # 4. Quantization
        quantized, y_max = numeric.quantize_symmetric(y, hsi.metadata.bit_depth)
        self.report_progress(0.6)

        # 5. Pack to bitstream
        stream = bitstream.pack_to_bit_depth(quantized, hsi.metadata.bit_depth)
        self.report_progress(0.95)

        # 6. Create output object
        compressed = CompressedHSI(
            bitstream=stream,
            metadata=hsi.metadata,
            side_information= {
                "hsi_min": hsi_min,
                "hsi_max": hsi_max,
                "y_shape": y.shape,
                "y_max": y_max,
                "seed": seed
            }
        )
        self.report_progress(1.0)
        
        return compressed


    def decompress(self, compressed: CompressedHSI) -> HSI:
        self.report_progress(0.0)
        
        # 1. Unpack bitstream
        quantized = bitstream.unpack_from_bit_depth(compressed.bitstream,
                                                    compressed.metadata.bit_depth,
                                                    compressed.side_information["y_shape"])
        self.report_progress(0.04)

        # 2. Dequantization
        y = numeric.dequantize_symmetric(quantized,
                                         compressed.metadata.bit_depth,
                                         compressed.side_information["y_max"])
        self.report_progress(0.05)

        # 3. Get measurement and sparse basis matrices
        n = compressed.metadata.shape[self.config.axis.value]
        p = compressed.side_information["y_shape"][self.config.axis.value]
        Phi = get_measurement(self.config.Phi, p, n, compressed.side_information["seed"])
        Psi = get_sparse_base(self.config.Psi, n)
        self.report_progress(0.06)

        # 4. Create normalized dictionary
        D = Phi @ Psi
        col_norms = np.linalg.norm(D, axis=0)
        col_norms[col_norms == 0] = 1.0
        S_inv = np.diag(1.0 / col_norms)
        D = D @ S_inv
        Psi_norm = Psi @ S_inv
        self.report_progress(0.1)

        # 5. Run sparse recovery algorithm
        y_unfolded = n_way_ops.mode_n_unfold(y, self.config.axis.value)
        num_pixels = y_unfolded.shape[1]
        x_unfolded = np.zeros((D.shape[1], num_pixels))
        for i in range(num_pixels):
            x_unfolded[:, i] = regression_algs.omp(D, y_unfolded[:, i], self.config.K, tol=1e-2)
            if i % 100 == 0:
                self.report_progress(0.1 + 0.85*(i / num_pixels))
        self.report_progress(0.95)

        # 6. Get reconstruction via inverse transform
        pixel_shape = list(compressed.metadata.shape)
        pixel_shape[self.config.axis.value] = D.shape[1]
        x = n_way_ops.mode_n_fold(x_unfolded, self.config.axis.value, pixel_shape)
        z = n_way_ops.mode_n_product(x, Psi_norm, self.config.axis.value)
        z = numeric.denormalize(z,
                                compressed.side_information["hsi_min"],
                                compressed.side_information["hsi_max"])
        self.report_progress(0.99)

        # 7. Create output object
        reconstruction = HSI(z, compressed.metadata)
        self.report_progress(1.0)

        return reconstruction

