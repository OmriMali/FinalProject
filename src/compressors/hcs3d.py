import numpy as np
from dataclasses import dataclass
from typing import Tuple

from src.compressors.base import Compressor, CompressorConfig
from src.compressors.registry import register_compressor
from src.core.hsi import HSI, CompressedHSI

from src.transforms.measurements import get_measurement
from src.transforms.sparse_bases import get_sparse_base
from src.math import regression_algs, n_way_ops, numeric
from src.utils import misc, bitstream


@dataclass(frozen=True)
class HCS3DConfig(CompressorConfig):
    """
    Configuration for HCS3D compressor.

    Parameters
    ----------
    K : int
        Sparsity of the entire HSI.

    sr : tuple of float (length 3)
        Sampling ratio for each dimension (H, W, B).

    Phis : tuple of str (length 3)
        Measurement matrices for each dimension (H, W, B).

    Psis : tuple of str (length 3)
        Sparse basis for each dimension (H, W, B).
    """
    K: int = 4000
    sr: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    Phis: Tuple[str, str, str] = ("SUBSAMPLING", "SUBSAMPLING", "SUBSAMPLING")
    Psis: Tuple[str, str, str] = ("IDCT", "IDCT", "IDCT")

@register_compressor
class HCS3D(Compressor):
    """
    3D compressed sensing based hyperspectral image compressor.
    """
    name = "hcs3d"
    Config = HCS3DConfig

    def __init__(self, config: HCS3DConfig, progress_callback=None):
        super().__init__(config, progress_callback)


    def compress(self, hsi: HSI) -> CompressedHSI:
        self.report_progress(0.0)

        # 1. Extract data 
        y, hsi_min, hsi_max = numeric.normalize(hsi.data)
        self.report_progress(0.1)
        
        # 2. Generate measurement matrices
        seeds = []
        Phis = []
        for i in range(3):
            n = hsi.shape[i]
            p = int(self.config.sr[i] * n)
            seeds.append(np.random.randint(0, 1_000_000))
            Phis.append(get_measurement(self.config.Phis[i], p, n, seeds[i]))
        self.report_progress(0.2)

        # 3. Get hsi measurements
        for i in range(3):
            y = n_way_ops.mode_n_product(y, Phis[i], i)
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
            side_information={
                "hsi_min": hsi_min,
                "hsi_max": hsi_max,
                "y_shape": y.shape,
                "y_max": y_max,
                "seeds": seeds
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

        # 3. Create normalized sparse bases and dictionaries
        Psis_norm = []
        Ds = []
        for i in range(3):
            n = compressed.metadata.shape[i]
            p = compressed.side_information["y_shape"][i]
            Phi = get_measurement(self.config.Phis[i], p, n, compressed.side_information["seeds"][i])
            Psi = get_sparse_base(self.config.Psis[i], n)
            D = Phi @ Psi

            col_norms = np.linalg.norm(D, axis=0)
            col_norms[col_norms == 0] = 1.0
            S_inv = np.diag(1.0 / col_norms)
            Ds.append(D @ S_inv)
            Psis_norm.append(Psi @ S_inv)
        self.report_progress(0.1)
        
        # 4. Run sparse recovery algorithm
        omp_callback = None
        if self._progress_callback:
            omp_callback = misc.scaled_callback(self.report_progress, 0.1, 0.95)
       
        x = regression_algs.n_bomp(Ds, y, self.config.K, tol=1e-2, progress_callback=omp_callback)

        # 5. Get reconstruction via inverse transforms
        z = x
        for n in range(3):
            z = n_way_ops.mode_n_product(z, Psis_norm[n], n)
        z = numeric.denormalize(z,
                                compressed.side_information["hsi_min"],
                                compressed.side_information["hsi_max"])
        self.report_progress(0.99)

        # 6. Create output object
        reconstruction = HSI(z, compressed.metadata)
        self.report_progress(1.0)

        return reconstruction

