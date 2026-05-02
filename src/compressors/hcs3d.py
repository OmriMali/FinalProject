import numpy as np

from src.compressors.base_compressor import BaseCompressor
from src.compressors.registry import register_compressor
from src.math import regression_algs, n_way_ops, transforms, measurement_matrices
from src.core.hsi import HSI
from src import util



@register_compressor("hcs3d")
class HCS3D(BaseCompressor):

    def __init__(self, K=3, sr=[0.5, 0.5, 1],
                 Phi_names=["SUBSAMPLING", "SUBSAMPLING", "IDENTITY"],
                 Psi_names=["IDCT", "IDCT", "IDCT"], progress_callback=None):
        super().__init__(progress_callback)
        self.K = K
        self.sr = sr
        self.Phi_names = Phi_names
        self.Psi_names = Psi_names

    def compress(self, hsi: HSI):
        if self.progress_callback:
            self.progress_callback(0.0)

        # 1. Extract data cube
        cube = hsi.get_norm_data()
        shape = hsi.shape
        if self.progress_callback:
            self.progress_callback(0.2)
        
        # 2. Generate Phis
        seeds = [np.random.randint(0, 1_000_000) for _ in range(len(shape))]
        Phis = []
        for i in range(len(shape)):
            Phis.append(measurement_matrices.get_measurement_matrix(
                    self.Phi_names[i], int(self.sr[i] * shape[i]), shape[i],seed=seeds[i]))
        if self.progress_callback:
            self.progress_callback(0.4)

        # 3. Get Measurement Array
        Y = cube.copy()
        for i in range(len(shape)):
            Y = n_way_ops.mode_n_product(Y, Phis[i], i)
        if self.progress_callback:
            self.progress_callback(0.6)

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
        if self.progress_callback:
            self.progress_callback(0.0)
            
        # 1. Unpack & Dequantize
        hsi_rec_dict = metadata["hsi_rec_dict"]
        shape = hsi_rec_dict["shape"]
        Y_shape = metadata["Y_shape"]
        Y_max = metadata["Y_max"]

        Y_quantized = util.unpack_from_bit_depth(bitstream, hsi_rec_dict["bitdepth"], Y_shape)
        max_int = (1 << hsi_rec_dict["bitdepth"]) - 1
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
            D = Phi @ Psi

            col_norms = np.linalg.norm(D, axis=0)
            col_norms[col_norms == 0] = 1.0
            S_inv = np.diag(1.0 / col_norms)
            Ds.append(D @ S_inv)
            Psis_norm.append(Psi @ S_inv)

        if self.progress_callback:
            self.progress_callback(0.1)
        
        # 3. Run recovery algorithm
        omp_callback = None
        if self.progress_callback:
            omp_callback = util.scaled_callback(self.progress_callback, 0.1, 0.95)
       
        X = regression_algs.n_bomp(Ds, Y, self.K, tol=1e-2 ,progress_callback=omp_callback)

        # 4. Recover the hsi
        Z = X
        for n in range(len(Psis_norm)):
            Z = n_way_ops.mode_n_product(Z, Psis_norm[n], n)
        
        hsi_rec = HSI.from_normalized(Z, hsi_rec_dict)
        
        if self.progress_callback:
            self.progress_callback(1.0)

        return hsi_rec

        





        

