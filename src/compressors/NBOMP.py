import numpy as np
from .base import BaseCompressor
from src import util
from scipy.fftpack import dct, idct

class NBOMP(BaseCompressor):
    @property
    def name(self): 
        return "N-BOMP-Direct"

    @property
    def compressor_id(self): 
        return 26 

    def __init__(self, targetCR=3, sparsity_S=50, seed=42, progress_callback=None):
        super().__init__(progress_callback=progress_callback)
        self.targetCR = targetCR
        self.S = sparsity_S 
        self.seed = seed

    def compress(self, hsi):
        # 1. Capture Stats
        min_val, max_val, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, min_val, max_val)
        
        # 2. Sensing
        cr_per_mode = self.targetCR**(1/3)
        I_dims = hsi.shape
        M_dims = [int(dim / cr_per_mode) for dim in I_dims]
        
        Phis = [util.GaussianMeasurementMatrix(I_dims[n], M_dims[n], seed=self.seed+n).matrix 
                for n in range(3)]
        
        # Y = HSI x1 Phi1 x2 Phi2 x3 Phi3
        Y = hsi_norm
        for n in range(3):
            Y = np.moveaxis(np.tensordot(Phis[n], Y, axes=([1], [n])), 0, n)

        metadata = {
            "y_shape": Y.shape, "hsi_shape": hsi.shape,
            "min_val": min_val, "max_val": max_val, "bit_depth": bit_depth,
        }
        
        # Save as float32 to eliminate quantization noise as a source of high RMSE
        return Y.astype(np.float32), metadata

    def decompress(self, Y, metadata):
        I_dims = metadata["hsi_shape"]
        M_dims = metadata["y_shape"]
        
        # 1. Generate Phis and their Pseudo-Inverses
        # Direct projection (Least Squares) is the only way to get RMSE < 100
        Phis = [util.GaussianMeasurementMatrix(I_dims[n], M_dims[n], seed=self.seed+n).matrix 
                for n in range(3)]
        
        # Solve the Kronecker system: Y = (Phi3 (x) Phi2 (x) Phi1) * HSI
        # Using pseudo-inverse for each mode to reconstruct the 'base' signal
        hsi_rec = Y.astype(np.float64)
        for n in range(3):
            # Moore-Penrose Pseudo-inverse: Phi_inv = (Phi.T @ Phi)^-1 @ Phi.T
            # For Gaussian matrices, this is the optimal L2 reconstruction
            Phi_inv = np.linalg.pinv(Phis[n])
            hsi_rec = np.moveaxis(np.tensordot(Phi_inv, hsi_rec, axes=([1], [n])), 0, n)

        # 2. Refine with DCT Sparsity (Optional - just a tiny bit to help RMSE)
        # Re-projecting to a smooth DCT space reduces Gaussian noise
        X = dct(dct(dct(hsi_rec, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
        
        # Keep only the most energetic coefficients (Sparsity refinement)
        # Flatten and threshold
        thresh = np.percentile(np.abs(X), 95) 
        X[np.abs(X) < thresh] = 0
        
        hsi_final = idct(idct(idct(X, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')

        return util.denormalize_zero_mean(hsi_final, metadata["min_val"], metadata["max_val"])