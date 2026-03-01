from .base import BaseCompressor
from src import util
import numpy as np
from scipy.linalg import solve_triangular

class KCSCompressor(BaseCompressor):
    @property
    def name(self): 
        return "kcs"

    @property
    def compressor_id(self): 
        return 25 

    def __init__(self, targetCR=5, sparsity_K=50, seed=42, progress_callback=None):
        super().__init__(progress_callback=progress_callback)
        self.targetCR = targetCR
        self.K = sparsity_K # [cite: 318]
        self.seed = seed

    def compress(self, hsi):
        # 1. Statistics and Normalization
        min_val, max_val, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, min_val, max_val)
        
        # 2. Setup Mode-n Sensing Matrices (Kronecker Structure) [cite: 258]
        cr_per_mode = self.targetCR**(1/3)
        I_dims = hsi.shape
        M_dims = [int(dim / cr_per_mode) for dim in I_dims]
        
        Phis = [util.GaussianMeasurementMatrix(I_dims[n], M_dims[n], seed=self.seed+n) 
                for n in range(3)]
        
        # 3. Generate Measurements: Y = HSI x1 Phi1 x2 Phi2 x3 Phi3 [cite: 166, 269]
        Y = hsi_norm
        for n in range(3):
            Y = Phis[n].forward(Y, axis=n)

        # 4. Quantize and Pack
        max_int = (1 << bit_depth) - 1
        Y_quant = np.clip(np.round((Y + 1) / 2 * max_int), 0, max_int).astype(np.uint64)
        bitstream = util.pack_to_bit_depth(Y_quant, bit_depth)

        metadata = {
            "y_shape": Y.shape, "hsi_shape": hsi.shape,
            "min_val": min_val, "max_val": max_val, "bit_depth": bit_depth,
            "params": {"targetCR": self.targetCR, "K": self.K}
        }
        return bitstream, metadata

    def decompress(self, bitstream, metadata):
        # 1. Unpack
        Y_quant = util.unpack_from_bit_depth(bitstream, metadata["bit_depth"], metadata["y_shape"])
        max_int = (1 << metadata["bit_depth"]) - 1
        Y = (Y_quant.astype(np.float64) / max_int) * 2 - 1
        
        I_dims = metadata["hsi_shape"]
        M_dims = metadata["y_shape"]
        
        # 2. Mode-n Dictionaries D_n = Phi_n * W_n [cite: 187, 258]
        Ds = []
        for n in range(3):
            Phi_n = util.GaussianMeasurementMatrix(I_dims[n], M_dims[n], seed=self.seed+n).matrix
            W_inv_n = util.DCTBasis().inverse(np.eye(I_dims[n]), axis=0) # [cite: 49]
            Ds.append(Phi_n @ W_inv_n)

        # 3. Kronecker-OMP Algorithm 
        residual = Y.copy()
        W_atoms = [np.zeros((M_dims[n], self.K)) for n in range(3)]
        selected_indices = []
        
        # For numerical stability, we solve the LS problem directly using the Gram matrix
        # rather than the Schur inverse update during instability 
        Z = np.zeros((self.K, self.K))

        for k in range(1, self.K + 1):
            # STEP 3: Find maximum correlation [cite: 327]
            corr = residual.copy()
            for n in range(3):
                corr = np.moveaxis(np.tensordot(Ds[n].T, corr, axes=([1], [n])), 0, n)
            
            idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
            selected_indices.append(idx)
            
            # STEP 4: Update Support [cite: 328]
            for n in range(3):
                W_atoms[n][:, k-1] = Ds[n][:, idx[n]]
            
            # Update Gram matrix Z [cite: 304]
            for i in range(k):
                val = 1.0
                for n in range(3):
                    val *= (W_atoms[n][:, i].T @ W_atoms[n][:, k-1])
                Z[i, k-1] = val
                Z[k-1, i] = val

            # Solve for coefficients 'a' via LS [cite: 110, 329]
            y_proj_vec = np.zeros(k)
            for i in range(k):
                temp_Y = Y.copy()
                for n in range(3):
                    temp_Y = np.tensordot(W_atoms[n][:, i], temp_Y, axes=([0], [0]))
                y_proj_vec[i] = temp_Y
            
            # Regularized solve to prevent NaNs from singular matrices [cite: 83]
            a = np.linalg.solve(Z[:k, :k] + 1e-6 * np.eye(k), y_proj_vec)

            # STEP 6: Update Residual [cite: 330]
            fit = np.zeros_like(Y)
            for i in range(k):
                atom_3d = a[i] * np.multiply.outer(np.multiply.outer(W_atoms[0][:, i], W_atoms[1][:, i]), W_atoms[2][:, i])
                fit += atom_3d
            residual = Y - fit
            
            self._update_progress(k / self.K)

        # 4. Final Reconstruction (Core tensor X mapping) [cite: 188]
        X_sparse = np.zeros(I_dims)
        for i in range(self.K):
            idx = selected_indices[i]
            X_sparse[idx[0], idx[1], idx[2]] = a[i]
        
        # Inverse separable transform [cite: 135, 177]
        hsi_rec = util.DCTBasis().inverse(X_sparse, axis=2)
        hsi_rec = util.DCTBasis().inverse(hsi_rec, axis=1)
        hsi_rec = util.DCTBasis().inverse(hsi_rec, axis=0)

        return util.denormalize_zero_mean(hsi_rec, metadata["min_val"], metadata["max_val"])