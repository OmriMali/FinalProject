import numpy as np
from src.compressors.base import BaseCompressor
from src import util
from scipy.fftpack import dct, idct

class NBOMP(BaseCompressor):
    @property
    def name(self): 
        return "Block-KCS-OMP"

    @property
    def compressor_id(self): 
        return 26 

    def __init__(self, targetCR=5, sparsity_S=50, block_size=8, seed=42, progress_callback=None):
        super().__init__(progress_callback=progress_callback)
        self.targetCR = targetCR
        self.S = sparsity_S 
        self.B = block_size 
        self.seed = seed

    def _mode_n_prod(self, tensor, matrix, mode):
        """Standard Mode-n product: multiplies a tensor by a matrix along a specific mode."""
        # Contract the matrix with the tensor at the specified mode
        res = np.tensordot(matrix, tensor, axes=([1], [mode]))
        # tensordot puts the new dimension at axis 0; move it back to the original mode position
        return np.moveaxis(res, 0, mode)

    def compress(self, hsi):
        min_val, max_val, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, min_val, max_val).astype(np.float32)
        
        H, W, C = hsi.shape
        cr_per_mode = self.targetCR**(1/3)
        M_b = max(1, int(self.B / cr_per_mode))
        M_c = max(1, int(C / cr_per_mode))

        Phi_s = util.GaussianMeasurementMatrix(self.B, M_b, seed=self.seed).matrix
        Phi_c = util.GaussianMeasurementMatrix(C, M_c, seed=self.seed+1).matrix
        
        pad_h = (self.B - H % self.B) % self.B
        pad_w = (self.B - W % self.B) % self.B
        hsi_padded = np.pad(hsi_norm, ((0, pad_h), (0, pad_w), (0, 0)))
        
        Y_blocks = []
        for i in range(0, hsi_padded.shape[0], self.B):
            for j in range(0, hsi_padded.shape[1], self.B):
                block = hsi_padded[i:i+self.B, j:j+self.B, :]
                # Mode-n sensing: y = block x1 Phi_s x2 Phi_s x3 Phi_c
                y = self._mode_n_prod(block, Phi_s, 0)
                y = self._mode_n_prod(y, Phi_s, 1)
                y = self._mode_n_prod(y, Phi_c, 2)
                Y_blocks.append(y.astype(np.float32))

        metadata = {
            "hsi_shape": hsi.shape, "padded_shape": hsi_padded.shape,
            "block_y_shape": Y_blocks[0].shape,
            "min_val": min_val, "max_val": max_val, "bit_depth": bit_depth,
            "params": {"targetCR": self.targetCR, "S": self.S}
        }
        return np.array(Y_blocks).tobytes(), metadata

    def decompress(self, bitstream, metadata):
        H, W, C = metadata["hsi_shape"]
        pH, pW, pC = metadata["padded_shape"]
        BY_shape = metadata["block_y_shape"]
        
        Y_blocks = np.frombuffer(bitstream, dtype=np.float32).reshape(-1, *BY_shape)
        Phi_s = util.GaussianMeasurementMatrix(self.B, BY_shape[0], seed=self.seed).matrix
        Phi_c = util.GaussianMeasurementMatrix(C, BY_shape[2], seed=self.seed+1).matrix
        
        # Dictionaries: D = Phi * Inv_DCT (Orthonormal)
        W_s = dct(np.eye(self.B), axis=0, norm='ortho')
        W_c = dct(np.eye(C), axis=0, norm='ortho')
        Ds = [Phi_s @ W_s, Phi_s @ W_s, Phi_c @ W_c]
        
        hsi_rec = np.zeros((pH, pW, pC))
        block_idx = 0
        
        for i in range(0, pH, self.B):
            for j in range(0, pW, self.B):
                y_block = Y_blocks[block_idx]
                block_idx += 1
                
                # Start with DC components to anchor the signal energy
                support = [[0], [0], [0]]
                residual = y_block.copy()
                
                for _ in range(self.S):
                    # Correlation: D.T @ Residual along each mode
                    corr = self._mode_n_prod(residual, Ds[0].T, 0)
                    corr = self._mode_n_prod(corr, Ds[1].T, 1)
                    corr = self._mode_n_prod(corr, Ds[2].T, 2)
                    
                    best_idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
                    for n in range(3):
                        if best_idx[n] not in support[n]: support[n].append(best_idx[n])
                    
                    # LS Solve on Support
                    Bn = [Ds[n][:, support[n]] for n in range(3)]
                    A = y_block.copy()
                    for n in range(3):
                        Gn = Bn[n].T @ Bn[n] + 1e-10 * np.eye(len(support[n]))
                        Pn = self._mode_n_prod(A, Bn[n].T, n)
                        Pn_unf = np.moveaxis(Pn, n, 0)
                        res = np.linalg.solve(Gn, Pn_unf.reshape(len(support[n]), -1))
                        A = np.moveaxis(res.reshape(Pn_unf.shape), 0, n)
                    
                    # Compute Fit: A x1 B1 x2 B2 x3 B3
                    fit = self._mode_n_prod(A, Bn[0], 0)
                    fit = self._mode_n_prod(fit, Bn[1], 1)
                    fit = self._mode_n_prod(fit, Bn[2], 2)
                    residual = y_block - fit

                X_sparse = np.zeros((self.B, self.B, C))
                X_sparse[np.ix_(*support)] = A
                # Inverse DCT to return to image domain
                hsi_rec[i:i+self.B, j:j+self.B, :] = idct(idct(idct(X_sparse, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
            
            self._update_progress(block_idx / len(Y_blocks))

        return util.denormalize_zero_mean(hsi_rec[:H, :W, :], metadata["min_val"], metadata["max_val"])