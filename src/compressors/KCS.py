from src.compressors.base import BaseCompressor
from src import util
import numpy as np


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
        self.K = sparsity_K
        self.seed = seed

    ###########################################################################
    # COMPRESS
    ###########################################################################

    def compress(self, hsi):

        # 1. Statistics + normalization
        min_val, max_val, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, min_val, max_val)

        # 2. Setup sensing matrices
        cr_per_mode = self.targetCR ** (1 / 3)

        I_dims = hsi.shape
        M_dims = [int(dim / cr_per_mode) for dim in I_dims]

        Phis = [
            util.GaussianMeasurementMatrix(I_dims[n], M_dims[n], seed=self.seed + n)
            for n in range(3)
        ]

        # 3. Mode-n sensing
        Y = hsi_norm
        for n in range(3):
            Y = Phis[n].forward(Y, axis=n)

        # 4. Quantization (dynamic range based)
        y_min = Y.min()
        y_max = Y.max()

        max_int = (1 << bit_depth) - 1

        Y_quant = np.round((Y - y_min) / (y_max - y_min) * max_int)
        Y_quant = np.clip(Y_quant, 0, max_int).astype(np.uint64)

        bitstream = util.pack_to_bit_depth(Y_quant, bit_depth)

        metadata = {
            "y_shape": Y.shape,
            "hsi_shape": hsi.shape,
            "min_val": min_val,
            "max_val": max_val,
            "bit_depth": bit_depth,
            "y_min": float(y_min),
            "y_max": float(y_max),
            "params": {
                "targetCR": self.targetCR,
                "K": self.K
            }
        }

        return bitstream, metadata

    ###########################################################################
    # DECOMPRESS
    ###########################################################################

    def decompress(self, bitstream, metadata):

        # 1. Unpack measurements
        Y_quant = util.unpack_from_bit_depth(
            bitstream,
            metadata["bit_depth"],
            metadata["y_shape"]
        )

        max_int = (1 << metadata["bit_depth"]) - 1

        y_min = metadata["y_min"]
        y_max = metadata["y_max"]

        Y = (Y_quant.astype(np.float64) / max_int) * (y_max - y_min) + y_min

        I_dims = metadata["hsi_shape"]
        M_dims = metadata["y_shape"]

        # 2. Build dictionaries D_n = Phi_n * W_n
        Ds = []

        for n in range(3):

            Phi = util.GaussianMeasurementMatrix(
                I_dims[n],
                M_dims[n],
                seed=self.seed + n
            ).matrix

            W_inv = util.DCTBasis().inverse(np.eye(I_dims[n]), axis=0)

            Ds.append(Phi @ W_inv)

        # 3. KCS / Kronecker-OMP

        residual = Y.copy()

        W_atoms = [np.zeros((M_dims[n], self.K)) for n in range(3)]

        selected_indices = []

        Z = np.zeros((self.K, self.K))

        coeffs = np.zeros(self.K)

        for k in range(1, self.K + 1):

            # ----- correlation search -----

            corr = residual.copy()

            for n in range(3):
                corr = np.moveaxis(
                    np.tensordot(Ds[n].T, corr, axes=([1], [n])),
                    0,
                    n
                )

            idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)

            selected_indices.append(idx)

            # ----- store atoms -----

            for n in range(3):
                W_atoms[n][:, k - 1] = Ds[n][:, idx[n]]

            # ----- update Gram matrix -----

            for i in range(k):

                val = 1.0

                for n in range(3):
                    val *= W_atoms[n][:, i].T @ W_atoms[n][:, k - 1]

                Z[i, k - 1] = val
                Z[k - 1, i] = val

            # ----- compute projections -----

            y_proj = np.zeros(k)

            for i in range(k):

                atom = np.multiply.outer(
                    np.multiply.outer(
                        W_atoms[0][:, i],
                        W_atoms[1][:, i]
                    ),
                    W_atoms[2][:, i]
                )

                y_proj[i] = np.sum(Y * atom)

            # ----- solve LS -----

            a = np.linalg.solve(
                Z[:k, :k] + 1e-6 * np.eye(k),
                y_proj
            )

            coeffs[:k] = a

            # ----- residual update -----

            fit = np.zeros_like(Y)

            for i in range(k):

                atom = np.multiply.outer(
                    np.multiply.outer(
                        W_atoms[0][:, i],
                        W_atoms[1][:, i]
                    ),
                    W_atoms[2][:, i]
                )

                fit += coeffs[i] * atom

            residual = Y - fit

            self._update_progress(k / self.K)

        # 4. Build sparse tensor

        X_sparse = np.zeros(I_dims)

        for i, idx in enumerate(selected_indices):
            X_sparse[idx[0], idx[1], idx[2]] = coeffs[i]

        # 5. Inverse separable DCT

        hsi_rec = util.DCTBasis().inverse(X_sparse, axis=2)
        hsi_rec = util.DCTBasis().inverse(hsi_rec, axis=1)
        hsi_rec = util.DCTBasis().inverse(hsi_rec, axis=0)

        # 6. Denormalize

        return util.denormalize_zero_mean(
            hsi_rec,
            metadata["min_val"],
            metadata["max_val"]
        )