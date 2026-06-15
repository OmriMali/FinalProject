import numpy as np
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from dataclasses import dataclass

from src.compressors.base import Compressor, CompressorConfig
from src.compressors.registry import register_compressor
from src.core.hsi import HSI, CompressedHSI

from src.utils import misc
from src.math import numeric, n_way_ops, regression_algs
from src.transforms import get_measurement, get_sparse_base


@dataclass(frozen=True)
class HybridConfig(CompressorConfig):
    """
    Configuration for hcs1d & ccsds123 hybrid compressor.

    Parameters
    ----------
    K : int
        Estimated sparsity for the spectral axis.

    sr : float
        Sampling rate in the spectral axis.

    Phi : str
        Measurement matrix name.

    Psi : str
        Sparse basis name.

    local_sum_mode : str
        Mode for local sum. accepts: 'column', 'neighbor'

    block size : int
        Block size for encoder.

    protect_bitstream: bool
        Protect sensitive parts of bitstream from BER.
    """
    K: int = 3
    sr: float = 0.1
    Phi: str = "BERNOULLI"
    Psi: str = "LEARNED"
    local_sum_mode: str = 'neighbor'
    block_size: int = 32
    protect_bitstream: bool = False

@register_compressor
class Hybrid(Compressor):
    """
    An hybrid of the CCSDS123 and HCS1D compressors. The spatial compression is done
    by a CCSDS123 like prediction loop, and the spectral compression by an HCS1D
    sparse recovery.
    """  
    name = "hybrid"
    Config = HybridConfig

    def __init__(self, config: HybridConfig, progress_callback=None):
        super().__init__(config, progress_callback)


    def compress(self, hsi: HSI) -> CompressedHSI:

        self.report_progress(0.0)
        
        # 1. Normalize data
        y, hsi_min, hsi_max = numeric.normalize(hsi.data)

        # 2. Generate measurement matrix
        n = hsi.shape[2]
        p = int(self.config.sr * n)
        seed = np.random.randint(0, 1_000_000)
        Phi = get_measurement(self.config.Phi, p, n, seed)
        self.report_progress(0.04)
        
        # 3. Take measurements
        y = n_way_ops.mode_n_product(y, Phi, 2)
        self.report_progress(0.08)

        # 4. Quantize samples
        y_quant, y_max = numeric.quantize_symmetric(y, hsi.metadata.bit_depth)
        self.report_progress(0.1)

        # 5. Prediction loop
        scaled_report = self._scaled_progress(0.1, 0.8)
        delta = self._encoder_predictor(y_quant, scaled_report)

        # 6. Convert residuals to bitstream
        scaled_report = self._scaled_progress(0.8, 0.99)
        stream, k_values = self._rice_encode(delta, scaled_report)

        # 7. Compute BER protection mask (optional)
        protection_mask = None 
        if self.config.protect_bitstream == True:
            protection_mask = self._calculate_protection_mask(delta, k_values)

        # 8. Create output object
        compressed = CompressedHSI(
            bitstream=stream,
            metadata=hsi.metadata,
            side_information={
                "hsi_min": hsi_min,
                "hsi_max": hsi_max,
                "y_max": y_max,
                "y_shape": y.shape,
                "seed": seed,
                "protection_mask": protection_mask
            }
        )
        self.report_progress(1.0)
        
        return compressed

    def decompress(self, compressed: CompressedHSI) -> HSI:

        self.report_progress(0.0)

        # 1. Get residuals by decoding bitstream
        scaled_report = self._scaled_progress(0.0, 0.1)
        delta = self._rice_decode(
            compressed.bitstream,
            compressed.side_information["y_shape"], scaled_report)

        # 2. Prediction loop
        scaled_report = self._scaled_progress(0.1, 0.5)
        y_quant = self._decoder_predictor(delta, scaled_report)
        
        # 3. Dequantization
        y = numeric.dequantize_symmetric(
            y_quant,
            compressed.metadata.bit_depth,
            compressed.side_information["y_max"])

        if self.config.sr < 1:
            # 3. Get measurement and sparse basis matrices
            n = compressed.metadata.shape[2]
            p = compressed.side_information["y_shape"][2]
            Phi = get_measurement(self.config.Phi, p, n, compressed.side_information["seed"])
            Psi = get_sparse_base(self.config.Psi, n)
            self.report_progress(0.52)

            # 4. Create normalized dictionary
            D = Phi @ Psi
            col_norms = np.linalg.norm(D, axis=0)
            col_norms[col_norms == 0] = 1.0
            S_inv = np.diag(1.0 / col_norms)
            D = D @ S_inv
            Psi_norm = Psi @ S_inv
            self.report_progress(0.54)

            # 5. Run sparse recovery algorithm
            y_unfolded = n_way_ops.mode_n_unfold(y, 2)
            num_pixels = y_unfolded.shape[1]
            x_unfolded = np.zeros((D.shape[1], num_pixels))
            for i in range(num_pixels):
                x_unfolded[:, i] = regression_algs.omp(D, y_unfolded[:, i], self.config.K, tol=1e-2)
                if i % 100 == 0:
                    self.report_progress(0.54 + 0.45*(i / num_pixels))
            self.report_progress(0.99)

            # 6. Get reconstruction via inverse transform
            pixel_shape = list(compressed.metadata.shape)
            pixel_shape[2] = D.shape[1]
            x = n_way_ops.mode_n_fold(x_unfolded, 2, pixel_shape)
            z = n_way_ops.mode_n_product(x, Psi_norm, 2)
        
        else:
            z = y
        
        # 7. Denormalize
        z = numeric.denormalize(
            z,
            compressed.side_information["hsi_min"],
            compressed.side_information["hsi_max"]
        )

        # 8. Create output object
        reconstruction = HSI(z, compressed.metadata)
        self.report_progress(1.0)

        return reconstruction

    def _predict_sample(self, S_rep_z, x: int, y: int) -> int:
        """
        Predict one sample from already reconstructed spatial neighbors.
        """
        H, W = S_rep_z.shape
        
        def rep(xx, yy):
            xx = min(max(xx, 0), W - 1)
            yy = min(max(yy, 0), H - 1)
            return int(S_rep_z[yy, xx])

        # Retrieve mode, default to original if not specified
        mode = getattr(self.config, 'local_sum_mode', 'hybrid_mean')

        if mode == 'column':
            if y > 0:
                return rep(x, y-1)
            elif x > 0:
                return rep(x-1, y)
            else:
                return 0

        elif mode == 'neighbor':
            if y == 0 and x == 0:
                return 0
            elif y == 0:
                return rep(x-1, y)
            elif x == 0:
                return (rep(x, y-1) + rep(x+1, y-1)) // 2
            elif x == W - 1:
                return (rep(x-1, y) + rep(x-1, y-1) + 2 * rep(x, y-1)) // 4
            else:
                return (rep(x-1, y) + rep(x-1, y-1) + rep(x, y-1) + rep(x+1, y-1)) // 4

        # Default to original hybrid predictor (hybrid_mean)
        else:
            if y == 0 and x == 0:
                return 0
            
            if y == 0:
                return rep(x-1, y)
            
            if x == 0:
                return rep(x, y-1)
            
            left = rep(x-1, y)
            up = rep(x, y-1)

            return (left + up) // 2

    def _encoder_predictor(self, S, report_callback=None):
        """
        Convert integer samples to mapped residuals.
        """
        S = np.asarray(S, dtype=np.int32)
        H, W, B = S.shape

        S_rep = np.zeros((H, W, B), dtype=np.int32)
        delta = np.zeros((H, W, B), dtype=np.int32)

        for z in range(B):
            if report_callback:
                report_callback(z / B)
            
            for y in range(H):
                for x in range(W):
                    pred = self._predict_sample(S_rep[:, :, z], x, y)
                    residual = S[y, x, z] - pred

                    if residual >= 0:
                        mapped = 2 * residual
                    else:
                        mapped = -2 * residual - 1

                    delta[y, x, z] = mapped

                    S_rep[y, x, z] = S[y, x, z]

        return delta    
    
    def _decoder_predictor(self, delta, report_callback=None):
        """
        Convert mapped prediction residuals back to integer samples.
        """
        delta = np.asarray(delta, dtype=np.int32)
        H, W, B = delta.shape

        S_rep = np.zeros((H, W, B), dtype=np.int32)

        for z in range(B):
            if report_callback:
                report_callback(z / B)

            for y in range(H):
                for x in range(W):
                    pred = self._predict_sample(S_rep[:, :, z], x, y)

                    mapped = delta[y, x, z]

                    if mapped % 2 == 0:
                        residual = mapped // 2
                    else:
                        residual = -((mapped + 1) // 2)

                    S_rep[y, x, z] = pred + residual

        return S_rep


    def _rice_encode(self, delta, report_callback=None):
        """
        Encode mapped residuals using adaptive Rice coding.

        Parameters
        ----------
        delta : np.ndarray
            Mapped prediction residuals.

        report_callback : callable | None, optional
            Progress callback function.

        Returns
        -------
        tuple[bytes, list[int]]
            Tuple containing:

            - Encoded bitstream
            - Rice parameter values for each block
        """

        bits = bitarray(endian='big')
        data = delta.flatten(order='F')
        N = len(data)
        k_values = []
        total_blocks = (N + self.config.block_size - 1) // self.config.block_size

        for b_idx in range(total_blocks):
            if report_callback:
                report_callback(b_idx / total_blocks)
       
            # Get block and calculate k (Rice parameter)
            block = data[b_idx * self.config.block_size : min((b_idx+1)*self.config.block_size, N)]
            med = np.median(block) if len(block) > 0 else 0
            k = int(max(0, np.floor(np.log2(med + 1)))) if med > 0 else 0
            k_values.append(k)
            # Write k to bitstream (using 4 bits for k)
            bits.extend(int2ba(k, length=4))

            for val in block:
                q = int(val) >> k
                r = int(val) & ((1 << k) - 1)
                
                # Unary part: q ones followed by a zero
                bits.extend('1' * q + '0')
                # Remainder part: k bits
                if k > 0:
                    bits.extend(int2ba(r, length=k))

        return bits.tobytes(), k_values

    def _rice_decode(self, bitstream_bytes, shape, report_callback=None):
        """
        Decode a Rice-coded residual bitstream.

        Parameters
        ----------
        bitstream_bytes : bytes
            Encoded residual bitstream.

        shape : tuple[int, int, int]
            Shape of the reconstructed residual cube.

        report_callback : callable | None, optional
            Progress callback function.

        Returns
        -------
        np.ndarray
            Decoded mapped residuals.
        """
        bits = bitarray(endian='big')
        bits.frombytes(bitstream_bytes)
        
        N = np.prod(shape)
        flat_delta = np.zeros(N, dtype=np.int32)
        
        bit_ptr = 0
        val_idx = 0
        total_bits = len(bits)
        while val_idx < N and bit_ptr < total_bits:
            if report_callback:
                report_callback(val_idx / N)
            
            # Read k (4 bits)
            k = ba2int(bits[bit_ptr : bit_ptr + 4])
            bit_ptr += 4
            
            # Read a block of samples
            for _ in range(self.config.block_size):
                if val_idx >= N or bit_ptr >= total_bits: break
                
                # Decode Unary (count ones until 0)
                q = 0
                while bit_ptr < total_bits and bits[bit_ptr]:
                    q += 1
                    bit_ptr += 1
                bit_ptr += 1 # skip the 0
                
                # Decode Remainder
                r = 0
                if k > 0:
                    r = ba2int(bits[bit_ptr : bit_ptr + k])
                    bit_ptr += k
                
                flat_delta[val_idx] = (q << k) + r
                val_idx += 1
                
        return flat_delta.reshape(shape, order='F')
    
    def _calculate_protection_mask(self, delta, k_values):
        """
        Build a bit-level BER protection mask for the encoded bitstream.

        Protected bits are marked with ``1`` and vulnerable bits
        are marked with ``0``.

        Parameters
        ----------
        delta : np.ndarray
            Mapped prediction residuals.

        k_values : list[int]
            Rice parameter values for each encoded block.

        Returns
        -------
        bitarray
            Bit-level protection mask aligned to the encoded bitstream.
        """
        mask = bitarray(endian='big')
        flat_delta = delta.flatten(order='F')
        delta_ptr = 0
        
        for k in k_values:
            # Mark the 4-bit Rice parameter header as PROTECTED
            mask.extend('1' * 4) 
            
            for _ in range(self.config.block_size):
                if delta_ptr >= len(flat_delta): 
                    break
                val = flat_delta[delta_ptr]
                q = int(val) >> k
                # Unary (q+1) and Remainder (k) are VULNERABLE
                mask.extend('0' * (q + 1 + k))
                delta_ptr += 1
        
        # --- NEW: Pad mask to match byte-alignment of tobytes() ---
        padding_needed = (8 - (len(mask) % 8)) % 8
        mask.extend('0' * padding_needed) # Padding is vulnerable
        # ---------------------------------------------------------
                
        return mask


    def _scaled_progress(self, start: float, end: float):
        """
        Create a scaled progress callback.

        Parameters
        ----------
        start : float
            Start value of scaled progress interval.

        end : float
            End value of scaled progress interval.

        Returns
        -------
        callable | None
            Scaled progress callback function.
        """
        if self._progress_callback:
            return misc.scaled_callback(self.report_progress, start, end)
        return None

