import numpy as np
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from dataclasses import dataclass

from src.compressors.base import Compressor, CompressorConfig
from src.compressors.registry import register_compressor
from src.core.hsi import HSI, CompressedHSI

from src.utils import misc


@dataclass(frozen=True)
class CCSDS123Config(CompressorConfig):
    """
    Configuration for CCSDS123 compressor.

    Parameters
    ----------
    local_sum_mode : str
        Mode for local sum. accepts: 'column', 'neighbor'

    P : int
        Number of spectral bands used in prediction loop.

    Omega : int
        Resolution of calculation.
    
    a : int
        Absolute error limit per pixel.
        At a=0, compression is lossless.

    block size : int
        Block size for encoder.

    protect_bitstream: bool
        Protect sensitive parts of bitstream from BER.
    """
    local_sum_mode: str = 'neighbor'
    P: int = 2
    Omega: int = 8
    a: int = 0
    block_size: int = 32
    protect_bitstream: bool = False

@register_compressor
class CCSDS123(Compressor):
    """
    Consulative Committee for Space Data Systems (CCSDS) standard 123
    for compression of hyperspectral images.
    """  
    name = "ccsds123"
    Config = CCSDS123Config

    def __init__(self, config: CCSDS123Config, progress_callback=None):
        super().__init__(config, progress_callback)

        self._validate_config()
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Build weights for CCSDS123 predictor loop.
        """
        weights = []
        w_0 = (7 * (1 << self.config.Omega)) >> 3
        weights.append(w_0)

        for i in range(1, self.config.P):
            w_prev = weights[-1]
            w_next = w_prev >> 3
            weights.append(w_next)
        
        self.W = np.array(weights, dtype=np.int32)     # Weights for predictor

    def _validate_config(self) -> None:
        """
        Validate CCSDS123 configuration parameters.
        """
        if self.config.local_sum_mode not in {"column", "neighbor"}:
            raise ValueError("local_sum_mode must be either 'column' or 'neighbor'")

        if self.config.P <= 0:
            raise ValueError("P must be positive")

        if self.config.Omega <= 0:
            raise ValueError("Omega must be positive")

        if self.config.a < 0:
            raise ValueError("a must be non-negative")

        if self.config.block_size <= 0:
            raise ValueError("block_size must be positive")

    def _validate_input(self, hsi: HSI) -> None:
        """
        Validate input hyperspectral image for CCSDS123 compression.
        """
        if not np.issubdtype(hsi.data.dtype, np.integer):
            raise ValueError("CCSDS123 requires integer-valued input data")


    def compress(self, hsi: HSI) -> CompressedHSI:
        """
        Compress a hyperspectral image using CCSDS123.

        Parameters
        ----------
        hsi : HSI
            Hyperspectral image to compress.

        Returns
        -------
        CompressedHSI
            Compressed hyperspectral image representation.
        """
        self.report_progress(0.0)

        # 1. Run predictor to get mapped residuals
        scaled_report = self._scaled_progress(0.0, 0.8)
        delta = self._encoder_predictor(hsi.data, scaled_report)

        # 2. Convert residuals to bitstream
        scaled_report = self._scaled_progress(0.8, 0.99)
        stream, k_values = self._rice_encode(delta, scaled_report)

        # 3. Compute BER protection mask (optional)
        protection_mask = None 
        if self.config.protect_bitstream == True:
            protection_mask = self._calculate_protection_mask(delta, k_values)

        # 4. Create output object
        compressed = CompressedHSI(
            bitstream=stream,
            metadata=hsi.metadata,
            side_information={
                "smin": np.min(hsi.data),
                "smax": np.max(hsi.data),
                "protection_mask": protection_mask
            }
        )
        self.report_progress(1.0)
        
        return compressed

    def decompress(self, compressed: CompressedHSI) -> HSI:
        """
        Reconstruct a hyperspectral image from a CCSDS123 bitstream.

        Parameters
        ----------
        compressed : CompressedHSI
            Compressed hyperspectral image representation.

        Returns
        -------
        HSI
            Reconstructed hyperspectral image.
        """
        self.report_progress(0.0)

        # 1. Get residuals by decoding bitstream
        scaled_report = self._scaled_progress(0.0, 0.2)
        delta = self._rice_decode(compressed.bitstream, compressed.metadata.shape, scaled_report)

        # 2. Run predictor to get reconstruction
        scaled_report = self._scaled_progress(0.2, 0.99)
        S_rec = self._decoder_predictor(delta,
                                        compressed.side_information["smin"],
                                        compressed.side_information["smax"],
                                        scaled_report)

        # 3. Create output object
        reconstruction = HSI(S_rec, compressed.metadata)
        self.report_progress(1.0)

        return reconstruction

    def decode_compressed_values(self, compressed: CompressedHSI) -> np.ndarray:
        return self._rice_decode(
            compressed.bitstream,
            compressed.metadata.shape,
        )

    def _encoder_predictor(self, S, report_callback=None):
        """
        Run the CCSDS123 predictive encoder.

        Parameters
        ----------
        S : np.ndarray
            Input hyperspectral image cube.

        report_callback : callable | None, optional
            Progress callback function.

        Returns
        -------
        np.ndarray
            Mapped prediction residuals.
        """
        Nx, Ny, Nz = S.shape
        smin = np.min(S)
        smax = np.max(S)

        S_rep = np.zeros_like(S, dtype=np.int32)
        delta = np.zeros_like(S, dtype=np.int32)
        U = np.zeros((Nx, Ny, self.config.P), dtype=np.int32)
        a_den = 2*self.config.a + 1

        for z in range(Nz):
            if report_callback:
                report_callback(z / Nz)

            for y in range(Ny):
                for x in range(Nx):
                    
                    sigma = self._calc_local_sum(S_rep[:, :, z], x, y, Nx, Ny)

                    d_hat = np.dot(self.W, U[x, y, :])

                    s_hat = (d_hat + (sigma << self.config.Omega)) >> (2 + self.config.Omega)
                    s_hat = np.clip(s_hat, smin, smax)

                    Delta = S[x, y, z] - s_hat
                    q = np.sign(Delta) * ((np.abs(Delta) + self.config.a) // a_den)
    
                    s_rep = np.clip(s_hat + q * a_den, smin, smax)
                    S_rep[x, y, z] = s_rep
                    
                    U[x, y, 1:] = U[x, y, :-1]
                    U[x, y, 0] = 4*s_rep - sigma

                    if q >= 0:
                        delta[x, y, z] = 2 * q
                    else:
                        delta[x, y, z] = 2 * abs(q) - 1

        return delta
    
    def _decoder_predictor(self, delta, smin, smax, report_callback=None):
        """
        Reconstruct an image cube from mapped prediction residuals.

        Parameters
        ----------
        delta : np.ndarray
            Mapped prediction residuals.

        smin : int
            Minimum sample value.

        smax : int
            Maximum sample value.

        report_callback : callable | None, optional
            Progress callback function.

        Returns
        -------
        np.ndarray
            Reconstructed hyperspectral image cube.
        """
        
        Nx, Ny, Nz = delta.shape

        S_rep = np.zeros((Nx, Ny, Nz), dtype=np.int32)
        U = np.zeros((Nx, Ny, self.config.P), dtype=np.int32)

        a_den = 2 * self.config.a + 1
        
        for z in range(Nz):
            if report_callback:
                report_callback(z / Nz)

            for y in range(Ny):
                for x in range(Nx):
                    
                    sigma = self._calc_local_sum(S_rep[:, :, z], x, y, Nx, Ny)

                    d_hat = np.dot(self.W, U[x, y, :])

                    s_hat = (d_hat + (sigma << self.config.Omega)) >> (2 + self.config.Omega)
                    s_hat = np.clip(s_hat, smin, smax)

                    delta_val = delta[x, y, z]

                    if delta_val % 2 == 0:
                        q = delta_val // 2      
                    else:
                        q = -(delta_val + 1) // 2

                    s_rep = np.clip(s_hat + q * a_den, smin, smax)
                    S_rep[x, y, z] = s_rep           

                    U[x, y, 1:] = U[x, y, :-1]
                    U[x, y, 0] = 4*s_rep - sigma

        return S_rep
    
    def _calc_local_sum(self, S_rep_z, x, y, Nx, Ny):
        """
        Compute the local prediction sum for a spatial sample.

        Parameters
        ----------
        S_rep_z : np.ndarray
            Reconstructed spectral slice.

        x : int
            Horizontal pixel coordinate.

        y : int
            Vertical pixel coordinate.

        Nx : int
            Image width.

        Ny : int
            Image height.

        Returns
        -------
        int
            Local prediction sum.
        """

        def rep(xx, yy):
            xx = min(max(xx, 0), Nx - 1)
            yy = min(max(yy, 0), Ny - 1)
            return S_rep_z[xx, yy]

        if self.config.local_sum_mode == 'column':
            if y > 0:
                return 4 * rep(x, y-1)
            elif x > 0:
                return 4 * rep(x-1, y)
            else:
                return 0

        elif self.config.local_sum_mode == 'neighbor':
            if y == 0 and x == 0:
                return 0
            elif y == 0:
                return 4 * rep(x-1, y)
            elif x == 0:
                return 2 * (rep(x, y-1) + rep(x+1, y-1))
            elif x == Nx - 1:
                return rep(x-1, y) + rep(x-1, y-1) + 2 * rep(x, y-1)
            else:
                return (
                    rep(x-1, y) +
                    rep(x-1, y-1) +
                    rep(x,   y-1) +
                    rep(x+1, y-1)
                )
    

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

