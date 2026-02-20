from .base import BaseCompressor
import numpy as np
from bitarray import bitarray
from bitarray.util import int2ba, ba2int

class CCSDS123(BaseCompressor):

    @property
    def name(self): return "ccsds123"

    @property
    def compressor_id(self): return 11

    def __init__(self, local_sum_mode='column', P=1, Omega=8, a=0, block_size=32, progress_callback=None):

        super().__init__(progress_callback=progress_callback)
        self.local_sum_mode = local_sum_mode            # Mode for local sum
        self.P = P                                      # Amount of spectral bands used in predictor
        self.Omega = Omega                              # Resolution of calculation
        self.a = a                                      # Absolute error limit (per pixel)
        self.block_size = block_size                    # Block size for encoder

        self._initialize_weights()    
    
    def _initialize_weights(self):

        weights = []
        w_0 = (7 * (1 << self.Omega)) >> 3
        weights.append(w_0)

        for i in range(1, self.P):
            w_prev = weights[-1]
            w_next = w_prev >> 3
            weights.append(w_next)
        
        self.W = np.array(weights, dtype=np.int32)     # Weights for predictor

    def _update_progress(self, fraction):
        if self.progress_callback:
            self.progress_callback(fraction)

    def _validate_input(self, hsi):
        
        if not isinstance(hsi, np.ndarray):
            raise TypeError("Input Image must be a numpy array")
        
        if not np.issubdtype(hsi.dtype, np.integer):
            raise TypeError(f"Input Image must be of integer type, got {hsi.dtype}")

        info = np.iinfo(hsi.dtype)
        if info.bits > 16:
            raise ValueError(f"Input Image integer depth exceeds 16 bits: {info.bits} bits")
        
        if hsi.ndim != 3:
            raise ValueError(f"Input Image must be 3D, got shape {hsi.shape}")
        
        return hsi.astype(np.int32, copy=False)

    def _encoder_predictor(self, S):
        
        Nx, Ny, Nz = S.shape
        smin = np.min(S)
        smax = np.max(S)

        S_rep = np.zeros_like(S, dtype=np.int32)
        delta = np.zeros_like(S, dtype=np.int32)
        U = np.zeros((Nx, Ny, self.P), dtype=np.int32)
        a_den = 2*self.a + 1

        for z in range(Nz):
            self._update_progress(z / Nz)

            for y in range(Ny):
                for x in range(Nx):
                    
                    sigma = self._calc_local_sum(S_rep[:, :, z], x, y, Nx, Ny)

                    d_hat = np.dot(self.W, U[x, y, :])

                    s_hat = (d_hat + (sigma << self.Omega)) >> (2 + self.Omega)
                    s_hat = np.clip(s_hat, smin, smax)

                    Delta = S[x, y, z] - s_hat
                    q = np.sign(Delta) * ((np.abs(Delta) + self.a) // a_den)
    
                    s_rep = np.clip(s_hat + q * a_den, smin, smax)
                    S_rep[x, y, z] = s_rep
                    
                    U[x, y, 1:] = U[x, y, :-1]
                    U[x, y, 0] = 4*s_rep - sigma

                    if q >= 0:
                        delta[x, y, z] = 2 * q
                    else:
                        delta[x, y, z] = 2 * abs(q) - 1

        return delta
    
    def _calc_local_sum(self, S_rep_z, x, y, Nx, Ny):

        def rep(xx, yy):
            xx = min(max(xx, 0), Nx - 1)
            yy = min(max(yy, 0), Ny - 1)
            return S_rep_z[xx, yy]

        if self.local_sum_mode == 'column':
            if y > 0:
                return 4 * rep(x, y-1)
            elif x > 0:
                return 4 * rep(x-1, y)
            else:
                return 0

        elif self.local_sum_mode == 'neighbor':
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

        else:
            raise ValueError(f"Invalid local sum mode: {self.local_sum_mode}")
    
    def _rice_encode(self, delta):

        bits = bitarray(endian='big')
        data = delta.flatten(order='F')
        N = len(data)
        
        total_blocks = (N + self.block_size - 1) // self.block_size

        for b_idx in range(total_blocks):
            if self.progress_callback: self.progress_callback(b_idx / total_blocks)
            
            # Get block and calculate k (Rice parameter)
            block = data[b_idx * self.block_size : min((b_idx+1)*self.block_size, N)]
            med = np.median(block) if len(block) > 0 else 0
            k = int(max(0, np.floor(np.log2(med + 1)))) if med > 0 else 0
            
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

        return bits.tobytes()

    def _rice_decode(self, bitstream_bytes, shape):

        bits = bitarray(endian='big')
        bits.frombytes(bitstream_bytes)
        
        N = np.prod(shape)
        flat_delta = np.zeros(N, dtype=np.int32)
        
        bit_ptr = 0
        val_idx = 0
        
        while val_idx < N:
            if self.progress_callback: self.progress_callback(val_idx / N)
            
            # Read k (4 bits)
            k = ba2int(bits[bit_ptr : bit_ptr + 4])
            bit_ptr += 4
            
            # Read a block of samples
            for _ in range(self.block_size):
                if val_idx >= N: break
                
                # Decode Unary (count ones until 0)
                q = 0
                while bits[bit_ptr]:
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
    
    def _decoder_predictor(self, delta, smin, smax):
        
        Nx, Ny, Nz = delta.shape

        S_rep = np.zeros((Nx, Ny, Nz), dtype=np.int32)
        U = np.zeros((Nx, Ny, self.P), dtype=np.int32)

        a_den = 2 * self.a + 1
        
        for z in range(Nz):

            self._update_progress(z / Nz)

            for y in range(Ny):
                for x in range(Nx):
                    
                    sigma = self._calc_local_sum(S_rep[:, :, z], x, y, Nx, Ny)

                    d_hat = np.dot(self.W, U[x, y, :])

                    s_hat = (d_hat + (sigma << self.Omega)) >> (2 + self.Omega)
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
    
    def compress(self, hsi):

        S = self._validate_input(hsi)
        shape = S.shape
        smin, smax = int(np.min(S)), int(np.max(S))

        # 1. Run Predictor to get mapped residuals (delta)
        original_cb = self.progress_callback
        self.progress_callback = lambda f: original_cb(f * 0.75) if original_cb else None
        delta = self._encoder_predictor(S)

        # 2. Convert residuals to actual bitstream
        self.progress_callback = lambda f: original_cb(0.75 + (f * 0.25)) if original_cb else None
        bitstream_bytes = self._rice_encode(delta)

        # 3. Package metadata for reconstruction
        metadata = {
            "shape": shape,
            "smin": smin,
            "smax": smax,
            "params": {
                "local_sum_mode": self.local_sum_mode,
                "P": self.P,
                "Omega": self.Omega,
                "a": self.a,
                "block_size": self.block_size
            }
        }

        return bitstream_bytes, metadata
    
    def decompress(self, bitstream_bytes, metadata):

        shape = metadata["shape"]
        smin, smax = metadata["smin"], metadata["smax"]

        # 1. Decode bitstream back to residuals
        original_cb = self.progress_callback
        self.progress_callback = lambda f: original_cb(f * 0.25) if original_cb else None
        delta = self._rice_decode(bitstream_bytes, shape)

        # 2. Run inverse predictor
        self.progress_callback = lambda f: original_cb(0.25 + (f * 0.75)) if original_cb else None
        S_rec = self._decoder_predictor(delta, smin, smax)
        
        return S_rec


    

