import numpy as np
import util


class CCSDS_123:
    def __init__(self, local_sum_mode='column', P=1, Omega=0, a=0, block_size=32):
        self.local_sum_mode = local_sum_mode
        self.P = P 
        self.Omega = Omega
        self.a = a
        self.block_size = block_size
        self.W = np.ones(P, dtype=np.int32)
    
    def load(self, Image):

        if not isinstance(Image, np.ndarray):
            raise TypeError("Input Image must be a numpy array")
        
        if not np.issubdtype(Image.dtype, np.integer):
            raise TypeError(f"Input Image must be of integer type, got {Image.dtype}")

        info = np.iinfo(Image.dtype)
        if info.bits > 16:
            raise ValueError(f"Input Image integer depth exceeds 16 bits: {info.bits} bits")
        
        if Image.ndim != 3:
            raise ValueError(f"Input Image must be 3D, got shape {Image.shape}")

        self.S = Image.astype(np.int32)

        self.smin, self.smax = util.get_bounds(self.S)

        self.Nx, self.Ny, self.Nz = self.S.shape

    def encoder_predictor(self):
        
        S_rep = np.zeros_like(self.S, dtype=np.int32)
        delta = np.zeros_like(self.S, dtype=np.int32)
        U = np.zeros((self.Nx, self.Ny, self.P), dtype=np.int32)
        a_den = 2*self.a + 1

        for z in range(self.Nz):
            for y in range(self.Ny):
                for x in range(self.Nx):
                    
                    sigma = self.calc_local_sum(S_rep[:, :, z], x, y)

                    d_hat = np.dot(self.W, U[x, y, :])

                    s_hat = s_hat = (d_hat + (sigma << self.Omega)) >> (2 + self.Omega)
                    s_hat = np.clip(s_hat, self.smin, self.smax)

                    Delta = self.S[x, y, z] - s_hat

                    q = np.sign(Delta) * ((np.abs(Delta) + self.a) // a_den)
    
                    s_rep = np.clip(s_hat + q * a_den, self.smin, self.smax)
                    S_rep[x, y, z] = s_rep
                    
                    U[x, y, 1:] = U[x, y, :-1]
                    U[x, y, 0] = 4*s_rep - sigma

                    # if (x==0 and y==0):
                    #     theta = min(s_hat - self.smin, self.smax - s_hat)
                    # else:
                    #     theta = min((s_hat - self.smin + self.a) // a_den,
                    #                 (self.smax - s_hat + self.a) // a_den)

                    # abs_q = abs(q)
                    # mid = (self.smin + self.smax) // 2

                    # if abs_q <= theta:
                    #     if q >= 0:
                    #         delta[x, y, z] = 2 * abs_q
                    #     else:
                    #         delta[x, y, z] = 2 * abs_q - 1
                    # elif (q > 0 and s_hat <= mid) or (q < 0 and s_hat > mid):
                    #     delta[x, y, z] = abs_q + theta
                    # else:
                    #     delta[x, y, z] = 2 * abs_q - 1

                    # 4. Pure ZigZag Mapping (No Theta)
                    if q >= 0:
                        delta[x, y, z] = 2 * q
                    else:
                        delta[x, y, z] = 2 * abs(q) - 1

        self.delta = delta
        return delta
    
    def calc_local_sum(self, S_rep_z, x, y):

        def rep(xx, yy):
            xx = min(max(xx, 0), self.Nx - 1)
            yy = min(max(yy, 0), self.Ny - 1)
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
            elif x == self.Nx - 1:
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
            
    def rice_encoder(self):

        if not hasattr(self, "delta"):
            raise RuntimeError("Mapped quantizer indices delta not computed yet")

        data = self.delta.transpose(2, 1, 0).flatten()
        blocks = [data[i:i+self.block_size] for i in range(0, len(data), self.block_size)]

        k = []
        bitstream = []  

        for block in blocks:
                med = int(np.median(block))
                k_i = max(0, int(np.floor(np.log2(med + 1))))
                k.append(k_i)

                for value in block:
                    q = value >> k_i
                    r = value & ((1 << k_i) - 1)

                    # unary(q)
                    bitstream.extend([1] * q)
                    bitstream.append(0)

                    # binary remainder
                    for b in reversed(range(k_i)):
                        bitstream.append((r >> b) & 1)

        self.k = np.array(k, dtype=np.int32)
        self.bitstream = np.array(bitstream, dtype=np.uint8)
        return self.k, self.bitstream
    
    def rice_decoder(self):

        if not hasattr(self, "k"):
            raise RuntimeError("Rice parameters k not available")

        if not hasattr(self, "bitstream"):
            raise RuntimeError("Rice bitstream not available")

        bitstream = self.bitstream
        k_list = self.k
        block_size = self.block_size

        delta = []
        bit_idx = 0

        for k_i in k_list:
            for _ in range(block_size):
                if bit_idx >= len(bitstream):
                    break  # last block may be partial

                # --- decode unary quotient q ---
                q = 0
                while bitstream[bit_idx] == 1:
                    q += 1
                    bit_idx += 1
                    if bit_idx >= len(bitstream):
                        raise ValueError("Unexpected end of bitstream during unary decoding")

                # consume the terminating zero
                bit_idx += 1

                # --- decode remainder r ---
                r = 0
                for _ in range(k_i):
                    if bit_idx >= len(bitstream):
                        raise ValueError("Unexpected end of bitstream during remainder decoding")
                    r = (r << 1) | bitstream[bit_idx]
                    bit_idx += 1

                # reconstruct delta
                value = (q << k_i) + r
                delta.append(value)

        delta = np.array(delta, dtype=np.int32)
        delta = delta.reshape((self.Nz, self.Ny, self.Nx)).transpose(2, 1, 0)
        self.delta_r = delta

        return delta

    def decoder_predictor(self):

        if not hasattr(self, "delta_r"):
                raise RuntimeError("Mapped quantizer indices delta_r not available")

        S_rep = np.zeros_like(self.S, dtype=np.int32)
        U = np.zeros((self.Nx, self.Ny, self.P), dtype=np.int32)
        a_den = 2*self.a + 1

        for z in range(self.Nz):
            for y in range(self.Ny):
                for x in range(self.Nx):
                    
                    sigma = self.calc_local_sum(S_rep[:, :, z], x, y)

                    d_hat = np.dot(self.W, U[x, y, :])

                    s_hat = s_hat = (d_hat + (sigma << self.Omega)) >> (2 + self.Omega)
                    s_hat = np.clip(s_hat, self.smin, self.smax)

                    delta_val = self.delta_r[x, y, z]

                    # if x == 0 and y == 0:
                    #     theta = min(s_hat - self.smin, self.smax - s_hat)
                    # else:
                    #     theta = min((s_hat - self.smin + self.a) // a_den,
                    #                 (self.smax - s_hat + self.a) // a_den)
                        
                    # if delta_val <= 2 * theta:
                    #     if delta_val % 2 == 0:
                    #         q = delta_val // 2
                    #     else:
                    #         q = -( (delta_val + 1) // 2 )
                    # else:
                    #     if s_hat <= (self.smin + self.smax) // 2:
                    #         q = delta_val - theta
                    #     else:
                    #         q = -(delta_val - theta)

                    # 3. Inverse ZigZag Mapping (No Theta)
                    if delta_val % 2 == 0:
                        q = delta_val // 2      # Even -> Positive
                    else:
                        q = -(delta_val + 1) // 2 # Odd -> Negative


                    s_rep = np.clip(s_hat + q * (2*self.a + 1), self.smin, self.smax)
                    S_rep[x, y, z] = s_rep           

                    U[x, y, 1:] = U[x, y, :-1]
                    U[x, y, 0] = 4*s_rep - sigma

        self.S_rec = S_rep
        return S_rep
    


