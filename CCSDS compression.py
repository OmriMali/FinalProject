"""
Simplified-but-realistic CCSDS-123 style compressor and decompressor (lossless / near-lossless).
- Implements: predictor (3-neighbour), closed-loop quantization, residual folding, adaptive Rice coding.
- Not a full reference implementation: meant for learning/prototyping. Use CCSDS doc for final conformance.
References: CCSDS 123.0-B specification (Issue 2) and Issue 1 background.
"""

import numpy as np
from math import floor, log2

# -------------------------
# Utilities
# -------------------------
def fold_signed_to_unsigned(e):
    """Map signed integer e -> nonnegative integer m."""
    if e >= 0:
        return 2 * e
    else:
        return -2 * e - 1

def unfold_unsigned_to_signed(m):
    """Inverse of fold_signed_to_unsigned."""
    if (m & 1) == 0:
        return m // 2
    else:
        return -((m + 1) // 2)

def write_bits_to_bytes(bitlist):
    """Simple pack bits (list of 0/1) into bytes (MSB first)."""
    b = bytearray()
    acc = 0
    cnt = 0
    for bit in bitlist:
        acc = (acc << 1) | (bit & 1)
        cnt += 1
        if cnt == 8:
            b.append(acc)
            acc = 0
            cnt = 0
    if cnt > 0:
        acc <<= (8 - cnt)
        b.append(acc)
    return bytes(b)

def read_bits_from_bytes(data):
    """Generator yielding bits from bytes (MSB first)."""
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1

# -------------------------
# Simple adaptive Rice coder
# -------------------------
class AdaptiveRiceEncoder:
    def __init__(self, init_k=2):
        self.k = init_k
        self.moving_sum = 0.0
        self.count = 1.0  # avoid divide by zero

        self.bits = []  # list of bits to be written out

    def encode_value(self, m):
        """Encode nonnegative integer m using Rice with parameter k."""
        q = m >> self.k
        r = m & ((1 << self.k) - 1)
        # unary q: q ones then zero
        self.bits.extend([1] * q)
        self.bits.append(0)
        # binary remainder r with k bits (MSB first)
        for i in range(self.k - 1, -1, -1):
            self.bits.append((r >> i) & 1)

        # update stats (exponential moving avg of m)
        self.moving_sum = 0.95 * self.moving_sum + 0.05 * m
        self.count = 0  # unused but kept for extensibility

        # adapt k: simple rule: choose k = max(0, floor(log2(avg/2)))
        avg = (self.moving_sum + 1e-9)
        new_k = max(0, int(floor(log2(avg / 2 + 1e-9))) if avg > 1 else 0)
        # small smoothing to avoid rapid swings
        self.k = int(round(0.8 * self.k + 0.2 * new_k))

    def flush_bits(self):
        return write_bits_to_bytes(self.bits)

class AdaptiveRiceDecoder:
    def __init__(self, init_k=2, bit_iter=None):
        self.k = init_k
        self.moving_sum = 0.0
        self.bit_iter = bit_iter  # generator of bits

    def decode_value(self):
        """Decode one Rice-coded nonnegative integer m. Requires bit_iter set."""
        # read unary q
        q = 0
        while True:
            bit = next(self.bit_iter)
            if bit == 1:
                q += 1
            else:
                break
        # read k-bit remainder
        r = 0
        for _ in range(self.k):
            bit = next(self.bit_iter)
            r = (r << 1) | bit
        m = (q << self.k) | r

        # update stats and k (same rule as encoder)
        self.moving_sum = 0.95 * self.moving_sum + 0.05 * m
        avg = (self.moving_sum + 1e-9)
        new_k = max(0, int(floor(log2(avg / 2 + 1e-9))) if avg > 1 else 0)
        self.k = int(round(0.8 * self.k + 0.2 * new_k))

        return m

# -------------------------
# Predictor (3-neighbour + adaptive weights via tiny LMS)
# -------------------------
class SimplePredictor:
    def __init__(self, bitdepth=12, mu=0.01):
        # weights: spectral(prev band), left, above
        self.w = np.array([0.5, 0.25, 0.25], dtype=float)
        self.mu = mu
        self.max_val = (1 << bitdepth) - 1

    def predict(self, neighbors):
        """neighbors: array-like [spec_prev, left, above] (use 0 if not available)"""
        # return float prediction (we cast to int where needed)
        return np.dot(self.w, neighbors)

    def adapt(self, neighbors, error):
        """LMS update: w += mu * error * neighbors_normalized"""
        # scale neighbors to avoid huge updates; simple normalization
        norm = np.dot(neighbors, neighbors) + 1e-12
        self.w += self.mu * error * (neighbors / norm)
        # optional: clip or normalize weights to reasonable range
        # keep weights non-negative and sum to 1 (simple heuristic)
        self.w = np.clip(self.w, -2.0, 2.0)
        s = np.sum(self.w)
        if abs(s) > 1e-8:
            self.w = self.w / s

# -------------------------
# Encoder / Decoder (BIL order)
# -------------------------
def ccsds123_encode(cube, bitdepth=12, near_lossless_Q=1):
    """
    Encode cube (Nx, Ny, Nb) using simplified CCSDS-123 style compressor.
    - near_lossless_Q = 1 => lossless (no quantization). Q > 1 => quantization step.
    Returns: header bytes (simple dict) and encoded bytes.
    """
    Nx, Ny, Nb = cube.shape
    header = {
        "Nx": Nx, "Ny": Ny, "Nb": Nb, "bitdepth": bitdepth, "Q": int(near_lossless_Q)
    }

    # We'll process in BIL: for y in lines, for x in samples, for b in bands
    encoder = AdaptiveRiceEncoder(init_k=2)
    predictor = SimplePredictor(bitdepth=bitdepth, mu=0.01)

    # We'll keep a reconstruction buffer to feed back quantized values (closed-loop)
    recon = np.zeros_like(cube, dtype=int)

    for y in range(Ny):
        for x in range(Nx):
            for b in range(Nb):
                val = int(cube[x, y, b])
                # build neighbours vector [spec_prev, left, above]
                spec_prev = recon[x, y, b - 1] if b > 0 else 0
                left = recon[x - 1, y, b] if x > 0 else 0
                above = recon[x, y - 1, b] if y > 0 else 0
                neighbors = np.array([spec_prev, left, above], dtype=float)

                # Default pred for first-sample & safe use later
                pred = 0.0

                if (x == 0 and y == 0 and b == 0):
                    # first sample: send raw (map to unsigned and write using Rice too)
                    residual = val
                else:
                    pred = predictor.predict(neighbors)
                    # round prediction to integer when computing residual
                    pred_int = int(round(pred))
                    residual = val - pred_int

                # near-lossless quantization (closed-loop): quantize residual
                Q = int(near_lossless_Q)
                if Q > 1:
                    qres = int(np.round(residual / Q))  # quantized residual integer
                    recon_val = int(pred) + qres * Q    # reconstructed sample used for feedback
                    # clip to valid range
                    recon_val = max(0, min(recon_val, (1 << bitdepth) - 1))
                    # store
                    recon[x, y, b] = recon_val
                    # encoder stores qres (not raw residual)
                    store_value = qres
                else:
                    # lossless
                    recon[x, y, b] = val
                    store_value = residual

                # map signed -> unsigned
                m = fold_signed_to_unsigned(store_value)
                encoder.encode_value(m)

                # adapt predictor using *reconstructed* neighbors and actual reconstruction error
                # skip adaptation for the very first sample (no meaningful pred)
                if (x == 0 and y == 0 and b == 0):
                    continue

                # use reconstructed value minus integer-rounded pred for adaptation
                err_for_adapt = recon[x, y, b] - int(round(pred))
                predictor.adapt(neighbors, err_for_adapt)

    coded = encoder.flush_bits()
    return header, coded

def ccsds123_decode(header, coded_bytes):
    """Decode bytes produced by ccsds123_encode."""
    Nx = header["Nx"]; Ny = header["Ny"]; Nb = header["Nb"]
    bitdepth = header["bitdepth"]; Q = int(header.get("Q", 1))

    # create bit iterator
    bit_iter = read_bits_from_bytes(coded_bytes)
    decoder = AdaptiveRiceDecoder(init_k=2, bit_iter=bit_iter)
    predictor = SimplePredictor(bitdepth=bitdepth, mu=0.01)

    recon = np.zeros((Nx, Ny, Nb), dtype=int)

    for y in range(Ny):
        for x in range(Nx):
            for b in range(Nb):
                # decode mapped m
                m = decoder.decode_value()
                store_value = unfold_unsigned_to_signed(m)

                spec_prev = recon[x, y, b - 1] if b > 0 else 0
                left = recon[x - 1, y, b] if x > 0 else 0
                above = recon[x, y - 1, b] if y > 0 else 0
                neighbors = np.array([spec_prev, left, above], dtype=float)

                # default pred
                pred = 0.0

                if (x == 0 and y == 0 and b == 0):
                    # first sample: store_value is raw sample if encoder used that convention
                    value = store_value
                    # store and skip adaptation
                    value = max(0, min(value, (1 << bitdepth) - 1))
                    recon[x, y, b] = int(value)
                    continue
                else:
                    pred = predictor.predict(neighbors)
                    pred_int = int(round(pred))

                if Q > 1:
                    qres = store_value
                    value = pred_int + qres * Q
                else:
                    value = pred_int + store_value

                # clip
                value = max(0, min(value, (1 << bitdepth) - 1))
                recon[x, y, b] = int(value)

                # adapt predictor using reconstructed value
                err_for_adapt = recon[x, y, b] - pred_int
                predictor.adapt(neighbors, err_for_adapt)

    return recon

# -------------------------
# Example usage (not run automatically)
# -------------------------
if __name__ == "__main__":
    # small synthetic cube
    Nx, Ny, Nb = 16, 8, 32
    rng = np.random.default_rng(0)
    # synthetic: smooth spectra + noise, 12-bit data
    cube = np.round(2048 * rng.random((Nx, Ny, Nb))).astype(int)  # random demo data

    header, coded = ccsds123_encode(cube, bitdepth=12, near_lossless_Q=1)  # Q=1 -> lossless
    recon = ccsds123_decode(header, coded)

    # verify
    print("Equal (lossless)?", np.array_equal(cube, recon))
    print("Header:", header)
    print("Compressed bytes:", len(coded))
