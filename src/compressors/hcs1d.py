from src.compressors.base import BaseCompressor
from src import util
import numpy as np
from scipy.sparse.linalg import LinearOperator
import spgl1

def get_A(phi, psi, n, m):
    """
    Creates a 1D LinearOperator for a single pixel (spectral vector).
    A = Phi @ Psi_inv
    """
    def forward_pass(s):
        x = psi.inverse(s, axis=0)
        y = phi.forward(x, axis=0)
        return y

    def adjoint_pass(y):
        x_adj = phi.adjoint(y, axis=0, n=n)
        s_adj = psi.forward(x_adj, axis=0)
        return s_adj.astype(psi.transform_dtype)

    return LinearOperator((m, n), matvec=forward_pass, rmatvec=adjoint_pass, dtype=psi.transform_dtype)

class HCS1D(BaseCompressor):
    
    # Map strings to classes
    BASIS_MAP = {"DFT": util.DFTBasis, "DCT": util.DCTBasis}
    MEASUREMENT_MAP = {"Subsampling": util.SubsamplingMatrix}

    # Map axis to spatial\spectral
    AXIS_MAP = ["Horizontal", "Vertical", "Spectral"]

    @property
    def name(self): return "hcs1d"

    @property
    def compressor_id(self): return 21

    def __init__(self, targetCR=2, axis=-1, measurement_matrix="Subsampling",
                 trasnform_basis="DFT", seed=42, progress_callback=None):
        super().__init__(progress_callback)
        self.targetCR = targetCR
        self.axis = axis
        self.Phi_name = measurement_matrix
        self.Psi_name = trasnform_basis
        self.seed = seed
        
    @classmethod
    def print_available_components(cls):
        """Prints a summary of all registered bases and measurement matrices."""
        print("\n" + "="*30)
        print(f"HCS1D AVAILABLE COMPONENTS")
        print("="*30)
        
        print("\n[Transform Bases]")
        for name, class_ref in cls.BASIS_MAP.items():
            # We instantiate briefly to check the dtype for information
            temp_obj = class_ref()
            dtype_str = "Complex" if np.iscomplexobj(np.array([], dtype=temp_obj.transform_dtype)) else "Real"
            print(f" - {name:<12} (Type: {dtype_str})")

        print("\n[Measurement Matrices]")
        for name in cls.MEASUREMENT_MAP.keys():
            print(f" - {name}")
        print("="*30 + "\n")

    def _setup_operators(self, n):
        """Instantiates the specific Phi and Psi objects."""
        m = int(n / self.targetCR)
        
        Phi = self.MEASUREMENT_MAP[self.Phi_name](n, m, seed=self.seed)
        Psi = self.BASIS_MAP[self.Psi_name]()
        return Phi, Psi

    def compress(self, hsi):
        if self.progress_callback:
                    self.progress_callback(0.0)

        # 1. Statistics and Normalization to [-1, 1]
        min_val, max_val, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, min_val, max_val)
        if self.progress_callback:
            self.progress_callback(0.2)

        # 2. Get Measurements
        n = hsi.shape[self.axis]
        Phi, _ = self._setup_operators(n)
        y = Phi.forward(hsi_norm, axis=self.axis)
        if self.progress_callback:
            self.progress_callback(0.8)

        # 3. Quantization & Bit Packing
        max_int = (1 << bit_depth) - 1
        y_quantized = np.clip(np.round((y + 1) / 2 * max_int).astype(np.uint64), 0, max_int)
        bitstream = util.pack_to_bit_depth(y_quantized, bit_depth)

        metadata = {
            "y_shape": y.shape,
            "hsi_shape": hsi.shape,
            "min_val": min_val,
            "max_val": max_val,
            "bit_depth": bit_depth,
            "params": {
                "transform basis": self.Psi_name,
                "measurement matrix": self.Phi_name,
                "seed": self.seed,
                "target CR": self.targetCR,
                "compression axis": self.AXIS_MAP[self.axis]
            }
        }
        return bitstream, metadata
    
    def decompress(self, bitstream, metadata):
        
        # 1. Unpack & Dequantize
        y_quantized = util.unpack_from_bit_depth(bitstream, 
                                                 metadata["bit_depth"], 
                                                 metadata["y_shape"])
        max_int = (1 << metadata["bit_depth"]) - 1
        y = (y_quantized.astype(np.float64) / max_int) * 2 - 1

        # 2. Setup Operators
        n = metadata["hsi_shape"][self.axis]
        m = metadata["y_shape"][self.axis]
        Phi, Psi = self._setup_operators(n)
        A = get_A(Phi, Psi, n, m)

        # 3. Reconstruction Setup
        y_flat = y.reshape(-1, m)
        num_pixels = y_flat.shape[0]
        s_hat_flat = np.zeros((num_pixels, n), dtype=Psi.transform_dtype)

        # 4. Reconstruction Loop
        for i in range(num_pixels):
            s_recon, resid, grad, info = spgl1.spg_bpdn(A, y_flat[i], sigma=0.01, iter_lim=100, verbosity=0)
            s_hat_flat[i] = s_recon
            self.progress_callback(i / num_pixels)
        
        # 5. Inverse Transform to Signal Domain
        s_hat = s_hat_flat.reshape(metadata["hsi_shape"])
        hsi_recon_norm = Psi.inverse(s_hat, axis=self.axis).real.astype(np.float64)

        return util.denormalize_zero_mean(hsi_recon_norm,
                                          metadata["min_val"],
                                          metadata["max_val"])
    

        





        

