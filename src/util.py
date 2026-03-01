import numpy as np
import scipy as sp
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from abc import ABC, abstractmethod


##### HSI Handling #####

def load_hsi(path):
    mat = sp.io.loadmat(path)
    mat_clean = {k: v for k, v in mat.items() if not k.startswith('__')}

    if len(mat_clean) == 1:
        data_array = next(iter(mat_clean.values()))
    else:
        data_array = max(mat_clean.values(), key=lambda x: getattr(x, 'size', 0))

    return data_array

def get_hsi_statistics(hsi, verbose=False):
    """
    Extract range and bit depth statistics based on the signal's dynamic span.

    Calculates the minimum and maximum pixel values and determines the 
    bit depth (D) as the number of bits required to represent the range 
    between the maximum and minimum values.

    Parameters
    ----------
    hsi : numpy.ndarray
        The input hyperspectral image cube.
    verbose : bool, optional
        If True, prints a summary of the statistics to the console, 
        by default False.

    Returns
    -------
    min_val : float
        The minimum pixel value found in the cube.
    max_val : float
        The maximum pixel value found in the cube.
    bit_depth : int
        The calculated bit depth (D) based on ceil(log2(max - min)).
    """
    min_val = float(np.min(hsi))
    max_val = float(np.max(hsi))
    
    # Calculate the span of the data
    span = int(max_val - min_val)
    
    # Determine bits required to represent that span
    # We use max(span, 1) to avoid bit_length(0) returning 0 for constant images
    bit_depth = span.bit_length() if span > 0 else 1

    if verbose:
        print(f"\n--- HSI Statistics (Span-based) ---")
        print(f"Min Value: {min_val:.2f} | Max Value: {max_val:.2f}")
        print(f"Data Span: {span} | Bit Depth: {bit_depth} bits")
        print(f"-----------------------------------\n")

    return min_val, max_val, bit_depth

def normalize_zero_mean(hsi, min_val, max_val):
    """
    Normalize HSI data to a zero-centered [-1, 1] range.

    Parameters
    ----------
    hsi : numpy.ndarray
        The input hyperspectral image cube.
    min_val : float
        The minimum pixel value in the original dataset.
    max_val : float
        The maximum pixel value in the original dataset.

    Returns
    -------
    numpy.ndarray
        The normalized HSI cube as float64, mapped to the [-1, 1] range.
        Returns an array of zeros if max_val equals min_val.
    """
    midpoint = (max_val + min_val) / 2.0
    half_range = (max_val - min_val) / 2.0
    
    if half_range == 0:
        return np.zeros_like(hsi, dtype=np.float64)
        
    return (hsi.astype(np.float64) - midpoint) / half_range

def denormalize_zero_mean(hsi_norm, min_val, max_val):
    """
    Rescale HSI data from the [-1, 1] range back to its original scale.

    Inverts the zero-centered normalization to recover the data in its 
    original radiance or reflectance units.

    Parameters
    ----------
    hsi_norm : numpy.ndarray
        The normalized HSI cube (typically the output of a CS reconstruction).
    min_val : float
        The original minimum pixel value.
    max_val : float
        The original maximum pixel value.

    Returns
    -------
    numpy.ndarray
        The HSI cube rescaled to its original dynamic range.
    """
    midpoint = (max_val + min_val) / 2.0
    half_range = (max_val - min_val) / 2.0
    
    return (hsi_norm * half_range) + midpoint

##### Metric Calculations #####

def calc_rmse(reference, target):
    """
    Calculate the Root Mean Square Error (RMSE) between two arrays.

    Parameters
    ----------
    reference : numpy.ndarray
        The original reference hyperspectral cube.
    target : numpy.ndarray
        The reconstructed or processed hyperspectral cube.

    Returns
    -------
    float
        The RMSE value in the original units of the data.
    """
    mse = np.mean((reference.astype(np.float64) - target.astype(np.float64)) ** 2)
    return np.sqrt(mse)

def calc_psnr(reference, target, bit_depth):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    This implementation uses the formula: 20 * log10(MAX_I / RMSE).

    Parameters
    ----------
    reference : numpy.ndarray
        The original ground truth HSI cube.
    target : numpy.ndarray
        The reconstructed HSI cube.
    bit_depth : int
        The bit depth (D) used to calculate the dynamic range (2^D - 1).

    Returns
    -------
    float
        The PSNR value in dB. Returns infinity if the images are identical.
    """
    max_i = float((1 << bit_depth) - 1)
    rmse_val = calc_rmse(reference, target)
    
    if rmse_val == 0:
        return float('inf')
        
    return 20 * np.log10(max_i / rmse_val)

def calc_sam(reference, target):
    """
    Calculate the Mean Spectral Angle Mapper (SAM) in degrees.
    
    SAM measures spectral similarity by calculating the angle between 
    spectral vectors. It is inherently scale-invariant.

    Parameters
    ----------
    reference : numpy.ndarray
        The original HSI cube, where the last dimension is the spectral axis.
    target : numpy.ndarray
        The reconstructed HSI cube.

    Returns
    -------
    float
        The mean spectral angle in degrees.
    """
    ref = reference.astype(np.float64)
    tgt = target.astype(np.float64)

    dot_product = np.sum(ref * tgt, axis=-1)
    norm_ref = np.linalg.norm(ref, axis=-1)
    norm_tgt = np.linalg.norm(tgt, axis=-1)

    # Calculate cosine similarity with epsilon to avoid division by zero
    cos_theta = dot_product / (norm_ref * norm_tgt + 1e-15)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angles_rad = np.arccos(cos_theta)
    return np.degrees(np.mean(angles_rad))

def calc_compression_ratio(original_cube, bitstream, bit_depth):
    """
    Calculate the Compression Ratio (CR) of the bitstream.

    Parameters
    ----------
    original_cube : numpy.ndarray
        The original HSI data.
    bitstream : bytes or list
        The compressed representation (bytes or bit list).
    bit_depth : int
        The number of bits used to represent the original pixel values.

    Returns
    -------
    float
        The compression ratio (Original Size / Compressed Size).
    """
    total_pixels = original_cube.size
    original_bits = total_pixels * bit_depth
    
    if isinstance(bitstream, (bytes, bytearray)):
        compressed_bits = len(bitstream) * 8
    else:
        compressed_bits = len(bitstream)

    return original_bits / compressed_bits

def compute_all_metrics(reference, target, bitstream):
    """
    Compute a comprehensive set of HSI performance metrics.

    Parameters
    ----------
    reference : numpy.ndarray
        The ground truth HSI cube.
    target : numpy.ndarray
        The reconstructed HSI cube.
    bitstream : bytes or list
        The resulting compressed bitstream.

    Returns
    -------
    dict
        A dictionary containing RMSE, PSNR, SAM, and CR.
    """
    _, _, bit_depth = get_hsi_statistics(reference, verbose=False)

    return {
        "rmse": calc_rmse(reference, target),
        "psnr": calc_psnr(reference, target, bit_depth),
        "sam": calc_sam(reference, target),
        "cr": calc_compression_ratio(reference, bitstream, bit_depth),
    }

##### Transform Bases #####

class TransformBasis(ABC):
    def __init__(self, name, transform_dtype):
        self.name = name
        self.transform_dtype = transform_dtype

    def _get_safe_axis(self, x, axis):
        return 0 if x.ndim == 1 else axis

    @abstractmethod
    def forward(self, x, axis=-1):
        """Psi * x (Signal to coefficients)"""
        pass
        
    @abstractmethod
    def inverse(self, s, axis=-1):
        """Psi_inv * s (Coefficients to signal)"""
        pass

class DFTBasis(TransformBasis):
    def __init__(self):
        super().__init__(name="DFT", transform_dtype=np.complex128)

    def forward(self, x, axis=-1):
        ax = self._get_safe_axis(x, axis)
        return sp.fft.fft(x, axis=ax, norm='ortho').astype(self.transform_dtype)

    def inverse(self, s, axis=-1):
        ax = self._get_safe_axis(s, axis)
        return sp.fft.ifft(s, axis=ax, norm='ortho').astype(self.transform_dtype)
    
class DCTBasis(TransformBasis):
    def __init__(self):
        super().__init__(name="DCT", transform_dtype=np.float64)

    def forward(self, x, axis=-1):
        ax = self._get_safe_axis(x, axis)
        return sp.fft.dct(x, axis=ax, type=2, norm='ortho').astype(self.transform_dtype)

    def inverse(self, s, axis=-1):
        ax = self._get_safe_axis(s, axis)
        return sp.fft.idct(s, axis=ax, type=2, norm='ortho').astype(self.transform_dtype)

##### Measurement Matrices #####

class MeasurementMatrix(ABC):
    def __init__(self, name):
        self.name = name

    def _get_safe_axis(self, x, axis):
        return 0 if x.ndim == 1 else axis

    @abstractmethod
    def forward(self, x, axis):
        """Perform y = Phi @ x."""
        pass

    @abstractmethod
    def adjoint(self, y, axis, n):
        """x_approx = Phi^T * y"""
        pass

class SubsamplingMatrix(MeasurementMatrix):
    def __init__(self, n, m, seed=42):
        super().__init__(name="Subsampling")
        self.n = n
        self.m = m
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.indices = np.sort(rng.choice(n, m, replace=False))

    def forward(self, x, axis):
        ax = self._get_safe_axis(x, axis)
        return np.take(x, self.indices, axis=ax)
    
    def adjoint(self, y, axis, n):
            ax = self._get_safe_axis(y, axis)
            out_dtype = np.result_type(y.dtype, np.float64)
            res = np.zeros(n, dtype=out_dtype)
            res[self.indices] = y
            return res
    
class GaussianMeasurementMatrix(MeasurementMatrix):
    """
    Implements a Gaussian random measurement matrix Phi where entries 
    are i.i.d. N(0, 1/m).
    """
    def __init__(self, n, m, seed=42):
        super().__init__(name="Gaussian")
        self.n = n
        self.m = m
        rng = np.random.RandomState(seed)
        # Normalize by sqrt(m) to maintain approximate unit-norm columns
        self.matrix = rng.randn(m, n) / np.sqrt(m)

    def forward(self, x, axis):
        """Perform y = Phi @ x along the specified axis."""
        ax = self._get_safe_axis(x, axis)
        x_swapped = np.moveaxis(x, ax, 0)
        shape_orig = x_swapped.shape
        x_flat = x_swapped.reshape(self.n, -1)
        y_flat = self.matrix @ x_flat
        y_swapped = y_flat.reshape(self.m, *shape_orig[1:])
        return np.moveaxis(y_swapped, 0, ax)

    def adjoint(self, y, axis, n):
        """Perform x_approx = Phi^T @ y."""
        ax = self._get_safe_axis(y, axis)
        y_swapped = np.moveaxis(y, ax, 0)
        shape_orig = y_swapped.shape
        y_flat = y_swapped.reshape(self.m, -1)
        x_flat = self.matrix.T @ y_flat
        x_swapped = x_flat.reshape(n, *shape_orig[1:])
        return np.moveaxis(x_swapped, 0, ax)
            
##### Bitstream Packing #####

def pack_to_bit_depth(data, bit_depth):
    """
    Packs a numpy array into a bitstream where each element 
    occupies exactly 'bit_depth' bits.
    """
    ba = bitarray()
    flat_data = data.flatten().tolist()
    
    for value in flat_data:
        # Convert integer to bitarray of length bit_depth
        ba.extend(int2ba(int(value), length=bit_depth, endian='big'))
    
    return ba.tobytes()

def unpack_from_bit_depth(byte_stream, bit_depth, shape):
    """
    Unpacks a byte-stream back into a numpy array of a specific shape.
    """
    ba = bitarray()
    ba.frombytes(byte_stream)
    
    total_elements = np.prod(shape)
    unpacked = np.zeros(total_elements, dtype=np.uint64)
    
    for i in range(total_elements):
        start = i * bit_depth
        end = start + bit_depth
        unpacked[i] = ba2int(ba[start:end])
        
    return unpacked.reshape(shape)




