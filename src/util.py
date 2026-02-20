import numpy as np
import scipy as sp
from bitarray import bitarray
from bitarray.util import int2ba, ba2int


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

class TransformBasis:
    """
    Abstract base class for sparsifying transforms in Compressive Sensing.
    """
    def __init__(self, name, axis=-1):
        self.name = name
        self.axis = axis

    def forward(self, x):
        """Map signal to sparse coefficient domain."""
        raise NotImplementedError
        
    def inverse(self, s):
        """Map sparse coefficients back to signal domain."""
        raise NotImplementedError

class DFTBasis(TransformBasis):
    """Discrete Fourier transform basis, computed with fft"""
    def __init__(self, axis=-1):
        super().__init__(name="DFT", axis=axis)

    def forward(self, x):
        return sp.fft.fft(x, axis=self.axis, norm='sqrtn')

    def inverse(self, s):
        return sp.fft.ifft(s, axis=self.axis, norm='sqrtn').real

##### Measurement Matrices #####

class MeasurementMatrix:
    """
    Abstract base class for CS Measurement Matrices (Phi).
    """
    def __init__(self, name, m, n, axis):
        self.name = name
        self.m = m  # Number of measurements
        self.n = n  # Original signal dimension along the target axis
        self.axis = axis

    def project(self, x):
        """Perform y = Phi @ x along the specified axis."""
        raise NotImplementedError

class SubsamplingMatrix(MeasurementMatrix):
    """
    Subsampling Measurement Matrix that picks M indices along a specific axis.
    """
    def __init__(self, axis=-1, seed=42):
        super().__init__(name="Subsampling", axis=axis)
        self.seed = seed
        self.indices = None
        self.m = None
        self.n = None
        self.axis=axis

    def initialize(self, n, m, axis=-1):
        """Explicitly set dimensions and axis. Generate indices based on seed."""
        self.axis = axis
        self.n = n
        self.m = m
        rng = np.random.RandomState(self.seed)
        self.indices = np.sort(rng.choice(n, m, replace=False))

    def project(self, x):
        if self.indices is None:
            raise RuntimeError("Phi not initialized. Call initialize(n, m) first.")
        return np.take(x, self.indices, axis=self.axis)


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
        # Using endian='big' ensures the most significant bit is first
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
        unpacked[i] = ba2int(ba[start:end], endian='big')
        
    return unpacked.reshape(shape)

### CS helpers ###

def linear_transform(x, Psi, axis=-1):
    """
    Applies a linear transformation matrix Psi to a specific axis of the array x.
    
    This function implements the operation y = Psi * v for every vector v 
    located along the specified axis of x.

    Parameters
    ----------
    x : ndarray
        Input array of shape (..., N, ...).
    Psi : ndarray
        Transformation matrix of shape (M, N).
        Note: The second dimension of Psi (N) must match the size of x 
        along the specified axis.
    axis : int, optional
        The axis along which to apply the transformation. Default is -1.

    Returns
    -------
    ndarray
        Transformed array of shape (..., M, ...), where the size of the 
        specified axis has changed from N to M.
    """
    x_s = np.moveaxis(x, axis, -1)
    x_transformed = x_s @ Psi.T
    return np.moveaxis(x_transformed, -1, axis)

def linear_transform_3D(f, Px, Py, Pz):
    """
    Applies a separable 3D linear transform to a volumetric signal.
    
    Parameters
    ----------
    f : ndarray
        Input 3D array of shape (Nx, Ny, Nz).
    Px : ndarray
        Linear transform matrix for the first (x) dimension, of shape
        (Mx, Nx).
    Py : ndarray
        Linear transform matrix for the second (y) dimension, of shape
        (My, Ny).
    Pz : ndarray
        Linear transform matrix for the third (z) dimension, of shape
        (Mz, Nz). The conjugate transpose is applied internally.

    Returns
    -------
    F : ndarray
        Transformed 3D array of shape (Mx, My, Mz).
    """
    F = f
    F = np.tensordot(Px, F, axes=(1, 0))
    F = np.tensordot(Py, F, axes=(1, 1))
    F = np.moveaxis(F, 0, 1)
    F = np.tensordot(Pz.conj(), F, axes=(1, 2))
    F = np.moveaxis(F, 0, 2)

    return F

def adjoint_linear_transform_3D(Y, Px, Py, Pz):
    """
    Adjoint (Hermitian) of linear_transform_3D.
    
    Y: (Mx, My, Mz) measured
    Px: (Mx, Nx)
    Py: (My, Ny)
    Pz: (Mz, Nz) – conjugate transpose used in forward
    """
    F = Y

    F = np.moveaxis(F, 2, 0)
    F = np.tensordot(Pz, F, axes=(0, 0))  # Pz * Y
    F = np.moveaxis(F, 0, 2)
    F = np.moveaxis(F, 1, 0)
    F = np.tensordot(Py.conj().T, F, axes=(1, 0))
    F = np.moveaxis(F, 0, 1)
    F = np.tensordot(Px.conj().T, F, axes=(1, 0))

    return F

def sparsify(x, Psi, T=1.0, axis=-1):
    """
    Transforms x into basis Psi and retains coefficients based on statistical
    thresholding relative to the mean and standard deviation of the coefficient magnitudes.

    Condition to keep coefficient s_i:
        |s_i| >= mean(|s|) + T * std(|s|)

    Parameters
    ----------
    x : ndarray
        Input data array.
    Psi : ndarray
        Transformation basis matrix.
    T : float, optional
        Sparsification factor. Controls the number of standard deviations above 
        the mean required to keep a coefficient.
        Default is 1.0.
    axis : int, optional
        The axis along which to apply the transform. Default is -1.

    Returns
    -------
    s : ndarray
        The full transformed array (dense).
    s_sparse : ndarray
        The sparsified transformed array.
    k : ndarray
        Integer array counting the number of kept coefficients 
        for each vector.
    """
    s = linear_transform(x, Psi, axis=axis)
 
    s_mag = np.abs(s)

    mu = np.mean(s_mag, axis=axis, keepdims=True)
    sigma = np.std(s_mag, axis=axis, keepdims=True)

    cutoff = mu + (T * sigma)
    mask = s_mag >= cutoff

    s_sparse = s * mask
    k = np.sum(mask, axis=axis)

    return s, s_sparse, k

def generate_subsampling_matrix(m, n, seed=None):
    """
    Generates a binary measurement matrix representing random subsampling.

    This matrix selects 'm' distinct components from a vector of size 'n'.
    Each row contains exactly one '1' and 'n-1' zeros. No two rows select 
    the same column index (sampling without replacement).

    Mathematically, if y = A @ x, then y is a vector containing m randomly 
    selected elements from x.

    Parameters
    ----------
    m : int
        The number of measurements (rows). Must be less than or equal to n.
    n : int
        The signal dimension (columns).
    seed : int or np.random.Generator, optional
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    ndarray
        A binary matrix of shape (m, n) with dtype=int.
        
    Raises
    ------
    ValueError
        If m > n (cannot select more unique samples than available dimensions).
    """
    if m > n:
        raise ValueError("Constraint violation: must have m ≤ n")

    rng = np.random.default_rng(seed)

    # Randomly choose m distinct columns
    cols = rng.choice(n, size=m, replace=False)

    A = np.zeros((m, n), dtype=int)
    A[np.arange(m), cols] = 1

    return A






