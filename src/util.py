import numpy as np
import scipy as sp
import math
import os
import tarfile
import matplotlib.pyplot as plt
from src import transforms
from src.hsi import HSI
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from abc import ABC, abstractmethod


##### HSI Handling #####

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

##### Data #####

def save_hsi(hsi: HSI, path: str):
    """
    Save an HSI object to a file.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image object to save.
    path : str
        Path to save the file. Should end with `.npy`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    np.save(path, {
        "data": hsi.data,
        "wavelengths": hsi.wavelengths,
        "metadata": hsi.metadata,
        "dtype": hsi.dtype
    })

def load_hsi(path: str) -> HSI:
    """
    Load an HSI object previously saved with `save_hsi`.
    """
    obj = np.load(path, allow_pickle=True).item()
    return HSI(
        data=obj["data"],
        wavelengths=obj["wavelengths"],
        dtype=obj["dtype"],
        metadata=obj["metadata"]
    )

def load_hsi_from_mat(mat_path: str, name: str, site: str, sensor: str) -> HSI:

    mat = sp.io.loadmat(mat_path)

    # Find the actual data key
    key = [k for k in mat.keys() if not k.startswith("__")][0]
    cube = mat[key]

    # ===== Wavelengths ===== #
    bands = cube.shape[2]

    # Option 1: simple approximation
    wavelengths = np.linspace(400, 2500, bands)

    # ===== Metadata ===== #
    metadata = {
        "name": name,
        "site": site,
        "sensor": sensor,
    }

    return HSI(
        data=cube,
        wavelengths=wavelengths,
        dtype=cube.dtype,
        metadata=metadata
    )

def save_array_to_path(arr, path, metadata=None):
    """
    Save a NumPy array to a file, optionally with metadata.

    Parameters
    ----------
    arr : ndarray
        Array to save.
    path : str
        File path to save the array to.
    metadata : dict, optional
        Dictionary of metadata to save alongside the array.
    """
    # Prepare save dictionary
    save_dict = {"array": arr}
    if metadata is not None:
        for key, value in metadata.items():
            save_dict[f"meta_{key}"] = value
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save as npz
    np.savez_compressed(path, **save_dict)

def load_array_from_path(path):
    """
    Load a NumPy array and optional metadata saved with `save_array_to_path`.

    Parameters
    ----------
    path : str
        Path to load.

    Returns
    -------
    arr : ndarray
        The main array.
    metadata : dict
        Metadata dictionary (empty if none saved).
    """
    with np.load(path, allow_pickle=True) as data:
        # Main array
        arr = data["array"]
        
        # Extract metadata
        metadata = {}
        for key in data.files:
            if key.startswith("meta_"):
                metadata[key[5:]] = data[key].item() if data[key].shape == () else data[key]
    
    return arr, metadata

def extract_tar_gz(path, out_dir):
    """
    Extract a .tar.gz dataset from path to out_dir.
    """
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(out_dir)

def load_aviris(folder_path: str) -> HSI:

    folder_path = os.path.abspath(folder_path)
    files = os.listdir(folder_path)

    # ===== Detect ort_img files ===== #
    img_files = [f for f in files if "ort_img" in f.lower() and not f.lower().endswith(".hdr")]
    hdr_files = [f for f in files if f.lower().endswith(".hdr")]
    spc_files = [f for f in files if f.lower().endswith(".spc")]
    info_files = [f for f in files if f.lower().endswith(".info")]

    if not img_files:
        raise FileNotFoundError("No ort_img image file found")

    if len(img_files) > 1:
        print(f"[WARNING] Multiple ort_img files found ({len(img_files)}). Using first.")

    img_file = img_files[0]
    img_path = os.path.join(folder_path, img_file)

    # ===== Find matching HDR ===== #
    base_name = os.path.splitext(img_file)[0]
    hdr_candidates = [f for f in hdr_files if base_name in f]
    if not hdr_candidates:
        raise FileNotFoundError("No matching HDR file found")
    hdr_file = hdr_candidates[0]
    hdr_path = os.path.join(folder_path, hdr_file)

    # ===== Parse HDR ===== #
    header = {}
    with open(hdr_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.split("=", 1)
                header[key.strip().lower()] = val.strip().lower()

    samples = int(header["samples"])
    lines = int(header["lines"])
    bands = int(header["bands"])
    interleave = header.get("interleave", "bip")
    data_type = int(header.get("data type", 2))
    byte_order = int(header.get("byte order", 1))

    # ===== Map dtype ===== #
    if data_type == 2:
        base_dtype = np.int16
    elif data_type == 4:
        base_dtype = np.float32
    else:
        raise ValueError(f"Unsupported ENVI data type: {data_type}")

    dtype = np.dtype(base_dtype).newbyteorder(">" if byte_order == 1 else "<")

    # ===== Load raw data ===== #
    cube = np.fromfile(img_path, dtype=dtype)
    expected = samples * lines * bands
    if cube.size != expected:
        raise ValueError(f"Size mismatch: got {cube.size}, expected {expected}")

    # ===== Reshape based on interleave ===== #
    if interleave == "bip":
        cube = cube.reshape((lines, samples, bands))
    elif interleave == "bil":
        cube = cube.reshape((lines, bands, samples))
        cube = np.transpose(cube, (0, 2, 1))
    elif interleave == "bsq":
        cube = cube.reshape((bands, lines, samples))
        cube = np.transpose(cube, (1, 2, 0))
    else:
        raise ValueError(f"Unknown interleave: {interleave}")

    # ===== Load wavelengths ===== #
    if spc_files:
        spc_path = os.path.join(folder_path, spc_files[0])
        wavelengths = np.loadtxt(spc_path, usecols=0)
    else:
        print("[WARNING] No SPC file found, using index wavelengths")
        wavelengths = np.arange(bands)

    # ===== Remove water absorption bands ===== #
    mask = np.ones(len(wavelengths), dtype=bool)
    bad_ranges = [(1340, 1445), (1790, 1955), (2480, 2600)]
    for lo, hi in bad_ranges:
        mask &= ~((wavelengths >= lo) & (wavelengths <= hi))

    cube = cube[:, :, mask]
    wavelengths = wavelengths[mask]

    # ===== Parse .info file for site ===== #
    site_name = "Unknown"
    if info_files:
        info_path = os.path.join(folder_path, info_files[0])
        with open(info_path, "r") as f:
            for line in f:
                if "site_name" in line:
                    _, val = line.split("=", 1)
                    site_name = val.strip()

    # ===== Extract dataset name from folder ===== #
    img_base = os.path.basename(img_path)
    if "rdn" in img_base:
        name = img_base.split("rdn")[0]
    elif "_" in img_base:
        name = img_base.split("_")[0]
    else:
        name = img_base

    # ===== Create HSI object ===== #
    hsi = HSI(
        data=cube,
        wavelengths=wavelengths,
        dtype=base_dtype,
        metadata={
            "name": name,
            "site": site_name,
            "sensor": "AVIRIS"
        }
    )

    print(f"Loaded AVIRIS: {name} | site={site_name} | shape={cube.shape} | dtype={dtype}")
    return hsi

def crop_aviris_scene(folder_path: str, section_size=(256, 256)):
    """
    Crop a full AVIRIS HSI scene into smaller sections and save them.

    Each section will be stored as an individual HSI object in a subfolder called 'sections'.
    Section metadata 'name' will be updated to include section index (s01, s02, ...).

    Parameters
    ----------
    folder_path : str
        Path to folder containing AVIRIS scene.
    section_size : tuple
        Size of each section (height, width). Default is (256, 256).
    """

    # ===== Load full scene ===== #
    hsi = load_aviris(folder_path)
    scene_name = hsi.metadata["name"]

    # ===== Prepare output folder ===== #
    sections_folder = os.path.join(folder_path, "sections")
    os.makedirs(sections_folder, exist_ok=True)

    H, W = hsi.height, hsi.width
    sh, sw = section_size

    # ===== Compute number of sections ===== #
    n_vert = math.ceil(H / sh)
    n_horiz = math.ceil(W / sw)

    sections = []
    for i in range(n_vert):
        y0 = i * sh
        y1 = min(y0 + sh, H)
        for j in range(n_horiz):
            x0 = j * sw
            x1 = min(x0 + sw, W)
            sections.append((y0, y1, x0, x1))

    print(f"Cropping HSI ({H}x{W}) into {len(sections)} sections of approx {sh}x{sw}...")

    # ===== Crop, update metadata, save ===== #
    for idx, (y0, y1, x0, x1) in enumerate(sections):
        section_hsi = hsi.crop((y0, y1), (x0, x1))

        # Update metadata with section number
        section_name = f"{scene_name}s{idx+1}"  # s1, s2, ...
        metadata = section_hsi.metadata.copy()
        metadata["name"] = section_name

        section_hsi = HSI(
            data=section_hsi.data,
            wavelengths=section_hsi.wavelengths,
            dtype=section_hsi.dtype,
            metadata=metadata
        )

        save_path = os.path.join(sections_folder, f"{section_name}")
        save_hsi(section_hsi, save_path)

    print(f"Saved {len(sections)} sections in '{sections_folder}'")

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

def compute_all_metrics(reference: HSI, target: HSI, bitstream):
    """
    Compute a comprehensive set of HSI performance metrics.

    Parameters
    ----------
    reference : HSI
        The ground truth HSI.
    target : HSI
        The reconstructed HSI.
    bitstream : bytes or list
        The resulting compressed bitstream.

    Returns
    -------
    dict
        A dictionary containing RMSE, PSNR, SAM, and CR.
    """
    ref_arr = reference.data
    target_arr = target.data
    bitdepth = reference.bitdepth

    return {
        "rmse": calc_rmse(ref_arr, target_arr),
        "psnr": calc_psnr(ref_arr, target_arr, bitdepth),
        "sam": calc_sam(ref_arr, target_arr),
        "cr": calc_compression_ratio(ref_arr, bitstream, bitdepth),
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

##### Sensing Operators #####

class SensingOperator:
    """
    Chains a MeasurementMatrix (Phi) and a TransformBasis (Psi) 
    into a single operator A = Phi @ Psi_inv.
    
    This class adheres to the LinearOperator-like interface required 
    for sparse recovery algorithms.
    """
    
    def __init__(self, phi, psi, n):
        self.phi = phi
        self.psi = psi
        self.n = n
        self.dtype = psi.transform_dtype

    def forward(self, s):
        """Coefficients 's' to measurements 'y': y = Phi(Psi_inv(s))"""
        return self.phi.forward(self.psi.inverse(s, axis=0), axis=0)

    def adjoint(self, y):
        """Measurements 'y' to coefficients 's_approx': s = Psi(Phi_adj(y))"""
        return self.psi.forward(self.phi.adjoint(y, axis=0, n=self.n), axis=0)

    def get_column(self, idx):
        """Extracts the i-th column of the operator A."""
        e_i = np.zeros(self.n, dtype=self.dtype)
        e_i[idx] = 1.0
        return self.forward(e_i)

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

##### Transform analysis #####

def transform_metrics(hsi, transforms, coeffs=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Analyzes the sparsity of a Hyperspectral Image (HSI) by applying specific 
    basis transforms, thresholding coefficients, and measuring reconstruction quality.

    Args:
        hsi (np.ndarray): The input hyperspectral data cube.
        transforms (list): A list of tuples, e.g., [("DFT", 0, 1), ("DCT", 2)], 
                           where the first element is the basis name and the 
                           following are the axes to transform.
        coeffs (list of float): Percentages (0.0 to 1.0) of the largest 
                                coefficients to keep.

    Returns:
        dict: A dictionary containing lists of 'rmse', 'psnr', 'sam', and 'coeffs'.
    """
    BASIS_MAP = {"DFT": DFTBasis(), "DCT": DCTBasis()}

    maxval, minval, depth = get_hsi_statistics(hsi)
    norm_hsi = normalize_zero_mean(hsi, minval, maxval)

    # 1. Forward Transforms
    transformed_hsi = norm_hsi.copy()
    for transform_name, *axes in transforms:
        for axis in axes:
            transformed_hsi = BASIS_MAP[transform_name].forward(transformed_hsi, axis)

    results = {"rmse": [], "psnr": [], "sam": [], "coeffs": coeffs, "transforms": transforms}

    # 2. Iterate through sparsity levels
    # Flatten to easily find the magnitude threshold
    flat_coeffs = transformed_hsi.flatten()
    abs_weights = np.abs(flat_coeffs)

    for p in coeffs:
        # Determine how many coefficients to keep
        n_keep = int(p * abs_weights.size)
        
        if n_keep < abs_weights.size:
            # Find the threshold value at the (1-p) percentile
            threshold = np.partition(abs_weights, -n_keep)[-n_keep]
            # Zero out elements below threshold
            sparse_hsi = np.where(np.abs(transformed_hsi) >= threshold, transformed_hsi, 0)
        else:
            sparse_hsi = transformed_hsi.copy()
            
        # 3. Inverse Transforms (Apply in REVERSE order of the forward transforms)
        reconstructed = sparse_hsi
        for transform_name, *axes in reversed(transforms):
            for axis in reversed(axes):
                reconstructed = BASIS_MAP[transform_name].inverse(reconstructed, axis)

        rec = denormalize_zero_mean(reconstructed, minval, maxval).real
        
        # 4. Evaluation
        # We compare against norm_hsi (the transformed/untransformed reference)
        results["rmse"].append(calc_rmse(hsi, rec))
        results["psnr"].append(calc_psnr(hsi, rec, depth))
        results["sam"].append(calc_sam(hsi, rec))
        
    return results

def plot_transform_metrics(all_results):
    """
    Plots RMSE, PSNR, and SAM trends for multiple transform configurations.

    Args:
        all_results (list of dict): A list where each dict contains 'rmse', 'psnr', 
                                   'sam', 'coeffs', and 'transforms'.
    """
    if isinstance(all_results, dict):
        all_results = [all_results]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("HSI Reconstruction Quality Comparison", fontsize=16, fontweight='bold')

    metrics = [("rmse", "RMSE"), 
               ("sam", "SAM"),
               ("psnr", "PSNR")]

    colors = plt.cm.tab10.colors 

    for idx, res in enumerate(all_results):
        x = [c * 100 for c in res["coeffs"]]
        
        # Build label: "DFT (0, 1)"
        label_parts = []
        for name, *axs in res.get("transforms", [("Unknown",)]):
            label_parts.append(f"{name} {tuple(axs)}")
        label = " | ".join(label_parts)
        
        color = colors[idx % len(colors)]

        for i, (key, title) in enumerate(metrics):
            axes[i].plot(x, res[key], marker='o', markersize=5, 
                         linestyle='-', color=color, label=label)
            axes[i].set_title(title, fontsize=13)
            axes[i].set_xlabel("% of Coefficients Kept")
            axes[i].grid(True, linestyle='--', alpha=0.6)
            # Show legend in each individual plot
            axes[i].legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def analyze_dictionary_coherence(D):
    """
    Calculates the coherence mu(D) and the upper bound for sparsity K.
    Assumes columns of D are already normalized to unit length.
    """
    # 1. Compute the Gram matrix G = D.T @ D
    # G_ij contains the inner product of column i and column j
    gram = np.dot(D.T, D)
    
    # 2. We only care about off-diagonal elements (i != j)
    # Set the diagonal to zero so they aren't picked by the max()
    np.fill_diagonal(gram, 0)
    
    # 3. Coherence is the maximum absolute value of off-diagonal entries
    mu = np.max(np.abs(gram))
    
    # 4. Calculate the upper bound for K: 0.5 * (1 + 1/mu)
    # Note: If mu is 0 (orthogonal matrix), K bound is technically infinite
    if mu > 0:
        k_bound = 0.5 * (1 + (1 / mu))
    else:
        k_bound = np.inf
        
    return mu, k_bound

def generate_synthetic_hsi(shape, K_sparse, transform_name="DCT"):
    """
    Generates a 3D HSI signal that is K-sparse in a transform domain.
    
    Parameters:
    -----------
    shape : tuple
        The dimensions of the HSI (e.g., (32, 32, 10))
    K_sparse : int
        The number of non-zero coefficients in the core tensor.
    transform_name : str
        The transform name (e.g., "DCT")

    Returns:
    --------
    Y : ndarray
        The synthetic signal in spatial/spectral domain.
    X_gt : ndarray
        The ground truth sparse core tensor.
    """
    # 1. Generate the dictionaries (Inverse DCT matrices act as atoms)
    # If Y is sparse in DCT, it is a linear combination of IDCT atoms.
    Ds = [transforms.get_inverse_transform(transform_name, n) for n in shape]
    
    # 2. Create the sparse core tensor X_gt
    X_gt = np.zeros(shape)
    total_elements = np.prod(shape)
    
    # Randomly pick K unique locations
    indices = np.random.choice(total_elements, K_sparse, replace=False)
    
    # Assign random coefficients (e.g., standard normal)
    X_gt.flat[indices] = np.random.normal(size=K_sparse)
    
    # 3. Synthesize the signal: Y = X_gt x1 D1 x2 D2 x3 D3
    Y = X_gt.copy()
    for n in range(len(shape)):
        Y = mode_n_product(Y, Ds[n], n)
        
    return Y, X_gt

##### N-Way Array Operations #####

def mode_n_product(X, U, n):
    """
    Mode-n tensor-matrix multiplication.

    X : ndarray, shape (I1, ..., In, ..., IN)
    U : ndarray, shape (J, In)
    n : int (mode, 0-based)

    Returns:
        Y : ndarray, shape (I1, ..., J, ..., IN)
    """
    # tensordot contracts axis n of X with axis 1 of U
    Y = np.tensordot(X, U, axes=([n], [1]))

    # Move the new axis (last) back to position n
    return np.moveaxis(Y, -1, n)

def mode_n_unfold(X, n):
    """
    Unfolds a tensor X into a matrix along mode n.
    
    X : ndarray, shape (I1, ..., In, ..., IN)
    n : int (0-based)
    
    Returns:
    X_n : ndarray, shape (In, -1)
    """
    return np.moveaxis(X, n, 0).reshape(X.shape[n], -1)

def mode_n_fold(X_n, n, original_shape):
    """
    Folds a mode-n matricized tensor back into its original shape.
    
    X_n : ndarray, the unfolded matrix
    n : int (0-based)
    original_shape : tuple, the shape of the tensor before unfolding
    
    Returns:
    X : ndarray, shape original_shape
    """
    reduced_shape = list(original_shape)
    mode_dim = reduced_shape.pop(n)
    intermediate_shape = [mode_dim] + reduced_shape
    X_inter = X_n.reshape(intermediate_shape)
    return np.moveaxis(X_inter, 0, n)

def generate_block_sparse_signal(D, S, N):
    """
    Generates a block-sparse signal Y based on the Tucker model.
    
    Parameters
    ----------
    D : ndarray
        The dictionary.
    S : int
        The number of active atoms in each dimension (sparsity per mode).
        The total number of non-zero coefficients will be S^N.
    N : int
        Number of dimensions.
        
    Returns
    -------
    Y : ndarray
        The synthesized signal of shape.
    X_gt : ndarray
        The ground truth sparse core tensor of shape.
    """
    M = D.shape[1]
    out_shape = []
    for n in range(N):
        out_shape.append(M)
    
    # 1. Randomly select S unique indices for each dimension
    active_indices = []
    for n in range(N):
        if S > M:
            raise ValueError(f"Sparsity S={S} cannot be greater than dictionary size M={M}")
        idx = np.sort(np.random.choice(M, S, replace=False))
        active_indices.append(idx)
        
    # 2. Create the core tensor
    X_gt = np.zeros(out_shape)
    
    # 3. Fill the S^N block with random coefficients (e.g., Gaussian)
    # np.ix_ creates a meshgrid of indices to address the specific block
    grid = np.ix_(*active_indices)
    X_gt[grid] = np.random.randn(*([S] * N))
    
    # 4. Synthesize the signal Y = X x1 D1 x2 D2 ... xN DN
    Y = X_gt.copy()
    for n in range(N):
        Y = mode_n_product(Y, D, n)
        
    return Y, X_gt

##### Progress Bar #####

def scaled_callback(base_callback, start, end):
    def wrapper(progress):
        base_callback(start + progress * (end - start))
    return wrapper










