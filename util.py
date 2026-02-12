import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import csv
import os
import json
from datetime import datetime

### Array Handling ###

def load_image(path):
    mat = sp.io.loadmat(path)
    mat_clean = {k: v for k, v in mat.items() if not k.startswith('__')}

    if len(mat_clean) == 1:
        data_array = next(iter(mat_clean.values()))
    else:
        data_array = max(mat_clean.values(), key=lambda x: getattr(x, 'size', 0))

    print(f'Image loaded as a np.array with dtype: {data_array.dtype} and shape:{data_array.shape}')
    return data_array

def normalize_image(I):
    I = I.astype(np.float64)
    max_val = max(np.abs(I.min()), np.abs(I.max()))
    if max_val <= 1:
        return I
    else:
        return I / max_val

def get_bounds(arr):
    """
    Returns the min & max values in the array.
    """
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError("Input array is empty")

    vmin = np.min(arr)
    vmax = np.max(arr)

    return vmin, vmax

### Metric Calculations ###

def calc_RMSE(I, I_hat):
    
    # determine normalization
    if np.issubdtype(I.dtype, np.integer):
        bitdepth = I.dtype.itemsize * 8
        norm = (2**bitdepth - 1)
        I_norm = I / norm
        I_hat_norm = I_hat / norm
    else:
        # float images: assume already scaled (e.g., [0,1])
        I_norm = I
        I_hat_norm = I_hat

    shape = I.shape
    factor = 1
    for dim in shape:
        factor *= dim
    
    return np.sqrt(np.sum(np.pow(I_norm - I_hat_norm, 2)) / factor)

def calc_SAM(I, I_hat):

    # dot product for each pixel (vector over Nz)
    dots = np.sum(I * I_hat, axis=-1)

    # norms
    norms = np.linalg.norm(I, axis=-1)
    norms_hat = np.linalg.norm(I_hat, axis=-1)

    # cosine
    cosines = dots / (norms * norms_hat)
    cosines = np.clip(cosines, -1.0, 1.0)

    # SAM per pixel
    angles = np.arccos(cosines)

    # average SAM
    return np.degrees(np.mean(angles))

def calc_compression_ratio(I, bitstream):
    
    max_val = np.max(I)
    original_bits = int(np.log2(max(1, max_val))) + 1
    original_total_bits = np.prod(I.shape) * original_bits
    bitstream_total_bits = len(bitstream)

    ratio = original_total_bits / bitstream_total_bits
    return ratio

def calc_PSNR(I, I_hat):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between a ground truth image 
    and a reconstructed image.

    Formula: PSNR = 20 * log10(MAX / RMSE)
    
    Parameters
    ----------
    I : ndarray
        Original image.
    I_hat : ndarray
        Reconstructed image (must have same shape as I).

    Returns
    -------
    float
        The PSNR value in decibels (dB).
    """
    I = normalize_image(I)
    I_hat = normalize_image(I_hat)

    rmse = calc_RMSE(I, I_hat)

    if rmse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(1 / rmse)
    
    return psnr

def calc_sweep_metrics(image, images_r, bitstreams):
    """
    Compute RMSE, SAM, and compression ratios for sweep results.

    Returns
    -------
    RMSEs : np.array
    SAMs : np.array
    ratios : np.array
    """

    RMSEs = []
    SAMs = []
    ratios = []

    for image_r, bitstream in zip(images_r, bitstreams):
        RMSEs.append(calc_RMSE(image, image_r))
        SAMs.append(calc_SAM(image, image_r))
        ratios.append(calc_compression_ratio(image, bitstream))

    return (
        np.array(RMSEs),
        np.array(SAMs),
        np.array(ratios)
    )

### Save Data ###

def save_sweep_results(param_name, param_values,
                       RMSEs, SAMs, ratios, complexities,
                       fixed_params,
                       directory, name,
                       titles=False):
    """
    Save sweep_CCSDS results:
    - Save CSV file
    - Save RMSE, SAM, and Compression Ratio plots as PNG files

    Parameters
    ----------
    param_name : str
        Name of the swept parameter.
    param_values : array-like
        Values used for sweeping.
    RMSEs, SAMs, ratios, complexities : array-like
        Results from sweep_CCSDS.
    fixed_params : dict
        Dictionary of fixed CCSDS parameters.
    directory : str
        Directory to save files.
    name : str
        folder name for the output files.
    titles : bool
        Whether to add titles to the plots.
    """

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # ============
    # 1. SAVE CSV
    # ============
    csv_path = os.path.join(directory, f"{name}.csv")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        # ---- Fixed-parameter metadata ----
        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_params.items())
        writer.writerow([f"# fixed_params: {fixed_str}"])
        writer.writerow([f"# swept_param: {param_name}"])

        # ---- Header ----
        writer.writerow([param_name, "RMSE", "SAM", "Compression Ratio", "Compression Time"])

        # ---- Data rows ----
        for p, r, s, c, t in zip(param_values, RMSEs, SAMs, ratios, complexities):
            writer.writerow([p, r, s, c, t])

    print(f"Saved results to {csv_path}")


    # ======================================
    # 2. PREPARE PLOT FILE NAMES (if needed)
    # ======================================
    plot_filenames = {
            "rmse":  f"{name}_rmse.png",
            "sam":   f"{name}_sam.png",
            "ratio": f"{name}_ratio.png",
            "time": f"{name}_time.png"
        }

    # ======================
    # 3. SAVE RMSE PLOT
    # ======================
    plt.figure()
    plt.plot(param_values, RMSEs, marker='o')
    plt.xlabel(param_name)
    plt.ylabel("RMSE")
    if titles:
        plt.title(f"RMSE vs {param_name}")
    plt.grid(True)
    rmse_path = os.path.join(directory, plot_filenames["rmse"])
    plt.savefig(rmse_path, dpi=300)
    plt.close()
    print(f"Saved RMSE plot to {rmse_path}")

    # ======================
    # 4. SAVE SAM PLOT
    # ======================
    plt.figure()
    plt.plot(param_values, SAMs, marker='o')
    plt.xlabel(param_name)
    plt.ylabel("SAM [deg]")
    if titles:
        plt.title(f"SAM vs {param_name}")
    plt.grid(True)
    sam_path = os.path.join(directory, plot_filenames["sam"])
    plt.savefig(sam_path, dpi=300)
    plt.close()
    print(f"Saved SAM plot to {sam_path}")

    # ======================
    # 5. SAVE RATIO PLOT
    # ======================
    plt.figure()
    plt.plot(param_values, ratios, marker='o')
    plt.xlabel(param_name)
    plt.ylabel("Compression Ratio")
    if titles:
        plt.title(f"Compression Ratio vs {param_name}")
    plt.grid(True)
    ratio_path = os.path.join(directory, plot_filenames["ratio"])
    plt.savefig(ratio_path, dpi=300)
    plt.close()
    print(f"Saved Ratio plot to {ratio_path}")

    # ======================
    # 6. SAVE TIME PLOT
    # ======================
    plt.figure()
    plt.plot(param_values, complexities, marker='o')
    plt.xlabel(param_name)
    plt.ylabel("Compression Time [s]")
    if titles:
        plt.title(f"Compression Time vs {param_name}")
    plt.grid(True)
    time_path = os.path.join(directory, plot_filenames["time"])
    plt.savefig(time_path, dpi=300)
    plt.close()
    print(f"Saved Ratio plot to {time_path}")

def save_histogram(array, directory, filename, log_scale=False):
    
    """
    Saves a histogram of `array` into the specified directory, under the given filename.
    No title is added. Optional log-scale on the y-axis and consistent styling.
    
    Parameters:
        array (np.ndarray): Input data array.
        directory (str): Directory path to save the histogram.
        filename (str): Output filename (e.g., 'hist.png').
        log_scale (bool): If True, plot the y-axis in log scale.
    """

    array = np.asarray(array).ravel()

    os.makedirs(directory, exist_ok=True)
    out_path = os.path.join(directory, filename + ".png")

    plt.figure(figsize=(6, 4), dpi=120)

    if np.issubdtype(array.dtype, np.integer):
        vmin = int(array.min())
        vmax = int(array.max())
        # One bin per integer value
        bins = np.arange(vmin - 0.5, vmax + 1.5, 1)
        plt.hist(array, bins=bins)

    else:
        raise ValueError("Array type is non-integer.")

    plt.xlabel("Value")
    plt.ylabel("Count")

    if log_scale:
        plt.yscale("log")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path

def save_images(image1, name1, image2, name2, filename, directory):
    """
    Plots two images side-by-side in a single figure and saves the figure 
    to a specified file path. Assumes image arrays are (Height, Width, Bands).
    """
    # 1. Create Figure and Subplots (fig is the object we MUST save)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

    # Apply style
    plt.style.use('seaborn-v0_8-paper')
    
    # 2. PLOT IMAGES - Assuming (Height, Width, Bands) format:
    # Selects the LAST band (slice)
    im1_data = image1[:,:,50] 
    im2_data = image2[:,:,50]
    
    # Plot 1
    im1 = axes[0].imshow(im1_data, cmap='gray')
    axes[0].set_title(name1)
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot 2
    im2 = axes[1].imshow(im2_data, cmap='gray') 
    axes[1].set_title(name2) 
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Finalize Layout
    plt.tight_layout()
    
    # 4. Save to Path
    os.makedirs(directory, exist_ok=True)
    out_path = os.path.join(directory, filename + ".png")

    # *** CRITICAL: Save the 'fig' object explicitly ***
    fig.savefig(out_path, bbox_inches='tight', dpi=120)

    # 5. Close the figure
    plt.close(fig)

def save_results(results: dict, folder_path: str):

    method_name = results.get("name", "unknown_method")
    dataset_name = results.get("dataset", "uknown_dataset")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(folder_path, f"{method_name}_{dataset_name}_{timestamp}")
    os.makedirs(folder, exist_ok=True)

    reconstructed = results.get("reconstructed")
    if reconstructed is not None:
        np.save(os.path.join(folder, "rec_HSI"), reconstructed)

    bitstream = results.get("bitstream")
    if bitstream is not None:
        
        bitstream_path = os.path.join(folder, "bitstream.bin")
        # If bitstream is bytes
        if isinstance(bitstream, (bytes, bytearray)):
            with open(bitstream_path, "wb") as f:
                f.write(bitstream)
        # If bitstream is numpy array of bits
        elif isinstance(bitstream, np.ndarray):
            bitstream.astype(np.uint8).tofile(bitstream_path)
        else:
            raise TypeError("Unsupported bitstream format")
    
    metadata = {
        "name": results.get("name"),
        "dataset": results.get("dataset"),
        "cube_shape": results.get("cube_shape"),
        "dtype": results.get("dtype"),
        "metrics": results.get("metrics"),
        "params": results.get("params")
    }

    with open(os.path.join(folder, "metadata_json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Results saved to: {folder}")

### Plot Data ###

def plot_log_error_heatmap(original, reconstructed, show=True):
    """
    Generates a log-scale spectral error heatmap between
    original and reconstructed hyperspectral cubes.

    Parameters:
        original       : ndarray (H, W, B)
        reconstructed  : ndarray (H, W, B)
        show           : bool, whether to display the plot

    Returns:
        error_map_log  : ndarray (H, W)
    """

    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed cubes must have same shape")

    if original.ndim != 3:
        raise ValueError("Inputs must be 3D hyperspectral cubes")

    # Convert to float for safety
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    # Spectral L2 error per pixel
    spectral_error = np.linalg.norm(original - reconstructed, axis=2)

    # Log scale
    error_map_log = np.log1p(spectral_error)

    if show:
        plt.figure(figsize=(6, 5))
        plt.imshow(error_map_log, cmap='inferno')
        plt.title("Log-Scale Spectral Error Heatmap")
        plt.colorbar(label="log(1 + L2 spectral error)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return error_map_log

def plot_false_color(I, bands=(30, 20, 10), show=True):
    """
    Generate a false-color RGB image from a hyperspectral cube.

    Parameters
    ----------
    I : ndarray
        Hyperspectral image of shape (H, W, B)
    bands : tuple of 3 ints
        Indices of bands to use as (R, G, B)

    Returns
    -------
    rgb: ndarray
        RGB image of shape (H, W, 3) scaled to [0, 1]
    """
    I = normalize_image(I)
    
    H, W, B = I.shape
    r, g, b = bands

    if max(bands) >= B:
        raise ValueError("Band index exceeds number of bands in HSI cube")
    
    R = I[:,:,r]
    G = I[:,:,g]
    B = I[:,:,b]

    rgb = np.stack([R,G,B], axis=-1)

    p_low, p_high = 2, 98  # percentile stretch

    for i in range(3):
        channel = rgb[:, :, i]
        low = np.percentile(channel, p_low)
        high = np.percentile(channel, p_high)
        rgb[:, :, i] = np.clip((channel - low) / (high - low), 0, 1)

    if show:
        plt.figure(figsize=(6, 5))
        plt.imshow(rgb)
        plt.title("False Color Image")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return rgb

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






