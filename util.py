import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import csv
import os

def load_image(path):
    mat = sp.io.loadmat(path)
    mat_clean = {k: v for k, v in mat.items() if not k.startswith('__')}

    if len(mat_clean) == 1:
        data_array = next(iter(mat_clean.values()))
    else:
        data_array = max(mat_clean.values(), key=lambda x: getattr(x, 'size', 0))

    print(f'Image loaded as a np.array with dtype: {data_array.dtype} and shape:{data_array.shape}')
    return data_array

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

def sparsify(x, Psi, T=0.99, axis=-1):
    """
    Transforms an array x into basis Psi and retains only the largest coefficients 
    required to preserve T% of the total energy.

    Parameters
    ----------
    x : ndarray
        Input data array of shape (..., N, ...).
    Psi : ndarray
        Transformation basis matrix of shape (M, N). 
    T : float, optional
        Energy preservation threshold (0 < T <= 1.0). 
        Default is 0.99 (99% energy).
    axis : int, optional
        The axis along which to apply the transform. Default is -1 (last axis).

    Returns
    -------
    s : ndarray
        The full transformed array (dense).
    s_sparse : ndarray
        The sparsified transformed array (zeros everywhere except top coefficients).
    k : ndarray
        An integer array of the same shape as x (minus the transform axis) 
        indicating how many coefficients were kept for each vector.
    """
    s = linear_transform(x, Psi, axis=axis)
 
    E_s = np.abs(s)**2
    E_tot = np.sum(E_s, axis=axis, keepdims=True)

    E_sorted = np.sort(E_s, axis=axis)
    E_sorted = np.flip(E_sorted, axis=axis)
    E_cumulative = np.cumsum(E_sorted, axis=axis)

    threshold = E_tot * T
    thres_mask = E_cumulative >= threshold
    
    k = np.argmax(thres_mask, axis=axis)
    
    k_expanded = np.expand_dims(k, axis=axis)
    cutoff_E_val = np.take_along_axis(E_sorted, k_expanded, axis=axis)
    idx_max = E_s >= cutoff_E_val
    s_sparse = s * idx_max

    k = k + 1

    return s, s_sparse, k








