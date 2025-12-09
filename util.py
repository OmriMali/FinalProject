import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import csv
import os

def load_image(path):
    mat = sp.io.loadmat(path)
    mat_clean = {k: v for k, v in mat.items() if not k.startswith('__')}
    # If there is only one remaining field, extract it
    if len(mat_clean) == 1:
        data_array = next(iter(mat_clean.values()))
    else:
        # If multiple fields, select the one with the largest array (often the data)
        data_array = max(mat_clean.values(), key=lambda x: getattr(x, 'size', 0))
    return data_array

def calc_RMSE(I, I_hat):
    
    shape = I.shape
    factor = 1
    for dim in shape:
        factor *= dim
    
    return np.sqrt(np.sum(np.pow(I - I_hat, 2)) / factor)

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

def save_sweep_results(param_name, param_values, RMSEs, SAMs, ratios,
                       directory, filename,
                       titles=False
                       ):
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
    RMSEs, SAMs, ratios : array-like
        Results from sweep_CCSDS.
    directory : str
        Directory to save files.
    filename : str
        Parent name for the output files.
    titles : bool
        Whether to add titles to the plots.
    """

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # ============
    # 1. SAVE CSV
    # ============
    csv_path = os.path.join(directory, filename)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([param_name, "RMSE", "SAM", "Compression_Ratio"])
        for p, r, s, c in zip(param_values, RMSEs, SAMs, ratios):
            writer.writerow([p, r, s, c])

    print(f"Saved CSV results to {csv_path}")


    # ======================================
    # 2. PREPARE PLOT FILE NAMES (if needed)
    # ======================================
    plot_filenames = {
            "rmse":  f"{filename}_rmse.png",
            "sam":   f"{filename}_sam.png",
            "ratio": f"{filename}_ratio.png",
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