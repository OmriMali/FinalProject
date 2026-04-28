import os
import csv
import pickle
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

from src import util
from src.hsi import HSI
from src.compressors.base import BaseCompressor


# ===== Helpers ===== #

def _sanitize(s: str) -> str:
    """Remove spaces and unsafe characters."""
    return str(s).replace(" ", "")

def _get_base_filename(results: dict) -> str:
    """Generate standardized filename."""
    sensor = _sanitize(results["sensor"])
    site = _sanitize(results["site"])
    name = _sanitize(results["name"])
    timestamp = results["timestamp"]

    return f"{sensor}_{site}_{name}_{timestamp}"

def _save_bitstream(bitstream, path: str):
    """Save bitstream in appropriate format."""
    if isinstance(bitstream, bytes):
        with open(path + ".bin", "wb") as f:
            f.write(bitstream)

    elif isinstance(bitstream, np.ndarray):
        np.save(path + ".npy", bitstream)

    else:
        # fallback
        with open(path + ".pkl", "wb") as f:
            pickle.dump(bitstream, f)

def _append_to_csv(csv_path: str, row: dict):
    """
    Append a row to CSV, dynamically expanding columns if needed.
    """
    file_exists = os.path.exists(csv_path)

    if file_exists:
        # Read existing header
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
    else:
        existing_fields = []

    # Merge fields
    new_fields = list(row.keys())
    all_fields = list(dict.fromkeys(existing_fields + new_fields))

    # If new fields appeared → rewrite file
    if set(all_fields) != set(existing_fields):
        rows = []

        if file_exists:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            writer.writerow(row)
    else:
        # Simple append
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

# ===== Running Expirements ===== #

def run_compression(hsi: HSI, compressor: BaseCompressor, ber=0, save_bitstream=False, save_reconstruction=False):
    """
    Run compression and decompression on an HSI object, compute metrics, display progress, log.

    Parameters
    ----------
    hsi : HSI
        Hyperspectral image object
    compressor : BaseCompressor
        Compressor instance implementing

    Returns
    -------
    dict
        Results dictionary containing:
        - run_id
        - bitstream
        - reconstructed_hsi
        - metrics (rmse, psnr, sam, cr, comp_time, decomp_time)
        - metadata (compressor parameters needed for reconstruction)
    """
    # ===== Start ===== #
    name = hsi.metadata.get("name", "unknown_name")
    site = hsi.metadata.get("site", "unknown_site")
    sensor = hsi.metadata.get("sensor", "unknown_sensor")
    print(f"\n{'='*20} Compressing {sensor}: {site} ({name}) with {compressor.name} {'='*20}")

    # ===== Compression ===== #
    with tqdm(total=100, desc="Compression", unit="%") as pbar:
        def progress_cb(fraction):
            pbar.n = int(fraction * 100)
            pbar.refresh()

        compressor.progress_callback = progress_cb
        start_comp = time.perf_counter()
        bitstream, metadata = compressor.compress(hsi)
        comp_time = time.perf_counter() - start_comp
        progress_cb(1.0)

    # ===== NOISY CHANNEL ===== #
    if ber > 0:
        print(f"[Channel] Adding noise with BER: {ber:.2e}")
        mask = metadata.get("protected_mask", None) 
        bitstream = util.add_bit_noise(data_bytes=bitstream, ber=ber, protected_mask=mask)
    # ===== Decompression ===== #
    with tqdm(total=100, desc="Decompression", unit="%") as pbar:
        def progress_cb_dec(fraction):
            pbar.n = int(fraction * 100)
            pbar.refresh()

        compressor.progress_callback = progress_cb_dec
        start_decomp = time.perf_counter()
        reconstructed_hsi = compressor.decompress(bitstream, metadata)
        decomp_time = time.perf_counter() - start_decomp
        progress_cb_dec(1.0)
    
    # ===== Metrics ===== #
    metrics = util.compute_all_metrics(hsi, reconstructed_hsi, bitstream)
    metrics.update({"comp_time": comp_time, "decomp_time": decomp_time})

    print(f"Compression Results | CR: {metrics['cr']:.3f} | RMSE: {metrics['rmse']:.3e} | "
          f"PSNR: {metrics['psnr']:.3f} dB | SAM: {metrics['sam']:.3e}°")
    print(f"Time | Compression: {comp_time:.2f}s | Decompression: {decomp_time:.2f}s")

    # ===== Wrap Results ===== #
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S")

    results = {
        "compressor": compressor.name,
        "sensor": sensor,
        "site": site,
        "name": name,
        "metrics": metrics,
        "compressor_metadata": metadata,
        "timestamp": timestamp,
        "bitstream": bitstream,
        "reconstructed_hsi": reconstructed_hsi
    }

    # ===== Log ===== #
    log_run_compression(results, save_bitstream=save_bitstream, save_reconstruction=save_reconstruction)

    return results

def learn_dictionary(Y: np.ndarray, dict_name: str,algorithm, base_dir = "results/dictionaries", **kwargs):
    """
    Top level generic dictionary learning function. Learns the dictionary and logs it.

    Parameters
    ----------
    Y : ndarray (M, N)
        Input signals (columns)
    dict_name : str
        Name of the dictionary
    algorithm : callable
        Dictionary learning function (e.g., k_svd)
    **kwargs :
        Algorithm-specific parameters

    Returns
    -------
    D : ndarray
        Learned dictionary
    metadata : dict
        Contains info about the algorithm and metrics.
    """

    algo_name = getattr(algorithm, "__name__", "unknown")

    print(f"\n{'='*20} Learning Dictionary {dict_name} using the {algo_name} Algorithm {'='*20}")

     # ===== Progress bar wrapper ===== #
    with tqdm(total=100, desc=algo_name, unit="%") as pbar:
        def progress_cb(fraction):
            pbar.n = int(fraction * 100)
            pbar.refresh()

        if "progress_callback" in algorithm.__code__.co_varnames:
            kwargs["progress_callback"] = progress_cb

        start = time.perf_counter()

        D, X = algorithm(Y, **kwargs)

        train_time = time.perf_counter() - start

    # ===== Metrics ===== #
    err = np.linalg.norm(Y - D @ X) / np.linalg.norm(Y)
    mean_sparsity = np.mean(np.count_nonzero(X, axis=0))

    print(f"Reconstruction Error: {err:.3e}")
    print(f"Mean Sparsity: {mean_sparsity:.2f}")
    print(f"Training Time: {train_time:.2f}s")


    # ===== Metadata ===== #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "name": dict_name,
        "algorithm": algo_name,
        "timestamp": timestamp,
        "params": {k: v for k, v in kwargs.items() if k != "progress_callback"},
        "metrics": {
            "reconstruction_error": err,
            "mean_sparsity": mean_sparsity,
            "train_time": train_time}
    }

    # ===== Log ===== #
    log_learn_dictionary(D, metadata, base_dir=base_dir)

    return D, metadata

# ===== Logging Results ===== #

def log_run_compression(results: dict, save_bitstream=False, save_reconstruction=False):
    """
    Log a compression run to disk.

    Parameters
    ----------
    results : dict
        Output from run_compression()
    save_bitstream : bool
    save_reconstruction : bool
    """

    compressor_name = _sanitize(results["compressor"])

    # ===== Paths ===== #
    root = os.getcwd()
    results_dir = os.path.join(root, "results", compressor_name)

    bitstream_dir = os.path.join(results_dir, "bitstreams")
    recon_dir = os.path.join(results_dir, "reconstructions")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(bitstream_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, f"{compressor_name}_log.csv")

    # ===== Filename ===== #
    base_name = _get_base_filename(results)

    # ===== Save Bitstream ===== #
    if save_bitstream:
        bitstream_path = os.path.join(bitstream_dir, base_name)
        _save_bitstream(results["bitstream"], bitstream_path)

    # ===== Save Reconstruction ===== #
    if save_reconstruction:
        recon_path = os.path.join(recon_dir, base_name)
        util.save_hsi(results["reconstructed_hsi"], recon_path)

    # ===== Prepare CSV Row ===== #
    metrics = results["metrics"]
    params = results["compressor_metadata"].get("params", {})

    row = {
        "sensor": results["sensor"],
        "site": results["site"],
        "name": results["name"],
        "timestamp": results["timestamp"],
        "rmse": metrics["rmse"],
        "psnr": metrics["psnr"],
        "sam": metrics["sam"],
        "cr": metrics["cr"],
        "comp_time": metrics["comp_time"],
        "decomp_time": metrics["decomp_time"],
    }

    # Add compressor params dynamically
    for k, v in params.items():
        row[_sanitize(k)] = v

    # ===== Write CSV ===== #
    _append_to_csv(csv_path, row)

    print(f"[LOG] Saved run: {compressor_name}")

def log_learn_dictionary(D: np.ndarray, metadata: dict, base_dir="results/dictionaries"):
    """
    Save dictionary + metadata and log to CSV.
    """
    # ---- Extract fields ---- #
    dict_name = metadata["name"]
    algorithm = metadata["algorithm"]
    timestamp = metadata["timestamp"]

    # ---- Sanitize ---- #
    safe_name = _sanitize(dict_name)
    safe_algo = _sanitize(algorithm)

    # ---- Paths ---- #
    base_name = f"{safe_name}_{safe_algo}_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    save_path = os.path.join(base_dir, base_name + ".npz")

    # ---- Save NPZ ---- #
    util.save_array_to_path(D, save_path, metadata=metadata)

    # ---- Prepare CSV row ---- #
    row = {
        "name": dict_name,
        "algorithm": algorithm,
        "timestamp": timestamp,
    }

    # Add metrics
    for k, v in metadata.get("metrics", {}).items():
        row[k] = v

    # Add params (dynamic)
    for k, v in metadata.get("params", {}).items():
        row[k] = v

    # ---- Append to CSV ---- #
    csv_path = os.path.join(base_dir, "dict_log.csv")
    _append_to_csv(csv_path, row)

    print(f"[LOG] Dictionary saved to {save_path}")

    return save_path
