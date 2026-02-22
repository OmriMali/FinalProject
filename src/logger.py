import os
import json
import csv
import numpy as np
from datetime import datetime

class DataLogger:
    """
    Handles the logging of experiment results, including CSV logs and artifact storage.

    Attributes
    ----------
    compressor_name : str
        Name of the compressor used for directory and file naming.
    compressor_id : int
        A 2-digit unique identifier for the compressor.
    comp_dir : str
        Path to the compressor-specific results directory.
    bitstream_dir : str
        Directory where compressed bitstreams (.bin) are stored.
    recons_dir : str
        Directory where reconstructed HSI cubes (.npz) are stored.
    metadata_dir : str
        Directory where experiment metadata (.json) is stored.
    log_file : str
        Path to the main CSV log file.
    current_counter : int
        The current sequence number for generating unique run IDs.
    """
    def __init__(self, compressor_name, compressor_id, base_dir="results"):
        """
        Initialize the DataLogger and create the necessary directory structure.

        Parameters
        ----------
        compressor_name : str
            Name of the compressor (e.g., 'ccsds123').
        compressor_id : int
            A 2-digit ID prefix for run identification (e.g., 11).
        base_dir : str, optional
            The root directory for all results, by default "results".
        """
        self.compressor_name = compressor_name
        self.compressor_id = compressor_id
        
        # Folder structure per compressor type
        self.comp_dir = os.path.join(base_dir, compressor_name)
        self.bitstream_dir = os.path.join(self.comp_dir, "bitstreams")
        self.recons_dir = os.path.join(self.comp_dir, "reconstructions")
        self.metadata_dir = os.path.join(self.comp_dir, "metadata")
        self.log_file = os.path.join(self.comp_dir, f"{compressor_name}_log.csv")
        
        os.makedirs(self.bitstream_dir, exist_ok=True)
        os.makedirs(self.recons_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        self.current_counter = self._get_starting_counter()

    def _get_starting_counter(self):
        """
        Scan the existing log file to find the next available sequence number.

        Returns
        -------
        int
            The starting counter (1 if no log exists, otherwise the next sequence).
        """
        if not os.path.exists(self.log_file):
            return 1
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1: return 1
                last_line = lines[-1]
                last_id = last_line.split(',')[0]
                # Extract the last 4 digits (the counter portion)
                return int(last_id[-4:]) + 1
        except (IndexError, ValueError):
            return 1

    def get_new_run_id(self):
        """
        Generate a unique Run ID combining the compressor ID and a 4-digit counter.

        Returns
        -------
        int
            The unique run ID (e.g., 110001).
        """
        run_id = int(f"{self.compressor_id}{self.current_counter:04d}")
        self.current_counter += 1
        return run_id

    def _init_csv(self, param_keys):
        """
        Initialize the CSV log file with standardized headers and specific parameters.

        Parameters
        ----------
        param_keys : list or dict_keys
            Keys representing compressor-specific parameters to include as headers.
        """
        headers = [
            "run_id", "timestamp", "dataset", 
            "rmse", "psnr_db", "sam_deg", "compression_ratio",
            "comp_time_s", "decomp_time_s"
        ] + list(param_keys)
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def save_run(self, run_id, dataset_name, metadata, metrics, 
                 bitstream=None, reconstructed_hsi=None):
        """
        Persist experiment artifacts and log metrics to the CSV file.

        Parameters
        ----------
        run_id : int
            The unique ID for this specific execution.
        dataset_name : str
            The name of the dataset processed.
        metadata : dict
            Dictionary containing 'params' and other run-specific info.
        metrics : dict
            Dictionary containing 'rmse', 'psnr', 'sam', 'cr', 'comp_time', 
            and 'decomp_time'.
        bitstream : bytes, optional
            The raw compressed data to be saved as a .bin file.
        reconstructed_hsi : numpy.ndarray, optional
            The reconstructed HSI cube to be saved as a compressed .npz file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_base = f"{run_id}_{dataset_name}"
        params = metadata.get('params', {})

        # Initialize CSV with specific compressor parameters as headers
        if not os.path.exists(self.log_file):
            self._init_csv(params.keys())

        # Save Artifacts
        meta_path = os.path.join(self.metadata_dir, f"{file_base}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        if bitstream is not None:
            bit_path = os.path.join(self.bitstream_dir, f"{file_base}.bin")
            with open(bit_path, "wb") as f:
                f.write(bitstream)

        if reconstructed_hsi is not None:
            recon_path = os.path.join(self.recons_dir, f"{file_base}_recon.npz")
            np.savez_compressed(recon_path, data=reconstructed_hsi)

        # Log to CSV using scientific notation for precision metrics
        log_row = [
            run_id,
            timestamp,
            dataset_name,
            f"{metrics['rmse']:.6e}",
            f"{metrics['psnr']:.6e}",
            f"{metrics['sam']:.6e}",
            f"{metrics['cr']:.6e}",
            f"{metrics['comp_time']:.4f}",
            f"{metrics['decomp_time']:.4f}"
        ] + [params.get(k, "") for k in params.keys()]

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_row)