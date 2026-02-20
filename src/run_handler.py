import time
from tqdm import tqdm
from src.util import compute_all_metrics

class RunHandler:
    """
    Orchestrates the compression and decompression workflow for HSI data.

    Attributes
    ----------
    compressor : object
        The compressor instance used for the experiment.
    logger : DataLogger
        The logger instance used for recording results.
    """
    def __init__(self, compressor, logger):
        """
        Initialize the RunHandler with a specific compressor and logger.

        Parameters
        ----------
        compressor : object
            Must implement .compress() and .decompress() methods.
        logger : DataLogger
            An initialized DataLogger instance.
        """
        self.compressor = compressor
        self.logger = logger

    def run_experiment(self, hsi, dataset_name="unknown", save_bitstream=True, save_reconstruction=True):
        """
        Execute a full compression-decompression cycle and evaluate performance.

        This includes time measurement, progress tracking, metric calculation, 
        and artifact logging.

        Parameters
        ----------
        hsi : numpy.ndarray
            The input hyperspectral cube to compress.
        dataset_name : str, optional
            Identifier for the dataset, by default "unknown".
        save_bitstream : bool, optional
            Whether to save the resulting bitstream to disk, by default True.
        save_reconstruction : bool, optional
            Whether to save the reconstructed HSI cube to disk, by default True.

        Returns
        -------
        dict
            The calculated metrics for the run (rmse, psnr, sam, cr, times).
        """
        run_id = self.logger.get_new_run_id()
        print(f"\n{'='*20} RUN {run_id} : {dataset_name} {'='*20}")

        # --- 1. COMPRESSION ---
        with tqdm(total=100, desc="Compression", unit="%") as pbar:
            def progress_cb(fraction):
                pbar.n = int(fraction * 100)
                pbar.refresh()

            self.compressor.progress_callback = progress_cb
            start_comp = time.perf_counter()
            bitstream_bytes, metadata = self.compressor.compress(hsi)
            comp_time = time.perf_counter() - start_comp
            pbar.n = 100
            pbar.refresh()

        # --- 2. DECOMPRESSION ---
        with tqdm(total=100, desc="Decompression", unit="%") as pbar:
            def progress_cb_dec(fraction):
                pbar.n = int(fraction * 100)
                pbar.refresh()
        
            self.compressor.progress_callback = progress_cb_dec
            start_decomp = time.perf_counter()
            reconstructed = self.compressor.decompress(bitstream_bytes, metadata)
            decomp_time = time.perf_counter() - start_decomp
            pbar.n = 100
            pbar.refresh()

        # --- 3. METRICS ---
        # Note: compute_all_metrics now maps values to [-1, 1] internally for RMSE and SAM
        results = compute_all_metrics(hsi, reconstructed, bitstream_bytes)
        metrics = {**results, "comp_time": comp_time, "decomp_time": decomp_time}

        # --- 4. LOGGING ---
        self.logger.save_run(
            run_id=run_id,
            dataset_name=dataset_name,
            metadata=metadata,
            metrics=metrics,
            bitstream=bitstream_bytes if save_bitstream else None,
            reconstructed_hsi=reconstructed if save_reconstruction else None
        )

        self._print_summary(metrics)
        return metrics

    def _print_summary(self, m):
        """
        Print a formatted performance summary to the console.

        Parameters
        ----------
        m : dict
            The metrics dictionary containing 'cr', 'rmse', 'psnr', 'sam', 
            and timing data.
        """
        print(f"CR: {m['cr']:.3f} | RMSE: {m['rmse']:.3e} | PSNR: {m['psnr']:.3f} dB | SAM: {m['sam']:.3e}°")
        print(f"Time: Comp {m['comp_time']:.2f}s | Decomp {m['decomp_time']:.2f}s")