import time
from datetime import datetime
from tqdm import tqdm

from src.core.compression_run_item import CompressionRunItem
from src import util
from src.compressors.registry import get_compressor
from src.math.metrics import compute_all_metrics

class CompressionRunner:
    def __init__(self, logger=None):
        self.logger = logger

    def run(self, item: CompressionRunItem) -> CompressionRunItem:
        """
        Runs compression + decompression experiment.
        """
        hsi = item.hsi
        compressor = get_compressor(
            item.compressor_name, **item.compressor_params)
        
        name = hsi.metadata.get("name", "unknown_name")
        site = hsi.metadata.get("site", "unknown_site")
        sensor = hsi.metadata.get("sensor", "unknown_sensor")

        print(
            f"\n{'='*20} Compressing {sensor}: {site} ({name}) "
            f"with {item.compressor_name} {'='*20}"
            )
        
        # 1. Compression
        item.update_status("compressing")

        with tqdm(total=100, desc="Compression", unit="%") as pbar:

            def progress_cb(fraction):
                pbar.n = int(fraction * 100)
                pbar.refresh()

            compressor.progress_callback = progress_cb

            start = time.perf_counter()
            bitstream, metadata = compressor.compress(hsi)
            comp_time = time.perf_counter() - start

            progress_cb(1.0)

        # 2. Channel noise
        if item.ber and item.ber > 0:
            print(f"Adding channel noise (BER: {item.ber:.2e})")

            mask = metadata.get("protected_mask", None)
            bitstream = util.add_bit_noise(
                data_bytes = bitstream,
                ber = item.ber,
                protected_mask = mask
            )
        
        # 3. Decompression
        item.update_status("reconstructing")

        with tqdm(total=100, desc="Decompression", unit="%") as pbar:

            def progress_cb_dec(fraction):
                pbar.n = int(fraction * 100)
                pbar.refresh()

            compressor.progress_callback = progress_cb_dec

            start = time.perf_counter()
            reconstructed = compressor.decompress(bitstream, metadata)
            decomp_time = time.perf_counter() - start

            progress_cb_dec(1.0)

        # 4. Metrics
        item.update_status("evaluating")

        metrics = compute_all_metrics(hsi, reconstructed, bitstream)
        metrics.update({
            "comp_time": comp_time,
            "decomp_time": decomp_time
        })

        print(
            f"Compression Results | CR: {metrics['cr']:.3f} | "
            f"RMSE: {metrics['rmse']:.3e} | PSNR: {metrics['psnr']:.3f} dB | "
            f"SAM: {metrics['sam']:.3e}"
        )

        # 6. Update ExperimentItem
        item.bitstream = bitstream
        item.metadata = metadata
        item.reconstructed = reconstructed
        item.metrics = metrics
        item.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        item.update_id()

        # 7. call logger
        if self.logger:
            self.logger.log_compression(item)

        item.update_status("done")

        return item

    
