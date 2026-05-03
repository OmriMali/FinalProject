import time
from datetime import datetime
from src.core.experiment_item import ExperimentItem
from src.registry.compressors import get_compressor
from src.math.metrics import compute_all_metrics
from src import util

class ExperimentRunner:

    def __init__(self, logger=None, reporter=None):
        self.logger = logger
        self.reporter = reporter

    def run(self, item: ExperimentItem):

        if item.task == "compression":
            self._run_compression(item)

        elif item.task == "dict_learning":
            self._run_dict_learning(item)

        else:
            raise ValueError(f"Unknown experiment: {item.task}")
        
        return item
    
    def _run_compression(self, item: ExperimentItem):

        self.reporter.prefix = "RUN"
        self.reporter.start("Compression Experiment")

        # 1. get experiment params
        ber = item.experiment_params.get("ber", 0)
        save_hsi = item.experiment_params.get("save_hsi", False)

        # 2. get compressor and initialize it
        cls = get_compressor(item.method)
        compressor = cls(item.config)
        
        # 3. compress
        self.reporter.create_bar("compression", desc="Compression")
        compressor.progress_callback = lambda p: self.reporter.update("compression", p)
        t0 = time.perf_counter()
        bitstream, metadata = compressor.compress(item.data)
        comp_time = time.perf_counter() - t0

        # 4. Add channel noise (optional)
        if ber > 0:
            self.reporter.message(f"Adding channnel noise (BER: {ber})")
            protected_mask = metadata.get("protected_mask", None)
            bitstream = util.add_bit_noise(bitstream, ber, protected_mask)
        
        # 5. decompress
        self.reporter.create_bar("decompression", desc="Decompression")
        compressor.progress_callback = lambda p: self.reporter.update("decompression", p)
        t0 = time.perf_counter()
        reconstructed = compressor.decompress(bitstream, metadata)
        decomp_time = time.perf_counter() - t0

        # 6. evaluate
        metrics = compute_all_metrics(item.data, reconstructed, bitstream)
        metrics.update({"comp_time": comp_time, "decomp_time": decomp_time})

        # 7. pack results
        item.timestamp = datetime.now().strftime("%Y/%m/%d_%H:%M:%S_%f")
        item.metrics = metrics
        if save_hsi:
            item.artifacts["reconstructed"] = reconstructed

        self.reporter.end("Compression Experiment")