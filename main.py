from src.compressors.ccsds123 import CCSDS123
from src.run_handler import RunHandler
from src.logger import DataLogger
from src import util
import numpy as np

# 1. Setup components
compressor = CCSDS123(P=2, local_sum_mode='column', Omega=8, a=2, block_size=32)
logger = DataLogger(compressor_name=compressor.name, compressor_id=compressor.compressor_id, base_dir="results")
handler = RunHandler(compressor, logger)

# 2. Prepare Data
# hsi_data = np.random.randint(0, 1000, (128, 128, 30), dtype=np.uint16)
hsi_data = util.load_hsi("raw\\Indian_pines_corrected.mat")
min_val, max_val, _ = util.get_hsi_statistics(hsi_data, verbose=True)
hsi_norm = util.normalize_zero_mean(hsi_data, min_val, max_val)
hsi_denorm = util.denormalize_zero_mean(hsi_norm, min_val, max_val)
util.get_hsi_statistics(hsi_denorm, verbose=True)


# # 3. Run
# a = [10]
# for a_val in a:
#     compressor.a = a_val
#     handler.run_experiment(hsi_data, dataset_name="Indian Pines")