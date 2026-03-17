from src.compressors.ccsds123 import CCSDS123
from src.compressors.hcs1d import HCS1D
from src.compressors.KCS import KCSCompressor
from src.compressors.NBOMP import NBOMP
from src.compressors.sparserep import SparseRep
from src.compressors.hcs3d import HCS3D
from src.run_handler import RunHandler
from src.logger import DataLogger
from src import util
from src import visuals
from src.sparse import Sparsity
import numpy as np
import time
import matplotlib.pyplot as plt

# 1. Setup components
compressor = HCS3D(10000, sr=[0.5, 0.5, 0.5], Phi_names=["SUBSAMPLING", "SUBSAMPLING", "SUBSAMPLING"], Psi_names=["DCT", "DCT", "DCT"])
logger = DataLogger(compressor_name=compressor.name, compressor_id=compressor.compressor_id, base_dir="results")
handler = RunHandler(compressor, logger)

# 2. Load HSI
hsi_data = util.load_hsi("raw\\Indian_pines_corrected.mat")
# hsi_data = hsi_data[:50, :50, :50]

# 3. Run:
# a. Single Run:
handler.run_experiment(hsi_data, dataset_name="IndianPines", save_reconstruction=True)

# b. Sweep Run:
# a = [0, 1, 2, 4, 8]
# for a in a:
#     compressor.a = a
#     handler.run_experiment(hsi_data, dataset_name=f"IndianPines")

orig = visuals.to_false_color(hsi_data)
rec = visuals.to_false_color(visuals.load_reconstruction("results\\HCS 3D\\reconstructions\\220013_IndianPines_recon.npz"))
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(orig)
plt.subplot(1, 2, 2)
plt.imshow(rec)
plt.show()