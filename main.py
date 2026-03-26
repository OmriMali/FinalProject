from src.compressors.ccsds123 import CCSDS123
from src.compressors.hcs1d import HCS1D
from src.compressors.KCS import KCSCompressor
from src.compressors.NBOMP import NBOMP
from src.compressors.sparserep import SparseRep
from src.compressors.hcs3d import HCS3D
from src import util
from src import workflow

import numpy as np
import time
import matplotlib.pyplot as plt

# # 1. Load HSI
# hsi_data = util.load_hsi("raw\\Indian_pines_corrected.mat")
# # hsi_data = hsi_data[:128, :128, :32]

# # 2. Setup components
# # compressor = HCS3D(40000,
# #                     sr=[0.57, 0.57, 0.57],
# #                     Phi_names=["SUBSAMPLING", "SUBSAMPLING", "SUBSAMPLING"],
# #                     Psi_names=["IDCT", "IDCT", "LEARNED:results\\learned_dicts\\D_ksvd.npz"])
# compressor = HCS1D(3, sr=0.25, axis=2, Phi_name="SUBSAMPLING", 
#                    Psi_name="LEARNED:results\\learned_dicts\\D_ksvd.npz")
# logger = DataLogger(compressor_name=compressor.name, compressor_id=compressor.compressor_id, base_dir="results")
# handler = RunHandler(compressor, logger)

# # 3. Run:
# handler.run_experiment(hsi_data, dataset_name="IndianPines", save_reconstruction=True)


# # visuals
# orig = visuals.to_false_color(hsi_data)
# rec = visuals.to_false_color(visuals.load_reconstruction("results\\hcs1d\\reconstructions\\210009_IndianPines_recon.npz"))
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(orig)
# plt.subplot(1, 2, 2)
# plt.imshow(rec)
# plt.show()

hsi = util.load_hsi(f"raw\\f060514t01p00r09rdn_c\\sections\\f060514t01p00r09s03.npy")
rgb, _, _= hsi.get_rgb()

# compressor = HCS1D(K=30, sr=0.25, axis=2, Phi_name="SUBSAMPLING", Psi_name="IDCT")
# # compressor = HCS3D(40000,
# #                    sr=[0.6, 0.6, 0.6],
# #                    Phi_names=["SUBSAMPLING", "SUBSAMPLING", "SUBSAMPLING"],
# #                    Psi_names=["IDCT", "IDCT", "IDCT"])
# # compressor = CCSDS123()

# results = workflow.run_compression(hsi, compressor)
# workflow.log_compression_run(results, save_bitstream=True, save_reconstruction=True)

rec = util.load_hsi(r"results\hcs1d\reconstructions\AVIRIS_JasperRidge_f060514t01p00r09s03_20260326_092201.hsi.npy")
rgb_rec, _, _ = rec.get_rgb()
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.subplot(1,2,2)
plt.imshow(rgb_rec)
plt.show()



