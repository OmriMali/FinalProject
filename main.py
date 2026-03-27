from src.compressors.ccsds123 import CCSDS123
from src.compressors.hcs1d import HCS1D
from src.compressors.KCS import KCSCompressor
from src.compressors.NBOMP import NBOMP
from src.compressors.sparserep import SparseRep
from src.compressors.hcs3d import HCS3D
from src import util, workflow, dictionary_learning
import numpy as np
import matplotlib.pyplot as plt

# Load HSI
hsi = util.load_hsi(r"raw\benchmarks\IndianPines.npy")

# Setup Compressor
D_path = r"results\dictionaries\IndianPines_k_svd_20260326_161718.npz"
compressor = HCS1D(K=3, sr=1, axis=2, Phi_name="SUBSAMPLING", Psi_name=f"LEARNED:path={D_path}")

# Run
for sr in [0.8]:
    compressor.sr = sr
    results = workflow.run_compression(hsi, compressor, save_bitstream=True, save_reconstruction=True)

# # Visualize
# rec_hsi = results["reconstructed_hsi"]
# rgb, _, _ = hsi.get_rgb()
# rec_rgb, _, _ = rec_hsi.get_rgb()

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(rgb)
# plt.subplot(1,2,2)
# plt.imshow(rec_rgb)
# plt.show()
