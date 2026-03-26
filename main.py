from src.compressors.ccsds123 import CCSDS123
from src.compressors.hcs1d import HCS1D
from src.compressors.KCS import KCSCompressor
from src.compressors.NBOMP import NBOMP
from src.compressors.sparserep import SparseRep
from src.compressors.hcs3d import HCS3D
from src import util, workflow, dictionary_learning
import numpy as np
import matplotlib.pyplot as plt

hsi = util.load_hsi(r"raw\benchmarks\IndianPines.npy")
rgb, _, _= hsi.get_rgb()

# Y_train = dictionary_learning.prep_hsi_for_dict_learning(hsi, N_train=1000, mode=2)
# D, dmetadata = workflow.learn_dictionary(Y_train, "IndianPines",
#                 dictionary_learning.k_svd, K=300, T_0=3, max_iter=50)


D_path = r"results\dictionaries\IndianPines_k_svd_20260326_161718.npz"
compressor = HCS1D(K=3, sr=0.1, axis=2, Phi_name="SUBSAMPLING", Psi_name=f"LEARNED:{D_path}")
# compressor = HCS3D(40000,
#                    sr=[0.6, 0.6, 0.6],
#                    Phi_names=["SUBSAMPLING", "SUBSAMPLING", "SUBSAMPLING"],
#                    Psi_names=["IDCT", "IDCT", "IDCT"])
# compressor = CCSDS123()

results = workflow.run_compression(hsi, compressor, save_bitstream=True, save_reconstruction=True)


rec = util.load_hsi(r"results\hcs1d\reconstructions\AVIRIS_IndianPines_IndianPines_20260326_172155.npy")
rgb_rec, _, _ = rec.get_rgb()
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.subplot(1,2,2)
plt.imshow(rgb_rec)
plt.show()



