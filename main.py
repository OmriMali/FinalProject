from src.compressors.ccsds123 import CCSDS123
from src.compressors.hcs1d import HCS1D
from src.compressors.KCS import KCSCompressor
from src.compressors.NBOMP import NBOMP
from src.compressors.sparserep import SparseRep
from src.compressors.hcs3d import HCS3D
from src import util, workflow, dictionary_learning
from src import transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# # Workflow 1: Dictionary Learning: 
# # Option A: From a given HSI
# # Load HSI
# hsi = util.load_hsi(r"C:\Users\omrim\Documents\FinalProject\raw\JasperRidge\sections\f060514t01p00r09s20.npy")
# # Preprocess: Unfold HSI and pick random fibers for training
# N_TRAIN = 10000 
# MODE = 2 # Spectral axis (bands)
# print(f"Preparing {N_TRAIN} training samples...")
# Y_train = dictionary_learning.prep_hsi_for_dict_learning(hsi, N_train=N_TRAIN, mode=MODE)
# # Execute high-level dictionary learning workflow
# # This handles: progress bars, timing, reconstruction error, and logging to results/dictionaries/
# D_learned, metadata = workflow.learn_dictionary(
#     Y=Y_train,
#     dict_name="JasperRidge",
#     algorithm=dictionary_learning.k_svd,
#     base_dir="results/dictionaries",
#     K=256,      # Dictionary size
#     T_0=10,     # Sparsity constraint
#     max_iter=30 # Iterations
# )

# # # Option B: From a created spectral library:
# # Preprocess: Unfold HSI and pick random fibers for training
# Y_train = util.build_diverse_spectral_library(
#     folder_path=r"C:\Users\omrim\Documents\FinalProject\raw\Mixed\sections\train", 
#     threshold=0.995,   # Relaxed threshold to allow more signatures
#     max_atoms=2000    # Target a much larger basis for redundancy
#     )
# D_learned, metadata = workflow.learn_dictionary(
#     Y=Y_train,
#     dict_name="Mixed",
#     algorithm=dictionary_learning.k_svd,
#     base_dir="results/dictionaries",
#     K=256  ,      # Dictionary size
#     T_0=3,     # Sparsity constraint
#     max_iter=50 # Iterations
# )

# # # Option C: K-SVD from multiple HSI
# # Improved Option B Logic
# all_Y = []
# folder = r"C:\Users\omrim\Documents\FinalProject\raw\JasperRidge\sections"
# for filename in os.listdir(folder)[:5]: # Take 5 patches
#     patch = util.load_hsi(os.path.join(folder, filename))
#     # This ensures every patch is normalized exactly like Option A
#     Y_patch = dictionary_learning.prep_hsi_for_dict_learning(patch, N_train=2000, mode=2)
#     all_Y.append(Y_patch)

# Y_train = np.hstack(all_Y) # Combined training set
# D_learned, metadata = workflow.learn_dictionary(
#     Y=Y_train,
#     dict_name="JasperRidge ",
#     algorithm=dictionary_learning.k_svd,
#     base_dir="results/dictionaries",
#     K=256  ,      # Dictionary size
#     T_0=3,     # Sparsity constraint
#     max_iter=50 # Iterations
# )

# Option 2: Run compression:
# Load HSI
hsi = util.load_hsi(r"C:\Users\omrim\Documents\FinalProject\raw\MoffetField\sections\test\f120507t01p00r13_test_s90.npy")
# Setup Compressor
D_path = r"C:\Users\omrim\Documents\FinalProject\results\dictionaries\Mixed_k_svd_20260428_145732.npz"
# D_path = r"C:\Users\omrim\Documents\FinalProject\results\dictionaries\MoffetField_k_svd_20260428_145325.npz"

compressor = HCS1D(K=3, sr=1, axis=2, Phi_name="SUBSAMPLING", Psi_name=f"LEARNED:path={D_path}")
# compressor = CCSDS123(P=2, a=8, local_sum_mode = "neighbor")
# compressor = HCS3D(K=4800, sr = [0.5, 0.5, 0.1], Phi_names=["SUBSAMPLING", "SUBSAMPLING", "SUBSAMPLING"], Psi_names=["IDCT", "IDCT", f"LEARNED:path={D_path}"])

# # Run   
for sr in [0.2]:
    ber = 0.00000
    compressor.sr = sr
    compressor.protected_bitstream = (ber > 0)
    results = workflow.run_compression(hsi, compressor, ber=ber, save_bitstream=True, save_reconstruction=True)

# Visualize
rec_hsi = results["reconstructed_hsi"]
rgb, _, _ = hsi.get_rgb()
rec_rgb, _, _ = rec_hsi.get_rgb()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.subplot(1,2,2)
plt.imshow(rec_rgb)
plt.show()

row, col = np.array(hsi.shape[:2]) // 2
plt.figure(figsize=(8, 4))
plt.plot(hsi.wavelengths, hsi.data[row, col, :], 'k-', label='Original')
plt.plot(hsi.wavelengths, rec_hsi.data[row, col, :], 'r--', label='Reconstructed')
plt.title(f"Spectral Comparison at Pixel ({row}, {col})")
plt.xlabel("Wavelength (nm)"); plt.ylabel("Intensity")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
