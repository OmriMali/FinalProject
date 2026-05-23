from src.compressors.ccsds123 import CCSDS123
from src.compressors.hcs1d import HCS1D
from src.compressors.KCS import KCSCompressor
from src.compressors.NBOMP import NBOMP
from src.compressors.sparserep import SparseRep
from src.compressors.hcs3d import HCS3D
from src import util, workflow, dictionary_learning, transforms, data_processing, visuals
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


# # Option 2: Run compression:
# # Load HSI
# hsi = util.load_hsi(r"C:\Users\omrim\Documents\FinalProject\raw\Mixed\sections\test\f060514t01p00r09_s11.npy")
# # Setup Compressor
# D_path = r"C:\Users\omrim\Documents\FinalProject\results\dictionaries\Mixed_k_svd_20260501_102323.npz"
# # D_path = r"C:\Users\omrim\Documents\FinalProject\results\dictionaries\MoffetField_k_svd_20260428_145325.npz"

# # compressor = HCS1D(K=3, sr=1, axis=2, Phi_name="SUBSAMPLING", Psi_name=f"LEARNED:path={D_path}")
# compressor = CCSDS123(P=2, a=400)
# # compressor = HCS3D(K=4800, sr = [0.5, 0.5, 0.1], Phi_names=["SUBSAMPLING", "SUBSAMPLING", "SUBSAMPLING"], Psi_names=["IDCT", "IDCT", f"LEARNED:path={D_path}"])

# # # Run   
# for sr in [0.090909]:
#     ber = 0.00000
#     compressor.sr = sr
#     compressor.protected_bitstream = (ber > 0)
#     results = workflow.run_compression(hsi, compressor, ber=ber, save_bitstream=True, save_reconstruction=True)

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

# row, col = np.array(hsi.shape[:2]) // 2
# plt.figure(figsize=(8, 4))
# plt.plot(hsi.wavelengths, hsi.data[row, col, :], 'k-', label='Original')
# plt.plot(hsi.wavelengths, rec_hsi.data[row, col, :], 'r--', label='Reconstructed')
# plt.title(f"Spectral Comparison at Pixel ({row}, {col})")
# plt.xlabel("Wavelength (nm)"); plt.ylabel("Intensity")
# plt.legend(); plt.grid(True, alpha=0.3)
# plt.show()


# # Option C: run sweep compression
# # 1. Setup Paths
# # Point to your deterministic test folder created by split_aviris_sections
# test_folder = r"C:\Users\omrim\Documents\FinalProject\raw\Mixed\sections\test"
# D_path = r"C:\Users\omrim\Documents\FinalProject\results\dictionaries\Mixed_k_svd_20260501_102323.npz"

# # Get list of all available test patches
# test_files = [f for f in os.listdir(test_folder) if f.endswith('.npy')]

# # 2. Define Benchmark Parameters
# # sr_values = [0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5]
# sr_values = [0.0833, 0.0667, 0.05]
# a_values = [0, 2, 8, 16, 32, 64, 128, 256, 400, 800]
# iterations = 10 

# # 3. Execution Loop for HCS1D
# print(f"\n{'='*20} RUNNING HCS1D BENCHMARK (RANDOM SAMPLING) {'='*20}")
# for sr in sr_values:
#     print(f"\n>>> Target SR: {sr}")
#     for i in range(iterations):
#         # Sample a random HSI from the test set for this iteration
#         random_file = random.choice(test_files)
#         hsi = util.load_hsi(os.path.join(test_folder, random_file))
        
#         print(f"Iteration {i+1}/{iterations} | File: {random_file}")
        
#         hcs = HCS1D(K=3, sr=sr, axis=2, Phi_name="SUBSAMPLING", Psi_name=f"LEARNED:path={D_path}")
#         workflow.run_compression(hsi, hcs, ber=0, save_bitstream=True, save_reconstruction=False)

# # # 4. Execution Loop for CCSDS123
# print(f"\n{'='*20} RUNNING CCSDS123 BENCHMARK (RANDOM SAMPLING) {'='*20}")
# for a_val in a_values:
#     print(f"\n>>> Error Limit (a): {a_val}")
#     for i in range(iterations):
#         # Sample a random HSI from the test set for this iteration
#         random_file = random.choice(test_files)
#         hsi = util.load_hsi(os.path.join(test_folder, random_file))
        
#         print(f"Iteration {i+1}/{iterations} | File: {random_file}")
        
#         ccs = CCSDS123(P=2, a=a_val)
#         workflow.run_compression(hsi, ccs, ber=0, save_bitstream=True, save_reconstruction=False)

# print("\nBenchmark complete. 180 total runs logged with randomized scene selection.")


# PLOTS:----------------------------------------------------------------------------------------
# 1. Setup paths to your result logs
hcs_csv = r"C:\Users\omrim\Documents\FinalProject\results\hcs1d\hcs1d_log_plot.csv"
ccs_csv = r"C:\Users\omrim\Documents\FinalProject\results\ccsds123\ccsds123_log_plot.csv"

# 2. Process HCS1D Data
# Grouping by 'samplingrate' to average the 10 iterations per point
hcs_rmse_series = data_processing.get_averaged_metric_series(
    csv_path=hcs_csv,
    x_metric="cr",
    y_metric="rmse",
    label="HCS1D",
    groupby_cols=["samplingrate"]
)

# 3. Process CCSDS123 Data
# Grouping by 'a' (error limit) to average the 10 iterations per point
ccs_rmse_series = data_processing.get_averaged_metric_series(
    csv_path=ccs_csv,
    x_metric="cr",
    y_metric="rmse",
    label="CCSDS123",
    groupby_cols=["a"]
)

# 4. Generate the Plot
# This will use the COLOR_MAP and error bar formatting from your module
data_processing.plot_multiple_series(
    series_list=[hcs_rmse_series, ccs_rmse_series],
    x_label="Compression Ratio (CR)",
    y_label="RMSE",
    connect_points=True,
    show_error = True
)


# 1. Fetch the data
hcs_series = data_processing.get_averaged_metric_series(
    hcs_csv, "cr", "comp_time", "HCS1D", ["samplingrate"]
)
ccs_series = data_processing.get_averaged_metric_series(
    ccs_csv, "cr", "comp_time", "CCSDS123", ["a"]
)

# 2. Plot with your custom function, passing the log scale!
data_processing.plot_multiple_series(
    [hcs_series, ccs_series], 
    x_label="Compression Ratio (CR)", 
    y_label="Compression Time (seconds)", 
    connect_points=True,
    show_error=True, 
    y_scale='log',
    y_max=1000
)



# # --- SIMPLIFIED POSTER VISUALS: IMAGES ---

# # 1. Fetch the last 2 runs for each
# hcs_runs = data_processing.fetch_recent(r"results\hcs1d\hcs1d_log.csv", n=2)
# ccs_runs = data_processing.fetch_recent(r"results\ccsds123\ccsds123_log.csv", n=2)

# # 2. Setup the lists for comparison
# # These keys (CR, RMSE, etc.) will show up automatically in the metadata box
# hcs_items = [{"hsi": util.load_hsi(e['rec_path']), 
#               "label": "HCS", 
#               "CR": round(e['cr'], 1), 
#               "RMSE": round(e['rmse'], 1),
#               "SAM": round(e['sam'], 2),
#               "Time": f"{e['comp_time']:.1f}s"} for e in hcs_runs]

# ccs_items = [{"hsi": util.load_hsi(e['rec_path']), 
#               "label": "CCSDS", 
#               "CR": round(e['cr'], 1), 
#               "RMSE": round(e['rmse'], 1),
#               "SAM": round(e['sam'], 2),
#               "Time": f"{e['comp_time']:.1f}s"} for e in ccs_runs]

# # 3. Plot them separately to fix the "weird" colors and keep titles clean
# # We use independent_norm=True so each image stretches itself correctly
# visuals.compare_hsis(hcs_items, independent_norm=True, fontsize=20)
# visuals.compare_hsis(ccs_items, independent_norm=True, fontsize=20)