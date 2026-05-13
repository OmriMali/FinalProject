import numpy as np
import matplotlib.pyplot as plt

from src.pipeline.runner import Runner
from src.ui.console.callback import ConsoleCallback
from src.loggers.csv_callback import CSVLoggerCallback

from src.core.dictionary import Axis
from src.compressors.hcs1d import HCS1D, HCS1DConfig
from src.compressors.hcs3d import HCS3D, HCS3DConfig
from src.compressors.ccsds123 import CCSDS123, CCSDS123Config
from src.dictionary_trainers.k_svd import K_SVD, K_SVDConfig
from src.io.hsi import load_hsi, save_hsi
from src.io.dictionary import load_dictionary, save_dictionary
from src.io.training_signals import load_training_signals, save_training_signals
from src.preprocessing.training_signals import sample_training_signals
from src.io import aviris
from src.preprocessing.hsi import trim_borders, crop_hsi_sections, filter_spectral_bands

# ========== Load Data ========== #

hsi = load_hsi(r"data\processed\JasperRidge\JasperRidge_r1_c1.npz")
dict_path = r"results\dictionaries\JasperRidge_0_Dict1.npz"

# ========== Compressors ========== #

config = HCS1DConfig(K=3, sr=0.1, axis=Axis.SPECTRAL, Phi="SUBSAMPLING", Psi=f"LEARNED:path={dict_path}")
compressor = HCS1D(config=config)

# config = HCS3DConfig(K=4000, sr=(0.8, 0.8, 0.1), Psis=("IDCT", "IDCT", f"LEARNED:path={dict_path}"))
# compressor = HCS3D(config=config)

# config = CCSDS123Config(a=50)
# compressor = CCSDS123(config=config)

# ========== Trainers ========== #

# config = K_SVDConfig(K=400, T_0=3)
# trainer = K_SVD(config)

# ========== Pipeline ========== #

runner = Runner(callbacks=[ConsoleCallback(), CSVLoggerCallback(log_dir="results/logs")])
result = runner.run_compression(hsi, compressor)
# result = runner.run_dictionary_training(signals, trainer)

# ========== Plot ========== #

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(hsi.data[:,:,10], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(result.reconstructed.data[:,:,10], cmap='gray')
plt.show()


# for scene in ["JasperRidge", "MoffetField", "Cuprite"]:
#     hsi = load_hsi(fr"data\processed\{scene}\{scene}.npz")
#     hsi = filter_spectral_bands(hsi, remove_ranges=[(104, 108), (150,163)], remove_bands=[220])
#     save_hsi(hsi, fr"data\processed\{scene}\{scene}.npz")
#     sections = crop_hsi_sections(hsi, (150, 150))
#     for sec in sections:
#         save_hsi(sec, fr"data\processed\{scene}\{scene}_r{sec.metadata.section_row}_c{sec.metadata.section_col}.npz")

# plt.figure()
# plt.subplot(2, 2, 1)
# plt.imshow(load_hsi(fr"data\processed\JasperRidge\JasperRidge_r4_c1.npz").data[:,:,10])
# plt.subplot(2, 2, 2)
# plt.imshow(load_hsi(fr"data\processed\JasperRidge\JasperRidge_r4_c2.npz").data[:,:,10])
# plt.subplot(2, 2, 3)
# plt.imshow(load_hsi(fr"data\processed\JasperRidge\JasperRidge_r5_c1.npz").data[:,:,10])
# plt.subplot(2, 2, 4)
# plt.imshow(load_hsi(fr"data\processed\JasperRidge\JasperRidge_r5_c2.npz").data[:,:,10])
# plt.show()