import numpy as np
import matplotlib.pyplot as plt

from src.pipeline.runner import Runner
from src.ui.console.callback import ConsoleCallback
from src.loggers.csv import CSVLoggerCallback
from src.loggers.artifacts import ArtifactLoggerCallback

from src.core.dictionary import Axis

from src import io, compressors, dictionary_trainers
from src.preprocessing.training_signals import sample_training_signals
from src.preprocessing.hsi import trim_borders, crop_hsi_sections, filter_spectral_bands

# ========== Load Data ========== #

hsi = io.load_hsi(directory=r"data\processed\JasperRidge\sections", name="JasperRidge_r1_c1")
signals = io.load_training_signals(r"data\training", "sampled_JasperRidge_r1_c1")
dict_dir = r"results\artifacts\dictionary\2026-05-16T16_18_40_ksvd_ksvd"
dict_name = r"dictionary"

# ========== Compressors ========== #

config = compressors.HCS1DConfig(K=3, sr=0.1, axis=Axis.SPECTRAL, Phi="SUBSAMPLING", Psi=f"LEARNED:directory={dict_dir},name={dict_name}")
compressor = compressors.HCS1D(config=config)

# config = compressors.HCS3DConfig(K=10000, sr=(0.5, 0.5, 0.1), Psis=("IDCT", "IDCT", f"LEARNED:directory={dict_dir},name={dict_name}"))
# compressor = compressors.HCS3D(config=config)

# config = compressors.CCSDS123Config(a=50)
# compressor = compressors.CCSDS123(config=config)

# ========== Trainers ========== #

# config = dictionary_trainers.K_SVDConfig(K=400, T_0=3)
# trainer = dictionary_trainers.K_SVD(config)

# ========== Pipeline ========== #

runner = Runner(callbacks=[
    ArtifactLoggerCallback(
        root_dir="results/artifacts",
    ),
    CSVLoggerCallback(
        log_dir="results/logs"
    ),
    ConsoleCallback()
])

result = runner.run_compression(hsi, compressor)
# result = runner.run_dictionary_training(signals, trainer)

# ========== Plot ========== #

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(hsi.data[:,:,10], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(result.reconstructed.data[:,:,10], cmap='gray')
plt.show()

# ========== Preprocessing HSIs ========== #

# for scene in ["JasperRidge", "MoffetField", "Cuprite"]:

#     hsi = io.load_aviris_folder(rf"data\raw\{scene}")
#     hsi = filter_spectral_bands(hsi, remove_ranges=[(104, 108), (150,163)], remove_bands=[220])
#     hsi = trim_borders(hsi, black_value=-50)
#     io.save_hsi(hsi, rf"data\processed\{scene}", f"{scene}")

#     sections = crop_hsi_sections(hsi, (256, 256), drop_incomplete=True)
#     for sec in sections:
#         io.save_hsi(sec, fr"data\processed\{scene}\sections", f"{scene}_r{sec.metadata.section_row}_c{sec.metadata.section_col}")

# plt.figure()
# plt.subplot(2, 2, 1)
# plt.imshow(io.load_hsi(fr"data\processed\JasperRidge\sections", "JasperRidge_r1_c1").data[:,:,10])
# plt.subplot(2, 2, 2)
# plt.imshow(io.load_hsi(fr"data\processed\JasperRidge\sections", "JasperRidge_r1_c2").data[:,:,10])
# plt.subplot(2, 2, 3)
# plt.imshow(io.load_hsi(fr"data\processed\JasperRidge\sections", "JasperRidge_r2_c1").data[:,:,10])
# plt.subplot(2, 2, 4)
# plt.imshow(io.load_hsi(fr"data\processed\JasperRidge\sections", "JasperRidge_r2_c2").data[:,:,10])
# plt.show()