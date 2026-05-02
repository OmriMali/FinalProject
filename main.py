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

D_path = r"results\dictionaries\JasperRidge_s53_k_svd_20260412_203247.npz"

hsi_path = r"raw\JasperRidge\sections\f060514t01p00r09s4.npy"
hsi = util.load_hsi(hsi_path)

compressor = HCS1D(K=3, sr=0.1, axis=2, Phi_name="SUBSAMPLING", Psi_name=f"LEARNED:path={D_path}")
# for sr in [0.5, 0.2, 0.1, 0.05, 0.133, 0.08, 0.0667, 0.0571]:
    # compressor.sr = sr
    # for i in range(1):
# workflow.run_compression(hsi, compressor, "test_1", save_reconstruction=True, save_bitstream=True)

compressor = CCSDS123(local_sum_mode="column", P=2, a=300)
# for a in [0, 10, 50, 300]:
#     compressor.a = a
#     for i in range(1):
# workflow.run_compression(hsi, compressor, "test_1", save_reconstruction=True, save_bitstream=True)


series = [data_processing.get_averaged_metric_series(r"results\hcs1d\hcs1d_log.csv", "cr", "rmse", "HCS1D", ["sensor", "site", "name", "samplingrate"], {"tag": "test_1"}),
          data_processing.get_averaged_metric_series(r"results\ccsds123\ccsds123_log.csv", "cr", "rmse", "CCSDS123", ["sensor", "site", "name", "a"], {"tag": "test_1"})]
data_processing.plot_multiple_series(series, "Compression Ratio", "RMSE", connect_points=True)


rec_hcs1d = visuals.load_recent_hsi(r"results\hcs1d\hcs1d_log.csv")
rec_ccsds123 = visuals.load_recent_hsi(r"results\ccsds123\ccsds123_log.csv")
visuals.compare_hsi_list([hsi, rec_hcs1d, rec_ccsds123], ["Original", "HCS1D", "CCSDS123"])
visuals.compare_spectra([hsi, rec_hcs1d, rec_ccsds123], ["Original", "HCS1D", "CCSDS123"], [(20,20), (50,50)])

# # plt.figure()
# # plt.plot(hsi.wavelengths)
# # plt.show()