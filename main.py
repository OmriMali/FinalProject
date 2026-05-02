import matplotlib.pyplot as plt

from src.compressors.ccsds123 import CCSDS123
from src.compressors.hcs1d import HCS1D
from src.compressors.hcs3d import HCS3D
from src.compressors.registry import list_compressors, get_compressor
from src.math.measurement_matrices import list_measurements, get_measurement_matrix
from src.math.transforms import list_transforms, get_transform
from src.io import loaders, aviris
from src.pipeline.compression_running import CompressionRunner
from src.pipeline.dictionary_learning import DictionaryLearner
from src.core.compression_run_item import CompressionRunItem
from src.pipeline.logger import Logger

# D_path = r"results\dictionaries\JasperRidge_s53_k_svd_20260412_203247.npz"

hsi = loaders.load_hsi(r"raw\JasperRidge\sections\f060514t01p00r09s4.npy")
compressor = get_compressor("hcs1d")

config = CompressionRunItem(
    hsi= hsi,
    compressor_name="hcs1d",
    compressor_params= {
        'K': 30,
        'sr': 0.1,
        'axis': 1,
        'Phi_name': "SUBSAMPLING",
        'Psi_name': "IDCT"
    },
    experiment_machine = 'Almog - Desktop',
    save_hsi=True
)


result = CompressionRunner(logger=Logger()).run(config)
recon = loaders.load_hsi(r"results\experiments\exp_20260502_172047\reconstruction.npy")
rgb, _, _ = recon.get_rgb()
plt.figure()
plt.imshow(rgb)
plt.show()


