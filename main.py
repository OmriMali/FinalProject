import matplotlib.pyplot as plt

from src.compressors.ccsds123 import CCSDS123
from src.compressors.hcs1d import HCS1D
from src.compressors.hcs3d import HCS3D
from src.io import loaders, aviris
from src.pipeline.experiment_runner import run_compression



hsi = loaders.load_hsi(r"raw\JasperRidge\sections\f060514t01p00r09s4.npy")

compressor = HCS1D(3, 0.1, 2)
run_compression(hsi, compressor)


rgb, _, _ = hsi.get_rgb()

plt.figure()
plt.imshow(rgb)
plt.show()