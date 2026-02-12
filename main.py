### version: 0.7.0
### date: 22/12/25
### author: Almog "hamelech" Sade
### description: Main script to test CCSDS compression on hyperspectral images with parameter sweep functionality.

import numpy as np
import CCSDS_compression
import CCSDS_new
import util
import matplotlib.pyplot as plt

# Load Data
image_path="data\\Indian_pines_corrected.mat"
image_name_short="IP"
image = util.load_image(image_path)
# image = image[:50, :50, :50]

# # Sweep
# sweep_param = 'a'
# sweep_vals = [0, 2, 4, 6, 8, 10]
# fixed_params = {'local_sum_mode': 'column', 'P': 2, 'Omega': 8, 'block_size': 32}
# bands_to_save = [50, 100, 150]

# compressor = CCSDS_new.CCSDS_123()
# compressor.sweep(
#     image=image,
#     image_name=image_name_short,
#     sweep_param=sweep_param,
#     sweep_values=sweep_vals,
#     fixed_params=fixed_params,
#     bands_snapshot=bands_to_save,
#     save_path="results\\CCSDS",
# )

# # Single run
# compressor = CCSDS_new.CCSDS_123(local_sum_mode='column', P=2, Omega=8, a=2, block_size=32)
# results = compressor.run(image, dataset_name='IndianPines')
# util.save_results(results, "results\\CCSDS")

recon = np.load("results\\CCSDS\\CCSDS_123_IndianPines_20260212_091506\\rec_HSI.npy")
recon_fc = util.plot_false_color(recon, bands=(160,80,0), show=False)
image_fc = util.plot_false_color(image, bands=(160,80,0), show=False)

plt.figure()
plt.subplot(2,1,1)
plt.imshow(image_fc)
plt.axis("off")
plt.title("Original HSI")
plt.subplot(2,1,2)
plt.imshow(recon_fc)
plt.axis("off")
plt.title("Reconstructed HSI")
plt.tight_layout()
plt.show()

