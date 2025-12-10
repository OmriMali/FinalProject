### version: 0.5.0
### date: 09/12/25
### author: Almog (the king)

import numpy as np
import CCSDS_compression
import util

image = util.load_image("data\\Indian_pines_corrected.mat")
image = image[:50, :50, :50]

param_name='Q'
param_values = [0, 1, 2, 3, 4, 8]
fixed_params = {
    "local_sum_mode": "wide",
    "P": 2,
    "Q": 8,
    "Omega": 1,
    "block_size": 64
}

images_r, bitstreams, deltas = CCSDS_compression.sweep_CCSDS(
    image, 
    param_name=param_name, 
    param_values=param_values, 
    fixed_params=fixed_params
    )
RMSE, SAM, ratio = util.calc_sweep_metrics(image, images_r, bitstreams)
util.save_sweep_results(
    param_name,
    param_values,
    RMSE,
    SAM,
    ratio,
    'results/CCSDS/test1',
    'test1'
)

# image_r, bit_stream, deltas=CCSDS_compression.CCSDS(image, "wide", P=3, W=np.array([0.8, 0.4, np.sqrt(0.2)]), Omega=1, Q=4, block_size=64)
# print(f"RMSE={util.calc_RMSE(image,image_r)}")
# print(f"SAM={util.calc_SAM(image,image_r)}")
# print(f"comp_ratio={util.calc_compression_ratio(image,bit_stream)}")

util.save_histogram(image, 'results/CCSDS/histograms', 'indian_pines_hist')
util.save_histogram(deltas, 'results/CCSDS/histograms', 'compressed_hist')