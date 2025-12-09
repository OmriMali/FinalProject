### version: 0.5.0
### date: 09/12/25
### author: Almog (the king)

import numpy as np
import CCSDS_compression
import util

image = util.load_image("data\\Indian_pines_corrected.mat")
image = image[:50, :50, :50]

param_values = [0, 1, 2, 3]
fixed_params = {
    "local_sum_mode": "col",
    "P": 1,
    "W": 0.5*np.ones(1),
    "Omega": 0,
    "block_size": 32
}

images_r, bitstreams, deltas = CCSDS_compression.sweep_CCSDS(
    image, 
    param_name='Q', 
    param_values=param_values, 
    fixed_params=fixed_params
    )
RMSE, SAM, ratio = util.calc_sweep_metrics(image, images_r, bitstreams)
util.save_sweep_results(
    'Q',
    param_values,
    RMSE,
    SAM,
    ratio,
    'results/CCSDS/test1',
    'test1'
)

util.save_histogram(image, 'results/CCSDS/histograms', 'indian_pines_hist')
util.save_histogram(deltas[0], 'results/CCSDS/histograms', 'compressed_hist')