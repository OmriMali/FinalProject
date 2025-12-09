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

RMSEs, SAMs, ratios = CCSDS_compression.sweep_CCSDS(
    image, 
    param_name='Q', 
    param_values=param_values, 
    fixed_params=fixed_params
    )

util.save_sweep_results(
    param_name='Q',
    param_values=param_values,
    RMSEs=RMSEs,
    SAMs=SAMs,
    ratios=ratios,
    filename='sweep_test_1',
    directory='results/CCSDS'
)