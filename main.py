### version: 0.6.0
### date: 09/12/25
### author: Omri "(Omi)" Malik
### description: Main script to test CCSDS compression on hyperspectral images with parameter sweep functionality.

import numpy as np
import CCSDS_compression
import util

# Load an image:
image_name="Indian_pines_corrected.mat"
image_name_short="IP"

image = util.load_image(f"data\\{image_name}")
# # Test on image's crop (faster):
# image = image[:50, :50, :50]

# Testing with sweep on a specific parameter:
param_name='P'
param_values = [0, 1, 2, 3, 4, 8]
fixed_params = {
    "local_sum_mode": "wide",
    "P": 2,
    "Q": 4,
    "Omega": 1, 
    "block_size": 64,
    "BER": 0.000}

# Option A: For testing with parameter sweep:

images_r, bitstreams, deltas, complexities = CCSDS_compression.sweep_CCSDS(
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
    f'results/CCSDS/{param_name}_sweep_{image_name_short}',
    f'{param_name}_sweep_{image_name_short}'
)

param_chosen_index=4

print(f"results for {image_name_short} with:")
for key, value in fixed_params.items():
    if(key==param_name):
        print(f"{param_name} = {param_values[param_chosen_index]}")
    else:
        print(f"{key} = {value}")
print(f"RMSE = {util.calc_RMSE(image,images_r[param_chosen_index])}")
print(f"SAM = {util.calc_SAM(image,images_r[param_chosen_index])}")
print(f"comp_ratio = {util.calc_compression_ratio(image,bitstreams[param_chosen_index])}")
print(f"compression_time_complexity = {complexities[param_chosen_index]}")

# # optional plots (carefull not to overide): 
# util.save_histogram(image, 'results/CCSDS/histograms', f'{image_name_short}_hist')
# util.save_histogram(deltas[param_chosen_index], 'results/CCSDS/histograms', f'{image_name_short}_compressed_hist')
# util.save_images(image, image_name_short, images_r[param_chosen_index], f"{image_name_short}_reconstructed", f"{image_name_short}_comparission", 'results/CCSDS/reconstructed_images')


# Option B: For testing with different W (without seep):

# image_r, bit_stream, deltas, compression_time_comlexity=CCSDS_compression.CCSDS(image, 
#                                                                                 fixed_params["local_sum_mode"],
#                                                                                 P=fixed_params["P"], 
#                                                                                 W=np.ones(fixed_params["P"]) / fixed_params["P"], 
#                                                                                 Omega=fixed_params["Omega"], 
#                                                                                 Q=fixed_params["Q"], 
#                                                                                 block_size=fixed_params["block_size"],
#                                                                                 BER=fixed_params["BER"])


# print(f"results for {image_name_short} with:")
# for key, value in fixed_params.items():
#     print(f"{key} = {value}")
# print(f"RMSE={util.calc_RMSE(image,image_r)}")
# print(f"SAM={util.calc_SAM(image,image_r)}")
# print(f"comp_ratio={util.calc_compression_ratio(image,bit_stream)}")
# print(f"compression_time_complexity = {compression_time_comlexity}")

# # # optional plots (carefull not to overide): 
# # util.save_histogram(image, 'results/CCSDS/histograms', f'{image_name_short}_hist')
# # util.save_histogram(deltas, 'results/CCSDS/histograms', f'{image_name_short}_compressed_hist')
# # util.save_images(image, image_name_short, image_r, f"{image_name_short}_reconstructed", f"{image_name_short}_comparission_with_BER", 'results/CCSDS/reconstructed_images')
