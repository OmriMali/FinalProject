### version: 0.6.0
### date: 09/12/25
### author: Omri "(Omi)" Malik
### description: Main script to test CCSDS compression on hyperspectral images with parameter sweep functionality.

import numpy as np
import CCSDS_compression
import CCSDS_new
import util

# Load Data
image_path="data\\SalinasA_corrected.mat"
image_name_short="SA"

image = util.load_image(image_path)
# image = image[:50, :50, :50]


# Define the Sweep
param_name = "P"
param_values = [1, 2, 3]

# Lists to store aggregated results
rmse_list = []
sam_list = []
ratio_list = []
time_list = []

print(f"--- Starting Sweep for {param_name} ---")

for val in param_values:

    compressor = CCSDS_new.CCSDS_123(P=val, a=0)
    print(f"Running P={val}...")
    res = compressor.run(image) 
  
    m = res["metrics"]
    rmse_list.append(m["RMSE"])
    sam_list.append(m["SAM"])
    ratio_list.append(m["Compression Ratio"])
    time_list.append(m["Compression Time"])

fixed_params = res["params"].copy()
fixed_params.pop("P", None)

output_dir = f"results\\CCSDS\\{image_name_short}_sweep_{param_name}"

util.save_sweep_results(
    param_name=param_name,
    param_values=param_values,
    RMSEs=rmse_list,
    SAMs=sam_list,
    ratios=ratio_list,
    complexities=time_list,
    fixed_params=fixed_params,
    directory=output_dir,
    name=f"{image_name_short}_sweep_{param_name}",
    titles=False
)

