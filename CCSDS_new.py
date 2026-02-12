### version: 0.7.0
### date: 22/12/25
### author: Almog "hamelech" Sade
### description: CCSDS-123 Compressor in a class implementation.

import numpy as np
import util
import time
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt


class CCSDS_123:
    def __init__(self, local_sum_mode='column', P=1, Omega=8, a=0, block_size=32):
        self.local_sum_mode = local_sum_mode            # Mode for local sum
        self.P = P                                      # Amount of spectral bands used in predictor
        self.Omega = Omega                              # Resolution of calculation
        self.a = a                                      # Absolute error limit (per pixel)
        self.block_size = block_size                    # Block size for encoder
        
        weights = []
        w_0 = (7 * (1 << self.Omega)) >> 3
        weights.append(w_0)

        for i in range(1, self.P):
            w_prev = weights[-1]
            w_next = w_prev >> 3
            weights.append(w_next)
        
        self.W = np.array(weights, dtype=np.int32)     # Weights for predictor

    def load(self, Image):

        if not isinstance(Image, np.ndarray):
            raise TypeError("Input Image must be a numpy array")
        
        if not np.issubdtype(Image.dtype, np.integer):
            raise TypeError(f"Input Image must be of integer type, got {Image.dtype}")

        info = np.iinfo(Image.dtype)
        if info.bits > 16:
            raise ValueError(f"Input Image integer depth exceeds 16 bits: {info.bits} bits")
        
        if Image.ndim != 3:
            raise ValueError(f"Input Image must be 3D, got shape {Image.shape}")

        self.S = Image.astype(np.int32)

        self.smin, self.smax = util.get_bounds(self.S)

        self.Nx, self.Ny, self.Nz = self.S.shape

    def encoder_predictor(self):
        
        S_rep = np.zeros_like(self.S, dtype=np.int32)
        delta = np.zeros_like(self.S, dtype=np.int32)
        U = np.zeros((self.Nx, self.Ny, self.P), dtype=np.int32)
        a_den = 2*self.a + 1

        for z in tqdm(range(self.Nz), desc="[1/4] Encoder Prediction", unit='band'):
            for y in range(self.Ny):
                for x in range(self.Nx):
                    
                    sigma = self.calc_local_sum(S_rep[:, :, z], x, y)

                    d_hat = np.dot(self.W, U[x, y, :])

                    s_hat = (d_hat + (sigma << self.Omega)) >> (2 + self.Omega)
                    s_hat = np.clip(s_hat, self.smin, self.smax)

                    Delta = self.S[x, y, z] - s_hat

                    q = np.sign(Delta) * ((np.abs(Delta) + self.a) // a_den)
    
                    s_rep = np.clip(s_hat + q * a_den, self.smin, self.smax)
                    S_rep[x, y, z] = s_rep
                    
                    U[x, y, 1:] = U[x, y, :-1]
                    U[x, y, 0] = 4*s_rep - sigma

                    if q >= 0:
                        delta[x, y, z] = 2 * q
                    else:
                        delta[x, y, z] = 2 * abs(q) - 1

        self.delta = delta
        return delta
    
    def calc_local_sum(self, S_rep_z, x, y):

        def rep(xx, yy):
            xx = min(max(xx, 0), self.Nx - 1)
            yy = min(max(yy, 0), self.Ny - 1)
            return S_rep_z[xx, yy]

        if self.local_sum_mode == 'column':
            if y > 0:
                return 4 * rep(x, y-1)
            elif x > 0:
                return 4 * rep(x-1, y)
            else:
                return 0

        elif self.local_sum_mode == 'neighbor':
            if y == 0 and x == 0:
                return 0
            elif y == 0:
                return 4 * rep(x-1, y)
            elif x == 0:
                return 2 * (rep(x, y-1) + rep(x+1, y-1))
            elif x == self.Nx - 1:
                return rep(x-1, y) + rep(x-1, y-1) + 2 * rep(x, y-1)
            else:
                return (
                    rep(x-1, y) +
                    rep(x-1, y-1) +
                    rep(x,   y-1) +
                    rep(x+1, y-1)
                )

        else:
            raise ValueError(f"Invalid local sum mode: {self.local_sum_mode}")
    
    def rice_encoder(self):
        if not hasattr(self, "delta"):
            raise RuntimeError("Run encoder_predictor first")

        data = []
        for z in range(self.Nz):
            for y in range(self.Ny):
                for x in range(self.Nx):
                    data.append(int(self.delta[x, y, z]))
        
        bitstream = []
        k_list = []
        idx = 0
        N = len(data)

        total_blocks = (N + self.block_size -1) // self.block_size

        with tqdm(total=total_blocks, desc="[2/4] Rice Encoding   ", unit="blk") as pbar:
            while idx < N:
                end = min(idx + self.block_size, N)
                block = data[idx:end]
                idx = end
                
                sorted_block = sorted(block)
                mid_idx = len(block) // 2
                med = sorted_block[mid_idx] 
                
                k = 0
                if med > 0:
                    k = med.bit_length() - 1
                    k = int(np.floor(np.log2(med + 1)))
                
                k_list.append(k)

                for value in block:
                    q = value >> k
                    r = value & ((1 << k) - 1)
                    
                    # Unary part
                    bitstream.extend([1] * q)
                    bitstream.append(0) 
                    
                    # Remainder part
                    for b in reversed(range(k)):
                        bitstream.append((r >> b) & 1)

                pbar.update(1)

        self.k = np.array(k_list, dtype=np.int32)
        self.bitstream = np.array(bitstream, dtype=np.uint8)
        return self.k, self.bitstream

    def rice_decoder(self):
        if not hasattr(self, "k"): raise RuntimeError("No k params")
        if not hasattr(self, "bitstream"): raise RuntimeError("No bitstream")

        bitstream = self.bitstream
        k_list = self.k
        
        total_pixels = self.Nx * self.Ny * self.Nz
        flat_delta = []
        
        bit_idx = 0
        k_idx = 0
        
        # Decode Loop
        with tqdm(total=total_pixels, desc="[3/4] Rice Decoding     ", unit="pix") as pbar:
            while len(flat_delta) < total_pixels:
                if k_idx >= len(k_list):
                    break
                    
                current_k = int(k_list[k_idx])
                k_idx += 1
                
                remaining = total_pixels - len(flat_delta)
                count = min(self.block_size, remaining)
                
                for _ in range(count):
                    # Decode Unary
                    q = 0
                    while bit_idx < len(bitstream) and bitstream[bit_idx] == 1:
                        q += 1
                        bit_idx += 1
                    bit_idx += 1 # skip delimiter 0
                    
                    # Decode Remainder
                    r = 0
                    for _ in range(current_k):
                        if bit_idx >= len(bitstream): break
                        r = (r << 1) | int(bitstream[bit_idx])
                        bit_idx += 1
                    
                    val = (q << current_k) + r
                    flat_delta.append(val)
                
                pbar.update(count)

        # Reconstruct 3D Array (Z, Y, X order)
        delta_r = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.int32)
        idx = 0
        for z in range(self.Nz):
            for y in range(self.Ny):
                for x in range(self.Nx):
                    if idx < len(flat_delta):
                        delta_r[x, y, z] = flat_delta[idx]
                        idx += 1
                    
        self.delta_r = delta_r
        return delta_r
    
    def decoder_predictor(self):

        if not hasattr(self, "delta_r"):
                raise RuntimeError("Mapped quantizer indices delta_r not available")

        S_rep = np.zeros_like(self.S, dtype=np.int32)
        U = np.zeros((self.Nx, self.Ny, self.P), dtype=np.int32)

        for z in tqdm(range(self.Nz), desc="[4/4] Decoder Prediction", unit="band"):
            for y in range(self.Ny):
                for x in range(self.Nx):
                    
                    sigma = self.calc_local_sum(S_rep[:, :, z], x, y)

                    d_hat = np.dot(self.W, U[x, y, :])

                    s_hat = (d_hat + (sigma << self.Omega)) >> (2 + self.Omega)
                    s_hat = np.clip(s_hat, self.smin, self.smax)

                    delta_val = self.delta_r[x, y, z]

                    if delta_val % 2 == 0:
                        q = delta_val // 2      
                    else:
                        q = -(delta_val + 1) // 2

                    s_rep = np.clip(s_hat + q * (2*self.a + 1), self.smin, self.smax)
                    S_rep[x, y, z] = s_rep           

                    U[x, y, 1:] = U[x, y, :-1]
                    U[x, y, 0] = 4*s_rep - sigma

        self.S_rec = S_rep
        return S_rep

    def run(self, Image, dataset_name=None):
        
        self.load(Image)

        start_time = time.time()

        self.encoder_predictor()
        _, bitstream = self.rice_encoder()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.rice_decoder()
        S_rec = self.decoder_predictor()

        rmse = util.calc_RMSE(self.S, S_rec)
        sam = util.calc_SAM(self.S, S_rec)
        ratio = util.calc_compression_ratio(self.S, bitstream)

        cube_shape = self.S.shape
        cube_dtype = str(Image.dtype)

        if dataset_name is None:
            dataset_name = "unknown_dataset"

        results = {
            "name": "CCSDS_123",
            "dataset": dataset_name,
            "cube_shape": cube_shape,
            "dtype": cube_dtype,
            "reconstructed": S_rec,
            "bitstream": bitstream,
            "metrics": {
                "RMSE": rmse,
                "SAM": sam,
                "Compression Ratio": ratio,
                "Compression Time": elapsed_time
            },
            "params": {
                "Local Sum Mode": self.local_sum_mode,
                "P": self.P,
                "Omega": self.Omega,
                "a": self.a,
                "Block Size": self.block_size
            }
        }
    
        return results
    
    def sweep(self, image, image_name, sweep_param, sweep_values, fixed_params, bands_snapshot=[], save_path=None):

        # 1. Setup Directories
        if save_path is None:
            save_path = os.getcwd()
            
        folder_name = f"{image_name}_sweep_{sweep_param}_CCSDS_123"
        full_output_path = os.path.join(save_path, folder_name)
        
        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)
            
        csv_filename = f"{folder_name}.csv"
        csv_path = os.path.join(full_output_path, csv_filename)

        # 2. Configure Bands (Limit to 4)
        target_bands = bands_snapshot[:4]
        
        # Dictionary to store slices: { band_idx: [ (label, slice_data), ... ] }
        snapshot_history = {b: [] for b in target_bands}

        # 3. Load Original Data & Store Original Slices
        self.load(image)
        for b in target_bands:
            if 0 <= b < self.Nz:
                # Copying is important so it doesn't get overwritten
                snapshot_history[b].append(
                    ("Original", self.S[:, :, b].copy())
                )

        # 4. Initialize CSV
        header_info = (
            f"# Image: {image_name}\n"
            f"# Shape: {self.S.shape}\n"
            f"# Compressor: CCSDS-123\n"
            f"# Fixed Parameters: {fixed_params}\n"
            f"# Sweep Parameter: {sweep_param}\n"
        )
        
        with open(csv_path, 'w') as f:
            f.write(header_info)
            
        columns = ["Sweep Value", "RMSE", "SAM", "Compression Ratio", "Compression Time"]
        pd.DataFrame(columns=columns).to_csv(csv_path, mode='a', index=False, header=True)

        print(f"Starting sweep for {sweep_param} over values: {sweep_values}")

        # --- Helper Function to Plot a Row ---
        def update_band_plot(band_idx, history_list):
            num_plots = len(history_list)
            # Create a figure: 1 row, N columns. Width expands with more plots.
            fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), squeeze=False)
            axes_flat = axes.flatten()
            
            for i, (label, img_slice) in enumerate(history_list):
                ax = axes_flat[i]
                ax.imshow(img_slice, cmap='gray')
                ax.set_title(label, fontsize=10)
                ax.axis('off')
            
            plt.tight_layout()
            # Overwrite the file so it grows with every step
            plt.savefig(os.path.join(full_output_path, f"band_{band_idx}_comparison.png"), bbox_inches='tight')
            plt.close(fig)

        # 5. Sweep Loop
        for val in sweep_values:
            print(f"\n--- Running sweep: {sweep_param} = {val} ---")
            
            # Update Params
            for param, fixed_val in fixed_params.items():
                setattr(self, param, fixed_val)
            setattr(self, sweep_param, val)    

            # Re-calc weights if needed
            if sweep_param in ['P', 'Omega'] or 'P' in fixed_params or 'Omega' in fixed_params:
                weights = []
                w_0 = (7 * (1 << self.Omega)) >> 3
                weights.append(w_0)
                for i in range(1, self.P):
                    w_prev = weights[-1]
                    w_next = w_prev >> 3
                    weights.append(w_next)
                self.W = np.array(weights, dtype=np.int32)

            # Run
            result = self.run(image)
            metrics = result['metrics']
            
            # Save to CSV
            row_data = {
                "Sweep Value": val,
                "RMSE": metrics['RMSE'],
                "SAM": metrics['SAM'],
                "Compression Ratio": metrics['Compression Ratio'],
                "Compression Time": metrics['Compression Time']
            }
            pd.DataFrame([row_data]).to_csv(csv_path, mode='a', index=False, header=False)
            
            # Save Slices & Update Plots
            S_rec = result['reconstructed']
            for b in target_bands:
                if 0 <= b < self.Nz:
                    # 1. Store the new slice
                    label_str = f"{sweep_param}={val}"
                    snapshot_history[b].append(
                        (label_str, S_rec[:, :, b].copy())
                    )
                    
                    # 2. Update the comparison image file immediately
                    update_band_plot(b, snapshot_history[b])

        print(f"\nSweep completed. Results saved to: {full_output_path}")
