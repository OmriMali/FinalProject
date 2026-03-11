import numpy as np
import matplotlib.pyplot as plt
import src.util as util
import src.visuals as visuals
import os
import csv
from datetime import datetime
from typing import Dict

class Sparsity:
    def __init__(self, base_dir: str = "results", verbose = True):
        self.verbose = verbose
        self.dir_path = os.path.join(base_dir, "sparsity")
        self.csv_path = os.path.join(self.dir_path, "sparsity_log.csv")
        self.axis_map = {0: "Vertical", 1: "Horizontal", 2: "Spectral", -1: "Spectral"}

        os.makedirs(self.dir_path, exist_ok=True)

    def analyze(self, basis, hsi: np.ndarray, name: str = "Uknown", axis: int = -1, T: float = 1.0) -> Dict:
        
        # 0. Normalize
        min_val, max_val, bit_depth = util.get_hsi_statistics(hsi)
        hsi_norm = util.normalize_zero_mean(hsi, min_val, max_val)
        
        # 1. Transform
        coeffs = basis.forward(hsi_norm, axis=axis)
        abs_coeffs = np.abs(coeffs)

        # 2. Pixel statistics
        mu = np.mean(abs_coeffs, axis=axis, keepdims=True)
        sigma = np.std(abs_coeffs, axis=axis, keepdims=True)

        # 3. Apply mask
        thres = mu + T * sigma
        mask = abs_coeffs >= thres
        kappa_map = np.sum(mask, axis=axis)
        mean_kappa = np.mean(kappa_map)

        # 4. Recover HSI
        sparse_coeffs = np.where(mask, coeffs, 0)
        recovered_hsi = np.real(basis.inverse(sparse_coeffs, axis=axis))
        recovered_hsi = util.denormalize_zero_mean(recovered_hsi, min_val, max_val)

        # 5. Wrap up
        results = {
            "dataset": name,
            "basis": basis.name,
            "axis": axis,
            "T_value": T,
            "mean_kappa": mean_kappa,
            "kappa_map": kappa_map,
            "recovered_hsi": recovered_hsi,
            "sparse_coeffs": sparse_coeffs,
            "rmse": util.calc_rmse(hsi, recovered_hsi),
            "psnr_db": util.calc_psnr(hsi, recovered_hsi, bit_depth),
            "sam_deg": util.calc_sam(hsi, recovered_hsi),
            "cr": hsi.size / np.sum(kappa_map)
        }

        if self.verbose: self._print_summary(results)
        self._log_to_csv(results)
        self._visualize_results(hsi, recovered_hsi, results)

        return results

    def _print_summary(self, res: Dict):
        """
        Print a formatted summary of the sparsity analysis results.
        """
        min_k = np.min(res['kappa_map'])
        max_k = np.max(res['kappa_map'])
        
        axis_label = self.axis_map.get(res['axis'], str(res['axis']))

        print("\n" + "="*35)
        print(f" SPARSITY ANALYSIS: {res['dataset']}")
        print("-"*35)
        
        # Metadata Block
        print(f"{'Basis:':<15} {res['basis']}")
        print(f"{'Axis:':<15} {axis_label}")
        print(f"{'Threshold (T):':<15} {res['T_value']:.2f}")
        
        print("-" * 35) # Sub-separator
        
        # Sparsity Statistics
        print(f"{'Mean Kappa:':<15} {res['mean_kappa']:.2f}")
        print(f"{'Min Kappa:':<15} {min_k}")
        print(f"{'Max Kappa:':<15} {max_k}")
        print(f"{'Comp. Ratio:':<15} {res['cr']:.2f}")
        
        print("-" * 35) # Sub-separator
        
        # Performance Metrics
        print(f"{'PSNR:':<15} {res['psnr_db']:.2f} dB")
        print(f"{'RMSE:':<15} {res['rmse']:.4f}")
        print(f"{'SAM:':<15} {res['sam_deg']:.2f}°")
        
        print("="*35 + "\n")
    
    def _log_to_csv(self, res: Dict):
        """
        Append the metrics from a single analysis run to the central CSV log file.
        """
        # 1. Define the columns in the order we want them
        fields = [
            "timestamp", "dataset", "basis", "axis", "T_value", 
            "mean_kappa", "min_kappa", "max_kappa", 
            "rmse", "psnr_db", "sam_deg", "cr"
        ]
        
        # 2. Check if we need to write a header
        file_exists = os.path.isfile(self.csv_path)
        
        # 3. Open and append
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            
            if not file_exists:
                writer.writeheader()
            
            # 4. Prepare the row data
            axis_label = self.axis_map.get(res['axis'], str(res['axis']))
            min_k = int(np.min(res['kappa_map']))
            max_k = int(np.max(res['kappa_map']))
            
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": res['dataset'],
                "basis": res['basis'],
                "axis": axis_label,
                "T_value": res['T_value'],
                "mean_kappa": f"{res['mean_kappa']:.4f}",
                "min_kappa": min_k,
                "max_kappa": max_k,
                "rmse": f"{res['rmse']:.6f}",
                "psnr_db": f"{res['psnr_db']:.4f}",
                "sam_deg": f"{res['sam_deg']:.4f}",
                "cr": f"{res['cr']:.4f}"
            }
            
            writer.writerow(row)

    def _get_case_studies(self, kappa_map, mean_kappa):
        """Finds indices for vectors representing max, min, and avg sparsity."""
        # These are indices into the kappa_map, NOT necessarily (y, x)
        idx_a, idx_b = np.unravel_index(np.argmax(kappa_map), kappa_map.shape)
        idx_min_a, idx_min_b = np.unravel_index(np.argmin(kappa_map), kappa_map.shape)
        
        # Find index closest to the mean
        idx_flat_avg = np.argmin(np.abs(kappa_map - mean_kappa))
        idx_avg_a, idx_avg_b = np.unravel_index(idx_flat_avg, kappa_map.shape)
        
        return {
            "high": (idx_a, idx_b, kappa_map[idx_a, idx_b]),
            "low":  (idx_min_a, idx_min_b, kappa_map[idx_min_a, idx_min_b]),
            "avg":  (idx_avg_a, idx_avg_b, kappa_map[idx_avg_a, idx_avg_b])
        }

    def _extract_profile(self, cube, coords, axis):
        """Extracts the specific 1D vector that was transformed."""
        a, b = coords[0], coords[1]
        if axis == 0:
            # Transformed along H: coords represent (W, C)
            return cube[:, a, b].real
        elif axis == 1:
            # Transformed along W: coords represent (H, C)
            return cube[a, :, b].real
        else: # axis == 2 or -1
            # Transformed along C: coords represent (H, W)
            return cube[a, b, :].real
        
    def _get_axis_labels(self, axis):
        """Determines labels for the coordinates based on the transform axis."""
        if axis == 0: return "x", "z"
        if axis == 1: return "y", "z"
        return "y", "x"

    def _visualize_results(self, hsi, recovered, res):
        """Generates a 2x3 dashboard of the sparsity analysis."""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3)
        
        # --- ENHANCED TITLE ---
        axis_name = self.axis_map.get(res['axis'], "Unknown")
        title_main = f"Sparsity Analysis: {res['dataset']}"
        title_sub = (f"Basis: {res['basis']}  |  Axis: {axis_name}  |  "
                     f"Threshold (T): {res['T_value']}  |  "
                     rf"Mean $\kappa$: {res['mean_kappa']:.1f}")
        
        fig.suptitle(f"{title_main}\n", fontsize=18, fontweight='bold', y=0.98)
        fig.text(0.5, 0.93, title_sub, ha='center', fontsize=12, alpha=0.8)

        # --- ROW 1: IMAGES & STATS ---
        # 1. Original
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(visuals.to_false_color(hsi))
        ax_orig.set_title("Original HSI")
        ax_orig.axis('off')

        # 2. Reconstructed
        ax_rec = fig.add_subplot(gs[0, 1])
        ax_rec.imshow(visuals.to_false_color(recovered))
        ax_rec.set_title("Recovered HSI")
        ax_rec.axis('off')

        # --- 3. PDF/CDF Plot ---
        ax_pdf = fig.add_subplot(gs[0, 2])
        vals, counts = np.unique(res['kappa_map'], return_counts=True)
        pdf = counts / np.sum(counts)
        
        # Plot PDF (Left Axis)
        ln1 = ax_pdf.plot(vals, pdf, color='teal', label='PDF', lw=2)
        ax_pdf.set_ylabel("PDF", color='black')
        ax_pdf.set_xlabel(rf"Sparsity ($\kappa$)")
        ax_pdf.set_ylim(bottom=0)

        # Mean Kappa Line (Added to ax_pdf)
        ln3 = [ax_pdf.axvline(res['mean_kappa'], color='red', linestyle=':', label=rf'Mean $\kappa$={res["mean_kappa"]:.1f}')]
        
        # Gather all line objects and their respective labels
        lns = ln1 + ln3
        labs = [l.get_label() for l in lns]
        ax_pdf.legend(lns, labs, loc='center right', fontsize='small', frameon=True)
        
        ax_pdf.set_title("Sparsity Distribution")

        # --- ROW 2: CASE STUDIES ---
        cases = self._get_case_studies(res['kappa_map'], res['mean_kappa'])
        titles = ["High", "Low", "Average"]
        colors = ["#e31a1c", "#33a02c", "#1f78b4"]
        lab_a, lab_b = self._get_axis_labels(res['axis'])
        
        for i, (key, color) in enumerate(zip(["high", "low", "avg"], colors)):
            ax = fig.add_subplot(gs[1, i])
            coords = cases[key]
            
            orig_p = self._extract_profile(hsi, coords, res['axis'])
            rec_p = self._extract_profile(recovered, coords, res['axis'])
            
            ax.plot(orig_p, color=color, alpha=0.7, lw=2, label='Original')
            ax.plot(rec_p, color='black', ls='--', lw=1, label='Recovered')
            
            # Dynamic label based on axis
            coord_str = f"{lab_a}={coords[0]}, {lab_b}={coords[1]}"
            ax.set_title(rf"{titles[i]} Complexity Vector ({coord_str}, $\kappa$={coords[2]})")
            ax.set_xlabel(f"Index along {self.axis_map[res['axis']]} axis")
            ax.legend(fontsize='x-small')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save and show
        save_path = os.path.join(self.dir_path, f"{res['dataset']}_{res['basis']}_{axis_name}_T{res['T_value']}.png")
        plt.savefig(save_path, dpi=150)
        if self.verbose: plt.show()