import os
import pandas as pd
import matplotlib.pyplot as plt

# Smart imports from your existing architecture
from src import io, data_processing
from src.visuals import hsi as hsi_visuals

def visualize_hsi_comparison(orig_hsi, ccsds_hsi, hcs1d_hsi, hybrid_hsi, metrics_dict=None):
    """
    Visualizes 4 HSI images (RGB) and their spectral plot at the center pixel.
    """
    hsis = [orig_hsi, ccsds_hsi, hcs1d_hsi, hybrid_hsi]
    
    # Internal labels match your log/style keys so compare_spectra styles them natively
    internal_labels = ["Original", "ccsds123", "hcs1d", "hybrid"]
    
    # Display names are strictly for clean UI text on the plots and legends
    display_names = ["Original", "CCSDS-123", "HCS1D", "Hybrid"]
    visual_labels = ["[a]", "[b]", "[c]", "[d]"]

    # 1. Plot RGB Comparison
    fig_rgb, axes_rgb = hsi_visuals.compare_rgb(
        hsis=hsis,
        labels=internal_labels,
        stretch=True,
        title="RGB Reconstruction Comparison"
    )

    # Apply structured labeling [a], [b], [c] to the subplots
    for i, ax in enumerate(axes_rgb.flat):
        if i < len(display_names):
            ax.set_title(f"{visual_labels[i]} {display_names[i]}", fontsize=12, pad=10)

    # Build and append the explanatory metrics caption at the bottom
    if metrics_dict:
        caption = "Performance Metrics:\n"
        for method, metrics in metrics_dict.items():
            caption += f"[{method}] CR: {metrics['CR']:.2f} | RMSE: {metrics['RMSE']:.2f}    "
        
        fig_rgb.text(0.5, 0.03, caption.strip(), ha='center', va='bottom', fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
        fig_rgb.subplots_adjust(bottom=0.18)

    # 2. Plot Spectral Comparison (Middle Pixel)
    height, width = orig_hsi.spatial_shape
    middle_pixel = (width // 2, height // 2)

    fig_spec, ax_spec = hsi_visuals.compare_spectra(
        hsis=hsis,
        labels=internal_labels, 
        pixel=middle_pixel,
        title=f"Spectrum Comparison at Pixel (x={middle_pixel[0]}, y={middle_pixel[1]})"
    )

    # Post-process the curves: Force the "Original" line to be black and in front
    lines = ax_spec.get_lines()
    for line, label in zip(lines, internal_labels):
        if label == "Original":
            line.set_color("black")
            line.set_zorder(10)

    # Refresh the legend with the clean UI names and updated colors
    ax_spec.legend(lines, display_names)

    plt.show()

def run_mode_csv(experiment: str, scene_name: str, section_row: int, section_col: int, target_cr: float):
    print(f"Running in CSV mode (Experiment: {experiment}, Scene: {scene_name}, Patch: r{section_row}_c{section_col}, Target CR: {target_cr})...")

    hybrid_path = r"results\compression\hybrid\log.csv"
    hcs1d_path = r"results\compression\hcs1d\log.csv"
    ccsds_path = r"results\compression\ccsds123\log.csv"

    df = data_processing.load_logs([hybrid_path, hcs1d_path, ccsds_path])

    # --- NEW: Handle Wildcard Experiment Matching ---
    if "*" in experiment:
        # Convert wildcard to a regex pattern (e.g., "comp*" becomes "^comp.*$")
        regex_pattern = "^" + experiment.replace("*", ".*") + "$"
        exp_filter = df["experiment"].str.match(regex_pattern, na=False)
    else:
        # Exact match
        exp_filter = (df["experiment"] == experiment)

    # Filter strictly by the experiment (or wildcard) AND the HSI image parameters
    df = df[
        exp_filter &
        (df["scene_name"] == scene_name) & 
        (df["section_row"] == section_row) & 
        (df["section_col"] == section_col)
    ]

    if "artifact_dir" in df.columns:
        df = df.dropna(subset=["artifact_dir"])
        df = df[df["artifact_dir"].str.strip() != ""]
    else:
        print("Error: 'artifact_dir' column not found in logs.")
        return

    if df.empty:
        print("Error: No saved artifacts found for this specific spatial patch in the specified experiment(s).")
        return

    def get_closest_run(method_name):
        method_df = df[df["method"] == method_name].copy()
        if method_df.empty:
            raise ValueError(f"No logged data found for method: {method_name}")
        
        method_df["cr_diff"] = (method_df["cr"] - target_cr).abs()
        
        # Find the absolute minimum difference
        min_diff = method_df["cr_diff"].min()
        
        # Filter the DataFrame to only the rows that match the minimum difference
        closest_runs = method_df[method_df["cr_diff"] == min_diff]
        
        # Return the last one in the subset (most recent from the bottom of the CSV)
        return closest_runs.iloc[-1]

    try:
        ccsds_row = get_closest_run("ccsds123")
        hcs1d_row = get_closest_run("hcs1d")
        hybrid_row = get_closest_run("hybrid")
    except ValueError as e:
        print(e)
        return

    print(f"Found closest matches:\n"
          f"  CCSDS-123 (Exp: {ccsds_row['experiment']}) CR: {ccsds_row['cr']:.2f}\n"
          f"  HCS1D     (Exp: {hcs1d_row['experiment']}) CR: {hcs1d_row['cr']:.2f}\n"
          f"  Hybrid    (Exp: {hybrid_row['experiment']}) CR: {hybrid_row['cr']:.2f}")

    metrics_dict = {
        "CCSDS-123": {"CR": ccsds_row["cr"], "RMSE": ccsds_row["rmse"]},
        "HCS1D": {"CR": hcs1d_row["cr"], "RMSE": hcs1d_row["rmse"]},
        "Hybrid": {"CR": hybrid_row["cr"], "RMSE": hybrid_row["rmse"]}
    }

    try:
        ccsds_hsi = io.load_hsi(ccsds_row["artifact_dir"], "reconstructed")
        hcs1d_hsi = io.load_hsi(hcs1d_row["artifact_dir"], "reconstructed")
        hybrid_hsi = io.load_hsi(hybrid_row["artifact_dir"], "reconstructed")
        
        clean_scene_name = scene_name.replace(" ", "")
        orig_dir = fr"data\sections\{clean_scene_name}"
        orig_name = f"{clean_scene_name}_r{section_row}_c{section_col}"
        orig_hsi = io.load_hsi(orig_dir, orig_name)

    except Exception as e:
        print(f"Error loading HSI file: {e}")
        return

    visualize_hsi_comparison(orig_hsi, ccsds_hsi, hcs1d_hsi, hybrid_hsi, metrics_dict)

def main():
    run_mode_csv(
        experiment="*visual_test",  # Added wildcard here as an example
        scene_name="Jusper Ridge", 
        section_row=6, 
        section_col=2, 
        target_cr=15.0
    )

if __name__ == "__main__":
    main()