from src import io
from src.preprocessing.training_signals import sample_diverse_training_signals


def main():
    # ===== Paths =====
    split_csv = r"resources\splits\cuprite_split_01.csv"
    output_dir = r"data\training_signals"

    # ===== Config =====
    split = "train"
    name = "cuprite_split_01_diverse_spectral"

    threshold = 0.9995
    max_signals = 5000
    candidates_per_hsi = 2000
    seed = 42

    # ===== Load train HSIs =====
    hsis = io.load_hsi_split(
        split_csv=split_csv,
        split=split,
    )

    # ===== Create training signals =====
    signals = sample_diverse_training_signals(
        hsis=hsis,
        name=name,
        threshold=threshold,
        max_signals=max_signals,
        candidates_per_hsi=candidates_per_hsi,
        seed=seed,
    )

    # ===== Save =====
    io.save_training_signals(
        signals,
        output_dir,
        name,
    )

    print(f"Saved training signals: {output_dir}\\{name}.npz")
    print(f"Num signals: {signals.num_signals}")


if __name__ == "__main__":
    main()