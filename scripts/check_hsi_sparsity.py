import numpy as np
import matplotlib.pyplot as plt

from src import io
from src.math import regression_algs
from src.transforms.sparse_bases import get_sparse_base


def main():

    # ===== HSI =====
    hsi_dir = r"data\sections\JasperRidge"
    hsi_name = "JasperRidge_r11_c1"

    # ===== Sparse bases =====
    analytic_basis_specs = [
        ("DCT", "DCT"),
    ]

    dict_dirs = [
        r"resources\dictionaries",
        r"resources\dictionaries",
        r"resources\dictionaries",
        r"resources\dictionaries",
        r"resources\dictionaries"
    ]
    dict_names = [
        r"jasper_ridge_split_01_K_1.npz",
        r"jasper_ridge_split_01_K_2.npz",
        r"jasper_ridge_split_01_K_3.npz",
        r"jasper_ridge_split_01_K_4.npz",
        r"jasper_ridge_split_01_K_5.npz",
    ]
    dict_labels = [
        "Learned K=1",
        "Learned K=2",
        "Learned K=3",
        "Learned K=4",
        "Learned K=5",
    ]

    basis_specs = analytic_basis_specs + _learned_basis_specs(
        dict_dirs,
        dict_names,
        dict_labels,
    )

    # ===== Analysis settings =====
    sample_signals = 2000
    random_seed = 0

    remove_signal_mean = False
    normalize_signal_norm = False

    cumulative_max_k = 30
    omp_tol = 1e-6

    # ===== Load and prepare =====
    hsi = io.load_hsi(hsi_dir, hsi_name)
    signals = _sample_spectral_signals(
        hsi.data,
        sample_signals=sample_signals,
        random_seed=random_seed,
    )
    signals = _preprocess_signals(
        signals,
        remove_signal_mean=remove_signal_mean,
        normalize_signal_norm=normalize_signal_norm,
    )

    results = []

    for label, basis_name in basis_specs:
        result = _analyze_basis(
            signals=signals,
            basis_name=basis_name,
            label=label,
            cumulative_max_k=cumulative_max_k,
            omp_tol=omp_tol,
        )
        results.append(result)

    _print_summary(
        hsi_shape=hsi.shape,
        hsi_dtype=hsi.data.dtype,
        signal_count=signals.shape[0],
        total_signal_count=hsi.data.shape[0] * hsi.data.shape[1],
        remove_signal_mean=remove_signal_mean,
        normalize_signal_norm=normalize_signal_norm,
        results=results,
    )

    _plot_energy(results)
    plt.show()


def _sample_spectral_signals(
    cube: np.ndarray,
    sample_signals: int | None,
    random_seed: int,
) -> np.ndarray:
    signals = cube.reshape(-1, cube.shape[2]).astype(float)

    if sample_signals is None:
        return signals

    if sample_signals >= len(signals):
        return signals

    rng = np.random.default_rng(random_seed)
    indices = rng.choice(len(signals), size=sample_signals, replace=False)

    return signals[indices]


def _preprocess_signals(
    signals: np.ndarray,
    remove_signal_mean: bool,
    normalize_signal_norm: bool,
) -> np.ndarray:
    signals = signals.copy()

    if remove_signal_mean:
        signals -= signals.mean(axis=1, keepdims=True)

    if normalize_signal_norm:
        norms = np.linalg.norm(signals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        signals /= norms

    return signals


def _learned_basis_specs(
    dict_dirs: list[str],
    dict_names: list[str],
    dict_labels: list[str],
) -> list[tuple[str, str]]:
    if not (len(dict_dirs) == len(dict_names) == len(dict_labels)):
        raise ValueError(
            "dict_dirs, dict_names, and dict_labels must have the same length"
        )

    return [
        (
            label,
            f"LEARNED:directory={directory},name={name}",
        )
        for directory, name, label in zip(dict_dirs, dict_names, dict_labels)
    ]


def _analyze_basis(
    signals: np.ndarray,
    basis_name: str,
    label: str,
    cumulative_max_k: int,
    omp_tol: float,
) -> dict:
    signal_length = signals.shape[1]
    basis = get_sparse_base(basis_name, signal_length)
    basis = _normalize_columns(basis)

    max_k = min(cumulative_max_k, basis.shape[1])
    coefficient_method = _coefficient_method(basis)

    print(f"Analyzing basis: {label}")

    if coefficient_method in {"orthonormal", "square"}:
        coefficients = _direct_coefficients(signals, basis, coefficient_method)
    else:
        coefficients = _omp_coefficients(
            signals,
            basis,
            max_k=max_k,
            tol=omp_tol,
            label=label,
        )

    cumulative_energy = _cumulative_energy(
        coefficients,
        max_k=min(cumulative_max_k, basis.shape[1]),
    )

    return {
        "label": label,
        "basis_shape": basis.shape,
        "coefficient_method": coefficient_method,
        "cumulative_energy": cumulative_energy,
    }


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms[norms == 0] = 1.0

    return matrix / norms


def _coefficient_method(basis: np.ndarray) -> str:
    rows, cols = basis.shape

    if rows == cols:
        gram = basis.T @ basis
        error = np.max(np.abs(gram - np.eye(cols)))

        if error < 1e-8:
            return "orthonormal"

        return "square"

    return "omp"


def _direct_coefficients(
    signals: np.ndarray,
    basis: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "orthonormal":
        return signals @ basis

    return np.linalg.solve(basis, signals.T).T


def _omp_coefficients(
    signals: np.ndarray,
    basis: np.ndarray,
    max_k: int,
    tol: float,
    label: str,
) -> np.ndarray:
    coefficients = np.zeros((signals.shape[0], basis.shape[1]))

    for index, signal in enumerate(signals):
        coefficients[index] = regression_algs.omp(
            basis,
            signal,
            K=max_k,
            tol=tol,
        )

        if (index + 1) % 500 == 0:
            print(f"{label} OMP progress: {index + 1}/{signals.shape[0]} spectra")

    return coefficients


def _cumulative_energy(
    coefficients: np.ndarray,
    max_k: int,
) -> dict[str, np.ndarray]:
    energy = np.sort(np.abs(coefficients) ** 2, axis=1)[:, ::-1]
    total_energy = energy.sum(axis=1, keepdims=True)
    total_energy[total_energy == 0] = 1.0

    cumulative = np.cumsum(energy[:, :max_k], axis=1) / total_energy

    return {
        "k": np.arange(1, max_k + 1),
        "mean": cumulative.mean(axis=0),
        "per_signal": cumulative,
    }


def _print_summary(
    hsi_shape: tuple[int, int, int],
    hsi_dtype,
    signal_count: int,
    total_signal_count: int,
    remove_signal_mean: bool,
    normalize_signal_norm: bool,
    results: list[dict],
) -> None:
    print()
    print("HSI sparsity check")
    print("==================")
    print(f"HSI shape:              {hsi_shape}")
    print(f"HSI dtype:              {hsi_dtype}")
    print(f"Analyzed spectra:       {signal_count} / {total_signal_count}")
    print(f"Bases compared:         {', '.join(result['label'] for result in results)}")
    print(f"Remove signal mean:     {remove_signal_mean}")
    print(f"Normalize signal norm:  {normalize_signal_norm}")
    print()

    _print_table("Basis details", _basis_summary_rows(results))
    _print_table("Atoms needed for coefficient energy", _energy_summary_rows(results))

    print()


def _basis_summary_rows(results: list[dict]) -> list[list[str]]:
    rows = [
        [
            "Basis",
            "Shape",
            "Method",
        ]
    ]

    for result in results:
        rows.append(
            [
                result["label"],
                _format_shape(result["basis_shape"]),
                result["coefficient_method"],
            ]
        )

    return rows


def _energy_summary_rows(results: list[dict]) -> list[list[str]]:
    targets = [0.90, 0.95, 0.99]
    rows = [["Basis"]]

    for target in targets:
        rows[0].extend(
            [
                f"{target:.0%} mean",
                f"{target:.0%} median",
                f"{target:.0%} p90",
            ]
        )

    for result in results:
        row = [result["label"]]
        cumulative_energy = result["cumulative_energy"]["per_signal"]

        for target in targets:
            needed = _atoms_needed_for_energy(cumulative_energy, target)
            row.extend(
                [
                    f"{needed.mean():.2f}",
                    f"{np.median(needed):.2f}",
                    f"{np.percentile(needed, 90):.2f}",
                ]
            )

        rows.append(row)

    return rows


def _print_table(title: str, rows: list[list[str]]) -> None:
    if not rows:
        return

    widths = [
        max(len(str(row[index])) for row in rows)
        for index in range(len(rows[0]))
    ]

    print(title)
    print("-" * len(title))

    for row_index, row in enumerate(rows):
        line = "  ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(row)
        )
        print(line)

        if row_index == 0:
            print(
                "  ".join("-" * width for width in widths)
            )

    print()


def _format_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in shape)


def _atoms_needed_for_energy(
    cumulative_energy: np.ndarray,
    target: float,
) -> np.ndarray:
    reached = cumulative_energy >= target
    indices = np.argmax(reached, axis=1) + 1
    indices[~reached.any(axis=1)] = cumulative_energy.shape[1]

    return indices


def _plot_energy(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for result in results:
        label = result["label"]
        cumulative_energy = result["cumulative_energy"]

        k = cumulative_energy["k"]
        ax.plot(k, 100 * cumulative_energy["mean"], label=label)

    ax.set_title("Cumulative Coefficient Energy")
    ax.set_xlabel("Largest coefficients kept")
    ax.set_ylabel("Mean cumulative energy (%)")
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


if __name__ == "__main__":
    main()
