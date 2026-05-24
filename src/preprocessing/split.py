from pathlib import Path
import random

import pandas as pd


def create_section_split_csv(
    output_path: str | Path,
    sections_dir: str | Path = r"data/sections",
    datasets: list[str] | None = None,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Create a train/test split CSV for HSI section files.

    Expected directory structure
    ----------------------------
    root_dir/
        DatasetA/
            *.npz
        DatasetB/
            *.npz

    Parameters
    ----------
    output_path : str | Path
        Output CSV path.

    section_dir : str | Path
        Directory containing dataset section folders.

    datasets : list[str] | None, optional
        Dataset folders to include. If None, all datasets are used.

    test_ratio : float, optional
        Fraction of sections assigned to the test split.

    seed : int, optional
        Random seed for reproducible splitting.
    """

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")

    sections_dir = Path(sections_dir)

    if not sections_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {sections_dir}")

    # ===== Collect section files =====

    if datasets is None:
        files = sorted(sections_dir.rglob("*.npz"))

    else:
        files = []

        for dataset in datasets:
            dataset_dir = sections_dir / dataset

            if not dataset_dir.exists():
                raise FileNotFoundError(
                    f"Dataset directory not found: {dataset_dir}"
                )

            files.extend(sorted(dataset_dir.rglob("*.npz")))

    if len(files) == 0:
        raise ValueError("No section files found")

    # ===== Random split =====

    rng = random.Random(seed)
    rng.shuffle(files)

    num_test = int(len(files) * test_ratio)

    test_files = set(files[:num_test])

    # ===== Build dataframe =====

    rows = []

    for path in files:

        split = "test" if path in test_files else "train"

        rows.append(
            {
                "section_path": str(path),
                "section_name": path.stem,
                "dataset": _infer_dataset_name(path),
                "section_row": _infer_section_index(path.stem, "r"),
                "section_col": _infer_section_index(path.stem, "c"),
                "split": split,
            }
        )

    df = pd.DataFrame(rows)

    # ===== Save =====

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


def _infer_dataset_name(path: Path) -> str:
    """
    Infer dataset name from a section path.
    """

    return path.parent.name


def _infer_section_index(
    stem: str,
    prefix: str,
) -> int | None:
    """
    Infer section row/column index from names like:

    Scene_r1_c2
    """

    parts = stem.split("_")

    for part in parts:

        if part.startswith(prefix):

            value = part[1:]

            if value.isdigit():
                return int(value)

    return None