from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.core.dictionary import Axis
from src.core.hsi import HSIMetadata
from src.core.training_signals import TrainingSignals
from src.io.common import make_npz_path, resolve_npz_path, list_npz_files


OBJECT_TYPE = "TRAINING_SIGNALS"


def save_training_signals(signals: TrainingSignals, directory: str | Path, name: str) -> None:
    """
    Save training signals to ``directory / f"{name}.npz"``.
    """
    path = make_npz_path(directory, name)

    payload = asdict(signals)
    payload["axis"] = signals.axis.value

    np.savez(
        path,
        object_type=OBJECT_TYPE,
        payload=payload,
    )


def load_training_signals(directory: str | Path, name: str) -> TrainingSignals:
    """
    Load a single training signals object from
    ``directory / f"{name}.npz"``.
    """
    path = resolve_npz_path(directory, name)

    return _load_single_training_signals(path)


def load_many_training_signals(directory: str | Path) -> list[TrainingSignals]:
    """
    Load all training signals objects from a directory.
    """
    signals_list = []

    for path in list_npz_files(directory):
        try:
            signals_list.append(_load_single_training_signals(path))
        except ValueError:
            continue

    return signals_list


def _load_single_training_signals(path: str | Path) -> TrainingSignals:
    """
    Load a single training signals object from an ``.npz`` file.
    """
    file = np.load(path, allow_pickle=True)

    object_type = file["object_type"].item()

    if object_type != OBJECT_TYPE:
        raise ValueError(
            f"Expected object_type {OBJECT_TYPE}, got {object_type}"
        )

    payload = file["payload"].item()

    payload["axis"] = Axis(payload["axis"])

    payload["sources"] = [
        HSIMetadata(**source)
        for source in payload["sources"]
    ]

def load_diverse_signals_library(folder_path: str, name: str, threshold: float = 0.95, max_atoms: int = 1000, seed: int | None = None) -> TrainingSignals:
    rng = np.random.default_rng(seed)
    path = Path(folder_path)
    
    # Locate all .npz patches
    all_files = list(path.rglob('*.npz'))
        
    rng.shuffle(all_files)
    
    library = []
    sources = []

    for file_path in all_files: 
        if len(library) >= max_atoms: break
            
        # Unpack the archive directly
        loaded = np.load(file_path, allow_pickle=True)
        data = loaded['data'].astype(np.float32)
        meta_dict = loaded['metadata'].item()
        
        # Instantiate HSIMetadata directly from the unpacked dictionary
        metadata = HSIMetadata(**meta_dict)
        sources.append(metadata)
        
        # Normalize data based on the extracted array
        if np.max(data) > 0:
            data /= np.max(data)  
            
        # Extract spectral signatures (using shape[2] for bands)
        Y = data.reshape(-1, data.shape[2]).T 
        N_pixels = Y.shape[1]
        
        sample_size = min(N_pixels, 2000) 
        pixel_indices = rng.choice(N_pixels, size=sample_size, replace=False)
        
        for idx in pixel_indices:
            if len(library) >= max_atoms: break
            current_pixel = Y[:, idx]
            
            if np.linalg.norm(current_pixel) < 1e-6: continue
            
            if len(library) == 0:
                library.append(current_pixel)
                continue
            
            # Vectorized correlation check
            lib_matrix = np.column_stack(library)
            curr_norm = current_pixel / np.linalg.norm(current_pixel)
            lib_norms = lib_matrix / np.linalg.norm(lib_matrix, axis=0)
            
            correlations = curr_norm @ lib_norms
            
            if np.max(np.abs(correlations)) < threshold:
                library.append(current_pixel)

    if not library:
        raise ValueError(f"No valid signals could be extracted from {folder_path}")

    return TrainingSignals(
        data=np.column_stack(library),
        axis=Axis.SPECTRAL,
        sources=sources,
        metadata={
            "name": name,
            "sampling": "diverse_correlation",
            "threshold": threshold,
            "seed": seed,
        }
    )