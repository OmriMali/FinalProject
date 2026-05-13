import numpy as np

from src.core.hsi import HSI, HSIMetadata


def load_hsi(path: str) -> HSI:
    """
    Load a hyperspectral image from an ``.npz`` file.
    """
    file = np.load(path, allow_pickle=True)

    metadata = HSIMetadata(
        shape=tuple(file["shape"]),
        wavelengths=file["wavelengths"],
        bit_depth=int(file["bit_depth"]),
        sensor=file["sensor"].item(),
        scene_id=file["scene_id"].item(),
        scene_name=file["scene_name"].item(),
        section_row=file["section_row"].item(),
        section_col=file["section_col"].item(),
        attributes=file["attributes"].item(),
    )
    return HSI(
        data=file["data"],
        metadata=metadata,
    )


def save_hsi(hsi: HSI, path: str) -> None:
    """
    Save a hyperspectral image to an ``.npz`` file.
    """
    np.savez(
        path,
        data=hsi.data,
        shape=np.array(hsi.metadata.shape),
        wavelengths=hsi.metadata.wavelengths,
        bit_depth=hsi.metadata.bit_depth,
        sensor=hsi.metadata.sensor,
        scene_id=hsi.metadata.scene_id,
        scene_name=hsi.metadata.scene_name,
        section_row=hsi.metadata.section_row,
        section_col=hsi.metadata.section_col,
        attributes=hsi.metadata.attributes,
    )