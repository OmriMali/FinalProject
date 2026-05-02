import numpy as np
import scipy as sp
import os

from src.core.hsi import HSI


def load_array(path):
    """
    Load a NumPy array and optional metadata saved with `save_array_to_path`.

    Parameters
    ----------
    path : str
        Path to load.

    Returns
    -------
    arr : ndarray
        The main array.
    metadata : dict
        Metadata dictionary (empty if none saved).
    """
    with np.load(path, allow_pickle=True) as data:
        # Main array
        arr = data["array"]
        
        # Extract metadata
        metadata = {}
        for key in data.files:
            if key.startswith("meta_"):
                metadata[key[5:]] = data[key].item() if data[key].shape == () else data[key]
    
    return arr, metadata


def load_hsi(path: str) -> HSI:
    """
    Load an HSI object previously saved with `save_hsi`.
    """
    obj = np.load(path, allow_pickle=True).item()
    return HSI(
        data=obj["data"],
        wavelengths=obj["wavelengths"],
        dtype=obj["dtype"],
        metadata=obj["metadata"]
    )

def load_hsi_from_mat(mat_path: str, name: str, site: str, sensor: str) -> HSI:
    """
    Load an HSI object from a .mat file.
    """

    mat = sp.io.loadmat(mat_path)

    # Find the actual data key
    key = [k for k in mat.keys() if not k.startswith("__")][0]
    cube = mat[key]

    # ===== Wavelengths ===== #
    bands = cube.shape[2]

    # Option 1: simple approximation
    wavelengths = np.linspace(400, 2500, bands)

    # ===== Metadata ===== #
    metadata = {
        "name": name,
        "site": site,
        "sensor": sensor,
    }

    return HSI(
        data=cube,
        wavelengths=wavelengths,
        dtype=cube.dtype,
        metadata=metadata
    )


def load_aviris(folder_path: str) -> HSI:
    """
    Load a raw AVIRIS dataset as an HSI object.
    """
    folder_path = os.path.abspath(folder_path)
    files = os.listdir(folder_path)

    # ===== Detect ort_img files ===== #
    img_files = [f for f in files if "ort_img" in f.lower() and not f.lower().endswith(".hdr")]
    hdr_files = [f for f in files if f.lower().endswith(".hdr")]
    spc_files = [f for f in files if f.lower().endswith(".spc")]
    info_files = [f for f in files if f.lower().endswith(".info")]

    if not img_files:
        raise FileNotFoundError("No ort_img image file found")

    if len(img_files) > 1:
        print(f"[WARNING] Multiple ort_img files found ({len(img_files)}). Using first.")

    img_file = img_files[0]
    img_path = os.path.join(folder_path, img_file)

    # ===== Find matching HDR ===== #
    base_name = os.path.splitext(img_file)[0]
    hdr_candidates = [f for f in hdr_files if base_name in f]
    if not hdr_candidates:
        raise FileNotFoundError("No matching HDR file found")
    hdr_file = hdr_candidates[0]
    hdr_path = os.path.join(folder_path, hdr_file)

    # ===== Parse HDR ===== #
    header = {}
    with open(hdr_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.split("=", 1)
                header[key.strip().lower()] = val.strip().lower()

    samples = int(header["samples"])
    lines = int(header["lines"])
    bands = int(header["bands"])
    interleave = header.get("interleave", "bip")
    data_type = int(header.get("data type", 2))
    byte_order = int(header.get("byte order", 1))

    # ===== Map dtype ===== #
    if data_type == 2:
        base_dtype = np.int16
    elif data_type == 4:
        base_dtype = np.float32
    else:
        raise ValueError(f"Unsupported ENVI data type: {data_type}")

    dtype = np.dtype(base_dtype).newbyteorder(">" if byte_order == 1 else "<")

    # ===== Load raw data ===== #
    cube = np.fromfile(img_path, dtype=dtype)
    expected = samples * lines * bands
    if cube.size != expected:
        raise ValueError(f"Size mismatch: got {cube.size}, expected {expected}")

    # ===== Reshape based on interleave ===== #
    if interleave == "bip":
        cube = cube.reshape((lines, samples, bands))
    elif interleave == "bil":
        cube = cube.reshape((lines, bands, samples))
        cube = np.transpose(cube, (0, 2, 1))
    elif interleave == "bsq":
        cube = cube.reshape((bands, lines, samples))
        cube = np.transpose(cube, (1, 2, 0))
    else:
        raise ValueError(f"Unknown interleave: {interleave}")

    # ===== Load wavelengths ===== #
    if spc_files:
        spc_path = os.path.join(folder_path, spc_files[0])
        wavelengths = np.loadtxt(spc_path, usecols=0)
    else:
        print("[WARNING] No SPC file found, using index wavelengths")
        wavelengths = np.arange(bands)

    # ===== Remove water absorption bands ===== #
    mask = np.ones(len(wavelengths), dtype=bool)
    bad_bands = [(104, 108), (150, 163)]

    # Update the mask to False for those specific indices
    for start, end in bad_bands:
        # end+1 because Python slicing is exclusive at the stop index
        mask[start:end+1] = False 

    # Apply the mask to the cube and wavelengths
    cube = cube[:, :, mask]
    wavelengths = wavelengths[mask]
    
    # ===== Parse .info file for site ===== #
    site_name = "Unknown"
    if info_files:
        info_path = os.path.join(folder_path, info_files[0])
        with open(info_path, "r") as f:
            for line in f:
                if "site_name" in line:
                    _, val = line.split("=", 1)
                    site_name = val.strip()

    # ===== Extract dataset name from folder ===== #
    img_base = os.path.basename(img_path)
    if "rdn" in img_base:
        name = img_base.split("rdn")[0]
    elif "_" in img_base:
        name = img_base.split("_")[0]
    else:
        name = img_base

    # ===== Create HSI object ===== #
    hsi = HSI(
        data=cube,
        wavelengths=wavelengths,
        dtype=base_dtype,
        metadata={
            "name": name,
            "site": site_name,
            "sensor": "AVIRIS"
        }
    )

    return hsi



