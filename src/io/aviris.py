import numpy as np
import os
import tarfile
import math
import random

from src.core.hsi import HSI
from src.io.savers import save_hsi


def extract_tar_gz(path, out_dir):
    """
    Extract a .tar.gz dataset from path to out_dir.
    """
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(out_dir)

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


def crop_aviris_scene(folder_path: str, section_size=(256, 256), train_ratio=0.8, seed=42):
    """
    Crop a full AVIRIS HSI scene into smaller sections and save them.
    Splits into train and test subfolders and filters black patches.
    """
    # ===== Load full scene ===== #
    hsi = load_aviris(folder_path)
    scene_name = hsi.metadata["name"]

    # ===== Prepare output folder ===== #
    base_folder = os.path.join(folder_path, "sections")
    # Added train/test subfolders
    train_folder = os.path.join(base_folder, "train")
    test_folder = os.path.join(base_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    H, W = hsi.height, hsi.width
    sh, sw = section_size

    # ===== Compute number of sections ===== #
    n_vert = math.ceil(H / sh)
    n_horiz = math.ceil(W / sw)

    coords = []
    for i in range(n_vert):
        y0 = i * sh
        y1 = min(y0 + sh, H)
        for j in range(n_horiz):
            x0 = j * sw
            x1 = min(x0 + sw, W)
            coords.append((y0, y1, x0, x1))

    # ===== Shuffle and Filter ===== #
    random.seed(seed)
    random.shuffle(coords) # Shuffling for randomized train/test split

    print(f"Cropping HSI ({H}x{W}) into sections of approx {sh}x{sw}...")

    # ===== Crop, update metadata, save ===== #
    # Filter black patches and split during save loop
    valid_count = 0
    for y0, y1, x0, x1 in coords:
        section_hsi = hsi.crop((y0, y1), (x0, x1))
        
        # Filter: Skip sections with zero dynamic range (black patches)
        if np.max(section_hsi.data) <= np.min(section_hsi.data):
            continue
            
        valid_count += 1
        is_train = valid_count <= (len(coords) * train_ratio)
        target_dir = train_folder if is_train else test_folder
        label = "train" if is_train else "test"

        # Update metadata with section number
        section_name = f"{scene_name}_{label}_s{valid_count}" 
        metadata = section_hsi.metadata.copy()
        metadata["name"] = section_name

        section_hsi = HSI(
            data=section_hsi.data,
            wavelengths=section_hsi.wavelengths,
            dtype=section_hsi.dtype,
            metadata=metadata
        )

        save_path = os.path.join(target_dir, f"{section_name}")
        save_hsi(section_hsi, save_path)

    print(f"Saved sections in '{base_folder}' (Train/Test split applied)")

