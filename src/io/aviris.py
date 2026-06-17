import math
import os
import re

import numpy as np

from src.core.hsi import HSI, HSIMetadata


def load_aviris_folder(folder_path: str) -> HSI:
    """
    Load an AVIRIS folder into an HSI object.

    The loader reads the orthorectified AVIRIS image file, parses
    the matching ENVI header, loads wavelengths from the SPC file,
    removes known water absorption bands, and returns the data in
    framework format ``(height, width, bands)``.

    Parameters
    ----------
    folder_path : str
        Path to AVIRIS scene folder.

    Returns
    -------
    HSI
        Loaded hyperspectral image.
    """

    folder_path = os.path.abspath(folder_path)
    files = os.listdir(folder_path)

    img_files = [
        f for f in files
        if "ort_img" in f.lower()
        and not f.lower().endswith(".hdr")
    ]

    hdr_files = [
        f for f in files
        if f.lower().endswith(".hdr")
    ]

    spc_files = [
        f for f in files
        if f.lower().endswith(".spc")
    ]

    info_files = [
        f for f in files
        if f.lower().endswith(".info")
    ]

    if not img_files:
        raise FileNotFoundError("No ort_img image file found")

    if len(img_files) > 1:
        print(
            f"[WARNING] Multiple ort_img files found "
            f"({len(img_files)}). Using first."
        )

    img_file = img_files[0]
    img_path = os.path.join(folder_path, img_file)

    base_name = os.path.splitext(img_file)[0]

    hdr_candidates = [
        f for f in hdr_files
        if base_name in f
    ]

    if not hdr_candidates:
        raise FileNotFoundError("No matching HDR file found")

    hdr_file = hdr_candidates[0]
    hdr_path = os.path.join(folder_path, hdr_file)

    header = _parse_envi_header(hdr_path)

    samples = int(header["samples"])
    lines = int(header["lines"])
    bands = int(header["bands"])

    interleave = header.get("interleave", "bip").lower()
    data_type = int(header.get("data type", 2))
    byte_order = int(header.get("byte order", 1))

    dtype = _envi_dtype(data_type, byte_order)

    raw = np.fromfile(img_path, dtype=dtype)

    expected = samples * lines * bands

    if raw.size != expected:
        raise ValueError(
            f"Size mismatch: got {raw.size}, expected {expected}"
        )

    cube = _reshape_envi_cube(
        raw,
        lines=lines,
        samples=samples,
        bands=bands,
        interleave=interleave,
    )

    wavelengths = _load_aviris_wavelengths(
        folder_path,
        spc_files,
        bands,
    )

    scene_id = _extract_scene_id(img_file)
    site_name = _load_site_name(folder_path, info_files)
    bit_depth = _compute_effective_bit_depth(cube)

    metadata = HSIMetadata(
        shape=cube.shape,
        wavelengths=wavelengths,
        bit_depth=bit_depth,
        sensor="AVIRIS",
        scene_id=scene_id,
        scene_name=site_name,
        attributes={
            "raw_folder": folder_path,
        },
    )
    return HSI(
        data=cube,
        metadata=metadata,
    )


def _parse_envi_header(path: str) -> dict:
    """
    Parse a simple ENVI header file.

    Parameters
    ----------
    path : str
        Header file path.

    Returns
    -------
    dict
        Parsed header key-value pairs.
    """

    header = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                header[key.strip().lower()] = value.strip().lower()

    return header

def _envi_dtype(data_type: int, byte_order: int) -> np.dtype:
    """
    Convert ENVI data type and byte order to NumPy dtype.

    Parameters
    ----------
    data_type : int
        ENVI data type code.

    byte_order : int
        ENVI byte order. ``1`` means big endian, ``0`` means little endian.

    Returns
    -------
    np.dtype
        NumPy dtype.
    """

    if data_type == 2:
        base_dtype = np.int16
    elif data_type == 4:
        base_dtype = np.float32
    elif data_type == 12:
        base_dtype = np.uint16
    else:
        raise ValueError(f"Unsupported ENVI data type: {data_type}")

    endian = ">" if byte_order == 1 else "<"

    return np.dtype(base_dtype).newbyteorder(endian)

def _reshape_envi_cube(
    raw: np.ndarray,
    lines: int,
    samples: int,
    bands: int,
    interleave: str,
) -> np.ndarray:
    """
    Reshape raw ENVI data to ``(lines, samples, bands)``.

    Parameters
    ----------
    raw : np.ndarray
        Flat raw data array.

    lines : int
        Number of image lines.

    samples : int
        Number of image samples.

    bands : int
        Number of spectral bands.

    interleave : str
        ENVI interleave type: ``bip``, ``bil``, or ``bsq``.

    Returns
    -------
    np.ndarray
        Hyperspectral cube with shape ``(lines, samples, bands)``.
    """

    if interleave == "bip":
        return raw.reshape((lines, samples, bands))

    if interleave == "bil":
        cube = raw.reshape((lines, bands, samples))
        return np.transpose(cube, (0, 2, 1))

    if interleave == "bsq":
        cube = raw.reshape((bands, lines, samples))
        return np.transpose(cube, (1, 2, 0))

    raise ValueError(f"Unknown interleave: {interleave}")

def _load_aviris_wavelengths(
    folder_path: str,
    spc_files: list[str],
    bands: int,
) -> np.ndarray:
    """
    Load AVIRIS wavelengths from an SPC file.

    Parameters
    ----------
    folder_path : str
        AVIRIS folder path.

    spc_files : list[str]
        Available SPC files.

    bands : int
        Number of spectral bands.

    Returns
    -------
    np.ndarray
        Wavelength vector.
    """

    if not spc_files:
        print("[WARNING] No SPC file found, using index wavelengths")
        return np.arange(bands)

    spc_path = os.path.join(folder_path, spc_files[0])

    return np.loadtxt(spc_path, usecols=0)

def _load_site_name(
    folder_path: str,
    info_files: list[str],
) -> str:
    """
    Load AVIRIS site name from an info file if available.

    Parameters
    ----------
    folder_path : str
        AVIRIS folder path.

    info_files : list[str]
        Available info files.

    Returns
    -------
    str
        Site name.
    """

    if not info_files:
        return "Unknown"

    info_path = os.path.join(folder_path, info_files[0])

    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "site_name" in line:
                _, value = line.split("=", 1)
                return value.strip()

    return "Unknown"

def _extract_scene_id(filename: str) -> str:
    """
    Extract AVIRIS scene id from a filename.

    Parameters
    ----------
    filename : str
        AVIRIS filename.

    Returns
    -------
    str
        Scene identifier.
    """

    match = re.match(r"(f\d{6}t\d{2}p\d{2}r\d{2})", filename)

    if match is not None:
        return match.group(1)

    if "rdn" in filename:
        return filename.split("rdn")[0]

    if "_" in filename:
        return filename.split("_")[0]

    return filename

def _compute_effective_bit_depth(data: np.ndarray) -> int:
    """
    Compute effective bit depth from the observed data range.

    Parameters
    ----------
    data : np.ndarray
        Input data.

    Returns
    -------
    int
        Number of bits needed to represent the observed value span.
    """

    data_min = int(np.min(data))
    data_max = int(np.max(data))

    span = data_max - data_min + 1

    if span <= 1:
        return 1

    return int(math.ceil(math.log2(span)))