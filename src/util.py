import numpy as np
import os
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import random

##### Bitstream Packing #####

def pack_to_bit_depth(data, bit_depth):
    """
    Packs a numpy array into a bitstream where each element 
    occupies exactly 'bit_depth' bits.
    """
    ba = bitarray()
    flat_data = data.flatten().tolist()
    
    for value in flat_data:
        # Convert integer to bitarray of length bit_depth
        ba.extend(int2ba(int(value), length=bit_depth, endian='big'))
    
    return ba.tobytes()

def unpack_from_bit_depth(byte_stream, bit_depth, shape):
    """
    Unpacks a byte-stream back into a numpy array of a specific shape.
    """
    ba = bitarray()
    ba.frombytes(byte_stream)
    
    total_elements = np.prod(shape)
    unpacked = np.zeros(total_elements, dtype=np.uint64)
    
    for i in range(total_elements):
        start = i * bit_depth
        end = start + bit_depth
        unpacked[i] = ba2int(ba[start:end])
        
    return unpacked.reshape(shape)

def add_bit_noise(data_bytes, ber, protected_mask=None):
    """
    Simulates a noisy channel where certain bits are protected from flipping.
    """
    if ber <= 0:
        return data_bytes
        
    bits = bitarray(endian='big')
    bits.frombytes(data_bytes)
    
    # Generate potential bit flips for the entire stream
    noise_mask = np.random.choice([True, False], size=len(bits), p=[ber, 1-ber])
    
    # If a protection mask is provided, zero out flips in protected zones
    if protected_mask is not None:
        # bitwise AND: only flip if noise is True AND protected is False
        # (Assuming protected_mask: 1 = protected, 0 = vulnerable)
        for i in range(len(bits)):
            if protected_mask[i]:
                noise_mask[i] = False
    
    for i in range(len(bits)):
        if noise_mask[i]:
            bits[i] = not bits[i]
            
    return bits.tobytes()

##### Progress Bar #####

def scaled_callback(base_callback, start, end):
    def wrapper(progress):
        base_callback(start + progress * (end - start))
    return wrapper

##### Registry #####

def make_registry():
    registry = {}

    def register(name):
        def decorator(func):
            key = name.upper()
            if key in registry:
                raise ValueError(f"{name} already registered")
            registry[key] = func
            return func
        return decorator

    return registry, register

##### Parsing #####

def _auto_cast(value):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value

def parse_config_string(name):
    """
    Parses strings like:
    'DCT'
    'LEARNED:path=abc.npz'
    'WAVELET:type=db4,level=3'
    """
    if ":" not in name:
        return name.upper(), {}
    
    base, param_str = name.split(":", 1)

    params = {}
    for item in param_str.split(","):
        key, value = item.split("=")
        params[key.strip()] = _auto_cast(value.strip())

    return base.upper(), params

def load_spectral_signature(file_path):
    """
    Opens a spectral library text file and returns the data as a NumPy array.
    
    Parameters
    ----------
    file_path : str
        Path to the .txt spectral signature file.
        
    Returns
    -------
    data : ndarray
        A 2D array where column 0 is wavelength and column 1 is reflectance.
    """
    try:
        # We use errors='ignore' to handle any non-standard characters in the header
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # ASTER library files contain 26 lines of metadata before the data
            data = np.loadtxt(f, skiprows=26)
        return data
    except Exception as e:
        print(f"Error loading spectral file {file_path}: {e}")
        return None

# In util.py
def build_diverse_spectral_library(folder_path, threshold=0.95, max_atoms=1000):
    """
    Creates a library of diverse spectral vectors using consistent normalization.
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    library = []
    
    random.shuffle(all_files)

    for file_name in all_files: 
        if len(library) >= max_atoms: break
            
        hsi = load_hsi(os.path.join(folder_path, file_name))
        
        Y = hsi.get_norm_data().reshape(-1, hsi.bands).T 
        N_pixels = Y.shape[1]
        
        sample_size = min(N_pixels, 2000) 
        pixel_indices = random.sample(range(N_pixels), sample_size)
        
        for idx in pixel_indices:
            if len(library) >= max_atoms: break
            current_pixel = Y[:, idx]
            
            if np.linalg.norm(current_pixel) < 1e-6: continue
            
            if len(library) == 0:
                library.append(current_pixel)
                continue
            
            # Vectorized correlation check using unit-norm versions
            lib_matrix = np.column_stack(library)
            curr_norm = current_pixel / np.linalg.norm(current_pixel)
            lib_norms = lib_matrix / np.linalg.norm(lib_matrix, axis=0)
            
            correlations = curr_norm @ lib_norms
            
            # If the pixel is unique enough, add it to the library
            if np.max(np.abs(correlations)) < threshold:
                library.append(current_pixel)

    return np.column_stack(library)