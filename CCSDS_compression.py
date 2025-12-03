### version: 0.3.0
### date: 03/12/25
### author: almog the king

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import metrics

def calc_local_sum(mode, idx, image_slice, Nx = None):
    """
    Calculate a local weighted sum depending on neighborhood mode.

    Parameters
    ----------
    mode : str
        'col', 'narrow', or 'wide'
    idx : tuple
        (x, y) pixel location
    image_slice : np.ndarray
        2D array of image values
    Nx : int, optional
        Width of the image (only needed for modes using horizontal neighbors)
    """
    x, y = idx
    
    # -------- FIRST ROW --------
    if y == 0:
        return 0 if x == 0 else 4 * image_slice[x-1, 0]

    # -------- COL MODE --------
    if mode == 'col':
        return 4 * image_slice[x, y-1]

    # -------- NARROW MODE --------
    if mode == 'narrow':
        if x == Nx - 1:
            return image_slice[x-1, y-1] + 3 * image_slice[x, y-1]
        else:
            return (image_slice[x-1, y-1] +
                    2 * image_slice[x, y-1] +
                    image_slice[x+1, y-1])

    # -------- WIDE MODE --------
    # same logic as yours, just grouped cleanly
    if x == Nx - 1:  # right border
        return (image_slice[x-1, y-1] +
                2 * image_slice[x, y-1] +
                image_slice[x-1, y])
    if x == 0:       # left border
        return (3 * image_slice[x, y-1] +
                image_slice[x+1, y-1])

    # middle
    return (image_slice[x-1, y-1] +
            image_slice[x, y-1] +
            image_slice[x+1, y-1] +
            image_slice[x-1, y])

def load_image(path):
    mat = sp.io.loadmat(path)
    mat_clean = {k: v for k, v in mat.items() if not k.startswith('__')}
    # If there is only one remaining field, extract it
    if len(mat_clean) == 1:
        data_array = next(iter(mat_clean.values()))
    else:
        # If multiple fields, select the one with the largest array (often the data)
        data_array = max(mat_clean.values(), key=lambda x: getattr(x, 'size', 0))
    return data_array
    
def predictor(image, local_sum_mode, P, W, Omega):
    
    Nx, Ny, Nz = image.shape
    sigma = np.zeros((Nx, Ny, Nz), dtype=np.int32)
    d = np.zeros((Nx, Ny, Nz), dtype=np.int32)
    image_hat =  np.zeros((Nx, Ny, Nz), dtype=np.int32)

    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                
                sigma[x,y,z] = calc_local_sum(local_sum_mode, (x, y), image[:, :, z], Nx)

                d[x, y, z] = 4 * image[x, y ,z] - sigma[x, y, z]
                
                U = np.zeros(P)
                for i in range(P):
                    if z-i-1 >= 0:
                        U[i] = d[x,y,z-i-1]           
                
                d_hat = np.dot(W, U)

                image_hat[x,y,z] = int((d_hat + 2**Omega * sigma[x,y,z]) / (2**(Omega+2)))

    return image_hat

def generate_positive_diff(image, image_hat, Q):
    
    Nx, Ny, Nz = image.shape 
    delta = np.zeros(image.shape, dtype=np.int32)
    delta_positive = np.zeros(image.shape, dtype=np.uint32)

    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                delta[x,y,z] = image[x,y,z] - image_hat[x,y,z]
                delta[x,y,z] = int(delta[x,y,z] / 2**Q)
                if delta[x,y,z] > 0:
                    delta_positive[x,y,z] = 2 * delta[x,y,z] - 1
                else:
                    delta_positive[x,y,z] = -2 * delta[x,y,z]
    
    return delta_positive

def unpack_positive_diff(delta_positive, Q):
    Nx, Ny, Nz = image.shape
    delta_r = np.zeros(image.shape, dtype=np.int32)
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                if delta_positive[x,y,z]%2 == 0:
                    delta_r[x,y,z] = - (delta_positive[x,y,z] / 2)
                else:
                    delta_r[x,y,z] = (delta_positive[x,y,z] + 1)/ 2

                delta_r[x,y,z] = delta_r[x,y,z] * 2**Q
    return delta_r

def reconstructor(delta_r, local_sum_mode, P, W, Omega):
    
    Nx, Ny, Nz = delta_r.shape
    sigma_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
    d_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
    image_hat_r =  np.zeros((Nx, Ny, Nz), dtype=np.int32)
    image_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)

    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):

                sigma_r[x,y,z] = calc_local_sum(local_sum_mode, (x, y), image_r[:, :, z], Nx)
                
                U = np.zeros(P)
                for i in range(P):
                    if z-i-1 >= 0:
                        U[i] = d_r[x,y,z-i-1]           
                
                d_hat = np.dot(W, U)
                image_hat_r[x,y,z] = int((d_hat + 2**Omega * sigma_r[x,y,z]) / (2**(Omega+2)))

                image_r[x,y,z] = image_hat_r[x,y,z] + delta_r[x,y,z]
                d_r[x, y, z] = 4 * image_r[x, y ,z] - sigma_r[x, y, z]

    return image_r

def rice_encoder(data, block_size):
    
    data = data.flatten()
    blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
    k = []
    bitstream = []

    for i in range(0, len(blocks)):
        k_i = int(np.floor(np.log2(1 + blocks[i].mean())))
        k.append(k_i)
        
        for j in range(i*block_size, i*block_size + len(blocks[i])):
            q_j = data[j] >> k_i
            r_j = data[j] & ((1 << k_i) - 1)

            # unary bits
            bitstream.extend([1] * q_j)
            bitstream.append(0)

            # remainder bits
            if k_i > 0:
                bits = [(r_j >> b) & 1 for b in reversed(range(k_i))]
                bitstream.extend(bits)


    # Convert bitstream to NumPy array
    bitstream = np.array(bitstream, dtype=np.uint8)

    return k, bitstream

def rice_decoder(bitstream, k, block_size, shape):
    
    total_len = np.prod(shape)
    data = np.zeros(total_len, dtype=np.uint16)

    # Compute block lengths
    num_blocks = len(k)
    block_lengths = [block_size] * num_blocks

    # adjust last block if needed
    last_block_len = total_len - block_size*(num_blocks-1)
    block_lengths[-1] = last_block_len

    bit_idx = 0
    out_idx = 0
    
    for i in range(num_blocks):
        k_i = k[i]
        block_len = block_lengths[i]
        
        for _ in range(block_len):
            # ---- Decode unary part ----
            q_val = 0
            while bitstream[bit_idx]:
                q_val += 1
                bit_idx += 1
            bit_idx += 1  # skip the terminating 0

            # ---- Decode remainder ----
            r_val = 0
            if k_i > 0:
                for b in range(k_i):
                    r_val = (r_val << 1) | int(bitstream[bit_idx])
                    bit_idx += 1

            # ---- Reconstruct value ----
            data[out_idx] = (q_val << k_i) + r_val
            out_idx += 1

    return data.reshape(shape)

def CCSDS(image, local_sum_mode, P, W, Omega, Q, block_size):
    """
    Parameters
    ----------
    image : ndarray
        3D Input image array of dtype uint16 or compatible integer type.
    local_sum_mode : str
        Mode for local sum calculation in the predictor (e.g., 'col', 'narrow', 'wide').
    P : int
        Predictor parameter defining the number of considered spectral bands.
    W : ndarray
        1D Weighting array (dim = P) used in the predictor.
    Omega : int
        Predictor parameter balancing spectral-spatial prediction.
    Q : int
        Quantization parameter for the positive difference calculation.
    block_size : int
        Number of elements per block for Rice block-adaptive encoding.

    Returns
    -------
    image_r : ndarray
        Reconstructed image after compression and decompression. Should be nearly identical
        to the input `image` if lossless parameters are used.
    bitstream : ndarray of uint8
        1D array of 0/1 bits representing the Rice-encoded positive differences of the image.
    """
    image_hat = predictor(image, local_sum_mode=local_sum_mode, P=P, W=W, Omega=Omega)
    delta_positive = generate_positive_diff(image, image_hat, Q=Q)

    k, bitstream = rice_encoder(delta_positive, block_size=block_size)
    delta_positive = rice_decoder(bitstream, k, block_size=block_size, shape=image.shape)

    delta_r = unpack_positive_diff(delta_positive, Q=Q)
    image_r = reconstructor(delta_r, local_sum_mode=local_sum_mode, P=P, W=W, Omega=Omega)

    return image_r, bitstream

image = load_image("data\\Indian_pines_corrected.mat")
#image = image [:50, :50, :50]

image_r, compressed_stream = CCSDS(image, local_sum_mode='col', P=1, W=0.5*np.ones(1), Omega=0, Q=0, block_size=32)

print(f'RMSE = {metrics.calc_RMSE(image, image_r)}')
print(f'SAM = {metrics.calc_SAM(image, image_r)}')
print(f'Ratio = {metrics.calc_compression_ratio(image, compressed_stream)}')

plt.subplot(1, 2, 1)
plt.imshow(image[:,:,20], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(image_r[:,:,20], cmap='gray')
plt.show()
