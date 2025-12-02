### version: 0.1.0
### date: 02/12/25
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

def rice_encoder(delta_positive, block_size=32):
    
    delta_positive = delta_positive.flatten()
    blocks = [delta_positive[i:i+block_size] for i in range(0, len(delta_positive), block_size)]

    q = np.zeros(delta_positive.shape, dtype=np.uint16)
    r = np.zeros(delta_positive.shape, dtype=np.uint16)
    k = []
    bitstrings = []

    for i in range(0, len(blocks)):
        k.append(int(np.log2(1 + np.mean(blocks[i]))))
        for j in range(i*block_size, i*block_size + len(blocks[i])):
            q[j] = delta_positive[j] >> k[i]
            r[j] = delta_positive[j] & ((1 << k[i]) - 1)
            unary = '1' * q[j] + '0'
            if k[i] > 0:
                remainder = format(r[j], f'0{k[i]}b')  # zero-padded binary
            else:
                remainder = ''
            bitstring = unary + remainder
            bitstrings.append(bitstring)
    
    return k, bitstrings

def rice_decoder(bitstrings, k, block_size, shape):
    
    total_len = np.prod(shape)
    delta_positive = np.zeros(total_len, dtype=np.uint16)

    # Compute block lengths
    num_blocks = len(k)
    block_lengths = [block_size] * num_blocks
    # adjust last block if needed
    last_block_len = total_len - block_size*(num_blocks-1)
    block_lengths[-1] = last_block_len

    bit_idx = 0
    for i in range(num_blocks):
        k_block = k[i]  # same k as used in encoder
        block_len = block_lengths[i]
        
        for j in range(i*block_size, i*block_size + block_len):
            bstr = bitstrings[bit_idx]
            
            # Decode unary part to get q
            q_val = bstr.find('0')  # count of leading '1's
            unary_len = q_val + 1   # include terminating '0'
            
            # Decode remainder
            if k_block > 0:
                r_val = int(bstr[unary_len:unary_len + k_block], 2)
            else:
                r_val = 0
            
            # Reconstruct δz
            delta_positive[j] = (q_val << k_block) + r_val
            bit_idx += 1

    return delta_positive.reshape(shape)

def CCSDS(image, local_sum_mode, P, W, Omega, Q, block_size):

    image_hat = predictor(image, local_sum_mode=local_sum_mode, P=P, W=W, Omega=Omega)
    delta_positive = generate_positive_diff(image, image_hat, Q=Q)

    k, bitstrings = rice_encoder(delta_positive, block_size=block_size)
    delta_positive = rice_decoder(bitstrings, k, block_size=block_size, shape=image.shape)

    delta_r = unpack_positive_diff(delta_positive, Q=Q)
    image_r = reconstructor(delta_r, local_sum_mode=local_sum_mode, P=P, W=W, Omega=Omega)

    return image_r, bitstrings

image = load_image("data\\Indian_pines_corrected.mat")
image = image [:50, :50, :50]

image_r, bitstrings = CCSDS(image, local_sum_mode='col', P=1, W=0.5*np.ones(1), Omega=0, Q=0, block_size=32)


print(f'RMSE = {metrics.calc_RMSE(image, image_r)}')
print(f'SAM = {metrics.calc_SAM(image, image_r)}')
print(f'Ratio = {metrics.calc_compression_ratio(image, bitstrings)}')

plt.subplot(1, 2, 1)
plt.imshow(image[:,:,20], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(image_r[:,:,20], cmap='gray')
plt.show()