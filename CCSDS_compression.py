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
    

image = sp.io.loadmat("final project\\Indian_pines_corrected.mat")
image = image['indian_pines_corrected']
Nx, Ny, Nz = image.shape

Nx = 100
Ny = 100
Nz = 50

image = image[:Nx, :Ny, :Nz]

sigma = np.zeros((Nx, Ny, Nz), dtype=np.int32)
d = np.zeros((Nx, Ny, Nz), dtype=np.int32)
image_hat =  np.zeros((Nx, Ny, Nz), dtype=np.int32)
delta = np.zeros((Nx, Ny, Nz), dtype=np.int32)
delta_positive = np.zeros((Nx,Ny,Nz), dtype=np.uint32)

local_sum_mode = 'wide'      # col, narrow, wide
P = 1
W = 0.5 * np.ones(P)
Omega = 0
Q = 1

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
            delta[x,y,z] = image[x,y,z] - image_hat[x,y,z]

            delta[x,y,z] = int(delta[x,y,z] / 2**Q)

            if delta[x,y,z] > 0:
                delta_positive[x,y,z] = 2 * delta[x,y,z] - 1
            else:
                delta_positive[x,y,z] = -2 * delta[x,y,z]

delta_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
sigma_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
d_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
image_hat_r =  np.zeros((Nx, Ny, Nz), dtype=np.int32)
image_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
test = np.zeros((Nx, Ny, Nz), dtype=np.int32)

for z in range(Nz):
    for y in range(Ny):
        for x in range(Nx):
            
            if delta_positive[x,y,z]%2 == 0:
                delta_r[x,y,z] = - (delta_positive[x,y,z] / 2)
            else:
                delta_r[x,y,z] = (delta_positive[x,y,z] + 1)/ 2

            delta_r[x,y,z] = delta_r[x,y,z] * 2**Q

            sigma_r[x,y,z] = calc_local_sum(local_sum_mode, (x, y), image_r[:, :, z], Nx)

            U = np.zeros(P)
            for i in range(P):
                if z-i-1 >= 0:
                    U[i] = d[x,y,z-i-1]           
            
            d_hat = np.dot(W, U)
            image_hat_r[x,y,z] = int((d_hat + 2**Omega * sigma_r[x,y,z]) / (2**(Omega+2)))

            image_r[x,y,z] = image_hat_r[x,y,z] + delta_r[x,y,z]

            d_r = 4 * image_r[x, y ,z] - sigma_r[x, y, z]

            test[x,y,z] = image[x,y,z] - image_r[x,y,z]


print(f'RMSE = {metrics.calc_RMSE(image, image_r)}')
print(f'SAM = {metrics.calc_SAM(image, image_r)}')

plt.subplot(1, 2, 1)
plt.imshow(image[:,:,20], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(image_r[:,:,20], cmap='gray')
plt.show()