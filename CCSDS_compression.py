
# @file: CCSDS.py
# @breif: Simplified-but-realistic CCSDS-123 style compressor and decompressor (lossless / near-lossless).
# Date Created: 28.11.25
# Version: 0.0.0

import numpy as np
import scipy as sp

image = sp.io.loadmat("Indian_pines_corrected.mat")
image = image['indian_pines_corrected']
Nx, Ny, Nz = image.shape

Nx = 10
Ny = 10
Nz = 5

sigma = np.zeros((Nx, Ny, Nz), dtype=np.int32)
d = np.zeros((Nx, Ny, Nz), dtype=np.int32)
image_hat = np.zeros((Nx, Ny, Nz), dtype=np.int32)
delta = np.zeros((Nx, Ny, Nz), dtype=np.int32)
delta_positive = np.zeros((Nx, Ny, Nz), dtype=np.uint32)

local_sum_mode = 'col'  # col, narrow, wide
P = 1
W = 0.5 * np.ones(P)
Omega = 0
Q = 2

for z in range(Nz):
    for y in range(Ny):
        for x in range(Nx):

            match local_sum_mode:
                case "col":
                    if y >= 1:
                        sigma[x, y, z] = 4 * image[x, y - 1, z]
                case "narrow":
                    if y >= 1:
                        if (x >= 1 and x <= Nx - 2):
                            sigma[x, y, z] = 2 * image[x, y - 1, z] + image[x - 1, y - 1, z] + image[x + 1, y - 1, z]
                        elif x == Nx - 1:
                            sigma[x, y, z] = 3 * image[x, y - 1, z] + image[x - 1, y - 1, z]
                        else:
                            sigma[x, y, z] = 3 * image[x, y - 1, z] + image[x + 1, y - 1, z]
                case "wide":
                    if y >= 1:
                        if (x >= 1 and x <= Nx - 2):
                            sigma[x, y, z] = image[x - 1, y, z] + image[x, y - 1, z] + image[x - 1, y - 1, z] + image[
                                x + 1, y - 1, z]
                        elif x == Nx - 1:
                            sigma[x, y, z] = image[x - 1, y, z] + 2 * image[x, y - 1, z] + image[x - 1, y - 1, z]
                        else:
                            sigma[x, y, z] = 3 * image[x, y - 1, z] + image[x + 1, y - 1, z]
                case _:
                    print("unvalid mode")

            d[x, y, z] = 4 * image[x, y, z] - sigma[x, y, z]
            U = np.zeros(P)
            for i in range(P):
                if z - i - 1 >= 0:
                    U[i] = d[x, y, z - i - 1]

            d_hat = np.dot(W, U)

            image_hat[x, y, z] = int((d_hat + 2 ** Omega * sigma[x, y, z]) / (2 ** (Omega + 2)))
            delta[x, y, z] = image[x, y, z] - image_hat[x, y, z]

            delta[x, y, z] = int(delta[x, y, z] / 2 ** Q)

            if delta[x, y, z] > 0:
                delta_positive[x, y, z] = 2 * delta[x, y, z] - 1
            else:
                delta_positive[x, y, z] = -2 * delta[x, y, z]

print(delta_positive)

delta_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
sigma_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
d_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
image_hat_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
image_r = np.zeros((Nx, Ny, Nz), dtype=np.int32)
test = np.zeros((Nx, Ny, Nz), dtype=np.int32)

for z in range(Nz):
    for y in range(Ny):
        for x in range(Nx):

            if delta_positive[x, y, z] % 2 == 0:
                delta_r[x, y, z] = - (delta_positive[x, y, z] / 2)
            else:
                delta_r[x, y, z] = (delta_positive[x, y, z] + 1) / 2

            delta_r[x, y, z] = delta_r[x, y, z] * 4

            match local_sum_mode:
                case "col":
                    if y >= 1:
                        sigma_r[x, y, z] = 4 * image_r[x, y - 1, z]
                case "narrow":
                    if y >= 1:
                        if (x >= 1 and x <= Nx - 2):
                            sigma_r[x, y, z] = 2 * image_r[x, y - 1, z] + image_r[x - 1, y - 1, z] + image_r[
                                x + 1, y - 1, z]
                        elif x == Nx - 1:
                            sigma_r[x, y, z] = 3 * image_r[x, y - 1, z] + image_r[x - 1, y - 1, z]
                        else:
                            sigma_r[x, y, z] = 3 * image_r[x, y - 1, z] + image_r[x + 1, y - 1, z]
                case "wide":
                    if y >= 1:
                        if (x >= 1 and x <= Nx - 2):
                            sigma_r[x, y, z] = image_r[x - 1, y, z] + image_r[x, y - 1, z] + image_r[x - 1, y - 1, z] + \
                                               image_r[x + 1, y - 1, z]
                        elif x == Nx - 1:
                            sigma_r[x, y, z] = image_r[x - 1, y, z] + 2 * image_r[x, y - 1, z] + image_r[
                                x - 1, y - 1, z]
                        else:
                            sigma_r[x, y, z] = 3 * image_r[x, y - 1, z] + image_r[x + 1, y - 1, z]
                case _:
                    print("unvalid mode")

            U = np.zeros(P)
            for i in range(P):
                if z - i - 1 >= 0:
                    U[i] = d[x, y, z - i - 1]

            d_hat = np.dot(W, U)
            image_hat_r[x, y, z] = int((d_hat + 2 ** Omega * sigma_r[x, y, z]) / (2 ** (Omega + 2)))
            image_r[x, y, z] = image_hat_r[x, y, z] + delta_r[x, y, z]
            d_r = 4 * image_r[x, y, z] - sigma_r[x, y, z]
            test[x, y, z] = image[x, y, z] - image_r[x, y, z]

print(test)