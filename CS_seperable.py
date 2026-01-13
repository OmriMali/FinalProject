import numpy as np
import matplotlib.pyplot as plt
import util
import gOMP

def generate_measurement_matrix(m, n, seed=None):
    if m > n:
        raise ValueError("Constraint violation: must have m ≤ n")

    rng = np.random.default_rng(seed)

    # Randomly choose m distinct columns
    cols = rng.choice(n, size=m, replace=False)

    A = np.zeros((m, n), dtype=int)
    A[np.arange(m), cols] = 1

    return A

image_path = "data\\Indian_pines_corrected.mat"
image = util.load_image(image_path)

# Normalize to [0, 1]
image = image[:, :, :]
image = image.astype(np.float64)
image = (image - image.min()) / (image.max() - image.min())

Nx, Ny, Nz = image.shape
F = np.zeros((Nz, Nx*Ny))

for nz in range(Nz):
    for ny in range(Ny):
        for nx in range(Nx):
            F[nz, nx + ny*Nx] = image[nx, ny, nz]

# F = np.reshape(image, (Nz, Nx*Ny), 'C', copy=True)

p = Nz // 4

Phi = generate_measurement_matrix(p, Nz)


y = Phi @ F
image_rec = np.zeros_like(image)
Psi = np.fft.ifft(np.eye(Nz))
A = Phi @ Psi

for ny in range(Ny):
    for nx in range(Nx):
        s = gOMP.gOMP(y[:, nx + ny*Nx], A, K = 15, G=10, eps=1e-6, N=Nz)
        x = np.real(np.fft.ifft(s))
        image_rec[nx, ny, :] = x
    print(f'finished row {ny}')


print(util.calc_RMSE(image, image_rec))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image[:,:,5], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(image_rec[:,:,5], cmap='gray')
plt.show()