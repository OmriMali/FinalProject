import numpy as np
import util
import matplotlib.pyplot as plt

image = util.load_image("data\\Indian_pines_corrected.mat")
image = image[:100, :100, :100]

# Create hyperspectral data matrix, size of MxL
L1, L2, M = image.shape
L = L1 * L2
X = image.reshape(L, M).T

# Matrix to vector, size of ML
x = X.reshape(-1, order='F')

def build_measurement_mat(L1, L2, M, r1, r2):
    L = int(L1 * L2)
    Lr1 = int(L1 / r1)
    Lr2 = int(L2 / r2)
    Lh = int(Lr1*Lr2)

    D = np.ones((1, M))
    B1 = np.kron(np.eye(Lr1), np.ones((r1, 1)))
    B2 = np.kron(np.eye(Lr2), np.ones((r2, 1)))
    B = np.kron(B2, B1)

    print(f'B Shape: {B.shape}')
    phi_top = np.kron(np.eye(L), D)
    phi_bot = np.kron(B.T, np.eye(M))

    print(f'phi top Shape: {phi_top.shape}')
    print(f'phi bot Shape: {phi_bot.shape}')

    phi = np.vstack([phi_top, phi_bot])
    
    print(f'phi Shape: {phi.shape}')
    return phi

phi = build_measurement_mat(L1, L2, M, 4, 4)

y = np.dot(phi, x)
print(f'x shape = {x.shape}')
print(f'y shape = {y.shape}')

# Vector to matrix, back to MxL
X_r = x.reshape(M, L, order='F')

# Matrix to hyperspectral im    age, back to (L1, L2, M)
image_r = X_r.T.reshape(L1, L2, M)

