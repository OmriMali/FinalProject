import numpy as np
import matplotlib.pyplot as plt
import pylops
import util 

# ==========================================
# 1. Load Data & Reshape to Pixel List
# ==========================================
print("Loading data...")
try:
    image_path = "data\\Indian_pines_corrected.mat"
    image = util.load_image(image_path)
except:
    image = np.random.rand(50, 50, 200)

# Normalize to [0, 1]
image = image.astype(np.float64)
image = (image - image.min()) / (image.max() - image.min())

# Reshape to (Num_Pixels, Num_Bands)
H, W, B = image.shape
N_pixels = H * W
f_original = image.reshape(N_pixels, B)

print(f"Data Shape: {N_pixels} pixels, {B} spectral bands each.")

# ==========================================
# 2. Pre-processing: Sparsification
# ==========================================
T_factor = 0.25
print(f"Sparsifying data (T={T_factor})...")

# A. Transform f -> x (IDFT domain)
x_dense = np.fft.ifft(f_original, axis=1)

# B. Calculate Stats
mu = np.mean(np.abs(x_dense), axis=1, keepdims=True)
sigma = np.std(np.abs(x_dense), axis=1, keepdims=True)

# C. Apply Threshold
threshold = mu + T_factor * sigma
mask = np.abs(x_dense) >= threshold
x_sparse_true = x_dense * mask

# D. Transform x -> f (Back to Spectral Domain)
f_sparse_true = np.fft.fft(x_sparse_true, axis=1).real
print(f"Sparsification complete. Avg sparsity: {np.mean(mask):.2%}")

# ==========================================
# 3. Define Operators (Pixel-wise Physics)
# ==========================================

# Measurement Indices (M bands out of B)
M = int(0.3 * B)
indices = np.random.choice(B, M, replace=False)
indices.sort()

# Generate Ground Truth Measurements (y)
y_true = f_sparse_true[:, indices] 


# Custom Operator: Maps x (Sparse IDFT coeffs) -> y (Measurements)
class SpectralMeasurementOp(pylops.LinearOperator):
    def __init__(self, n_pixels, n_bands, valid_indices, dtype='complex128'):
        self.N = n_pixels
        self.B = n_bands
        self.M = len(valid_indices)
        self.idx = valid_indices
        
        # Calculate shape: (Total Measurements, Total Coefficients)
        shape = (self.N * self.M, self.N * self.B)
        
        # CRITICAL FIX: Initialize the parent PyLops class!
        super().__init__(dtype=np.dtype(dtype), shape=shape)
        
    def _matvec(self, x):
        # Forward: x (flat) -> reshape -> FFT -> Subsample -> y (flat)
        X = x.reshape(self.N, self.B)
        # norm='ortho' ensures the spectral norm is 1.0
        F = np.fft.fft(X, axis=1, norm='ortho')
        Y = F[:, self.idx]
        return Y.ravel()
    
    def _rmatvec(self, y):
        # Adjoint: y (flat) -> reshape -> ZeroFill -> IFFT -> x (flat)
        Y = y.reshape(self.N, self.M)
        F_rec = np.zeros((self.N, self.B), dtype=self.dtype)
        F_rec[:, self.idx] = Y
        X_rec = np.fft.ifft(F_rec, axis=1, norm='ortho')
        return X_rec.ravel()

# Instantiate the Operator
A = SpectralMeasurementOp(N_pixels, B, indices)

# ==========================================
# 4. Solve with FISTA
# ==========================================
print("Solving Inverse Problem (Pixel-wise FISTA)...")

# FISTA solves: min || A x - y ||^2 + eps || x ||_1
# Note: eps is the Lambda parameter. 
# Since our data is normalized [0,1], 0.001 is a good starting point.
x_rec_flat, niter, cost = pylops.optimization.sparsity.fista(
    A,
    y_true.ravel(),
    niter=100,
    eps=0.1,  
    tol=1e-5,
    show=True
)

# ==========================================
# 5. Reconstruction & Visualization
# ==========================================

# x_rec is in IDFT domain. Transform back to Spectral Domain (f)
x_rec = x_rec_flat.reshape(N_pixels, B)
f_rec = np.fft.fft(x_rec, axis=1, norm='ortho').real

# Reshape back to Image Cube
f_rec_image = f_rec.reshape(H, W, B)
f_sparse_image = f_sparse_true.reshape(H, W, B)

# Calculate RMSE
rmse = np.sqrt(np.mean((f_sparse_image - f_rec_image)**2))
print(f"Reconstruction RMSE: {rmse:.6f}")

# Plotting
band_idx = 30
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image[:, :, band_idx], cmap='gray')
plt.title("Original Raw")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(f_sparse_image[:, :, band_idx], cmap='gray')
plt.title(f"Sparsified Input (T={T_factor})")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(f_rec_image[:, :, band_idx], cmap='gray')
plt.title(f"Reconstructed (RMSE={rmse:.4f})")
plt.axis('off')

plt.tight_layout()
plt.show()

# Spectral Signature Check
plt.figure(figsize=(8, 4))
px_y, px_x = 20, 20
plt.plot(f_sparse_image[px_y, px_x, :], label='Ground Truth (Sparse)')
plt.plot(f_rec_image[px_y, px_x, :], '--', label='Reconstruction')
plt.legend()
plt.title(f"Spectral Signature Pixel ({px_y}, {px_x})")
plt.show()