import numpy as np
from src import util
from src.hsi import  HSI
from tqdm import tqdm
from src import recovery_algorithms
import os
import random
from scipy.interpolate import CubicSpline

def k_svd(Y, K, T_0, tol=1e-6, max_iter=100, progress_callback=None):
    """
    K-SVD algorithm for dictionary learning.

    Parameters   
    ----------
    Y : ndarray
        Input signals to learn, arranged as column vectors of a 2D arra of shape (M, N)
    K : int
        Size of the output dictionary.
    T_0 : int
        The sparsity level to be obtained by the dictionary for the input signals.
    tol : float, optional
        Stopping threshold based on relative error of Y - DX.
    max_iter : int, optional
        Maximum iterations for the algorithm.
    progress_callback : float, optional
        Updates an external progress bar.

    Returns
    -------
    D : ndarray
        The dictionary which sparsly represent the input signals, of shape (M, K)
    X : ndarray
        The coefficient vectors arragned in a matrix of size (K, N) satisfying Y~=DX. 
    """
    # Step 1: Initialize
    M, N = Y.shape

    # # Trivial Dictrionary Initialization
    # D = Y[:, :K].copy()
    # D /= np.linalg.norm(D, axis=0, keepdims=True)

    # SVD Dictrionary Initialization
    U, _, _ = np.linalg.svd(Y, full_matrices=False)
    D = np.zeros((M, K))
    D[:, :M] = U
    if K > M:
        idx = np.random.choice(Y.shape[1], K - M, replace=False)
        D[:, M:] = Y[:, idx]
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1
    D /= norms
    
    X = np.zeros((K, N))
    Y_norms = np.linalg.norm(Y)

    # Step 2: Loop
    J = 1
    while J <= max_iter:
        # Step 3: Sparse Coding
        for i in range(N):
            y = Y[:, i].copy()
            X[:, i] = recovery_algorithms.omp(D, y, T_0)
        
        # Step 4: Codebook Update
        for k in range(K):
            # Step 4.1: Find the signals that use the current atom
            eps = 1e-10
            omega_k = np.where(np.abs(X[k, :]) > eps)[0]
            
            # (Optional) Replace dead atoms with the worst represented signal
            if len(omega_k) == 0:
                R = Y - D @ X
                errors = np.linalg.norm(R, axis=0)
                worst_idx = np.argmax(errors)
                D[:, k] = Y[:, worst_idx] / np.linalg.norm(Y[:, worst_idx])
                X[k, :] = 0
                X[k, worst_idx] = np.dot(D[:, k], Y[:, worst_idx])
                continue
            
            # Step 4.2: Compute the overall error matrix
            E_k = Y - D @ X + np.outer(D[:, k], X[k, :])

            # Step 4.3: Obtain the restricted error matrix
            E_k_R = E_k[:, omega_k]

            # Step 4.4: Apply SVD and obtain dictionary column and coefficent vector
            U, S, V_T = np.linalg.svd(E_k_R, full_matrices=False)
            D[:, k] = U[:, 0].copy()
            X[k, omega_k] = S[0] * V_T[0, :].T

        
        # (Optional) Update external progress function
        if progress_callback:
            progress_callback(J / max_iter)
        
        # Step 5: Check for stop condition, increment loop
        err = np.linalg.norm(Y - D @ X) / Y_norms
        if err < tol:
            if progress_callback:
                progress_callback(1.0)
            break
        J += 1

    return D, X

def k_svd_aster_paper_hybrid(Y, folder_path, hsi, K=128, T_0=3, max_iter=50, progress_callback=None, **kwargs):
    """
    Implements the sparse basis construction method from the ASTER library paper.
    Uses your original k_svd function for the training phase.
    """
    M, N = Y.shape
    hsi_wl = hsi.wavelengths
    # Convert nm to microns for ASTER library compatibility
    hsi_wl_microns = hsi_wl / 1000.0 if np.max(hsi_wl) > 20 else hsi_wl
    
    # --- 1. Construct Training Set from Library ---
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    library_signals = []
    
    for file_name in all_files:
        try:
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # ASTER headers are 26 lines
                data_lines = [l.strip() for l in lines[26:] if l.strip()]
                if len(data_lines) < 2: 
                    continue
                    
                data = np.loadtxt(data_lines)
                lib_wl, lib_refl = data[:, 0], data[:, 1]
                
                # Only use files that cover the full HSI wavelength range
                if lib_wl.min() <= hsi_wl_microns.min() and lib_wl.max() >= hsi_wl_microns.max():
                    # Cubic Spline Interpolation as specified in the paper
                    cs = CubicSpline(lib_wl, lib_refl)
                    atom = cs(hsi_wl_microns)
                    
                    # Normalize atom to unit norm
                    norm = np.linalg.norm(atom)
                    if norm > 0:
                        library_signals.append(atom / norm)
        except: 
            continue

    if not library_signals:
        raise ValueError("No matching ASTER library signals found for this sensor's range.")

    # 2. Combine Library signals (W) and Target HSI pixels (Y)
    W = np.column_stack(library_signals)
    Training_Set = np.hstack([W, Y])
    
    # 3. Use YOUR original K-SVD to learn the dictionary
    # Pass Training_Set to your existing k_svd function. 
    # Note: K must be <= M to avoid the broadcast error in your original k_svd.
    D_learned, X_learned = k_svd(
        Training_Set, 
        K=K, 
        T_0=T_0, 
        max_iter=max_iter, 
        progress_callback=progress_callback
    )
    
    # Return dictionary and coefficients for the original target Y
    X_target = np.linalg.pinv(D_learned) @ Y
    return D_learned, X_target

def _synth_test_k_svd(M=20, K=10, N=200, T_0=3):

    pbar = tqdm(total=100)
    def progress_bar(fraction):
        pbar.n = int(100 * fraction)
        pbar.refresh()

    # Dictionary Generation
    np.random.seed(0)
    D_true = np.random.randn(M, K)
    D_true /= np.linalg.norm(D_true, axis=0, keepdims=True)

    # Sparse Coefficients Generation
    X_true = np.zeros((K, N))
    for i in range(N):
        idx = np.random.choice(K, T_0, replace=False)
        X_true[idx, i] = np.random.randn(T_0)
    
    # Generate Signals
    Y = D_true @ X_true

    # Run K-SVD
    D_learned, X_learned = k_svd(Y, K, T_0, max_iter=100, progress_callback=progress_bar)
    pbar.close()

    # Reconstruction error
    err = np.linalg.norm(Y - D_learned @ X_learned) / np.linalg.norm(Y)
    print("Reconstruction error:", err)

    # Dictionary Correlation
    C = np.abs(D_true.T @ D_learned)
    max_corr = np.max(C, axis=1)
    print("Mean atom recovery:", np.mean(max_corr))
    print("Min atom recovery:", np.min(max_corr))

    # Sparsity Check
    avg_sparsity = np.mean(np.count_nonzero(X_learned, axis=0))
    print("Average sparsity:", avg_sparsity)

def prep_hsi_for_dict_learning(hsi: HSI, N_train: int, mode: int):
    """
    Preprocess HSI to use as training data for dictionary.
    Preprocessing includes: Unfolding the HSI along mode, randomly choosing N_train fibers, returning the data shaped as a 2D matrix of N_train column vectors.
    
    Parameters
    ----------
    hsi : HSI
        HSI to use as training data.
    N_train : int
       Max number of fibers to train on.
    mode : int
        Axis on which to train.
    
    Returns
    --------
    Y : ndarray
        A 2D array of shape (:, N_train).
    """
    
    Y = util.mode_n_unfold(hsi.get_norm_data(), n=mode)
    if N_train >= Y.shape[1]:
        return Y
    else:
        idx = np.random.choice(Y.shape[1], N_train, replace=False)
        return Y[:, idx]

def prep_multi_hsi_for_dict_learning(folder_path, N_train_per_hsi, mode=2):
    """
    Aggregates training signals from all .npy HSI files in a directory.
    """
    all_Y = []
    # Find all .npy HSI files saved via util.save_hsi
    hsi_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    if not hsi_files:
        raise FileNotFoundError(f"No HSI files found in {folder_path}")

    for f_name in hsi_files:
        try:
            hsi = util.load_hsi(os.path.join(folder_path, f_name))
            # Use existing prep function to sample fibers from this section
            Y_sub = prep_hsi_for_dict_learning(hsi, N_train=N_train_per_hsi, mode=mode)
            all_Y.append(Y_sub)
        except Exception as e:
            print(f"Skipping {f_name} due to error: {e}")

    # Stack all collected fibers into a single large training matrix (Bands x Total_Fibers)
    return np.hstack(all_Y)

def get_keywords_for_scene(scene_type: str):
    """
    Returns a list of keywords appropriate for a given scene type 
    based on the spectral library categories.
    """
    scene_type = scene_type.lower()
    
    # Mapping scenes to init.txt categories
    registry = {
        "field": [
            'vegetation', 'soil', 'grass', 'shrub', 'alfalfa', 
            'non-photosynthetic', 'mineral', 'forb'
        ],
        "village": [
            'concrete', 'asphalt', 'roof', 'brick', 'wood', 
            'metal', 'paint', 'grass', 'tree'
        ],
        "forest": [
            'vegetation', 'tree', 'shrub', 'leaf', 'conifer', 
            'bark', 'soil', 'liquid.water'
        ],
        "urban": [
            'concrete', 'asphalt', 'metal', 'paint', 'glass', 
            'brick', 'tile', 'plastic', 'soil'
        ],
        "lake": [
            'liquid.water',      # Pure and natural water signatures
            'algae',             # Critical for biological content in lakes
            'vegetation',        # Riparian/shoreline plants
            'mineral',           # Sediments and lake-bed materials
            'soil',              # Wet soil at the water's edge
            'non-photosynthetic'  # Decaying organic matter in the water
        ]
    }
    
    if scene_type not in registry:
        print(f"[WARNING] Scene '{scene_type}' not found. Using general vegetation and soil.")
        return ['vegetation', 'soil']
        
    return registry[scene_type]

def k_svd_from_spectral_library(Y, folder_path, hsi, K=128, T_0=3, max_iter=50, progress_callback=None, **kwargs):
    """
    Learns a dictionary exclusively from the ASTER spectral library as described in the paper.
    Adjusts library signals to match sensor resolution and physical scale.
    """
    M, N = Y.shape
    hsi_wl = hsi.wavelengths
    # ASTER/JPL library files are in micrometers. Convert nm to microns if needed.
    hsi_wl_microns = hsi_wl / 1000.0 if np.max(hsi_wl) > 20 else hsi_wl
    
    # 1. Targeted Selection: Target only VSWIR signals and avoid .ancillary text files
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.spectrum.txt')]
    library_signals = []
    
    for file_name in all_files:
        # Avoid TIR-only files as they don't cover the visible/shortwave bands
        if 'tir' in file_name.lower() and 'vswir' not in file_name.lower():
            continue

        try:
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # Skip exactly 26 lines of the ASTER header to reach numeric data
                data_lines = [l.strip() for l in lines[26:] if l.strip()]
                if len(data_lines) < 2: 
                    continue
                
                data = np.loadtxt(data_lines)
                lib_wl, lib_refl = data[:, 0], data[:, 1]
                
                # Range Check: The library signal must cover the entire sensor range
                if lib_wl.min() <= hsi_wl_microns.min() and lib_wl.max() >= hsi_wl_microns.max():
                    # Cubic Spline Interpolation for piecewise smoothness
                    cs = CubicSpline(lib_wl, lib_refl)
                    atom = cs(hsi_wl_microns)
                    
                    # SCALE ADJUSTMENT: Normalize library atoms to [0, 1] range to fix 
                    # the mix of 'Percentage' and 'Fractional' units in files
                    atom = (atom - np.min(atom)) / (np.max(atom) - np.min(atom) + 1e-6)
                    
                    # K-SVD requires unit-norm atoms for the update step
                    norm = np.linalg.norm(atom)
                    if norm > 0:
                        library_signals.append(atom / norm)
        except Exception:
            continue

    if not library_signals:
        raise ValueError("No valid library signals found matching the sensor range.")

    # W is the training set constructed EXCLUSIVELY from library samples
    W = np.column_stack(library_signals)
    
    # 2. Train K-SVD on the library data W
    # This adaptively constructs the basis D from physical spectral samples
    D_final, _ = k_svd(W, K, T_0, max_iter=max_iter, progress_callback=progress_callback)
    
    # 3. Calculate X for the dummy Y to satisfy workflow metric requirements
    # Using OMP ensures the Mean Sparsity metric reflects the T_0=3 setting
    X_val = np.zeros((K, N))
    for i in range(N):
        X_val[:, i] = recovery_algorithms.omp(D_final, Y[:, i], T_0)
        
    return D_final, X_val