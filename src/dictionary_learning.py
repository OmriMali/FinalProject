import numpy as np
from src import util
from src.hsi import  HSI
from tqdm import tqdm
from src import recovery_algorithms
import os
import random

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

def from_spectral_library_targeted(Y, folder_path, hsi, limit=512, correlation_threshold=0.98, keywords=None, **kwargs):
    """
    New function: Extracts specific material signatures based on keywords 
    and handles potential empty/malformed files.
    """
    import random
    target_wavelengths = hsi.wavelengths
    target_wl_microns = target_wavelengths / 1000.0 if np.max(target_wavelengths) > 20 else target_wavelengths
    
    atoms = []
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if keywords:
        all_files = [f for f in all_files if any(key.lower() in f.lower() for key in keywords)]
    
    random.shuffle(all_files)
    
    for file_name in all_files:
        if len(atoms) >= limit: break
        try:
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                data_content = [l.strip() for l in lines[26:] if l.strip()]
                if not data_content: continue
                data = np.loadtxt(data_content)
                atom = np.interp(target_wl_microns, data[:, 0], data[:, 1])
                norm = np.linalg.norm(atom)
                if norm > 0:
                    atom /= norm
                    if not atoms or np.max(np.abs(np.dot(np.array(atoms), atom))) < correlation_threshold:
                        atoms.append(atom)
        except: continue

    D = np.column_stack(atoms)
    X = np.linalg.pinv(D) @ Y
    return D, X

def k_svd_hybrid(Y, K, T_0, D_init, tol=1e-6, max_iter=100, progress_callback=None):
    """
    New function: Performs K-SVD starting from a physical library 
    initialization instead of SVD or random signals.
    """
    from src import recovery_algorithms
    M, N = Y.shape
    
    # Initialize D from the provided physical dictionary
    D = D_init[:, :K].copy()
    if D.shape[1] < K:
        idx = np.random.choice(N, K - D.shape[1], replace=False)
        D = np.hstack([D, Y[:, idx]])
    
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    D /= np.where(norms == 0, 1, norms)
    X = np.zeros((K, N))
    Y_norms = np.linalg.norm(Y)

    for J in range(1, max_iter + 1):
        # Sparse Coding Step
        for i in range(N):
            X[:, i] = recovery_algorithms.omp(D, Y[:, i], T_0)
        
        # Dictionary Update Step
        for k in range(K):
            omega_k = np.where(np.abs(X[k, :]) > 1e-10)[0]
            if len(omega_k) == 0: continue
            
            E_k = Y - D @ X + np.outer(D[:, k], X[k, :])
            U, S, V_T = np.linalg.svd(E_k[:, omega_k], full_matrices=False)
            D[:, k] = U[:, 0]
            X[k, omega_k] = S[0] * V_T[0, :]

        if progress_callback: progress_callback(J / max_iter)
        if (np.linalg.norm(Y - D @ X) / Y_norms) < tol: break

    return D, X