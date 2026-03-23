import numpy as np
from tqdm import tqdm
from src import recovery_algorithms

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
