import numpy as np
from src import util


def omp(D, y, K, tol=1e-6, progress_callback=None):
    """
    OMP algorithm for sparse approximation of a vector.

    Parameters
    ----------
    D : ndarray
        A dictionary of shape (I, M). Columns must be normalized.
    y : ndarray
        Input vector of length I.
    K : int
        Sparsity level.
    tol : float, optional
        Stopping threshold based on norm of the residual.
    progress_callback : float, optional
        Updates an external progress bar.

    Returns
    -------
    x : ndarray
        Sparse vector such that y ~= Dx.
    """
    # Step 1: Initialization
    I, M = D.shape
    idx_list = []
    r = y.copy()
    x = np.zeros(M)
    a = np.array([])

    # Step 2: Loop
    k = 1
    r_norm = np.linalg.norm(r)
    while k <= K and r_norm > tol:

        # Step 3: Find maximum correlated atom
        corr = D.T @ r
        idx = np.argmax(np.abs(corr))

        # Step 4: Update index list
        if idx in idx_list:
            break

        idx_list.append(idx)

        # Step 5: Compute coefficients (least squares)
        D_sub = D[:, idx_list]
        a, _, _, _ = np.linalg.lstsq(D_sub, y, rcond=None)

        # Step 6: Update residual
        r = y - D_sub @ a
        r_norm = np.linalg.norm(r)

        # Optional progress callback
        if progress_callback:
            progress_callback(k / K)
        
        # Step 7: Increment k
        k += 1

    # Step 8: Compute x
    if len(idx_list) > 0:
        x[idx_list] = a

    return x

def gomp(y, operator, K, N, eps=1e-6):
    """
    Generalized Orthogonal Matching Pursuit (gOMP) implementation.

    Parameters
    ----------
    y : numpy.ndarray
        The measurement vector (m,).
    operator : SensingOperator
        The operator A providing .forward(), .adjoint(), and .get_column().
    K : int
        Target sparsity (max number of atoms to select).
    N : int
        Step size (number of atoms to select per iteration).
    eps : float
        Convergence threshold for the residual norm.

    Returns
    -------
    s_hat : numpy.ndarray
        The reconstructed sparse coefficient vector (n,).
    """
    m = y.shape[0]
    n = operator.n
    
    # Initialize
    k = 0
    r = y.copy().astype(operator.dtype)
    Lambda = set()
    s_ls = np.array([])
    Lambda_list = []

    # Optimization Loop
    while np.linalg.norm(r) > eps and k < min(K, m // N):
        k += 1
        
        # 1. Identification: Correlation between residual and atoms
        correlations = np.abs(operator.adjoint(r))
        
        # Mask already selected indices to prevent re-selection
        if len(Lambda) > 0:
            correlations[list(Lambda)] = -1.0
            
        # 2. Selection: Get N best indices
        new_indices = np.argsort(correlations)[-N:]
        Lambda.update(new_indices)
        Lambda_list = sorted(list(Lambda))
        
        # 3. Estimation: Least Squares on the sub-dictionary
        A_Lambda = np.column_stack([operator.get_column(i) for i in Lambda_list])
        s_ls, _, _, _ = np.linalg.lstsq(A_Lambda, y, rcond=None)
        
        # 4. Residual Update
        r = y - (A_Lambda @ s_ls)
        
    # Build full output vector
    s_hat = np.zeros(n, dtype=operator.dtype)
    if s_ls.size > 0:
        s_hat[Lambda_list] = s_ls
        
    return s_hat

def kronecker_omp(Ds, Y, K, tol=1e-6, progress_callback=None):
    """
    Kronecker-OMP algorithm for sparse approximation of an N-dimensional tensor.

    This algorithm finds a sparse representation of a tensor Y using separable
    dictionaries along each mode.

    Parameters
    ----------
    Ds : list of ndarray
        List of N dictionaries. Each D_n has shape (I_n, M_n), where I_n matches
        the size of Y along mode n. Columns must be normalized.
    Y : ndarray
        Input tensor of shape (I_1, I_2, ..., I_N).
    K : int
        Maximum number of atoms (sparsity level).
    tol : float, optional
        Stopping threshold based on Frobenius norm of the residual.
    progress_callback : float, optional
        Updates an external progress bar.

    Returns
    -------
    X : ndarray
        Sparse N-way array.
    """

    # Step 1: Initialization
    shape = Y.shape
    N = len(shape)
    if len(Ds) != N:
        raise ValueError("Amount of dictionaries does not match the signal dimension")

    Is = []
    Ws = []
    for n in range(N):
        Is.append([])
        Ws.append(np.zeros(shape=(shape[n],K)))    
    a = None 
    R = Y.copy()
    Z_inv = None
    p = np.array([])

    # Step 2: Loop
    k = 1
    while k <= K and np.linalg.norm(R) > tol:

        # Step 3: Find atom indices with max correlation
        corr = R.copy()
        for n, D in enumerate(Ds):
            corr = util.mode_n_product(corr, D.T, n)
        indices = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
        
        # Step 4: Update indices lists and W matrices
        for n in range(N):
            Is[n].append(indices[n])
            Ws[n][:, k-1] = Ds[n][:, indices[n]]
        
        # Step 5: Compute coefficient vector
        ws = []
        for n in range(N):
            ws.append(Ws[n][:, k-1])

        p_temp = Y.copy()
        for n in range(N):
            p_temp = util.mode_n_product(p_temp, ws[n].reshape(1, -1), n)
        p = np.append(p, p_temp.item())

        if k == 1:
            Z_inv = np.array([[1.0]])
        else:
            b = np.ones(k-1)
            for n in range(N):
                b *= (Ws[n][:, :k-1].T @ ws[n])
            c = 1.0 - b.T @ b
            if c < 1e-12:
                c = 1e-12   
            d = -Z_inv @ b

            topleft = Z_inv + np.outer(d, d) / c
            topright = (d / c).reshape(-1, 1)
            bottomleft = (d.T / c).reshape(1, -1)
            bottomright = np.array([[1 / c]])

            Z_inv = np.block([[topleft, topright],
                              [bottomleft, bottomright]])
            
        a = Z_inv @ p

        # Step 6: Update residual
        Y_hat = np.zeros(shape)
        for j in range(k):
            rank1 = a[j]
            for n in range(N):
                v = Ws[n][:, j]
                rank1 = np.multiply.outer(rank1, v)
            Y_hat += rank1

        R = Y - Y_hat

        # Step 7: Advance k
        k += 1
        if progress_callback and (k % 5 == 0 or k == K):
            progress_callback(k / K)

    # Step 8: Compute X
    full_shape = [D.shape[1] for D in Ds]
    X = np.zeros(full_shape)
    for j in range(len(a)):
        coord = tuple(Is[n][j] for n in range(len(Is)))
        X[coord] = a[j]

    return X
        
def n_bomp(Ds, Y, K, tol=1e-6, progress_callback=None):
    """
    N-BOMP algorithm for sparse approximation of an N-dimensional tensor.

    Parameters
    ----------
    Ds : list of ndarray
        List of N dictionaries. Each D_n has shape (I_n, M_n), where I_n matches
        the size of Y along mode n. Columns must be normalized.
    Y : ndarray
        Input tensor of shape (I_1, I_2, ..., I_N).
    K : int
        Maximum number of atoms (sparsity level).
    tol : float, optional
        Stopping threshold based on Frobenius norm of the residual.
    progress_callback : float, optional
        Updates an external progress bar.

    Returns
    -------
    X : ndarray
        Sparse N-way array.
    """
    def _get_curr_k(Is):
        k = 1
        for I in Is:
            k *= len(I)
        return k

    # Step 1: Initialization
    shape = Y.shape
    N = len(shape)
    if len(Ds) != N:
        raise ValueError("Amount of dictionaries does not match the signal dimension")

    Is = []
    for n in range(N):
        Is.append([])  
    
    R = Y.copy()
    A = None
    # Step 2: Loop
    k = 1
    while _get_curr_k(Is) < K and np.linalg.norm(R) > tol:

        # Step 3: Find atom indices with max correlation
        corr = R.copy()
        for n, D in enumerate(Ds):
            corr = util.mode_n_product(corr, D.T, n)
        indices = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
        
        # Step 4: Update indices lists and B matrices
        Bs = []
        for n in range(N):
            if indices[n] not in Is[n]:
                Is[n].append(indices[n])
            B = Ds[n][:, Is[n]]
            Bs.append(B)
        
        # Step 5: Compute coefficient vector
        Z_prev = Y.copy()
        for n in range(N):
            B = Bs[n]
            G = B.T @ B

            Z_prev = util.mode_n_product(Z_prev, B.T, n)
            Z_prev_n = util.mode_n_unfold(Z_prev, n)
            
            L = np.linalg.cholesky(G + 1e-10*np.eye(G.shape[0]))
            Y_tmp = np.linalg.solve(L, Z_prev_n)
            Z_n = np.linalg.solve(L.T, Y_tmp)
            
            new_shape = list(Z_prev.shape)
            new_shape[n] = B.shape[1]
            Z_prev = util.mode_n_fold(Z_n, n, new_shape)

        A = Z_prev

        # Step 6: Update residual
        Y_hat = A
        for n in range(N):
            Y_hat = util.mode_n_product(Y_hat, Bs[n], n)
        
        R = Y - Y_hat

        # Step 7: Advance k
        k += 1
        if progress_callback:
            progress_callback(_get_curr_k(Is) / K)

    # Step 8: Compute X
    full_shape = [D.shape[1] for D in Ds]
    X = np.zeros(full_shape)
    grids = np.ix_(*Is)
    X[grids] = A
    if progress_callback:
        progress_callback(1.0)
    
    return X








