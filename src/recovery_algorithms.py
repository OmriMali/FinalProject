import numpy as np
from src import util


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
        the size of Y along mode n.
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
    Is : list of lists
        Selected indices per mode. Is[n][k] is the index chosen from D_n at iteration k.
    a : ndarray
        Coefficient vector corresponding to selected atoms.
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

    return Is, a
        
