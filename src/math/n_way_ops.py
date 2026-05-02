import numpy as np


def mode_n_product(X, U, n):
    """
    Mode-n tensor-matrix multiplication.

    X : ndarray, shape (I1, ..., In, ..., IN)
    U : ndarray, shape (J, In)
    n : int (mode, 0-based)

    Returns:
        Y : ndarray, shape (I1, ..., J, ..., IN)
    """
    # tensordot contracts axis n of X with axis 1 of U
    Y = np.tensordot(X, U, axes=([n], [1]))

    # Move the new axis (last) back to position n
    return np.moveaxis(Y, -1, n)

def mode_n_unfold(X, n):
    """
    Unfolds a tensor X into a matrix along mode n.
    
    X : ndarray, shape (I1, ..., In, ..., IN)
    n : int (0-based)
    
    Returns:
    X_n : ndarray, shape (In, -1)
    """
    return np.moveaxis(X, n, 0).reshape(X.shape[n], -1)

def mode_n_fold(X_n, n, original_shape):
    """
    Folds a mode-n matricized tensor back into its original shape.
    
    X_n : ndarray, the unfolded matrix
    n : int (0-based)
    original_shape : tuple, the shape of the tensor before unfolding
    
    Returns:
    X : ndarray, shape original_shape
    """
    reduced_shape = list(original_shape)
    mode_dim = reduced_shape.pop(n)
    intermediate_shape = [mode_dim] + reduced_shape
    X_inter = X_n.reshape(intermediate_shape)
    return np.moveaxis(X_inter, 0, n)

def generate_block_sparse_signal(D, S, N):
    """
    Generates a block-sparse signal Y based on the Tucker model.
    
    Parameters
    ----------
    D : ndarray
        The dictionary.
    S : int
        The number of active atoms in each dimension (sparsity per mode).
        The total number of non-zero coefficients will be S^N.
    N : int
        Number of dimensions.
        
    Returns
    -------
    Y : ndarray
        The synthesized signal of shape.
    X_gt : ndarray
        The ground truth sparse core tensor of shape.
    """
    M = D.shape[1]
    out_shape = []
    for n in range(N):
        out_shape.append(M)
    
    # 1. Randomly select S unique indices for each dimension
    active_indices = []
    for n in range(N):
        if S > M:
            raise ValueError(f"Sparsity S={S} cannot be greater than dictionary size M={M}")
        idx = np.sort(np.random.choice(M, S, replace=False))
        active_indices.append(idx)
        
    # 2. Create the core tensor
    X_gt = np.zeros(out_shape)
    
    # 3. Fill the S^N block with random coefficients (e.g., Gaussian)
    # np.ix_ creates a meshgrid of indices to address the specific block
    grid = np.ix_(*active_indices)
    X_gt[grid] = np.random.randn(*([S] * N))
    
    # 4. Synthesize the signal Y = X x1 D1 x2 D2 ... xN DN
    Y = X_gt.copy()
    for n in range(N):
        Y = mode_n_product(Y, D, n)
        
    return Y, X_gt
