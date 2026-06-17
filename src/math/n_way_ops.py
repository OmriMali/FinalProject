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