import numpy as np
from src import recovery_algorithms

def k_svd(Y, K, T_0, tol=1e-6, progress_callback=None):
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
        Stopping threshold based on Frobenius norm of the minimzation value.
    progress_callback : float, optional
        Updates an external progress bar.

    Returns
    -------
    D : ndarray
        The dictionary which sparsly represent the input signals, of shape (M, K)
    """

    
