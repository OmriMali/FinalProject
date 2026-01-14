import numpy as np
from scipy.fft import fft, ifft

def gOMP(y, A, K, G, eps, N):
    """
    y: measurements (M,)
    A: 
    K: Sparsity parameter (final number of non-zero coefficients)
    G: Number of atoms to pick per iteration
    eps: Error tolerance
    N: Total number of pixels in the band
    """
    
    # Initialization
    res = y.copy()        # r0 = y
    support = []          # c0 = empty
    x_output = np.zeros(N, dtype=complex)
    delta = 1.0
    
    while delta >= eps:
        res_old = res.copy()
        
        # 1. Match: Project residual onto atoms
        # p = A* @ r_{i-1}
        p = A.conj().T @ res
        
        # Identify G largest indexes
        new_indices = np.argsort(np.abs(p))[-G:]
        support = list(set(support) | set(new_indices)) # c_i = Theta_i U c_{i-1}
        
        # 2. Estimate: Least Squares on the support
        Bi = A[:, support]
        # s_i = Bi+ @ y (Pseudo-inverse)
        si = np.linalg.lstsq(Bi, y, rcond=None)[0]
        
        # 3. Prune: Keep only K strongest (to ensure K-sparsity)
        x_temp = np.zeros(N, dtype=complex)
        x_temp[support] = si
        
        q_i = np.argsort(np.abs(x_temp))[-K:] # q_i = argmax_K(|x|)
        
        # Final update for this iteration
        Bi_final = A[:, q_i]
        si_final = np.linalg.lstsq(Bi_final, y, rcond=None)[0]
        
        x_output = np.zeros(N, dtype=complex)
        x_output[q_i] = si_final
        
        # 4. Update Residual
        res = y - A @ x_output
        delta = np.linalg.norm(res - res_old)
        
    # Return to spatial domain
    return x_output