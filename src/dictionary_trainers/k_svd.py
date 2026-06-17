import numpy as np
from dataclasses import dataclass

from src.dictionary_trainers.base import DictionaryTrainer, DictionaryTrainerConfig
from src.dictionary_trainers.registry import register_dictionary_trainer
from src.core.dictionary import Dictionary
from src.core.training_signals import TrainingSignals
from src.math import regression_algs



@dataclass(frozen=True)
class K_SVDConfig(DictionaryTrainerConfig):
    """
    Configuration for K_SVD dictionary learner.

    Parameters
    ----------
    K : int
        Number of atoms in output dictionary.

    T_0 : int
        Target sparsity level for the input signals.

    tol : float
        Stopping threshold based on relative error of Y - DX.
    
    max_iter : int
        Stopping condition based on maximum iterations.
    """
    K: int
    T_0: int
    tol: float = 1e-2
    max_iter: int = 50

@register_dictionary_trainer
class K_SVD(DictionaryTrainer):
    """
    Trains a dictionary with the K-SVD algorithm.
    """
    name = "ksvd"
    Config = K_SVDConfig

    def __init__(self, config: K_SVDConfig, progress_callback=None):
        super().__init__(config, progress_callback)
        self._validate_config()
    
    def _validate_config(self) -> None:
        if self.config.K <= 0:
            raise ValueError("K must be positive")
        if self.config.T_0 <= 0:
            raise ValueError("T_0 must be positive")
        if self.config.T_0 > self.config.K:
            raise ValueError("T_0 cannot be larger than K")
        if self.config.tol <= 0:
            raise ValueError("tol must be positive")
        if self.config.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        
    def _validate_input(self, signals: TrainingSignals) -> None:
        if signals.data.ndim != 2:
            raise ValueError("Training signals must have shape (signal_length, num_signals)")
        if signals.num_signals < self.config.K:
            raise ValueError("Number of training signals must be at least K")


    def fit(self, signals: TrainingSignals):
        """
        Train a dictionary on input signals.

        Parameters
        ----------
        signals : TrainingSignals
            Training signals for the dictionary.

        Returns
        -------
        Dictionary
            Trained dictionary.
        """
        self._validate_input(signals)
        self.report_progress(0.0)

        # 1. Initialization
        M, N = signals.data.shape

        # # 1.a. trivial initialization
        # D = np.copy(signals.data[:, :self.config.K])
        # D /= np.linalg.norm(D, axis=0, keepdims=True)
        # self.report_progress(0.05)

        # 1.b. svd initialization
        U, _, _ = np.linalg.svd(signals.data, full_matrices=False)
        atoms_from_svd = min(M, self.config.K)
        D = np.zeros((M, self.config.K))
        D[:, :atoms_from_svd] = U[:, :atoms_from_svd]
        if self.config.K > atoms_from_svd:
            idx = np.random.choice(signals.data.shape[1], self.config.K - M, replace=False)
            D[:, M:] = signals.data[:, idx]
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        norms[norms == 0] = 1
        D /= norms
        self.report_progress(0.05)

        # 2. Loop
        X = np.zeros((self.config.K, N))
        Y_norms = np.linalg.norm(signals.data)
        J = 1
        while J <= self.config.max_iter:
            # 3. Sparse Coding
            for i in range(N):
                y = signals.data[:, i].copy()
                X[:, i] = regression_algs.omp(D, y, self.config.T_0)
            
            # 4. Codebook Update
            for k in range(self.config.K):
                # 4.a. Find the signals that use the current atom
                eps = 1e-10
                omega_k = np.where(np.abs(X[k, :]) > eps)[0]
                
                # (Optional) Replace dead atoms with the worst represented signal
                if len(omega_k) == 0:
                    R = signals.data - D @ X
                    errors = np.linalg.norm(R, axis=0)
                    worst_idx = np.argmax(errors)
                    norm = np.linalg.norm(signals.data[:, worst_idx])
                    if norm == 0:
                        continue
                    D[:, k] = signals.data[:, worst_idx] / norm
                    X[k, :] = 0
                    X[k, worst_idx] = np.dot(D[:, k], signals.data[:, worst_idx])
                    continue
                
                # 4.b. Compute the overall error matrix
                E_k = signals.data - D @ X + np.outer(D[:, k], X[k, :])

                # 4.c. Obtain the restricted error matrix
                E_k_R = E_k[:, omega_k]

                # 4.d. Apply SVD and obtain dictionary column and coefficent vector
                U, S, V_T = np.linalg.svd(E_k_R, full_matrices=False)
                D[:, k] = U[:, 0].copy()
                X[k, omega_k] = S[0] * V_T[0, :].T
            
            self.report_progress(J / self.config.max_iter)

            # 5. Check for stop condition, increment loop
            err = np.linalg.norm(signals.data - D @ X) / Y_norms
            if err < self.config.tol:
                break
            J += 1

        # 6. Create output object
        dictionary = Dictionary(D, signals.axis, name=self.name)
        self.report_progress(1.0)

        return dictionary, X