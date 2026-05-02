import time
import numpy as np

from src.core.dictionary_learning_item import DictionaryLearningItem
from src.dictionary.k_svd import k_svd


class DictionaryLearner:
    def __init__(self, logger=None):
        self.logger = logger

    def run(self, item: DictionaryLearningItem):


        item.update_status("running")
        print(f"\n=== Learning dictionary: {item.dict_name} (K-SVD) ===")


        # 1. Run K-SVD
        start = time.perf_counter()

        D, X = k_svd(
            item.Y,
            **item.algorithm_params
        )

        train_time = time.perf_counter() - start

        # 2. Metrics
        item.update_status("evaluating")

        err = np.linalg.norm(item.Y - D @ X) / np.linalg.norm(item.Y)
        sparsity = np.mean(np.count_nonzero(X, axis=0))

        item.metrics = {
            "reconstruction_error": float(err),
            "mean_sparsity": float(sparsity),
            "train_time": train_time
        }

        print(f"Reconstruction Error: {err:.3e}")
        print(f"Mean Sparsity: {sparsity:.2f}")
        print(f"Training Time: {train_time:.2f}s")


        # 3. Save outputs
        item.D = D
        item.set_timestamp()
        item.update_id()

        # 4. Log results
        if self.logger:
            self.logger.log_dictionary(item)
        
        item.update_status("done")

        return item