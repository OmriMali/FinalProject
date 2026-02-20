from tqdm import tqdm


class ExperimentHandler:
    """
    Orchestrates running compressors on datasets and logging results.
    """
    def __init__(self, logger):
        self.logger = logger

    def run_experiment(self, compressor, dataset_name, hsi, **kwargs):
        """
        Runs a single compressor on a dataset and logs results.
        """
        # Run compression
        result = compressor.run(hsi, dataset_name=dataset_name, **kwargs)

        # Pass result to logger
        self.logger.log(
            dataset_name=dataset_name,
            compressor_name=result["name"],
            params=result["params"],
            metrics=result["metrics"],
            reconstructed=result.get("reconstructed", None),
            bitstream=result.get("bitstream", None)
        )
        return result
