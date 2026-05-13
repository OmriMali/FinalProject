from src.pipeline.callbacks import RunnerCallback
from src.core.results import CompressionRunResult, DictionaryTrainingResult


class ArtifactLoggerCallback(RunnerCallback):
    def __init__(self, root_dir: str,
                 save_reconstruction: bool = True,
                 save_compressed: bool = False,
                 save_dictionary: bool = True,
                 save_coefficients: bool = False):
        self.root_dir = root_dir
        self.save_reconstruction = save_reconstruction
        self.save_compressed = save_compressed
        self.save_dictionary = save_dictionary
        self.save_coefficients = save_coefficients

    def on_compression_end(self, result: CompressionRunResult):
        pass

    def on_dictionary_training_end(self, result: DictionaryTrainingResult):
        pass