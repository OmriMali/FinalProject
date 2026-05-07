import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.core.dictionary import Dictionary



@dataclass(frozen=True)
class LearnerConfig:
    pass

class DictionaryLearner(ABC):
    def __init__(self, config: LearnerConfig, progress_callback):
        self.config = config
        self.progress_callback = progress_callback

    @abstractmethod
    def run(self, Y: np.ndarray) -> Dictionary:
        pass

    @abstractmethod
    def update_progress(self, value):
        if self.progress_callback:
            self.progress_callback(value)