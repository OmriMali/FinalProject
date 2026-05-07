from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from src.core.dictionary import Dictionary
from src.core.training_signals import TrainingSignals



@dataclass(frozen=True)
class DictionaryTrainerConfig:
    """
    Base class for dictionary trainer configuration objects.
    """
    pass

class DictionaryTrainer(ABC):
    """
    Base interface for dictionary trainers.
    """

    name: str
    Config = DictionaryTrainerConfig

    def __init__(self, config: DictionaryTrainerConfig, progress_callback: Callable[[float], None] | None = None):
        self.config = config
        self._progress_callback = progress_callback

    @abstractmethod
    def fit(self, signals: TrainingSignals) -> Dictionary:
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
        raise NotImplementedError

    def report_progress(self, value: float):
        """
        Report training progress.
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("Progress value must be between 0.0 and 1.0")
        
        if self._progress_callback:
            self._progress_callback(value)