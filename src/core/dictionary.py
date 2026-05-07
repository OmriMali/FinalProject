import numpy as np
from dataclasses import dataclass
from enum import Enum

class Axis(Enum):
    """
    Signal axis used for dictionary learning.
    """

    VERTICAL = 0
    HORIZONTAL = 1
    SPECTRAL = 2
    

@dataclass(frozen=True)
class Dictionary:
    """
    Immutable learned dictionary object.

    Parameters
    ----------
    data : np.ndarray
        Dictionary matrix of shape
        (signal_length, num_atoms).

    axis : Axis
        Signal axis the dictionary was learned for.
    """
    data: np.ndarray
    axis: Axis

    def __post_init__(self):
        """
        Validate object consistency after initialization.
        """
        if self.data.ndim != 2:
            raise ValueError("Dictionary data must have shape "
                             "(signal_length, num_atoms)")

    @property
    def shape(self):
        """
        tuple[int, int] : Shape of the dictionary matrix.
        """
        return self.data.shape
    
    @property
    def atoms(self):
        """
        int : Number of atoms in the dictionary.
        """
        return self.data.shape[1]
    
    @property
    def signal_length(self):
        """
        int : Length of the axis the dictionary is inteded for.
        """
        return self.data.shape[0]