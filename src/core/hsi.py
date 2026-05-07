import numpy as np
from dataclasses import dataclass, field

@dataclass(frozen=True)
class HSI:

    data: np.ndarray
    wavelengths: np.ndarray
    bit_depth: int
    
    sensor: str | None = None
    scene_name: str | None = None
    section: str | None = None

    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.data.ndim != 3:
            raise ValueError("HSI data must have shape (H, W, B)")
        
        if len(self.wavelengths) != self.data.shape[2]:
            raise ValueError("Number of wavelengths must match spectral bands")

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def spatial_shape(self):
        return self.data.shape[:2]
    
    @property
    def bands(self):
        return self.data.shape[2]