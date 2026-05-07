import numpy as np
from dataclasses import dataclass, field

@dataclass(frozen=True)
class HSI:
    """
    Immutable hyperspectral image container.

    Stores hyperspectral image data in (H, W, B) format together with
    spectral wavelength information and acquisition metadata.

    Parameters
    ----------
    data : np.ndarray
        Hyperspectral image cube with shape (height, width, bands).

    wavelengths : np.ndarray
        Wavelength corresponding to each spectral band, in nanometers.

    bit_depth : int
        Number of bits used to represent each sample.

    sensor : str | None, optional
        Sensor or acquisition system name.

    scene_id : str | None, optional
        Scene identifier used by the original dataset.
    
    scene_name : str | None, optional
        Human-readable scene name.

    section : int | None, optional
        In case the hsi was divided to section, section number

    metadata : dict, optional
        Additional dataset-specific metadata.
    """
    data: np.ndarray
    wavelengths: np.ndarray
    bit_depth: int
    
    sensor: str | None = None
    scene_id: str | None = None
    scene_name: str | None = None
    section: int | None = None

    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Validate object consistency after initialization.
        """
        if self.data.ndim != 3:
            raise ValueError("HSI data must have shape (H, W, B)")
        
        if len(self.wavelengths) != self.data.shape[2]:
            raise ValueError("Number of wavelengths must match spectral bands")

    @property
    def shape(self):
        """
        tuple[int, int, int] : Shape of the hyperspectral cube.
        """
        return self.data.shape
    
    @property
    def spatial_shape(self):
        """
        tuple[int, int] : Spatial shape of the hyperspectral image.
        """
        return self.data.shape[:2]
    
    @property
    def bands(self):
        """
        int : Number of spectral bands.
        """
        return self.data.shape[2]