import numpy as np
from dataclasses import dataclass, field



@dataclass(frozen=True)
class HSIMetadata:
    """
    Metadata describing a hyperspectral image.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the original hyperspectral image, as (height, width, bands).

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

    section_idx : int | None, optional
        In case the hsi was divided to sections, section number.

    attributes : dict, optional
        Additional dataset-specific metadata.
    """
    shape: tuple[int, int, int]
    wavelengths: np.ndarray
    bit_depth: int
    
    sensor: str | None = None
    scene_id: str | None = None
    scene_name: str | None = None
    section_idx: int | None = None

    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Validate object consistency after initialization.
        """
        if len(self.shape) != 3:
            raise ValueError("Shape must be (H, W, B)")
        
        if len(self.wavelengths) != self.shape[2]:
            raise ValueError("Number of wavelengths must match spectral bands")
        
        if self.bit_depth <= 0:
            raise ValueError("Bit depth must be positive")



@dataclass(frozen=True)
class HSI:
    """
    Immutable hyperspectral image container.

    Stores hyperspectral image data in (H, W, B) format together with
    metadata about the HSI.

    Parameters
    ----------
    data : np.ndarray
        Hyperspectral image cube with shape (height, width, bands).

    metadata : HSIMetadata
        Metadata about the HSI, like wavelengths, shape, scene_name...
    """
    data: np.ndarray
    metadata: HSIMetadata

    def __post_init__(self):
        """
        Validate object consistency after initialization.
        """
        if self.data.shape != self.metadata.shape:
            raise ValueError("Metadata shape does not match data shape")

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



@dataclass(frozen=True)
class CompressedHSI:
    """
    Compressed hyperspectral image representation.

    Parameters
    ----------
    bitstream : bytes
        Encoded binary representation of the hyperspectral image.

    metadata : HSIMetadata
        Metadata required for reconstruction.

    side_information : dict, optional
        Additional compressor-specific information about the encoded
        representation.
    """
    bitstream: bytes
    metadata: HSIMetadata

    side_information: dict = field(default_factory=dict)