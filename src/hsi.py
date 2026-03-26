import numpy as np
import math

class HSI:
    """
    Hyperspectral Image (HSI) data container.

    Stores raw hyperspectral data in (H, W, B) format and provides
    utilities for normalization, visualization, and tensor operations.

    The class is immutable: all transformations return new HSI objects.
    """
    def __init__(self, data, wavelengths, dtype=None, metadata=None):
        """
        Initialize HSI from raw data.

        Parameters
        ----------
        data : np.ndarray
            Hyperspectral cube of shape (H, W, B), raw values.
        wavelengths : np.ndarray
            Array of shape (B,) representing wavelengths in nm.
        dtype : np.dtype, optional
            Original container dtype. If None, inferred from data.
        metadata : dict, optional
            Additional metadata (e.g., dataset name, sensor).
        """
        data = np.asarray(data)
        wavelengths = np.asarray(wavelengths)

        self._validate_inputs(data, wavelengths)

        self._data = data.copy()
        self._wavelengths = wavelengths.copy()

        self._dtype = dtype if dtype is not None else data.dtype

        self._metadata = metadata.copy() if metadata is not None else {}

        self._min = None
        self._max = None
        self._compute_min_max()


    @classmethod
    def from_normalized(cls, data: np.ndarray, rec_dict: dict):
        """
        Create an HSI object from normalized data using a reconstruction dict.
        
        Parameters
        ----------
        data : np.ndarray
            Normalized HSI data in [0,1] float.
        rec_dict : dict
            Dictionary containing min, max, dtype, wavelengths, metadata

        Returns
        -------
        HSI
        """
        return cls(
            data=(data * (rec_dict["max"] - rec_dict["min"]) + rec_dict["min"]),
            dtype=rec_dict["dtype"],
            wavelengths=rec_dict["wavelengths"],
            metadata=rec_dict["metadata"]
        )
    
    def to_dict(self) -> dict:
        """
        Returns a dictionary containing all information needed to
        reconstruct the HSI from normalized data using from_normalized().
        """
        return {
            "min": self.min,
            "max": self.max,
            "dtype": self.dtype,
            "bitdepth": self.bitdepth,
            "shape": self.shape,
            "wavelengths": self.wavelengths,
            "metadata": self.metadata
        }

    # ===== Internal Helpers ===== #

    def _validate_inputs(self, data, wavelengths):
        """Ensure shape consistency."""
        if data.ndim != 3:
            raise ValueError(f"data must be 3D (H, W, B), got shape {data.shape}")
        
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1D array")
        
        if data.shape[2] != len(wavelengths):
            raise ValueError(
                f"Mismatch: data has {data.shape[2]} bands but "
                f"{len(wavelengths)} wavelengths provided")
        
        if len(wavelengths) == 0:
            raise ValueError("wavelengths cannot be empty")

    def _compute_min_max(self):
        """Compute and cache min/max."""
        self._min = self._data.min()
        self._max = self._data.max()

    # ===== Core Properties ===== #

    @property
    def data(self):
        """Return a copy of raw data (H, W, B)."""
        return self._data.copy()
    
    @property
    def wavelengths(self):
        """Return a copy of wavelengths array."""
        return self._wavelengths.copy()
    
    @property
    def dtype(self):
        """Original container dtype."""
        return self._dtype

    @property
    def metadata(self):
        """Metadata dictionary (shallow copy)."""
        return self._metadata.copy()
    
    # ===== Shape & Dimensions ===== #

    @property
    def shape(self):
        """Return (H, W, B)."""
        return self._data.shape

    @property
    def height(self):
        """Return H."""
        return self._data.shape[0]

    @property
    def width(self):
        """Return W."""
        return self._data.shape[1]
    
    @property
    def bands(self):
        """Return B."""
        return self._data.shape[2]
    
    @property
    def num_pixels(self):
        """Return H * W."""
        return self.height * self.width

    @property
    def size(self):
        """Return H * W * B."""
        return self.height * self.width * self.bands
    
    # ===== Statistics ===== #

    @property
    def min(self):
        return self._min
    
    @property
    def max(self):
        return self._max
    
    @property
    def bitdepth(self):
        """
        Minimal number of bits required to represent the data span:
            ceil(log2(max - min + 1))
        """
        span = int(self._max) - int(self._min)
        if span == 0:
            return 1  # edge case: constant image
        return math.ceil(math.log2(span + 1))

    # ===== Data Access Methods ===== #

    def get_norm_data(self):
        """
        Return globally normalized data (H, W, B) in [0,1] as float32.
        """
        if self._max == self._min:
            return np.zeros_like(self._data, dtype=np.float32)
        
        norm = (self._data.astype(np.float32) - self._min) / (self._max - self._min)
        return norm

    def get_band(self, idx, normalize=True):
        """
        Return a single band as (H, W).

        Parameters
        ----------
        idx : int
            Band index.
        normalize : bool
            If True, return normalized [0,1].
        """
        if not (0 <= idx < self.bands):
            raise IndexError(f"Band index {idx} out of range [0, {self.bands})")

        band = self._data[:, :, idx]

        if not normalize:
            return band.copy()

        if self._max == self._min:
            return np.zeros((self.height, self.width), dtype=np.float32)

        band = band.astype(np.float32)
        band = (band - self._min) / (self._max - self._min)
        return band

    # ===== Visualization Methods ===== #

    def get_rgb(self, bands=None, percentiles=(2, 98)):
        """
        Return RGB image for visualization.

        Parameters
        ----------
        bands : tuple of 3 ints, optional
            (R, G, B) band indices. If None, automatically selected.
        percentiles : tuple
            Percentile stretch (low, high) for contrast enhancement.

        Returns
        -------
        rgb : np.ndarray
            (H, W, 3) float32 image in [0,1]
        band_indices : tuple
        wavelengths_used : tuple
        """
        if bands is None:
            target_wavelengths = [650, 550, 450]

            band_indices = tuple(
                int(np.argmin(np.abs(self._wavelengths - wl)))
                for wl in target_wavelengths
            )
        else:
            if len(bands) != 3:
                raise ValueError("bands must be a tuple of 3 indices (R, G, B)")

            for b in bands:
                if not (0 <= b < self.bands):
                    raise IndexError(f"Band index {b} out of range")

            band_indices = tuple(bands)

        rgb = np.stack(
            [self._data[:, :, i] for i in band_indices],
            axis=2
        ).astype(np.float32)

        # Apply percentile stretching per channel
        low_p, high_p = percentiles

        for c in range(3):
            channel = rgb[:, :, c]

            lo = np.percentile(channel, low_p)
            hi = np.percentile(channel, high_p)

            if hi == lo:
                rgb[:, :, c] = 0.0
            else:
                channel = (channel - lo) / (hi - lo)
                channel = np.clip(channel, 0.0, 1.0)
                rgb[:, :, c] = channel

        wavelengths_used = tuple(self._wavelengths[i] for i in band_indices)

        return rgb, band_indices, wavelengths_used

    # ===== Structural Transformations ===== #

    def subset_bands(self, indices):
        """
        Return a new HSI with selected spectral bands.

        Parameters
        ----------
        indices : array-like
            Indices of bands to keep.

        Returns
        -------
        HSI
            New HSI object with selected bands.
        """
        indices = np.asarray(indices)

        if indices.ndim != 1:
            raise ValueError("indices must be a 1D array")

        if len(indices) == 0:
            raise ValueError("indices cannot be empty")

        if np.any(indices < 0) or np.any(indices >= self.bands):
            raise IndexError("Band indices out of range")

        new_data = self._data[:, :, indices]
        new_wavelengths = self._wavelengths[indices]

        new_metadata = self._metadata.copy()

        return HSI(new_data, new_wavelengths, dtype=self._dtype, metadata=new_metadata)

    def crop(self, y_range, x_range):
        """
        Return a spatially cropped HSI.

        Parameters
        ----------
        y_range : tuple (start, end)
            Vertical slice (rows)
        x_range : tuple (start, end)
            Horizontal slice (cols)

        Returns
        -------
        HSI
            Cropped HSI object.
        """
        y0, y1 = y_range
        x0, x1 = x_range

        if not (0 <= y0 < y1 <= self.height):
            raise ValueError(f"Invalid y_range {y_range}")

        if not (0 <= x0 < x1 <= self.width):
            raise ValueError(f"Invalid x_range {x_range}")

        new_data = self._data[y0:y1, x0:x1, :]

        new_wavelengths = self._wavelengths
        new_metadata = self._metadata.copy()

        return HSI(new_data, new_wavelengths, dtype=self._dtype, metadata=new_metadata)

