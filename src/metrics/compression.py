import numpy as np

from src.metrics.base import Metric, MetricResult
from src.metrics.registry import register_metric
from src.core.results import CompressionRunResult


@register_metric
class RMSE(Metric):
    """
    Root mean squared error metric.
    """
    name = "Root Mean Squared Error"
    short_name = "RMSE"
    unit = None

    def compute(self, target: CompressionRunResult) -> MetricResult:
        """
        Compute RMSE between original and reconstructed HSI data.

        Parameters
        ----------
        target : CompressionRunResult
            Compression run result to evaluate.

        Returns
        -------
        MetricResult
            RMSE metric result.
        """
        diff = target.original.data.astype(float) - target.reconstructed.data.astype(float)
        value = np.sqrt(np.mean(diff ** 2))

        return MetricResult(
            name=self.name,
            short_name=self.short_name,
            value=float(value),
            unit=self.unit,
        )

@register_metric
class PSNR(Metric):
    """
    Peak signal-to-noise ratio metric.
    """
    name = "Peak Signal to Noise Ratio"
    short_name = "PSNR"
    unit = "dB"

    def compute(self, target: CompressionRunResult) -> MetricResult:
        """
        Compute PSNR between original and reconstructed HSI data.

        Parameters
        ----------
        target : CompressionRunResult
            Compression run result to evaluate.

        Returns
        -------
        MetricResult
            PSNR metric result.
        """
        diff = target.original.data.astype(float) - target.reconstructed.data.astype(float)
        rmse = np.sqrt(np.mean(diff ** 2))
        max_i = float((1 << target.original.metadata.bit_depth) - 1)
        if rmse == 0:
            value = float('inf')
        else:
            value = 20 * np.log10(max_i / rmse)

        return MetricResult(
            name=self.name,
            short_name=self.short_name,
            value=float(value),
            unit=self.unit,
        )
    
@register_metric
class SAM(Metric):
    """
    Spectral angle mapper metric.
    """
    name = "Mean Spectral Angle Map"
    short_name = "SAM"
    unit = "°"

    def compute(self, target: CompressionRunResult) -> MetricResult:
        """
        Compute mean spectral angle between original and reconstructed HSI data.

        Parameters
        ----------
        target : CompressionRunResult
            Compression run result to evaluate.

        Returns
        -------
        MetricResult
            SAM metric result in degrees.
        """
        ref = target.original.data.astype(float)
        tgt = target.reconstructed.data.astype(float)
        
        dot_product = np.sum(ref * tgt, axis=2)
        norm_ref = np.linalg.norm(ref, axis=2)
        norm_tgt = np.linalg.norm(tgt, axis=2)

        valid = (norm_ref > 0) & (norm_tgt > 0)
        if not np.any(valid):
            value = 0.0
        else:
            cos_theta = dot_product[valid] / (norm_ref[valid] * norm_tgt[valid])
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angles_rad = np.arccos(cos_theta)
            value = np.degrees(np.mean(angles_rad))

        return MetricResult(
            name=self.name,
            short_name=self.short_name,
            value=float(value),
            unit=self.unit,
        )

@register_metric
class CompressionRate(Metric):
    """
    Compression rate metric.
    """
    name = "Compression Rate"
    short_name = "CR"
    unit = None

    def compute(self, target: CompressionRunResult) -> MetricResult:
        """
        Compute compression rate as original size divided by compressed size.

        Parameters
        ----------
        target : CompressionRunResult
            Compression run result to evaluate.

        Returns
        -------
        MetricResult
            Compression rate metric result.
        """
        total_pixels = target.original.data.size
        original_bits = total_pixels * target.original.metadata.bit_depth        
        compressed_bits = len(target.compressed.bitstream) * 8
        value = original_bits / compressed_bits

        return MetricResult(
            name=self.name,
            short_name=self.short_name,
            value=float(value),
            unit=self.unit,
        )


DEFAULT_COMPRESSION_METRICS = [RMSE(), PSNR(), SAM(), CompressionRate()]