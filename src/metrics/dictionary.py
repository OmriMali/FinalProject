import numpy as np

from src.metrics.base import Metric, MetricResult
from src.metrics.registry import register_metric
from src.core.results import DictionaryTrainingResult


@register_metric
class RepresentationError(Metric):
    """
    Representation Error.
    """
    name = "representation error"
    short_name = "REP_ERR"
    unit = "%"

    def compute(self, target: DictionaryTrainingResult) -> MetricResult:
        """
        Compute representation error between training signals and coefficients.

        Parameters
        ----------
        target : DictionaryTrainingResult
            Dictionary training result to evaluate.

        Returns
        -------
        MetricResult
            RepresentationError metric result.
        """
        value = 100 * np.linalg.norm(target.signals.data - target.dictionary.data @ target.coefficients) / np.linalg.norm(target.signals.data)

        return MetricResult(
            name=self.name,
            short_name=self.short_name,
            value=float(value),
            unit=self.unit,
        )
    
@register_metric
class MeanSparsity(Metric):
    """
    Mean sparsity of sparse coefficient vectors.
    """
    name = "mean sparsity"
    short_name = "MEAN_K"
    unit = None

    def compute(self, target: DictionaryTrainingResult) -> MetricResult:
        """
        Compute the average number of nonzero coefficients
        per sparse representation vector.

        Parameters
        ----------
        target : DictionaryTrainingResult
            Dictionary training result to evaluate.

        Returns
        -------
        MetricResult
            Mean sparsity metric result.
        """

        eps = 1e-10

        nonzero_counts = np.sum(
            np.abs(target.coefficients) > eps,
            axis=0
        )

        value = np.mean(nonzero_counts)

        return MetricResult(
            name=self.name,
            short_name=self.short_name,
            value=float(value),
            unit=self.unit,
        )
    
@register_metric
class DictionaryCoherence(Metric):
    """
    Mutual coherence of dictionary atoms.
    """
    name = "dictionary coherence"
    short_name = "MU"
    unit = None

    def compute(self, target: DictionaryTrainingResult) -> MetricResult:
        """
        Compute the mutual coherence of the dictionary.

        Mutual coherence is defined as the maximum absolute
        inner product between distinct normalized atoms.

        Parameters
        ----------
        target : DictionaryTrainingResult
            Dictionary training result to evaluate.

        Returns
        -------
        MetricResult
            Dictionary coherence metric result.
        """

        D = target.dictionary.data.astype(np.float64)

        norms = np.linalg.norm(D, axis=0, keepdims=True)
        norms[norms == 0] = 1.0

        D_norm = D / norms

        G = np.abs(D_norm.T @ D_norm)

        np.fill_diagonal(G, 0.0)

        value = np.max(G)

        return MetricResult(
            name=self.name,
            short_name=self.short_name,
            value=float(value),
            unit=self.unit,
        )