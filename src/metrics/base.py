from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricResult:
    """
    Result produced by a metric.

    Parameters
    ----------
    name : str
        Name of the metric.
    short_name : str
        Shortend version of metric name.
    value : float
        Value of computed metric.
    unit : str | None
        Metric units.
    """
    name: str
    short_name: str
    value: float
    unit: str | None = None

class Metric(ABC):
    """
    Base interface for all metrics.
    
    Parameters
    ----------
    name : str
        Name of the metric.
    short_name : str
        Shortend version of metric name.
    value : float
        Value of computed metric.
    unit : str | None
        Metric units.
    """
    name: str
    short_name: str
    unit: str | None = None

    @abstractmethod
    def compute(self, target: Any) -> MetricResult:
        """
        Compute the metric for a result object.
        """
        raise NotImplementedError