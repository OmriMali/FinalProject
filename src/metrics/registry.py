from src.metrics.base import Metric


_METRICS: dict[str, type[Metric]] = {}


def register_metric(cls: type[Metric]) -> type[Metric]:
    short_name = getattr(cls, "short_name", None)

    if not short_name:
        raise ValueError("Metric class must define a non-empty short_name")

    if short_name in _METRICS:
        raise ValueError(f"Metric already registered: {short_name}")

    _METRICS[short_name] = cls

    return cls


def get_metric(short_name: str) -> type[Metric]:
    if short_name not in _METRICS:
        raise KeyError(f"Unknown metric: {short_name}")

    return _METRICS[short_name]


def available_metrics() -> list[str]:
    return sorted(_METRICS.keys())