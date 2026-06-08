from src.core.dictionary import Axis


PHI_OPTIONS = [
    "SUBSAMPLING",
    "GAUSSIAN",
    "BERNOULLI",
]

PSI_OPTIONS = [
    "IDCT",
    "DCT",
    "LEARNED",
]

LOCAL_SUM_MODE_OPTIONS = [
    "column",
    "neighbor",
    "hybrid_mean",
]

AXIS_OPTIONS = {
    "Vertical": Axis.VERTICAL,
    "Horizontal": Axis.HORIZONTAL,
    "Spectral": Axis.SPECTRAL,
}