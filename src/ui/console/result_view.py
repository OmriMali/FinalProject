from src.compressors.base import Compressor
from src.core.hsi import HSI
from src.core.results import CompressionRunResult, DictionaryTrainingResult


def print_compression_run_header(
    hsi: HSI,
    compressor: Compressor,
) -> None:
    """
    Print a short summary before a compression run.
    """
    metadata = hsi.metadata

    print()
    print("=" * 60)
    print("COMPRESSION RUN")
    print("=" * 60)

    if metadata.scene_name is not None:
        print(f"Scene:\t\t{metadata.scene_name}")

    if metadata.section_idx is not None:
        print(f"Section:\t\t{metadata.scene_name}")

    if metadata.sensor is not None:
        print(f"Sensor:\t\t{metadata.sensor}")

    print(f"Shape:\t\t({metadata.shape[0]},{metadata.shape[1]},{metadata.shape[2]})")
    print(f"Compressor:\t{compressor.name}")
    print("=" * 60)
    print()


def print_compression_result(
    result: CompressionRunResult,
) -> None:
    """
    Print compression run metrics.
    """
    print()
    print("=" * 60)
    print("COMPRESSION RESULTS")
    print("=" * 60)

    for metric in result.metrics.values():
        value = f"{metric.value:.4f}"

        if metric.unit is not None:
            value += f"\t\t[{metric.unit}]"

        print(f"{metric.name:<25}\t{value}")

    print("=" * 60)
    print()