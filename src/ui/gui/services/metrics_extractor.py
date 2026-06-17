from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.hsi import HSI


@dataclass(frozen=True)
class LoadedMetric:
    """
    Lightweight metric object for metrics loaded from logs.
    """

    value: Any
    unit: str = ""


class MetricsExtractor:
    """
    Extract metrics for loaded reconstructed HSIs.
    """

    METRIC_ALIASES = {
        "RMSE": ("RMSE", "rmse"),
        "PSNR": ("PSNR", "psnr"),
        "SAM": ("SAM", "sam"),
        "CR": ("CR", "cr"),
        "comp_time": ("comp_time", "compression_time"),
        "decomp_time": ("decomp_time", "decompression_time"),
    }

    METRIC_UNITS = {
        "RMSE": "",
        "PSNR": "dB",
        "SAM": "deg",
        "CR": "",
        "comp_time": "s",
        "decomp_time": "s",
    }

    def extract_for_hsi(
        self,
        hsi: HSI,
        path: Path | None,
    ) -> dict[str, LoadedMetric] | None:
        metrics = self._extract_from_metadata(hsi)

        if metrics:
            return metrics

        if path is None:
            return None

        return self._extract_from_log(path)

    def _extract_from_metadata(
        self,
        hsi: HSI,
    ) -> dict[str, LoadedMetric] | None:
        attrs = hsi.metadata.attributes

        if "metrics" in attrs and isinstance(attrs["metrics"], dict):
            return {
                name: LoadedMetric(
                    value=getattr(value, "value", value),
                    unit=getattr(value, "unit", self.METRIC_UNITS.get(name, "")),
                )
                for name, value in attrs["metrics"].items()
            }

        found = {}

        for metric_name, aliases in self.METRIC_ALIASES.items():
            for alias in aliases:
                if alias not in attrs:
                    continue

                found[metric_name] = LoadedMetric(
                    value=attrs[alias],
                    unit=self.METRIC_UNITS.get(metric_name, ""),
                )
                break

        return found or None

    def _extract_from_log(
        self,
        reconstructed_path: Path,
    ) -> dict[str, LoadedMetric] | None:
        artifact_dir = reconstructed_path.parent
        log_path = self._find_nearby_log_path(reconstructed_path)

        if log_path is None:
            return None

        df = pd.read_csv(log_path)

        if "artifact_dir" not in df.columns:
            return None

        artifact_dir_text = str(artifact_dir)

        matching = df[
            df["artifact_dir"].astype(str).apply(
                lambda value: self._same_or_endswith(value, artifact_dir_text)
            )
        ]

        if matching.empty:
            return None

        row = matching.iloc[-1]

        metrics = {}

        for metric_name, aliases in self.METRIC_ALIASES.items():
            value = self._get_row_value_by_alias(row, aliases)

            if value is None:
                continue

            metrics[metric_name] = LoadedMetric(
                value=value,
                unit=self.METRIC_UNITS.get(metric_name, ""),
            )

        return metrics or None

    def _find_nearby_log_path(self, path: Path) -> Path | None:
        for parent in path.parents:
            candidate = parent / "log.csv"

            if candidate.exists():
                return candidate

        return None

    def _same_or_endswith(
        self,
        log_value: str,
        artifact_dir: str,
    ) -> bool:
        log_path = Path(log_value)

        if log_path == Path(artifact_dir):
            return True

        return artifact_dir.endswith(str(log_path)) or log_value.endswith(
            Path(artifact_dir).name
        )
    
    def _get_row_value_by_alias(
        self,
        row,
        aliases: tuple[str, ...],
    ):
        for alias in aliases:
            if alias in row.index:
                value = row[alias]

                if pd.isna(value):
                    return None

                return value

        return None