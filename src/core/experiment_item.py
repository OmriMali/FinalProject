from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
from src.core.hsi import HSI

VALID_STATUSES = {
    "initialized", "compressing", "reconstructing", "evaluating",
    "saving", "done", "error"
    }

@dataclass
class ExperimentItem:



    # Input
    hsi: HSI

    # Compressor
    compressor_name: str
    compressor_params: Dict[str, Any]

    # Experiment specifics
    save_hsi: bool = False
    ber: Optional[float] = None
    experiment_id: Optional[str] = None

    # HSI metadata
    sensor: Optional[str] = None
    site: Optional[str] = None
    name: Optional[str] = None

    # Compression Output
    bitstream: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None

    # Reconstruction
    reconstructed: Optional[HSI] = None

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Outputs
    output_dir: Optional[Path] = None

    # Status
    status: str = "initialized"

    def setup_output_dir(self, base_dir: Path):
        if self.experiment_id is None:
            raise ValueError("experiment_id must be set")
        
        self.output_dir = base_dir / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def update_status(self, status: str):
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        self.status = status


