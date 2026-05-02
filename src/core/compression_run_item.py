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
class CompressionRunItem:
    
    # Provided on initialization
    hsi: HSI

    compressor_name: str
    compressor_params: Dict[str, Any]

    experiment_machine: str
    save_hsi: bool = False
    ber: float = 0
    tag: Optional[str] = None

    # Updated during experiment
    bitstream: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None

    reconstructed: Optional[HSI] = None

    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[str] = None
    output_dir: Optional[Path] = None

    run_id: Optional[str] = None

    status: str = "initialized"


    def update_status(self, status: str):
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        self.status = status

    def update_id(self):
        if self.timestamp is None:
            raise ValueError(f"No timestamp generated")
        
        if self.tag is None:
            self.run_id = f"run_{self.timestamp}"
        else:
            self.run_id = f"run_{self.tag}_{self.timestamp}"



