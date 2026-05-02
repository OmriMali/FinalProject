from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np


@dataclass
class DictionaryLearningItem:

    # Input
    Y: np.ndarray
    dict_name: str
    algorithm_name: str
    algorithm_params: Dict[str, Any]

    # Experiment info
    experiment_machine: str
    tag: Optional[str] = None

    # Outputs
    D: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    # Metadata
    timestamp: Optional[str] = None
    run_id: Optional[str] = None
    output_dir: Optional[Path] = None

    status: str = "initialized"

    # helpers
    def update_status(self, status: str):
        self.status = status

    def set_timestamp(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def update_id(self):
        if self.tag is None:
            self.run_id = f"dict_{self.dict_name}"
        else:
            self.run_id = f"dict_{self.dict_name}_{self.tag}"