from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

@dataclass
class ExperimentItem:

    # identity
    task: str
    method: str
    config: Any

    # data
    data: Any

    # experiment specifics
    experiment_params: dict = field(default_factory=dict)
    machine: str = ""
    tag: str | None = None
    timestamp: str | None = None

    # outputs
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)

    # optional runtime info
    output_dir: Path | None = None