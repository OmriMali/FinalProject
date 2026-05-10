from dataclasses import dataclass
from enum import Enum
from typing import Callable

class RunEventType(Enum):
    """
    Type of runner status event.
    """
    MESSAGE = "message"
    PROGRESS_START = "progress_start"
    PROGRESS_UPDATE = "progress_update"
    PROGRESS_END = "progress_end"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class RunStatus:
    """
    Status event emitted by the runner.

    Parameters
    ----------
    event_type : RunEventType
        Type of status event.

    stage : str
        Current execution stage, for example ``"compression"``,
        ``"decompression"``, ``"metrics"``, or ``"done"``.

    message : str | None, optional
        Optional human-readable status message.

    progress : float | None, optional
        Progress value in the range [0, 1]. Only relevant for
        progress update events.
    """
    event_type: RunEventType
    stage: str
    message: str | None = None
    progress: float | None = None


StatusCallback = Callable[[RunStatus], None]