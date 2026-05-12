from dataclasses import dataclass


@dataclass(frozen=True)
class RunProgress:
    """
    Progress update emitted by the runner.

    Parameters
    ----------
    stage : str
        Current execution stage, for example ``"compression"``,
        ``"decompression"``, ``"metrics"``, or ``"dictionary_training"``.

    value : float
        Progress value in the range [0, 1].

    message : str | None, optional
        Optional human-readable progress message.
    """

    stage: str
    value: float
    message: str | None = None