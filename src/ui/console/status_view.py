from tqdm import tqdm

from src.pipeline.status import RunStatus, RunEventType


class ConsoleStatusView:
    """
    Console-based status view for runner events.

    Handles runner status events by printing messages and displaying
    progress bars for long-running stages.
    """
    def __init__(self):
        self._current_stage: str | None = None
        self._bar: tqdm | None = None

    def handle(self, status: RunStatus) -> None:
        """
        Handle a runner status event.

        Parameters
        ----------
        status : RunStatus
            Status event emitted by the runner.
        """
        if status.event_type == RunEventType.MESSAGE:
            self._handle_message(status)

        elif status.event_type == RunEventType.PROGRESS_START:
            self._handle_progress_start(status)

        elif status.event_type == RunEventType.PROGRESS_UPDATE:
            self._handle_progress_update(status)

        elif status.event_type == RunEventType.PROGRESS_END:
            self._handle_progress_end(status)

        elif status.event_type == RunEventType.DONE:
            self._handle_done(status)

        elif status.event_type == RunEventType.ERROR:
            self._handle_error(status)

    def _handle_message(self, status: RunStatus) -> None:
        """
        Print a normal status message.
        """
        if status.message:
            print(status.message)

    def _handle_progress_start(self, status: RunStatus) -> None:
        """
        Start a progress bar for a stage.
        """
        self._close_bar()

        self._current_stage = status.stage

        self._bar = tqdm(
            total=100,
            desc=status.message or status.stage,
            unit="%",
        )

    def _handle_progress_update(self, status: RunStatus) -> None:
        """
        Update the active progress bar.
        """
        if self._bar is None:
            return

        if status.stage != self._current_stage:
            return

        if status.progress is None:
            return

        target = int(max(0.0, min(1.0, status.progress)) * 100)
        delta = target - self._bar.n

        if delta > 0:
            self._bar.update(delta)

    def _handle_progress_end(self, status: RunStatus) -> None:
        """
        Complete and close the active progress bar.
        """
        if self._bar is not None:
            delta = 100 - self._bar.n
            if delta > 0:
                self._bar.update(delta)

        self._close_bar()

        if status.message:
            print(status.message)

    def _handle_done(self, status: RunStatus) -> None:
        """
        Handle a completed run event.
        """
        self._close_bar()

        if status.message:
            print(status.message)

    def _handle_error(self, status: RunStatus) -> None:
        """
        Handle an error event.
        """
        self._close_bar()

        if status.message:
            print(f"ERROR: {status.message}")

    def _close_bar(self) -> None:
        """
        Close the active progress bar.
        """
        if self._bar is not None:
            self._bar.close()
            self._bar = None

        self._current_stage = None