from __future__ import annotations

import json
import sys
import tempfile

from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


class CompressionController(QObject):
    """
    Runs compression jobs in a separate Python process.

    The controller owns:
    - QProcess lifecycle
    - temporary job file
    - stdout JSON parsing
    - abort/kill behavior

    The GUI owns:
    - selecting source items
    - displaying messages/progress
    - loading returned artifacts into the workspace
    """

    started = Signal()
    progress_changed = Signal(float)
    message_changed = Signal(str)
    failed = Signal(str)
    finished_payload = Signal(dict)
    run_ended = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.process: QProcess | None = None
        self.current_process_buffer = ""
        self.current_job_path: Path | None = None
        self.abort_requested = False

    @property
    def is_running(self) -> bool:
        return self.process is not None

    def start(
        self,
        source_path: Path,
        compressor_name: str,
        config_values: dict,
        experiment_settings: dict,
    ):
        if self.process is not None:
            self.failed.emit("A compression process is already running.")
            return

        job_file_path = self._write_job_file(
            source_path=source_path,
            compressor_name=compressor_name,
            config_values=config_values,
            experiment_settings=experiment_settings,
        )

        self.current_job_path = job_file_path
        self.current_process_buffer = ""
        self.abort_requested = False

        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(
            [
                "-m",
                "src.ui.gui.processes.compression_job",
                str(job_file_path),
            ]
        )

        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)

        self.started.emit()
        self.message_changed.emit("Starting...")

        self.process.start()

    def abort(self):
        if self.process is None:
            return

        self.abort_requested = True
        self.message_changed.emit("Aborting...")

        self.process.terminate()

        QTimer.singleShot(3000, self._kill_if_needed)

    def _write_job_file(
        self,
        source_path: Path,
        compressor_name: str,
        config_values: dict,
        experiment_settings: dict,
    ) -> Path:
        job = {
            "source_path": str(source_path),
            "compressor_name": compressor_name,
            "config_values": self._make_json_safe(config_values),
            "experiment_settings": experiment_settings,
        }

        job_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )

        with job_file:
            json.dump(job, job_file)

        return Path(job_file.name)

    def _make_json_safe(self, value):
        if isinstance(value, dict):
            return {
                key: self._make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, tuple):
            return [
                self._make_json_safe(item)
                for item in value
            ]

        if isinstance(value, list):
            return [
                self._make_json_safe(item)
                for item in value
            ]

        if hasattr(value, "name") and hasattr(value, "value"):
            return {
                "__enum__": value.__class__.__name__,
                "name": value.name,
            }

        return value

    def _on_stdout(self):
        if self.process is None:
            return

        data = bytes(
            self.process.readAllStandardOutput()
        ).decode("utf-8")

        self.current_process_buffer += data

        while "\n" in self.current_process_buffer:
            line, self.current_process_buffer = (
                self.current_process_buffer.split("\n", 1)
            )

            line = line.strip()

            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            self._handle_process_message(payload)

    def _handle_process_message(self, payload: dict):
        message_type = payload.get("type")

        if message_type == "progress":
            self.progress_changed.emit(float(payload.get("value", 0.0)))
            return

        if message_type == "message":
            self.message_changed.emit(payload.get("message", "Running"))
            return

        if message_type == "error":
            message = payload.get("message", "Unknown error")
            self.failed.emit(message)
            self.message_changed.emit("Failed")
            return

        if message_type == "finished":
            self.finished_payload.emit(payload)
            return

    def _on_stderr(self):
        if self.process is None:
            return

        data = bytes(
            self.process.readAllStandardError()
        ).decode("utf-8")

        if data.strip():
            print(data)

    def _on_error(self, error):
        self.failed.emit(f"Process error: {error}")
        self.message_changed.emit(f"Process error: {error}")

    def _on_finished(self, exit_code: int, exit_status):
        self.process = None

        self._cleanup_job_file()

        if self.abort_requested:
            status = "aborted"
            self.message_changed.emit("Aborted")

        elif exit_code == 0:
            status = "finished"
            self.progress_changed.emit(1.0)
            self.message_changed.emit("Finished")

        else:
            status = "failed"
            self.message_changed.emit("Failed")

        self.abort_requested = False
        self.run_ended.emit(status)

    def _kill_if_needed(self):
        if self.process is None:
            return

        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.message_changed.emit("Killed")

    def _cleanup_job_file(self):
        if self.current_job_path is None:
            return

        self.current_job_path.unlink(missing_ok=True)
        self.current_job_path = None