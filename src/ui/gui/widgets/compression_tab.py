from __future__ import annotations

from dataclasses import fields, is_dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from src.compressors.registry import get_compressor, list_compressors
from src.ui.gui.widgets.config_widgets import create_config_widget, read_widget_value


class CompressionTab(QWidget):
    """
    Compression run tab.

    Owns experiment settings, compressor settings, run buttons,
    and progress display.
    """

    compress_decompress_requested = Signal()
    abort_requested = Signal()

    def __init__(self):
        super().__init__()

        self.config_widgets = {}

        self._can_compress = False
        self._can_decompress = False
        self._can_compress_decompress = False
        self._is_running = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_experiment_panel())
        layout.addWidget(self._build_compressor_panel(), stretch=1)
        layout.addWidget(self._build_run_actions_panel())

    def _build_experiment_panel(self) -> QGroupBox:
        box = QGroupBox("Experiment Settings")
        layout = QFormLayout(box)

        self.experiment_edit = QLineEdit("gui_test")

        self.ber_spin = QDoubleSpinBox()
        self.ber_spin.setRange(0.0, 1.0)
        self.ber_spin.setDecimals(6)
        self.ber_spin.setSingleStep(0.001)
        self.ber_spin.setValue(0.0)

        self.results_dir_edit = QLineEdit("results")
        self.browse_results_dir_button = QPushButton("Browse")
        self.browse_results_dir_button.clicked.connect(
            self._on_browse_results_dir
        )

        results_dir_widget = QWidget()
        results_dir_layout = QHBoxLayout(results_dir_widget)
        results_dir_layout.setContentsMargins(0, 0, 0, 0)
        results_dir_layout.addWidget(self.results_dir_edit)
        results_dir_layout.addWidget(self.browse_results_dir_button)

        self.save_reconstructed_check = QCheckBox()
        self.save_reconstructed_check.setChecked(True)

        self.save_compressed_check = QCheckBox()
        self.save_compressed_check.setChecked(False)

        self.save_config_check = QCheckBox()
        self.save_config_check.setChecked(False)

        self.save_metadata_check = QCheckBox()
        self.save_metadata_check.setChecked(False)

        layout.addRow("Experiment", self.experiment_edit)
        layout.addRow("BER", self.ber_spin)
        layout.addRow("Results directory", results_dir_widget)
        layout.addRow("Save reconstructed", self.save_reconstructed_check)
        layout.addRow("Save compressed", self.save_compressed_check)
        layout.addRow("Save config", self.save_config_check)
        layout.addRow("Save metadata", self.save_metadata_check)

        return box

    def _build_compressor_panel(self) -> QGroupBox:
        box = QGroupBox("Compressor Settings")
        layout = QVBoxLayout(box)

        compressor_form = QFormLayout()

        self.compressor_combo = QComboBox()
        self.compressor_combo.addItems(list_compressors())
        self.compressor_combo.currentTextChanged.connect(
            self._on_compressor_changed
        )

        compressor_form.addRow("Compressor", self.compressor_combo)
        layout.addLayout(compressor_form)

        layout.addWidget(self._horizontal_separator())

        self.compressor_params_form = QFormLayout()

        layout.addLayout(self.compressor_params_form)
        layout.addStretch()

        self._on_compressor_changed(self.compressor_combo.currentText())

        return box

    def _build_run_actions_panel(self) -> QGroupBox:
        box = QGroupBox("Run Actions")
        layout = QVBoxLayout(box)

        button_layout = QHBoxLayout()

        self.compress_button = QPushButton("Compress")
        self.decompress_button = QPushButton("Decompress")
        self.compress_decompress_button = QPushButton("Compress + Decompress")
        self.abort_button = QPushButton("Abort")

        self.compress_decompress_button.clicked.connect(
            self.compress_decompress_requested.emit
        )
        self.abort_button.clicked.connect(self.abort_requested.emit)

        button_layout.addWidget(self.compress_button)
        button_layout.addWidget(self.decompress_button)
        button_layout.addWidget(self.compress_decompress_button)
        button_layout.addWidget(self.abort_button)

        self.run_progress_bar = QProgressBar()
        self.run_progress_bar.setRange(0, 100)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setTextVisible(True)
        self.run_progress_bar.setFormat("%p%")

        self.run_status_label = QLabel("Ready")
        self.run_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addLayout(button_layout)
        layout.addWidget(self.run_progress_bar)
        layout.addWidget(self.run_status_label)

        self._refresh_run_buttons()

        return box

    def read_experiment_settings(self) -> dict:
        experiment = self.experiment_edit.text().strip()
        results_dir = self.results_dir_edit.text().strip()

        if not experiment:
            raise ValueError("Experiment name cannot be empty")

        if not results_dir:
            raise ValueError("Results directory cannot be empty")

        return {
            "experiment": experiment,
            "ber": self.ber_spin.value(),
            "results_dir": results_dir,
            "save_reconstructed": self.save_reconstructed_check.isChecked(),
            "save_compressed": self.save_compressed_check.isChecked(),
            "save_config": self.save_config_check.isChecked(),
            "save_metadata": self.save_metadata_check.isChecked(),
        }

    def read_compressor_config_values(self) -> dict:
        values = {}

        for name, widget in self.config_widgets.items():
            values[name] = read_widget_value(widget)

        return values

    def current_compressor_name(self) -> str:
        return self.compressor_combo.currentText()

    def set_action_availability(
        self,
        can_compress: bool = False,
        can_decompress: bool = False,
        can_compress_decompress: bool = False,
    ):
        self._can_compress = can_compress
        self._can_decompress = can_decompress
        self._can_compress_decompress = can_compress_decompress

        self._refresh_run_buttons()

    def set_running(self, running: bool):
        self._is_running = running

        if running:
            self.run_progress_bar.setValue(0)
            self.run_status_label.setText("Starting...")

        self._refresh_run_buttons()

    def set_progress(self, value: float):
        value = max(0.0, min(1.0, value))
        self.run_progress_bar.setValue(int(value * 100))

    def set_message(self, message: str):
        if not message:
            message = "Running"

        self.run_status_label.setText(message)

    def _refresh_run_buttons(self):
        if self._is_running:
            self.compress_button.setEnabled(False)
            self.decompress_button.setEnabled(False)
            self.compress_decompress_button.setEnabled(False)
            self.abort_button.setEnabled(True)
            return

        self.compress_button.setEnabled(self._can_compress)
        self.decompress_button.setEnabled(self._can_decompress)
        self.compress_decompress_button.setEnabled(
            self._can_compress_decompress
        )
        self.abort_button.setEnabled(False)

    def _on_compressor_changed(self, compressor_name: str):
        self._clear_compressor_params_form()

        if not compressor_name:
            return

        compressor_cls = get_compressor(compressor_name)
        config_cls = compressor_cls.Config

        if not is_dataclass(config_cls):
            raise TypeError(
                f"Config for compressor '{compressor_name}' must be a dataclass"
            )

        self.config_widgets = {}

        for field in fields(config_cls):
            default = field.default
            widget = create_config_widget(field, default)

            self.config_widgets[field.name] = widget
            self.compressor_params_form.addRow(field.name, widget)

    def _clear_compressor_params_form(self):
        while self.compressor_params_form.rowCount() > 0:
            self.compressor_params_form.removeRow(0)

    def _on_browse_results_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select results directory",
            self.results_dir_edit.text(),
        )

        if directory:
            self.results_dir_edit.setText(directory)

    def _horizontal_separator(self) -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator