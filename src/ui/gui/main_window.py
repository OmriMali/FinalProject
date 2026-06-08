from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui.gui.controllers.compression_controller import CompressionController
from src.ui.gui.widgets.config_widgets import create_config_widget, read_widget_value
from src.ui.gui.workers.compression_worker import CompressionWorker

class MainWindow(QMainWindow):
    """
    Main GUI window for hyperspectral image compression.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("HSI Compression GUI")
        self.resize(1200, 750)

        self.controller = CompressionController()
        self.selected_hsi_path: Path | None = None

        self.compression_thread: QThread | None = None
        self.compression_worker: CompressionWorker | None = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)

        main_layout = QHBoxLayout()
        root_layout.addLayout(main_layout)

        main_layout.addWidget(self._build_file_panel(), stretch=2)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self._build_experiment_panel())
        right_layout.addWidget(self._build_compressor_panel())
        right_layout.addStretch()

        main_layout.addWidget(right_panel, stretch=1)

        root_layout.addWidget(self._build_run_panel())
        root_layout.addWidget(self._build_results_panel())

    def _build_file_panel(self) -> QGroupBox:
        box = QGroupBox("Input HSI Files")
        layout = QVBoxLayout(box)

        button_layout = QHBoxLayout()

        self.add_file_button = QPushButton("Add")
        self.clear_file_button = QPushButton("Clear")

        self.add_file_button.clicked.connect(self._on_add_file)
        self.clear_file_button.clicked.connect(self._on_clear_file)

        button_layout.addWidget(self.add_file_button)
        button_layout.addWidget(self.clear_file_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.file_table = QTableWidget(0, 4)
        self.file_table.setHorizontalHeaderLabels(
            ["Filename", "Path", "Status", "Shape"]
        )
        self.file_table.horizontalHeader().setStretchLastSection(True)
        self.file_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.file_table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout.addWidget(self.file_table)

        return box

    def _build_compressor_panel(self) -> QGroupBox:
        box = QGroupBox("Compression")
        layout = QVBoxLayout(box)

        # Fixed general settings
        form = QFormLayout()

        self.compressor_combo = QComboBox()
        self.compressor_combo.addItems(self.controller.available_compressors())
        self.compressor_combo.currentTextChanged.connect(
            self._on_compressor_changed
        )

        form.addRow("Compressor", self.compressor_combo)
        layout.addLayout(form)

        # Dynamic compressor-specific parameters
        self.config_form = QFormLayout()
        self.config_widgets = {}

        layout.addLayout(self.config_form)
        layout.addStretch()

        # Build parameters for initially selected compressor
        self._on_compressor_changed(self.compressor_combo.currentText())

        return box

    def _build_experiment_panel(self) -> QGroupBox:
        box = QGroupBox("Experiment")
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

    def _read_experiment_settings(self) -> dict:
        return {
            "experiment": self.experiment_edit.text().strip(),
            "ber": self.ber_spin.value(),
            "results_dir": self.results_dir_edit.text().strip(),
            "save_reconstructed": self.save_reconstructed_check.isChecked(),
            "save_compressed": self.save_compressed_check.isChecked(),
            "save_config": self.save_config_check.isChecked(),
            "save_metadata": self.save_metadata_check.isChecked(),
        }

    def _on_browse_results_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select results directory",
            self.results_dir_edit.text(),
        )

        if path:
            self.results_dir_edit.setText(path)

    def _on_compressor_changed(self, compressor_name: str):
        self._clear_config_form()

        if not compressor_name:
            return

        self.config_widgets = {}

        for field in self.controller.get_config_fields(compressor_name):
            default = field.default

            widget = create_config_widget(field, default)

            self.config_form.addRow(field.name, widget)
            self.config_widgets[field.name] = widget

    def _clear_config_form(self):
        while self.config_form.rowCount() > 0:
            self.config_form.removeRow(0)

    def _build_run_panel(self) -> QGroupBox:
        box = QGroupBox("Run")
        layout = QHBoxLayout(box)

        self.run_button = QPushButton("Compress && Decompress")
        self.run_button.clicked.connect(self._on_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(self.run_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        return box

    def _build_results_panel(self) -> QGroupBox:
        box = QGroupBox("Results")
        layout = QVBoxLayout(box)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        layout.addWidget(self.results_text)

        return box

    def _on_add_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HSI file",
            "",
            "NumPy files (*.npz);;All files (*.*)",
        )

        if not path:
            return

        self.selected_hsi_path = Path(path)

        self.file_table.setRowCount(0)
        self.file_table.insertRow(0)

        self.file_table.setItem(
            0, 0, QTableWidgetItem(self.selected_hsi_path.name)
        )
        self.file_table.setItem(
            0, 1, QTableWidgetItem(str(self.selected_hsi_path))
        )
        self.file_table.setItem(
            0, 2, QTableWidgetItem("Selected")
        )
        self.file_table.setItem(
            0, 3, QTableWidgetItem("not loaded yet")
        )

        self.status_label.setText(f"Selected {self.selected_hsi_path.name}")

    def _on_clear_file(self):
        self.selected_hsi_path = None
        self.file_table.setRowCount(0)
        self.results_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")

    def _on_run(self):
        if self.selected_hsi_path is None:
            QMessageBox.warning(
                self,
                "No file selected",
                "Please select an HSI file first.",
            )
            return

        config_values = self._read_config_values()
        experiment_settings = self._read_experiment_settings()

        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running compression...")
        self.results_text.clear()

        self.compression_thread = QThread()
        self.compression_worker = CompressionWorker(
            hsi_path=self.selected_hsi_path,
            compressor_name=self.compressor_combo.currentText(),
            config_values=config_values,
            experiment_settings=experiment_settings,
        )

        self.compression_worker.moveToThread(self.compression_thread)

        self.compression_thread.started.connect(self.compression_worker.run)

        self.compression_worker.progress_changed.connect(self._set_progress)
        self.compression_worker.status_changed.connect(self.status_label.setText)
        self.compression_worker.finished.connect(self._on_compression_finished)
        self.compression_worker.failed.connect(self._on_compression_failed)

        self.compression_worker.finished.connect(self.compression_thread.quit)
        self.compression_worker.failed.connect(self.compression_thread.quit)

        self.compression_thread.finished.connect(self.compression_worker.deleteLater)
        self.compression_thread.finished.connect(self.compression_thread.deleteLater)
        self.compression_thread.finished.connect(self._on_compression_thread_finished)

        self.compression_thread.start()

    def _on_compression_finished(self, result: dict):
        self._show_result(result)
        self.status_label.setText("Finished")
        self.progress_bar.setValue(100)

    def _on_compression_failed(self, error_message: str):
        QMessageBox.critical(
            self,
            "Compression failed",
            error_message,
        )
        self.status_label.setText("Failed")

    def _on_compression_thread_finished(self):
        self.run_button.setEnabled(True)
        self.compression_thread = None
        self.compression_worker = None

    def _read_config_values(self) -> dict:
        values = {}

        for name, widget in self.config_widgets.items():
            values[name] = read_widget_value(widget)

        return values

    def _parse_config_value(self, text: str):
        text = text.strip()

        if text.lower() == "none":
            return None

        if text.lower() == "true":
            return True

        if text.lower() == "false":
            return False

        try:
            return int(text)
        except ValueError:
            pass

        try:
            return float(text)
        except ValueError:
            pass

        return text

    def _set_progress(self, value: float):
        value = max(0.0, min(1.0, value))
        self.progress_bar.setValue(int(value * 100))

    def _show_result(self, result: dict):
        lines = []

        for key, value in result.items():
            lines.append(f"{key}: {value}")

        self.results_text.setPlainText("\n".join(lines))