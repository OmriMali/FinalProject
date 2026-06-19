from __future__ import annotations

from dataclasses import fields, is_dataclass
from itertools import product
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.compressors.registry import get_compressor, list_compressors
from src.ui.gui.widgets.config_widgets import create_config_widget, read_widget_value


class CompressionTab(QWidget):
    """
    Compression run tab.

    Owns experiment settings, compressor settings, optional sweep settings,
    run buttons, and progress display.
    """

    compress_decompress_requested = Signal()
    abort_requested = Signal()

    RUN_MODE_SINGLE = "Single"
    RUN_MODE_SWEEP = "Sweep"

    def __init__(self):
        super().__init__()

        self.config_widgets: dict[str, QWidget] = {}
        self.config_field_names: list[str] = []

        self._can_compress_decompress = False
        self._is_running = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.experiment_panel = self._build_experiment_panel()
        self.compressor_panel = self._build_compressor_panel()
        self.sweep_panel = self._build_sweep_panel()

        layout.addWidget(self.experiment_panel)
        layout.addWidget(self.compressor_panel, stretch=1)
        layout.addWidget(self.sweep_panel)
        layout.addWidget(self._build_run_actions_panel())

        self._on_run_mode_changed(self.run_mode_combo.currentText())

    def _build_experiment_panel(self) -> QGroupBox:
        box = QGroupBox("Experiment Settings")
        layout = QFormLayout(box)

        self.experiment_edit = QLineEdit("gui_test")

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

        self.save_result_check = QCheckBox()
        self.save_result_check.setChecked(True)

        self.run_mode_combo = QComboBox()
        self.run_mode_combo.addItems([self.RUN_MODE_SINGLE, self.RUN_MODE_SWEEP])
        self.run_mode_combo.currentTextChanged.connect(self._on_run_mode_changed)

        layout.addRow("Experiment", self.experiment_edit)
        layout.addRow("Results directory", results_dir_widget)
        layout.addRow("Save result", self.save_result_check)
        layout.addRow("Run mode", self.run_mode_combo)

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

    def _build_sweep_panel(self) -> QGroupBox:
        box = QGroupBox("Sweep Settings")
        layout = QVBoxLayout(box)

        button_layout = QHBoxLayout()

        self.add_sweep_row_button = QPushButton("Add Parameter")
        self.remove_sweep_row_button = QPushButton("Remove Parameter")
        self.add_sweep_row_button.clicked.connect(self._add_sweep_row)
        self.remove_sweep_row_button.clicked.connect(self._remove_selected_sweep_rows)

        button_layout.addWidget(self.add_sweep_row_button)
        button_layout.addWidget(self.remove_sweep_row_button)
        button_layout.addStretch()

        self.sweep_table = QTableWidget(0, 2)
        self.sweep_table.setHorizontalHeaderLabels(["Parameter", "Values"])
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.sweep_table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
            | QTableWidget.EditTrigger.AnyKeyPressed
        )
        self.sweep_table.itemChanged.connect(self._update_sweep_preview)

        header = self.sweep_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self.sweep_preview_label = QLabel("1 run")
        self.sweep_preview_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addLayout(button_layout)
        layout.addWidget(self.sweep_table)
        layout.addWidget(self.sweep_preview_label)

        return box

    def _build_run_actions_panel(self) -> QGroupBox:
        box = QGroupBox("Run Actions")
        layout = QVBoxLayout(box)

        button_layout = QHBoxLayout()

        self.compress_decompress_button = QPushButton("Compress + Decompress")
        self.abort_button = QPushButton("Abort")

        self.compress_decompress_button.clicked.connect(
            self.compress_decompress_requested.emit
        )
        self.abort_button.clicked.connect(self.abort_requested.emit)

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
            "ber": 0.0,
            "results_dir": results_dir,
            "save_result": self.save_result_check.isChecked(),
        }

    def read_compressor_config_values(self) -> dict:
        values = {}

        for name, widget in self.config_widgets.items():
            values[name] = read_widget_value(widget)

        return values

    def read_config_variants(self) -> list[tuple[str, dict[str, Any]]]:
        base_config = self.read_compressor_config_values()

        if not self.is_sweep_mode():
            return [("", base_config)]

        sweep_rows = self._read_sweep_rows(base_config)

        if not sweep_rows:
            raise ValueError("Add at least one sweep parameter or use Single mode.")

        variants = []

        for values in product(*(row_values for _, row_values in sweep_rows)):
            config = dict(base_config)
            label_parts = []

            for (name, _), value in zip(sweep_rows, values):
                config[name] = value
                label_parts.append(f"{name}-{self._value_label(value)}")

            variants.append(("__".join(label_parts), config))

        return variants

    def current_compressor_name(self) -> str:
        return self.compressor_combo.currentText()

    def is_sweep_mode(self) -> bool:
        return self.run_mode_combo.currentText() == self.RUN_MODE_SWEEP

    def set_action_availability(
        self,
        can_compress_decompress: bool = False,
    ):
        self._can_compress_decompress = can_compress_decompress

        self._refresh_run_buttons()

    def set_running(self, running: bool):
        self._is_running = running

        self.experiment_panel.setEnabled(not running)
        self.compressor_panel.setEnabled(not running)
        self.sweep_panel.setEnabled(not running)

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
            self.compress_decompress_button.setEnabled(False)
            self.abort_button.setEnabled(True)
            return

        self.compress_decompress_button.setEnabled(
            self._can_compress_decompress
        )
        self.abort_button.setEnabled(False)

    def _on_compressor_changed(self, compressor_name: str):
        self._clear_compressor_params_form()

        if not compressor_name:
            self.config_widgets = {}
            self.config_field_names = []
            self._update_sweep_parameter_options()
            return

        compressor_cls = get_compressor(compressor_name)
        config_cls = compressor_cls.Config

        if not is_dataclass(config_cls):
            raise TypeError(
                f"Config for compressor '{compressor_name}' must be a dataclass"
            )

        self.config_widgets = {}
        self.config_field_names = []

        for field in fields(config_cls):
            default = field.default
            widget = create_config_widget(field, default)

            self.config_widgets[field.name] = widget
            self.config_field_names.append(field.name)
            self.compressor_params_form.addRow(field.name, widget)

        self._update_sweep_parameter_options()

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

    def _on_run_mode_changed(self, mode: str):
        is_sweep = mode == self.RUN_MODE_SWEEP
        self.sweep_panel.setVisible(is_sweep)
        self._update_sweep_preview()

    def _add_sweep_row(self):
        row = self.sweep_table.rowCount()
        self.sweep_table.insertRow(row)

        parameter_combo = QComboBox()
        parameter_combo.addItems(self.config_field_names)
        parameter_combo.currentTextChanged.connect(self._update_sweep_preview)

        values_item = QTableWidgetItem("")

        self.sweep_table.setCellWidget(row, 0, parameter_combo)
        self.sweep_table.setItem(row, 1, values_item)
        self._update_sweep_preview()

    def _remove_selected_sweep_rows(self):
        selected_rows = {
            index.row()
            for index in self.sweep_table.selectedIndexes()
        }

        if not selected_rows:
            return

        for row in sorted(selected_rows, reverse=True):
            self.sweep_table.removeRow(row)

        self._update_sweep_preview()

    def _update_sweep_parameter_options(self):
        if not hasattr(self, "sweep_table"):
            return

        for row in range(self.sweep_table.rowCount()):
            combo = self.sweep_table.cellWidget(row, 0)

            if not isinstance(combo, QComboBox):
                continue

            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self.config_field_names)

            if current in self.config_field_names:
                combo.setCurrentText(current)

            combo.blockSignals(False)

        self._update_sweep_preview()

    def _read_sweep_rows(
        self,
        base_config: dict[str, Any],
    ) -> list[tuple[str, list[Any]]]:
        rows = []
        seen_names = set()

        for row in range(self.sweep_table.rowCount()):
            combo = self.sweep_table.cellWidget(row, 0)
            values_item = self.sweep_table.item(row, 1)

            if not isinstance(combo, QComboBox):
                continue

            name = combo.currentText()

            if not name:
                raise ValueError("Sweep parameter cannot be empty")

            if name in seen_names:
                raise ValueError(f"Sweep parameter '{name}' is listed more than once")

            if name not in base_config:
                raise ValueError(f"Unknown sweep parameter: {name}")

            text = values_item.text() if values_item is not None else ""
            values = self._parse_sweep_values(name, text, base_config[name])

            rows.append((name, values))
            seen_names.add(name)

        return rows

    def _parse_sweep_values(
        self,
        name: str,
        text: str,
        base_value: Any,
    ) -> list[Any]:
        text = text.strip()

        if not text:
            raise ValueError(f"Sweep values for '{name}' cannot be empty")

        if isinstance(base_value, tuple):
            return [
                self._parse_tuple_value(item, base_value)
                for item in self._split_values(text, separator=";")
            ]

        return [
            self._parse_scalar_value(item, base_value)
            for item in self._split_values(text, separator=",")
        ]

    def _parse_tuple_value(
        self,
        text: str,
        base_value: tuple,
    ) -> tuple:
        parts = self._split_values(text, separator=",")

        if len(parts) != len(base_value):
            raise ValueError(
                f"Tuple sweep value '{text}' must have {len(base_value)} values"
            )

        return tuple(
            self._parse_scalar_value(part, base_item)
            for part, base_item in zip(parts, base_value)
        )

    def _parse_scalar_value(self, text: str, base_value: Any) -> Any:
        text = text.strip()

        if isinstance(base_value, bool):
            normalized = text.lower()

            if normalized in {"true", "1", "yes", "y"}:
                return True

            if normalized in {"false", "0", "no", "n"}:
                return False

            raise ValueError(f"Cannot parse boolean sweep value: {text}")

        if isinstance(base_value, int):
            return int(text)

        if isinstance(base_value, float):
            return float(text)

        if hasattr(base_value, "name") and hasattr(base_value.__class__, "__members__"):
            return base_value.__class__[text]

        return text

    def _split_values(self, text: str, separator: str) -> list[str]:
        return [
            item.strip()
            for item in text.split(separator)
            if item.strip()
        ]

    def _update_sweep_preview(self, *_args):
        if not hasattr(self, "sweep_preview_label"):
            return

        if not self.is_sweep_mode():
            self.sweep_preview_label.setText("1 run")
            return

        try:
            total = len(self.read_config_variants())
        except ValueError:
            total = 0

        self.sweep_preview_label.setText(f"{total} runs")

    def _value_label(self, value: Any) -> str:
        if isinstance(value, tuple):
            value = "-".join(str(item) for item in value)

        return str(value).strip().replace(" ", "_").replace("/", "_")

    def _horizontal_separator(self) -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator
