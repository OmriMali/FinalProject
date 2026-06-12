import math

from pathlib import Path
from dataclasses import fields, is_dataclass

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QTableWidgetItem,
    QFileDialog,
    QMessageBox,
    QSpinBox,
)

from src.core.hsi import HSI
from src.visuals.hsi import plot_rgb, select_rgb_bands, compare_spectra
from src.visuals.style import DEFAULT_STYLE
from src.compressors.registry import get_compressor, list_compressors

from src.ui.gui.models.workspace_item import WorkspaceItem, WorkspaceItemKind, WorkspaceItemRole
from src.ui.gui.services.workspace_loader import WorkspaceLoader, WorkspaceLoadError
from src.ui.gui.workers.compression_worker import CompressionWorker
from src.ui.gui.widgets.config_widgets import create_config_widget, read_widget_value
from src.ui.gui.widgets.metrics_table import MetricsTableWidget


class MainWindow(QMainWindow):
    """
    Main GUI window for hyperspectral image compression.

    This version defines only the visual layout. Application logic,
    file loading, compression, decompression, plotting, and callbacks
    should be connected later.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("HSI Compression GUI")
        self.resize(1500, 850)

        self.next_workspace_number = 1

        self.current_display_mode = None
        self.current_spectra_hsis = None
        self.current_spectra_labels = None

        self.spectra_update_timer = QTimer(self)
        self.spectra_update_timer.setSingleShot(True)
        self.spectra_update_timer.setInterval(200)
        self.spectra_update_timer.timeout.connect(self._refresh_current_spectra_plot)

        self.workspace_items: list[WorkspaceItem] = []
        self.workspace_loader = WorkspaceLoader()

        self.compression_thread = None
        self.compression_worker = None
        self.is_running = False

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self._build_config_column())
        splitter.addWidget(self._build_workspace_column())
        splitter.addWidget(self._build_results_column())

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)

        root_layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Column 1: configuration and run actions
    # ------------------------------------------------------------------

    def _build_config_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_experiment_panel())
        layout.addWidget(self._build_compressor_panel(), stretch=1)
        layout.addWidget(self._build_run_actions_panel())

        return column

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
        
        self.browse_results_dir_button.clicked.connect(self._on_browse_results_dir)

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
        self.config_widgets = {}

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

        self.compress_decompress_button.clicked.connect(self._on_compress_decompress)

        button_layout.addWidget(self.compress_button)
        button_layout.addWidget(self.decompress_button)
        button_layout.addWidget(self.compress_decompress_button)

        self.compress_button.setEnabled(False)
        self.decompress_button.setEnabled(False)
        self.compress_decompress_button.setEnabled(False)

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

        return box

    # ------------------------------------------------------------------
    # Column 2: data workspace and file controls
    # ------------------------------------------------------------------

    def _build_workspace_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
     
        layout.addWidget(self._build_file_controls_panel())
        layout.addWidget(self._build_loaded_items_panel(), stretch=1)

        return column

    def _build_loaded_items_panel(self) -> QGroupBox:
        box = QGroupBox("Loaded Items")
        layout = QVBoxLayout(box)

        headers = [""] + WorkspaceItem.table_headers()

        self.loaded_items_table = QTableWidget(0, len(headers))
        self.loaded_items_table.setHorizontalHeaderLabels(headers)

        header = self.loaded_items_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        header.setSectionsMovable(True)

        self.loaded_items_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection
        )
        self.loaded_items_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.loaded_items_table.verticalHeader().setVisible(False)
        self.loaded_items_table.itemChanged.connect(self._on_loaded_item_changed)

        layout.addWidget(self.loaded_items_table)

        return box

    def _build_file_controls_panel(self) -> QGroupBox:
        box = QGroupBox("File Controls")
        layout = QHBoxLayout(box)

        self.load_hsi_button = QPushButton("Load HSI")
        self.load_compressed_button = QPushButton("Load CompressedHSI")
        self.remove_selected_button = QPushButton("Remove Selected")
        self.clear_items_button = QPushButton("Clear")

        self.load_hsi_button.clicked.connect(self._on_load_hsi)
        self.load_compressed_button.clicked.connect(self._on_load_compressed_hsi)
        self.remove_selected_button.clicked.connect(self._on_remove_selected_items)
        self.clear_items_button.clicked.connect(self.clear_workspace_items)

        layout.addWidget(self.load_hsi_button)
        layout.addWidget(self.load_compressed_button)
        layout.addWidget(self.remove_selected_button)
        layout.addWidget(self.clear_items_button)
        layout.addStretch()

        return box

    # ------------------------------------------------------------------
    # Column 3: results and visualization
    # ------------------------------------------------------------------

    def _build_results_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_visualization_controls_panel())
        layout.addWidget(self._build_metrics_panel())
        layout.addWidget(self._build_display_panel(), stretch=1)

        return column

    def _build_visualization_controls_panel(self) -> QGroupBox:
        box = QGroupBox("Visualization Controls")
        layout = QHBoxLayout(box)

        self.show_rgb_button = QPushButton("Show RGB")
        self.compare_selected_button = QPushButton("Compare Selected")
        self.compare_last_result_button = QPushButton("Compare Last Result")
        self.plot_spectra_button = QPushButton("Plot Spectra")
        self.clear_canvas_button = QPushButton("Clear Canvas")

        self.spectrum_x_spin = QSpinBox()
        self.spectrum_x_spin.setRange(0, 10000)
        self.spectrum_x_spin.setValue(100)
        self.spectrum_x_spin.valueChanged.connect(self._schedule_spectra_update)

        self.spectrum_y_spin = QSpinBox()
        self.spectrum_y_spin.setRange(0, 10000)
        self.spectrum_y_spin.setValue(100)
        self.spectrum_y_spin.valueChanged.connect(self._schedule_spectra_update)

        self.show_rgb_button.clicked.connect(self._on_show_rgb)
        self.compare_selected_button.clicked.connect(self._on_compare_selected)
        self.clear_canvas_button.clicked.connect(self._clear_canvas)
        self.plot_spectra_button.clicked.connect(self._on_plot_spectra)

        self.show_rgb_button.setEnabled(False)
        self.compare_selected_button.setEnabled(False)
        self.compare_last_result_button.setEnabled(False)
        self.plot_spectra_button.setEnabled(False)

        layout.addWidget(self.show_rgb_button)
        layout.addWidget(self.compare_selected_button)
        layout.addWidget(self.compare_last_result_button)

        layout.addWidget(QLabel("x"))
        layout.addWidget(self.spectrum_x_spin)

        layout.addWidget(QLabel("y"))
        layout.addWidget(self.spectrum_y_spin)

        layout.addWidget(self.plot_spectra_button)
        layout.addStretch()

        layout.addWidget(self.clear_canvas_button)

        return box

    def _build_metrics_panel(self) -> QGroupBox:
        box = QGroupBox("Metrics")
        layout = QVBoxLayout(box)

        self.metrics_table = MetricsTableWidget()
        layout.addWidget(self.metrics_table)

        return box

    def _build_display_panel(self) -> QGroupBox:
        box = QGroupBox("Display")
        layout = QVBoxLayout(box)

        self.display_figure = Figure(figsize=(6, 5))
        self.display_canvas = FigureCanvas(self.display_figure)

        self.display_toolbar = NavigationToolbar(
            self.display_canvas,
            self,
        )

        layout.addWidget(self.display_toolbar)
        layout.addWidget(self.display_canvas, stretch=1)

        return box

    # ------------------------------------------------------------------
    # Workspace item helpers
    # ------------------------------------------------------------------

    def add_workspace_item(self, item: WorkspaceItem):
        if item.number is None:
            item.number = self.next_workspace_number
            self.next_workspace_number += 1

        self.workspace_items.append(item)
        self._append_workspace_item_row(item)
        self._update_action_buttons()

    def clear_workspace_items(self):
        self.workspace_items.clear()
        self.loaded_items_table.setRowCount(0)
        self.next_workspace_number = 1

        self._clear_canvas()
        self._update_action_buttons()

    def _append_workspace_item_row(self, item: WorkspaceItem):
        self.loaded_items_table.blockSignals(True)

        row = self.loaded_items_table.rowCount()
        self.loaded_items_table.insertRow(row)

        check_item = QTableWidgetItem()
        check_item.setFlags(
            Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEnabled
        )
        check_item.setCheckState(Qt.CheckState.Unchecked)
        check_item.setData(Qt.ItemDataRole.UserRole, item.item_id)

        self.loaded_items_table.setItem(row, 0, check_item)

        for col, value in enumerate(item.table_values(), start=1):
            table_item = QTableWidgetItem(str(value))
            table_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.loaded_items_table.setItem(row, col, table_item)

        self.loaded_items_table.blockSignals(False)

    def _on_remove_selected_items(self):
        rows_to_remove = []

        for row in range(self.loaded_items_table.rowCount()):
            check_item = self.loaded_items_table.item(row, 0)

            if check_item is None:
                continue

            if check_item.checkState() == Qt.CheckState.Checked:
                rows_to_remove.append(row)

        if not rows_to_remove:
            return

        item_ids_to_remove = set()

        for row in rows_to_remove:
            item_id = self.loaded_items_table.item(row, 0).data(
                Qt.ItemDataRole.UserRole
            )
            item_ids_to_remove.add(item_id)

        for row in sorted(rows_to_remove, reverse=True):
            self.loaded_items_table.removeRow(row)

        self.workspace_items = [
            item
            for item in self.workspace_items
            if item.item_id not in item_ids_to_remove
        ]

        self._update_action_buttons()

    def checked_workspace_items(self) -> list[WorkspaceItem]:
        checked_ids = set()

        for row in range(self.loaded_items_table.rowCount()):
            check_item = self.loaded_items_table.item(row, 0)

            if check_item is None:
                continue

            if check_item.checkState() == Qt.CheckState.Checked:
                item_id = check_item.data(Qt.ItemDataRole.UserRole)
                checked_ids.add(item_id)

        return [
            item
            for item in self.workspace_items
            if item.item_id in checked_ids
        ]

    def checked_workspace_item(self) -> WorkspaceItem | None:
        checked = self.checked_workspace_items()

        if len(checked) != 1:
            return None

        return checked[0]

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _horizontal_separator(self) -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator
   
    def _set_run_progress(self, value: float):
        if value <= 1.0:
            value *= 100

        self.run_progress_bar.setValue(int(value))

    def _set_run_progress_message(self, message: str):
        if not message:
            message = "Running"

        self.run_status_label.setText(message)

    def _set_running(self, running: bool):
        self.is_running = running

        if running:
            self.run_progress_bar.setValue(0)
            self.run_progress_bar.setFormat("%p%")
            self.run_status_label.setText("Starting...")

            self.compress_button.setEnabled(False)
            self.decompress_button.setEnabled(False)
            self.compress_decompress_button.setEnabled(False)

            self.load_hsi_button.setEnabled(False)
            self.load_compressed_button.setEnabled(False)
            self.remove_selected_button.setEnabled(False)
            self.clear_items_button.setEnabled(False)

            return

        self.load_hsi_button.setEnabled(True)
        self.load_compressed_button.setEnabled(True)
        self.remove_selected_button.setEnabled(True)
        self.clear_items_button.setEnabled(True)

        self.run_status_label.setText("Ready")
        self._update_action_buttons()

    def _on_compression_thread_finished(self):
        self.compression_thread = None
        self.compression_worker = None
        self._set_running(False)

    # ------------------------------------------------------------------
    # Loading actions
    # ------------------------------------------------------------------
        
    def _on_load_hsi(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load HSI",
            "",
            "NumPy files (*.npz);;All files (*.*)",
        )

        if not path:
            return

        try:
            item = self.workspace_loader.inspect_hsi(Path(path))
        except WorkspaceLoadError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.add_workspace_item(item)

    def _on_load_compressed_hsi(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load CompressedHSI",
            "",
            "NumPy files (*.npz);;All files (*.*)",
        )

        if not path:
            return

        try:
            item = self.workspace_loader.inspect_compressed_hsi(Path(path))
        except WorkspaceLoadError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.add_workspace_item(item)

    def _on_loaded_item_changed(self, item):
        if item.column() != 0:
            return

        self._update_action_buttons()
        self._update_metrics_from_checked_items()

    # ------------------------------------------------------------------
    # Visualization actions
    # ------------------------------------------------------------------

    def _on_show_rgb(self):
        item = self.checked_workspace_item()

        if item is None:
            QMessageBox.warning(
                self,
                "Show RGB",
                "Please check exactly one HSI item.",
            )
            return

        try:
            obj = self.workspace_loader.load_object(item)
        except WorkspaceLoadError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        if not isinstance(obj, HSI):
            QMessageBox.warning(
                self,
                "Show RGB",
                "The selected item is not an HSI.",
            )
            return

        self._display_rgb(obj, title=item.name)

    def _display_rgb(self, hsi: HSI, title: str | None = None):

        self.current_display_mode = "rgb"
        self.current_spectra_hsis = None
        self.current_spectra_labels = None

        self.display_figure.clear()

        ax = self.display_figure.add_subplot(1, 1, 1)

        plot_rgb(
            hsi=hsi,
            style=DEFAULT_STYLE,
            ax=ax,
            title=title,
            show_axis=False,
        )

        self.display_figure.tight_layout()
        self.display_canvas.draw()

    def _on_compare_selected(self):
        items = self.checked_workspace_items()

        if len(items) < 2:
            QMessageBox.warning(
                self,
                "Compare Selected",
                "Please check at least two HSI items.",
            )
            return

        hsis = []
        labels = []

        for item in items:
            try:
                obj = self.workspace_loader.load_object(item)
            except WorkspaceLoadError as exc:
                QMessageBox.critical(self, "Load failed", str(exc))
                return

            if not isinstance(obj, HSI):
                QMessageBox.warning(
                    self,
                    "Compare Selected",
                    "Only HSI items can be compared.",
                )
                return

            hsis.append(obj)
            labels.append(item.name)

        self._display_rgb_comparison(hsis, labels)

    def _display_rgb_comparison(
        self,
        hsis: list[HSI],
        labels: list[str],
    ):
        
        self.current_display_mode = "rgb"
        self.current_spectra_hsis = None
        self.current_spectra_labels = None

        self.display_figure.clear()

        n_images = len(hsis)

        n_cols = min(3, n_images)
        n_rows = math.ceil(n_images / n_cols)

        axes = self.display_figure.subplots(
            n_rows,
            n_cols,
            squeeze=False,
        ).ravel()

        bands = select_rgb_bands(hsis[0])

        for ax, hsi, label in zip(axes, hsis, labels):
            plot_rgb(
                hsi=hsi,
                bands=bands,
                style=DEFAULT_STYLE,
                ax=ax,
                title=label,
                show_axis=False,
            )

        for ax in axes[n_images:]:
            ax.set_axis_off()

        self.display_figure.tight_layout()
        self.display_canvas.draw()

    def _clear_canvas(self):
        self.current_display_mode = None
        self.current_spectra_hsis = None
        self.current_spectra_labels = None

        self.display_figure.clear()
        self.display_canvas.draw_idle()

    def _on_plot_spectra(self):
        items = self.checked_workspace_items()

        if not items:
            QMessageBox.warning(
                self,
                "Plot Spectra",
                "Please check at least one HSI item.",
            )
            return

        hsis = []
        labels = []

        for item in items:
            try:
                obj = self.workspace_loader.load_object(item)
            except WorkspaceLoadError as exc:
                QMessageBox.critical(self, "Load failed", str(exc))
                return

            if not isinstance(obj, HSI):
                QMessageBox.warning(
                    self,
                    "Plot Spectra",
                    "Only HSI items can be used for spectral plots.",
                )
                return

            hsis.append(obj)
            labels.append(item.name)

        pixel = (
            self.spectrum_x_spin.value(),
            self.spectrum_y_spin.value(),
        )

        self.current_display_mode = "spectra"
        self.current_spectra_hsis = hsis
        self.current_spectra_labels = labels

        self._set_spectrum_pixel_limits(hsis[0])
        self._display_spectra(hsis, labels, pixel)

    def _display_spectra(
        self,
        hsis: list[HSI],
        labels: list[str],
        pixel: tuple[int, int],
    ):
        self.display_figure.clear()

        ax = self.display_figure.add_subplot(1, 1, 1)

        compare_spectra(
            hsis=hsis,
            labels=labels,
            pixel=pixel,
            style=DEFAULT_STYLE,
            ax=ax,
            title=f"Spectrum at pixel ({pixel[0]}, {pixel[1]})",
            show_legend=True,
        )

        self.display_figure.tight_layout()
        self.display_canvas.draw_idle()

    def _set_spectrum_pixel_limits(self, hsi: HSI):
        height, width = hsi.spatial_shape

        self.spectrum_x_spin.blockSignals(True)
        self.spectrum_y_spin.blockSignals(True)

        self.spectrum_x_spin.setRange(0, width - 1)
        self.spectrum_y_spin.setRange(0, height - 1)

        self.spectrum_x_spin.blockSignals(False)
        self.spectrum_y_spin.blockSignals(False)

    def _schedule_spectra_update(self):
        if self.current_display_mode != "spectra":
            return

        if self.current_spectra_hsis is None:
            return

        self.spectra_update_timer.start()

    def _refresh_current_spectra_plot(self):
        if self.current_display_mode != "spectra":
            return

        if self.current_spectra_hsis is None:
            return

        pixel = (
            self.spectrum_x_spin.value(),
            self.spectrum_y_spin.value(),
        )

        try:
            self._display_spectra(
                hsis=self.current_spectra_hsis,
                labels=self.current_spectra_labels,
                pixel=pixel,
            )
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Experiment setting actions
    # ------------------------------------------------------------------

    def _on_browse_results_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select results directory",
            self.results_dir_edit.text(),
        )

        if directory:
            self.results_dir_edit.setText(directory)

    def _read_experiment_settings(self) -> dict:
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

    # ------------------------------------------------------------------
    # Compressor settings actions
    # ------------------------------------------------------------------

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

    def _read_compressor_config_values(self) -> dict:
        values = {}

        for name, widget in self.config_widgets.items():
            values[name] = read_widget_value(widget)

        return values

    # ------------------------------------------------------------------
    # Compression actions
    # ------------------------------------------------------------------
    
    def _on_compress_decompress(self):
        item = self.checked_workspace_item()

        if item is None:
            QMessageBox.warning(
                self,
                "Compress + Decompress",
                "Please check exactly one HSI item.",
            )
            return

        if item.kind != WorkspaceItemKind.HSI:
            QMessageBox.warning(
                self,
                "Compress + Decompress",
                "Compress + Decompress requires an HSI item.",
            )
            return

        try:
            config_values = self._read_compressor_config_values()
            experiment_settings = self._read_experiment_settings()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        self._start_compression_worker(
            source_item=item,
            compressor_name=self.compressor_combo.currentText(),
            config_values=config_values,
            experiment_settings=experiment_settings,
        )

    def _on_compression_finished(self, gui_result):
        self.run_progress_bar.setValue(100)

        result = gui_result.result
        method = gui_result.compressor_name
        artifact_dir = gui_result.artifact_dir
        source_name = gui_result.source_item.name

        compressed_item = WorkspaceItem.from_compressed_hsi(
            compressed=result.compressed,
            name=f"{source_name}_{method}_compressed",
            method=method,
            directory=artifact_dir,
            keep_cached=True,
        )

        reconstructed_item = WorkspaceItem.from_hsi(
            hsi=result.reconstructed,
            name=f"{source_name}_{method}_reconstructed",
            role=WorkspaceItemRole.RECONSTRUCTION,
            method=method,
            directory=artifact_dir,
            metrics=result.metrics,
            keep_cached=True,
        )

        self.add_workspace_item(compressed_item)
        self.add_workspace_item(reconstructed_item)

        self.metrics_table.show_item_metrics(reconstructed_item)

    def _on_compression_failed(self, message: str):
        QMessageBox.critical(
            self,
            "Compression failed",
            message,
        )

        if hasattr(self, "run_status_label"):
            self.run_status_label.setText("Failed")

    # ------------------------------------------------------------------
    # Button Updating
    # ------------------------------------------------------------------

    def _update_action_buttons(self):

        if self.is_running:
            self.compress_button.setEnabled(False)
            self.decompress_button.setEnabled(False)
            self.compress_decompress_button.setEnabled(False)
            self.show_rgb_button.setEnabled(False)
            self.compare_selected_button.setEnabled(False)
            self.compare_last_result_button.setEnabled(False)
            self.plot_spectra_button.setEnabled(False)
            return
        
        checked_items = self.checked_workspace_items()

        checked_hsis = [
            item
            for item in checked_items
            if item.kind == WorkspaceItemKind.HSI
        ]

        checked_compressed = [
            item
            for item in checked_items
            if item.kind == WorkspaceItemKind.COMPRESSED_HSI
        ]

        n_checked = len(checked_items)
        n_hsis = len(checked_hsis)
        n_compressed = len(checked_compressed)

        # ------------------------------------------------------------
        # Run actions
        # ------------------------------------------------------------
        exactly_one_hsi = n_checked == 1 and n_hsis == 1
        exactly_one_compressed = n_checked == 1 and n_compressed == 1

        self.compress_button.setEnabled(False)
        self.compress_decompress_button.setEnabled(exactly_one_hsi)
        self.decompress_button.setEnabled(False)

        # ------------------------------------------------------------
        # Visualization actions
        # ------------------------------------------------------------
        self.show_rgb_button.setEnabled(exactly_one_hsi)
        self.plot_spectra_button.setEnabled(n_hsis >= 1 and n_compressed == 0)
        self.compare_selected_button.setEnabled(n_hsis >= 2 and n_compressed == 0)

        # This will be enabled later when we track the latest run result.
        self.compare_last_result_button.setEnabled(False)

    # ------------------------------------------------------------------
    # Worker actions
    # ------------------------------------------------------------------
    
    def _start_compression_worker(
        self,
        source_item: WorkspaceItem,
        compressor_name: str,
        config_values: dict,
        experiment_settings: dict,
    ):
        self._set_running(True)

        self.compression_thread = QThread(self)
        self.compression_worker = CompressionWorker(
            source_item=source_item,
            compressor_name=compressor_name,
            config_values=config_values,
            experiment_settings=experiment_settings,
        )

        self.compression_worker.moveToThread(self.compression_thread)

        self.compression_thread.started.connect(
            self.compression_worker.run
        )

        self.compression_worker.progress_changed.connect(
            self._set_run_progress
        )

        self.compression_worker.progress_message_changed.connect(
            self._set_run_progress_message
        )

        self.compression_worker.finished.connect(
            self._on_compression_finished
        )
        self.compression_worker.failed.connect(
            self._on_compression_failed
        )

        self.compression_worker.finished.connect(
            self.compression_thread.quit
        )
        self.compression_worker.failed.connect(
            self.compression_thread.quit
        )

        self.compression_thread.finished.connect(
            self.compression_worker.deleteLater
        )
        self.compression_thread.finished.connect(
            self.compression_thread.deleteLater
        )
        self.compression_thread.finished.connect(
            self._on_compression_thread_finished
        )

        self.compression_thread.start()

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    
    def _update_metrics_from_checked_items(self):
        items = [
            item
            for item in self.checked_workspace_items()
            if item.metrics is not None
        ]

        self.metrics_table.show_metrics_comparison(items)
