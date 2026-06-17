import math
import json
import sys
import tempfile

from pathlib import Path
from dataclasses import fields, is_dataclass

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QTimer, QProcess
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

from src.core.hsi import HSI, CompressedHSI
from src.visuals.hsi import plot_rgb, select_rgb_bands, compare_spectra, plot_histogram, plot_compressed_histogram
from src.visuals.style import DEFAULT_STYLE
from src.compressors.registry import get_compressor, list_compressors

from src.ui.gui.models.workspace_item import WorkspaceItem, WorkspaceItemKind, WorkspaceItemRole
from src.ui.gui.services.workspace_loader import WorkspaceLoader, WorkspaceLoadError
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

        self.abort_requested = False

        self.current_display_mode = None
        self.current_spectra_hsis = None
        self.current_spectra_labels = None

        self.spectra_update_timer = QTimer(self)
        self.spectra_update_timer.setSingleShot(True)
        self.spectra_update_timer.setInterval(200)
        self.spectra_update_timer.timeout.connect(self._refresh_current_spectra_plot)

        self.workspace_items: list[WorkspaceItem] = []
        self.workspace_loader = WorkspaceLoader()

        self.compression_process: QProcess | None = None
        self.current_process_buffer = ""
        
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
        self.abort_button = QPushButton("Abort")

        self.compress_decompress_button.clicked.connect(self._on_compress_decompress)
        self.abort_button.clicked.connect(self._abort_compression_process)

        button_layout.addWidget(self.compress_button)
        button_layout.addWidget(self.decompress_button)
        button_layout.addWidget(self.compress_decompress_button)
        button_layout.addWidget(self.abort_button)

        self.compress_button.setEnabled(False)
        self.decompress_button.setEnabled(False)
        self.compress_decompress_button.setEnabled(False)
        self.abort_button.setEnabled(False)

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
        self._set_table_headers(self.loaded_items_table, headers)

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
            
        self.plot_histogram_button = QPushButton("Histogram")
        self.plot_histogram_button.clicked.connect(self._on_plot_histogram)
        self.plot_histogram_button.setEnabled(False)

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

        layout.addWidget(self.plot_spectra_button)
        layout.addWidget(self.plot_histogram_button)
        layout.addWidget(self.clear_canvas_button)

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
        check_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.loaded_items_table.setItem(row, 0, check_item)

        for col, value in enumerate(item.table_values(), start=1):
            table_item = QTableWidgetItem(str(value))
            table_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            table_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
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
        value = max(0.0, min(1.0, value))
        self.run_progress_bar.setValue(int(value * 100))

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
            self.abort_button.setEnabled(True)

            self.load_hsi_button.setEnabled(False)
            self.load_compressed_button.setEnabled(False)
            self.remove_selected_button.setEnabled(False)
            self.clear_items_button.setEnabled(False)

            return

        self.load_hsi_button.setEnabled(True)
        self.load_compressed_button.setEnabled(True)
        self.remove_selected_button.setEnabled(True)
        self.clear_items_button.setEnabled(True)

        self.abort_button.setEnabled(False)

        self._update_action_buttons()

    def _set_table_headers(self, table: QTableWidget, headers: list[str]):
        table.setColumnCount(len(headers))

        for col, header in enumerate(headers):
            item = QTableWidgetItem(header)
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setHorizontalHeaderItem(col, item)

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

        self._display_rgb(obj, title=item.plot_label)

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
            labels.append(item.plot_label)

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
            labels.append(item.plot_label)

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

    def _on_plot_histogram(self):
        item = self.checked_workspace_item()

        if item is None:
            QMessageBox.warning(
                self,
                "Histogram",
                "Please check exactly one item.",
            )
            return

        try:
            obj = self.workspace_loader.load_object(item)
        except WorkspaceLoadError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        if isinstance(obj, HSI):
            self._display_hsi_histogram(
                hsi=obj,
                title=f"{item.plot_label}_Histogram",
            )
            return

        if isinstance(obj, CompressedHSI):
            self._display_compressed_histogram(
                compressed=obj,
                title=f"{item.plot_label}_Compressed_Histogram",
            )
            return

        QMessageBox.warning(
            self,
            "Histogram",
            "Selected item is not an HSI or CompressedHSI.",
        )

    def _display_hsi_histogram(
        self,
        hsi: HSI,
        title: str | None = None,
    ):
        self.current_display_mode = "histogram"
        self.current_spectra_hsis = None
        self.current_spectra_labels = None

        self.display_figure.clear()
        ax = self.display_figure.add_subplot(1, 1, 1)

        plot_histogram(
            hsi=hsi,
            band=None,
            bins=256,
            style=DEFAULT_STYLE,
            ax=ax,
            title=title,
        )

        self.display_figure.tight_layout()
        self.display_canvas.draw_idle()

    def _display_compressed_histogram(
        self,
        compressed: CompressedHSI,
        title: str | None = None,
    ):
        try:
            compressor = self._compressor_from_compressed_hsi(compressed)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Histogram",
                f"Could not prepare compressed histogram decoder:\n{exc}",
            )
            return

        self.current_display_mode = "compressed_histogram"
        self.current_spectra_hsis = None
        self.current_spectra_labels = None

        self.display_figure.clear()
        ax = self.display_figure.add_subplot(1, 1, 1)

        try:
            plot_compressed_histogram(
                compressed=compressed,
                compressor=compressor,
                bins=256,
                style=DEFAULT_STYLE,
                ax=ax,
                title=title,
            )
        except NotImplementedError as exc:
            QMessageBox.warning(self, "Histogram", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Histogram",
                f"Could not decode compressed values:\n{exc}",
            )
            return

        self.display_figure.tight_layout()
        self.display_canvas.draw_idle()

    def _compressor_from_compressed_hsi(
        self,
        compressed: CompressedHSI,
    ):
        run_info = compressed.metadata.attributes.get("run")

        if run_info is None:
            raise ValueError(
                "CompressedHSI metadata does not contain run information."
            )

        method = run_info.get("method")
        algorithm_config = run_info.get("algorithm_config", {})

        if not method:
            raise ValueError(
                "CompressedHSI run metadata does not contain a compressor method."
            )

        compressor_cls = get_compressor(method)
        config = compressor_cls.Config(**algorithm_config)

        return compressor_cls(config=config)

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
    # -------------------------------------s-----------------------------

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

        self._start_compression_process(
            source_item=item,
            compressor_name=self.compressor_combo.currentText(),
            config_values=config_values,
            experiment_settings=experiment_settings,
        )

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
            self.plot_histogram_button.setEnabled(False)
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
        exactly_one_histogram_item = exactly_one_hsi or exactly_one_compressed

        self.compress_button.setEnabled(False)
        self.compress_decompress_button.setEnabled(exactly_one_hsi)
        self.decompress_button.setEnabled(False)

        # ------------------------------------------------------------
        # Visualization actions
        # ------------------------------------------------------------
        self.show_rgb_button.setEnabled(exactly_one_hsi)
        self.plot_spectra_button.setEnabled(n_hsis >= 1 and n_compressed == 0)
        self.compare_selected_button.setEnabled(n_hsis >= 2 and n_compressed == 0)
        self.plot_histogram_button.setEnabled(exactly_one_histogram_item)

        # This will be enabled later when we track the latest run result.
        self.compare_last_result_button.setEnabled(False)

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------
    
    def _start_compression_process(
        self,
        source_item: WorkspaceItem,
        compressor_name: str,
        config_values: dict,
        experiment_settings: dict,
    ):
        
        if self.compression_process is not None:
            QMessageBox.warning(
                self,
                "Process already running",
                "A compression process is already running.",
            )
            return
        
        if source_item.path is None:
            QMessageBox.warning(
                self,
                "Cannot run compression",
                "This item has no file path. Save or reload it from disk first.",
            )
            return

        self._set_running(True)

        job = {
            "source_path": str(source_item.path),
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

        self.current_job_path = Path(job_file.name)

        with job_file:
            json.dump(job, job_file)

        self.current_process_buffer = ""

        self.compression_process = QProcess(self)
        self.compression_process.setProgram(sys.executable)
        self.compression_process.setArguments([
            "-m",
            "src.ui.gui.processes.compression_job",
            job_file.name,
        ])

        self.compression_process.readyReadStandardOutput.connect(
            self._on_process_stdout
        )
        self.compression_process.readyReadStandardError.connect(
            self._on_process_stderr
        )
        self.compression_process.finished.connect(
            self._on_process_finished
        )
        self.compression_process.errorOccurred.connect(
            self._on_process_error
        )

        self.compression_process.start()

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

    def _on_process_stdout(self):
        if self.compression_process is None:
            return

        data = bytes(
            self.compression_process.readAllStandardOutput()
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
            self._set_run_progress(float(payload.get("value", 0.0)))
            return

        if message_type == "message":
            self._set_run_progress_message(payload.get("message", "Running"))
            return

        if message_type == "error":
            QMessageBox.critical(
                self,
                "Compression failed",
                payload.get("message", "Unknown error"),
            )
            self._set_run_progress_message("Failed")
            return

        if message_type == "finished":
            self._on_process_compression_finished(payload)
            return

    def _on_process_error(self, error):
        self._set_run_progress_message(f"Process error: {error}")
        self._set_running(False)
        self.compression_process = None

    def _on_process_stderr(self):
        if self.compression_process is None:
            return

        data = bytes(
            self.compression_process.readAllStandardError()
        ).decode("utf-8")

        if data.strip():
            print(data)

    def _on_process_finished(self, exit_code: int, exit_status):
        self.compression_process = None

        if getattr(self, "current_job_path", None) is not None:
            self.current_job_path.unlink(missing_ok=True)
            self.current_job_path = None

        if self.abort_requested:
            self._set_run_progress_message("Aborted")
        elif exit_code == 0:
            self.run_progress_bar.setValue(100)
            self._set_run_progress_message("Finished")
        else:
            self._set_run_progress_message("Failed")

        self.abort_requested = False
        self._set_running(False)

    def _abort_compression_process(self):

        if self.compression_process is None:
            return
        
        self.abort_requested = True
        self._set_run_progress_message("Aborting...")

        self.compression_process.terminate()

        QTimer.singleShot(3000, self._kill_compression_process_if_needed)

    def _kill_compression_process_if_needed(self):
        if self.compression_process is None:
            return

        if self.compression_process.state() != QProcess.ProcessState.NotRunning:
            self.compression_process.kill()
            self._set_run_progress_message("Killed")

    def _on_process_compression_finished(self, payload: dict):
        method = payload.get("compressor_name", "unknown")
        metrics = payload.get("metrics", {})

        compressed_path = payload.get("compressed_path")
        reconstructed_path = payload.get("reconstructed_path")

        compressed_item = None
        reconstructed_item = None

        if compressed_path:
            compressed_item = self._load_process_compressed_output(
                Path(compressed_path),
                method,
            )

        if reconstructed_path:
            reconstructed_item = self._load_process_reconstructed_output(
                Path(reconstructed_path),
                method,
                metrics,
            )

        if reconstructed_item is not None:
            self.metrics_table.show_item_metrics(reconstructed_item)

    def _load_process_reconstructed_output(
        self,
        path: Path,
        method: str,
        metrics: dict,
    ) -> WorkspaceItem | None:
        try:
            item = self.workspace_loader.inspect_hsi(path)
        except WorkspaceLoadError as exc:
            QMessageBox.warning(
                self,
                "Could not load reconstructed output",
                str(exc),
            )
            return None

        item.role = WorkspaceItemRole.RECONSTRUCTION
        item.method = method
        item.metrics = self._deserialize_process_metrics(metrics)

        self.add_workspace_item(item)

        return item

    def _load_process_compressed_output(
        self,
        path: Path,
        method: str,
    ) -> WorkspaceItem | None:
        try:
            item = self.workspace_loader.inspect_compressed_hsi(path)
        except WorkspaceLoadError as exc:
            QMessageBox.warning(
                self,
                "Could not load compressed output",
                str(exc),
            )
            return None

        item.method = method

        self.add_workspace_item(item)

        return item

    def _deserialize_process_metrics(self, metrics: dict) -> dict:
        from src.ui.gui.services.metrics_extractor import LoadedMetric

        return {
            name: LoadedMetric(
                value=metric.get("value"),
                unit=metric.get("unit", ""),
            )
            for name, metric in metrics.items()
        }

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
