from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
    QFileDialog,
    QMessageBox,
    QTabWidget,
)

from src.core.hsi import HSI, CompressedHSI
from src.compressors.registry import get_compressor

from src.ui.gui.models import WorkspaceItem, WorkspaceItemKind, WorkspaceItemRole
from src.ui.gui.services import WorkspaceLoader, WorkspaceLoadError
from src.ui.gui.widgets import WorkspacePanel, CompressionTab, ResultsTab
from src.ui.gui.controllers import CompressionController

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

        self.workspace_loader = WorkspaceLoader()

        self.is_running = False

        self.compression_controller = CompressionController(self)

        self._build_ui()

        self._connect_compression_controller()


    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        workspace_column = self._build_workspace_column()
        main_tabs = self._build_main_tabs()

        # Cross-widget connections that require both widgets to exist.
        self.workspace_panel.cleared.connect(self.results_tab.clear_canvas)

        splitter.addWidget(workspace_column)
        splitter.addWidget(main_tabs)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([420, 1080])

        root_layout.addWidget(splitter)


    def _build_main_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        tabs.addTab(self._build_compression_tab(), "Compression")
        tabs.addTab(self._build_results_tab(), "Results")

        return tabs

    def _build_compression_tab(self) -> QWidget:
        self.compression_tab = CompressionTab()

        self.compression_tab.compress_decompress_requested.connect(
            self._on_compress_decompress
        )
        self.compression_tab.abort_requested.connect(
            self._abort_compression_process
        )

        return self.compression_tab

    def _build_results_tab(self) -> QWidget:
        self.results_tab = ResultsTab()

        self.results_tab.show_rgb_requested.connect(self._on_show_rgb)
        self.results_tab.compare_selected_requested.connect(
            self._on_compare_selected
        )
        self.results_tab.plot_spectra_requested.connect(self._on_plot_spectra)
        self.results_tab.plot_histogram_requested.connect(self._on_plot_histogram)

        return self.results_tab


    def _build_workspace_column(self) -> QWidget:
        self.workspace_panel = WorkspacePanel()

        self.workspace_panel.load_hsi_requested.connect(self._on_load_hsi)
        self.workspace_panel.load_compressed_hsi_requested.connect(
            self._on_load_compressed_hsi
        )

        self.workspace_panel.selection_changed.connect(
            self._on_workspace_selection_changed
        )
        self.workspace_panel.workspace_changed.connect(
            self._on_workspace_changed
        )

        return self.workspace_panel
    
    def _on_workspace_selection_changed(self):
        self._update_action_buttons()
        self._update_metrics_from_checked_items()

    def _on_workspace_changed(self):
        self._update_action_buttons()
        self._update_metrics_from_checked_items()

    # ------------------------------------------------------------------
    # Workspace item helpers
    # ------------------------------------------------------------------

    def add_workspace_item(self, item: WorkspaceItem):
        self.workspace_panel.add_workspace_item(item)

    def clear_workspace_items(self):
        self.workspace_panel.clear_workspace_items()

    def checked_workspace_items(self) -> list[WorkspaceItem]:
        return self.workspace_panel.checked_workspace_items()

    def checked_workspace_item(self) -> WorkspaceItem | None:
        return self.workspace_panel.checked_workspace_item()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

   
    def _set_run_progress(self, value: float):
        self.compression_tab.set_progress(value)

    def _set_run_progress_message(self, message: str):
        self.compression_tab.set_message(message)

    def _set_running(self, running: bool):
        self.is_running = running

        self.compression_tab.set_running(running)
        self.workspace_panel.set_controls_enabled(not running)

        self._update_action_buttons()

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

        self.workspace_panel.add_workspace_item(item)

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

        self.workspace_panel.add_workspace_item(item)

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

        self.results_tab.display_rgb(obj, title=item.plot_label)

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

        self.results_tab.display_rgb_comparison(hsis, labels)

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

        self.results_tab.start_spectra_plot(hsis, labels)

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
            self.results_tab.display_hsi_histogram(
                hsi=obj,
                title=f"{item.plot_label}_Histogram",
            )
            return

        if isinstance(obj, CompressedHSI):
            try:
                compressor = self._compressor_from_compressed_hsi(obj)
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Histogram",
                    f"Could not prepare compressed histogram decoder:\n{exc}",
                )
                return

            self.results_tab.display_compressed_histogram(
                compressed=obj,
                compressor=compressor,
                title=f"{item.plot_label}_Compressed_Histogram",
            )
            return

        QMessageBox.warning(
            self,
            "Histogram",
            "Selected item is not an HSI or CompressedHSI.",
        )

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
            config_values = self.compression_tab.read_compressor_config_values()
            experiment_settings = self.compression_tab.read_experiment_settings()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        self._start_compression_process(
            source_item=item,
            compressor_name=self.compression_tab.current_compressor_name(),
            config_values=config_values,
            experiment_settings=experiment_settings,
        )

    # ------------------------------------------------------------------
    # Button Updating
    # ------------------------------------------------------------------

    def _update_action_buttons(self):

        if self.is_running:
            self.compression_tab.set_action_availability(
                can_compress=False,
                can_decompress=False,
                can_compress_decompress=False,
            )
            self.results_tab.set_action_availability()
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

        exactly_one_hsi = n_checked == 1 and n_hsis == 1
        exactly_one_compressed = n_checked == 1 and n_compressed == 1
        exactly_one_histogram_item = exactly_one_hsi or exactly_one_compressed

    
        self.compression_tab.set_action_availability(
            can_compress=False,
            can_decompress=False,
            can_compress_decompress=exactly_one_hsi,
        )

        self.results_tab.set_action_availability(
            can_show_rgb=exactly_one_hsi,
            can_compare_selected=n_hsis >= 2 and n_compressed == 0,
            can_compare_last_result=False,
            can_plot_spectra=n_hsis >= 1 and n_compressed == 0,
            can_plot_histogram=exactly_one_histogram_item,
        )


    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------
        
    def _connect_compression_controller(self):
        self.compression_controller.started.connect(
            lambda: self._set_running(True)
        )

        self.compression_controller.progress_changed.connect(
            self._set_run_progress
        )

        self.compression_controller.message_changed.connect(
            self._set_run_progress_message
        )

        self.compression_controller.failed.connect(
            self._on_compression_failed
        )

        self.compression_controller.finished_payload.connect(
            self._on_process_compression_finished
        )

        self.compression_controller.run_ended.connect(
            self._on_compression_run_ended
        )

    def _on_compression_failed(self, message: str):
        QMessageBox.critical(
            self,
            "Compression failed",
            message,
        )

    def _on_compression_run_ended(self, status: str):
        self._set_running(False)


    def _start_compression_process(
        self,
        source_item: WorkspaceItem,
        compressor_name: str,
        config_values: dict,
        experiment_settings: dict,
    ):
        if self.compression_controller.is_running:
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

        self.compression_controller.start(
            source_path=source_item.path,
            compressor_name=compressor_name,
            config_values=config_values,
            experiment_settings=experiment_settings,
        )

    def _abort_compression_process(self):
        self.compression_controller.abort()


    def _on_process_compression_finished(self, payload: dict):
        method = payload.get("compressor_name", "unknown")
        metrics = payload.get("metrics", {})

        reconstructed_path = payload.get("reconstructed_path")
        reconstructed_item = None

        if reconstructed_path:
            reconstructed_item = self._load_process_reconstructed_output(
                Path(reconstructed_path),
                method,
                metrics,
            )

        if reconstructed_item is not None:
            self.results_tab.show_item_metrics(reconstructed_item)

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

        self.results_tab.show_metrics_comparison(items)
