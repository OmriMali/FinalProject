from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
    QFileDialog,
    QTabWidget,
)

from src.ui.gui.models import WorkspaceItem
from src.ui.gui.services import WorkspaceLoader, WorkspaceLoadError
from src.ui.gui.widgets import WorkspacePanel, CompressionTab, ResultsTab
from src.ui.gui.controllers import CompressionController, VisualizationController, ArtifactController
from src.ui.gui.utils import show_error, show_warning

class MainWindow(QMainWindow):
    """
    Main GUI shell for hyperspectral image compression.

    Owns the top-level layout, shared services, controllers, and signal wiring.
    Workflow logic is delegated to widgets and controllers.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("HSI Compression GUI")
        self.resize(1500, 850)

        self.workspace_loader = WorkspaceLoader()

        self.is_running = False

        self.compression_controller = CompressionController(self)

        self._build_ui()

        self.visualization_controller = VisualizationController(
            workspace_loader=self.workspace_loader,
            results_tab=self.results_tab,
            parent=self,
            )
        
        self.artifact_controller = ArtifactController(
            workspace_loader=self.workspace_loader,
            parent=self,
        )

        self._connect_signals()
        self._update_action_buttons()


    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        workspace_column = self._build_workspace_column()
        main_tabs = self._build_main_tabs()

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

    # ------------------------------------------------------------------
    # Compression Tab
    # ------------------------------------------------------------------

    def _build_compression_tab(self) -> QWidget:
        self.compression_tab = CompressionTab()
        return self.compression_tab
    
    def _on_compress_decompress(self):
        item = self.selected_workspace_item()

        if item is None:
            show_warning(
                self,
                "Compress + Decompress",
                "Please check exactly one HSI item.",
            )
            return

        if not item.is_hsi:
            show_warning(
                self,
                "Compress + Decompress",
                "Compress + Decompress requires an HSI item.",
            )
            return

        try:
            config_values = self.compression_tab.read_compressor_config_values()
            experiment_settings = self.compression_tab.read_experiment_settings()
        except ValueError as exc:
            show_warning(self, "Invalid settings", str(exc))
            return

        self._start_compression(
            source_item=item,
            compressor_name=self.compression_tab.current_compressor_name(),
            config_values=config_values,
            experiment_settings=experiment_settings,
        )

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
    # Results Tab
    # ------------------------------------------------------------------

    def _build_results_tab(self) -> QWidget:
        self.results_tab = ResultsTab()
        return self.results_tab

    def _on_show_rgb(self):
        self.visualization_controller.show_rgb(
            self.selected_workspace_items()
        )

    def _on_show_band(self):
        self.visualization_controller.show_band(
            self.selected_workspace_items(),
            band=self.results_tab.current_band(),
        )

    def _on_plot_spectra(self):
        self.visualization_controller.plot_spectra(
            self.selected_workspace_items()
        )

    def _on_plot_histogram(self):
        self.visualization_controller.plot_histogram(
            self.selected_workspace_items()
        )

    # ------------------------------------------------------------------
    # Workspace Column
    # ------------------------------------------------------------------

    def _build_workspace_column(self) -> QWidget:
        self.workspace_panel = WorkspacePanel()
        return self.workspace_panel
    
    def _on_workspace_selection_changed(self):
        self._update_action_buttons()
        self._update_metrics_from_selected_items()

    def _on_workspace_changed(self):
        self._update_action_buttons()
        self._update_metrics_from_selected_items()

    def add_workspace_item(self, item: WorkspaceItem):
        self.workspace_panel.add_workspace_item(item)

    def clear_workspace_items(self):
        self.workspace_panel.clear_workspace_items()

    def selected_workspace_items(self) -> list[WorkspaceItem]:
        return self.workspace_panel.selected_workspace_items()

    def selected_workspace_item(self) -> WorkspaceItem | None:
        return self.workspace_panel.selected_workspace_item()
      
    def _on_load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load HSI or CompressedHSI files",
            "",
            "NumPy files (*.npz);;All files (*.*)",
        )

        if not paths:
            return

        for path_text in paths:
            path = Path(path_text)

            try:
                item = self._inspect_workspace_file(path)
            except WorkspaceLoadError as exc:
                show_error(self, "Load failed", f"{path}\n\n{exc}")
                continue

            self.workspace_panel.add_workspace_item(item)

    def _inspect_workspace_file(self, path: Path) -> WorkspaceItem:
        try:
            return self.workspace_loader.inspect_hsi(path)
        except WorkspaceLoadError as hsi_error:
            try:
                return self.workspace_loader.inspect_compressed_hsi(path)
            except WorkspaceLoadError as compressed_error:
                raise WorkspaceLoadError(
                    "File could not be loaded as either HSI or CompressedHSI.\n\n"
                    f"HSI error: {hsi_error}\n\n"
                    f"CompressedHSI error: {compressed_error}"
                )


    # ------------------------------------------------------------------
    # Process Handling
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
            self.artifact_controller.handle_compression_finished
        )

        self.compression_controller.run_ended.connect(
            self._on_compression_finished
        )

    def _on_compression_failed(self, message: str):
        show_error(
            self,
            "Compression failed",
            message,
        )

    def _on_compression_finished(self, status: str):
        self._set_running(False)

    def _start_compression(
        self,
        source_item: WorkspaceItem,
        compressor_name: str,
        config_values: dict,
        experiment_settings: dict,
    ):
        if self.compression_controller.is_running:
            show_warning(
                self,
                "Process already running",
                "A compression process is already running.",
            )
            return

        if source_item.path is None:
            show_warning(
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

    def _abort_compression(self):
        self.compression_controller.abort()

    # ------------------------------------------------------------------
    # General
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

        selected_items = self.selected_workspace_items()

        selected_hsis = [
            item
            for item in selected_items
            if item.is_hsi
        ]

        selected_compressed = [
            item
            for item in selected_items
            if item.is_compressed
        ]

        n_selected = len(selected_items)
        n_hsis = len(selected_hsis)
        n_compressed = len(selected_compressed)

        exactly_one_hsi = n_selected == 1 and n_hsis == 1
        exactly_one_compressed = n_selected == 1 and n_compressed == 1
        exactly_one_histogram_item = exactly_one_hsi or exactly_one_compressed

    
        self.compression_tab.set_action_availability(
            can_compress=False,
            can_decompress=False,
            can_compress_decompress=exactly_one_hsi,
        )

        self.results_tab.set_action_availability(
            can_show_rgb=n_hsis >= 1 and n_compressed == 0,
            can_show_band=n_hsis >= 1 and n_compressed == 0,
            can_plot_spectra=n_hsis >= 1 and n_compressed == 0,
            can_plot_histogram=exactly_one_histogram_item,
        )

    def _connect_signals(self):
        self.workspace_panel.load_requested.connect(self._on_load_files)
        self.workspace_panel.selection_changed.connect(self._on_workspace_selection_changed)
        self.workspace_panel.workspace_changed.connect(self._on_workspace_changed)
        self.workspace_panel.cleared.connect(self.results_tab.clear_canvas)

        self.compression_tab.compress_decompress_requested.connect(self._on_compress_decompress)
        self.compression_tab.abort_requested.connect(self._abort_compression)

        self.results_tab.show_rgb_requested.connect(self._on_show_rgb)
        self.results_tab.show_band_requested.connect(self._on_show_band)
        self.results_tab.plot_spectra_requested.connect(self._on_plot_spectra)
        self.results_tab.plot_histogram_requested.connect(self._on_plot_histogram)

        self._connect_compression_controller()
        self._connect_artifact_controller()

    def _update_metrics_from_selected_items(self):
        items = [
            item
            for item in self.selected_workspace_items()
            if item.metrics is not None
        ]

        self.results_tab.show_metrics_comparison(items)

    def _connect_artifact_controller(self):
        self.artifact_controller.item_ready.connect(
            self.add_workspace_item
        )

        self.artifact_controller.metrics_item_ready.connect(
            self.results_tab.show_item_metrics
        )

        self.artifact_controller.warning.connect(
            self._show_warning
        )

    def _show_warning(self, title: str, message: str):
        show_warning(self, title, message)