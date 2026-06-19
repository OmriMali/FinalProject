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

from src.ui.gui.models import WorkspaceItem, CompressionRunSpec
from src.ui.gui.services import WorkspaceLoader, WorkspaceLoadError
from src.ui.gui.widgets import (
    CompressionTab,
    DataAnalysisTab,
    VisualizationTab,
    WorkspacePanel,
)
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
        self.run_queue: list[CompressionRunSpec] = []
        self.current_run_number = 0
        self.total_run_count = 0

        self.compression_controller = CompressionController(self)

        self._build_ui()

        self.visualization_controller = VisualizationController(
            workspace_loader=self.workspace_loader,
            visualization_tab=self.visualization_tab,
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
        tabs.addTab(self._build_visualization_tab(), "Visualization")
        tabs.addTab(self._build_data_analysis_tab(), "Data Analysis")

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
            run_specs = self._build_run_specs(item)
        except ValueError as exc:
            show_warning(self, "Invalid settings", str(exc))
            return

        self._start_run_specs(run_specs)

    def _build_run_specs(
        self,
        source_item: WorkspaceItem,
    ) -> list[CompressionRunSpec]:
        if source_item.path is None:
            raise ValueError(
                "This item has no file path. Save or reload it from disk first."
            )

        experiment_settings = self.compression_tab.read_experiment_settings()
        config_variants = self.compression_tab.read_config_variants()
        compressor_name = self.compression_tab.current_compressor_name()
        base_experiment = experiment_settings["experiment"]

        run_specs = []

        for label, config_values in config_variants:
            run_settings = dict(experiment_settings)

            if label:
                run_settings["experiment"] = f"{base_experiment}__{label}"

            run_specs.append(
                CompressionRunSpec(
                    source_path=source_item.path,
                    compressor_name=compressor_name,
                    config_values=config_values,
                    experiment_settings=run_settings,
                    label=label,
                )
            )

        return run_specs

    def _on_compression_progress(self, value: float):
        if self.total_run_count > 1:
            completed = max(0, self.current_run_number - 1)
            value = (completed + value) / self.total_run_count

        self.compression_tab.set_progress(value)

    def _on_compression_message(self, message: str):
        if self.total_run_count > 1:
            message = (
                f"Run {self.current_run_number}/{self.total_run_count}: "
                f"{message}"
            )

        self.compression_tab.set_message(message)

    def _set_running(self, running: bool):
        self.is_running = running

        self.compression_tab.set_running(running)
        self.workspace_panel.set_controls_enabled(not running)

        self._update_action_buttons()

    # ------------------------------------------------------------------
    # Visualization / Analysis Tabs
    # ------------------------------------------------------------------

    def _build_visualization_tab(self) -> QWidget:
        self.visualization_tab = VisualizationTab()
        return self.visualization_tab

    def _build_data_analysis_tab(self) -> QWidget:
        self.data_analysis_tab = DataAnalysisTab()
        return self.data_analysis_tab

    def _on_show_rgb(self):
        self.visualization_controller.show_rgb(
            self.selected_workspace_items()
        )

    def _on_show_band(self):
        self.visualization_controller.show_band(
            self.selected_workspace_items(),
            band=self.visualization_tab.current_band(),
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
            self._on_compression_started
        )

        self.compression_controller.progress_changed.connect(
            self._on_compression_progress
        )

        self.compression_controller.message_changed.connect(
            self._on_compression_message
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

    def _on_compression_started(self):
        if not self.is_running:
            self._set_running(True)

    def _on_compression_failed(self, message: str):
        self.run_queue.clear()

        show_error(
            self,
            "Compression failed",
            message,
        )

    def _on_compression_finished(self, status: str):
        if status == "finished" and self.run_queue:
            self._start_next_run_spec()
            return

        self.run_queue.clear()
        self.current_run_number = 0
        self.total_run_count = 0
        self._set_running(False)

    def _start_run_specs(
        self,
        run_specs: list[CompressionRunSpec],
    ):
        if self.compression_controller.is_running:
            show_warning(
                self,
                "Process already running",
                "A compression process is already running.",
            )
            return

        if not run_specs:
            return

        self.run_queue = list(run_specs)
        self.current_run_number = 0
        self.total_run_count = len(run_specs)
        self._start_next_run_spec()

    def _start_next_run_spec(self):
        if not self.run_queue:
            return

        spec = self.run_queue.pop(0)
        self.current_run_number += 1

        self.compression_controller.start(
            source_path=spec.source_path,
            compressor_name=spec.compressor_name,
            config_values=spec.config_values,
            experiment_settings=spec.experiment_settings,
        )

    def _abort_compression(self):
        self.run_queue.clear()
        self.compression_controller.abort()

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    def _update_action_buttons(self):
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
        can_plot_histogram = (n_hsis + n_compressed) > 0

        self.compression_tab.set_action_availability(
            can_compress_decompress=exactly_one_hsi and not self.is_running,
        )

        self.visualization_tab.set_action_availability(
            can_show_rgb=n_hsis >= 1 and n_compressed == 0,
            can_show_band=n_hsis >= 1 and n_compressed == 0,
            can_plot_spectra=n_hsis >= 1 and n_compressed == 0,
            can_plot_histogram=can_plot_histogram,
        )

    def _connect_signals(self):
        self.workspace_panel.load_requested.connect(self._on_load_files)
        self.workspace_panel.selection_changed.connect(self._on_workspace_selection_changed)
        self.workspace_panel.workspace_changed.connect(self._on_workspace_changed)
        self.workspace_panel.cleared.connect(self.visualization_tab.clear_canvas)
        self.workspace_panel.cleared.connect(self.compression_tab.clear_workspace_metrics)

        self.compression_tab.compress_decompress_requested.connect(self._on_compress_decompress)
        self.compression_tab.abort_requested.connect(self._abort_compression)

        self.visualization_tab.show_rgb_requested.connect(self._on_show_rgb)
        self.visualization_tab.show_band_requested.connect(self._on_show_band)
        self.visualization_tab.plot_spectra_requested.connect(self._on_plot_spectra)
        self.visualization_tab.plot_histogram_requested.connect(self._on_plot_histogram)

        self._connect_compression_controller()
        self._connect_artifact_controller()

    def _update_metrics_from_selected_items(self):
        items = [
            item
            for item in self.selected_workspace_items()
            if item.metrics is not None
        ]

        self.compression_tab.show_metrics_comparison(items)

    def _connect_artifact_controller(self):
        self.artifact_controller.item_ready.connect(
            self.add_workspace_item
        )

        self.artifact_controller.metrics_item_ready.connect(
            self.compression_tab.show_item_metrics
        )

        self.artifact_controller.warning.connect(
            self._show_warning
        )

    def _show_warning(self, title: str, message: str):
        show_warning(self, title, message)
