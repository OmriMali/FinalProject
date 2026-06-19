from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

from src.core.hsi import HSI, CompressedHSI
from src.compressors.registry import get_compressor

from src.ui.gui.models import WorkspaceItem
from src.ui.gui.services import WorkspaceLoader, WorkspaceLoadError
from src.ui.gui.widgets import ResultsTab


class VisualizationController:
    """
    Handles visualization actions for workspace items.

    The controller owns:
    - loading selected workspace objects
    - validating object types
    - choosing the correct visualization path
    - constructing compressors for compressed-domain histograms

    ResultsTab owns:
    - buttons
    - metrics table
    - matplotlib canvas
    - actual display methods
    """

    def __init__(
        self,
        workspace_loader: WorkspaceLoader,
        results_tab: ResultsTab,
        parent: QWidget | None = None,
    ):
        self.workspace_loader = workspace_loader
        self.results_tab = results_tab
        self.parent = parent

    def show_rgb(self, items: list[WorkspaceItem]):
        if not items:
            self._warn(
                "Show RGB",
                "Please check at least one HSI item.",
            )
            return

        hsis = []
        labels = []

        for item in items:
            obj = self._load_object(item)
            if obj is None:
                return

            if not isinstance(obj, HSI):
                self._warn(
                    "Show RGB",
                    "Only HSI items can be displayed as RGB.",
                )
                return

            hsis.append(obj)
            labels.append(item.plot_label)

        if len(hsis) == 1:
            self.results_tab.display_rgb(
                hsis[0],
                title=labels[0],
            )
            return

        self.results_tab.display_rgb_comparison(hsis, labels)

    def show_band(
        self,
        items: list[WorkspaceItem],
        band: int,
    ):
        if not items:
            self._warn(
                "Show Band",
                "Please check at least one HSI item.",
            )
            return

        hsis = []
        labels = []

        for item in items:
            obj = self._load_object(item)
            if obj is None:
                return

            if not isinstance(obj, HSI):
                self._warn(
                    "Show Band",
                    "Only HSI items can be displayed by band.",
                )
                return

            hsis.append(obj)
            labels.append(item.plot_label)

        self.results_tab.set_band_limits_for_hsis(hsis)
        band = self.results_tab.current_band()

        try:
            if len(hsis) == 1:
                self.results_tab.display_band(
                    hsi=hsis[0],
                    band=band,
                    label=labels[0],
                )
                return

            self.results_tab.display_band_comparison(
                hsis=hsis,
                labels=labels,
                band=band,
            )
        except ValueError as exc:
            self._warn("Show Band", str(exc))


    def plot_spectra(self, items: list[WorkspaceItem]):
        if not items:
            self._warn(
                "Plot Spectra",
                "Please check at least one HSI item.",
            )
            return

        hsis = []
        labels = []

        for item in items:
            obj = self._load_object(item)
            if obj is None:
                return

            if not isinstance(obj, HSI):
                self._warn(
                    "Plot Spectra",
                    "Only HSI items can be used for spectral plots.",
                )
                return

            hsis.append(obj)
            labels.append(item.plot_label)

        self.results_tab.start_spectra_plot(hsis, labels)

    def plot_histogram(self, items: list[WorkspaceItem]):
        item = self._single_item(
            items,
            title="Histogram",
            message="Please check exactly one item.",
        )

        if item is None:
            return

        obj = self._load_object(item)
        if obj is None:
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
                self._warn(
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

        self._warn(
            "Histogram",
            "Selected item is not an HSI or CompressedHSI.",
        )

    def _load_object(self, item: WorkspaceItem):
        try:
            return self.workspace_loader.load_object(item)
        except WorkspaceLoadError as exc:
            QMessageBox.critical(
                self.parent,
                "Load failed",
                str(exc),
            )
            return None

    def _single_item(
        self,
        items: list[WorkspaceItem],
        title: str,
        message: str,
    ) -> WorkspaceItem | None:
        if len(items) != 1:
            self._warn(title, message)
            return None

        return items[0]

    def _compressor_from_compressed_hsi(self, compressed: CompressedHSI):
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

    def _warn(self, title: str, message: str):
        QMessageBox.warning(
            self.parent,
            title,
            message,
        )