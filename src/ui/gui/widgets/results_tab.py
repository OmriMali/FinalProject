from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from src.core.hsi import HSI, CompressedHSI
from src.compressors.base import Compressor
from src.visuals.hsi import (
    plot_rgb,
    select_rgb_bands,
    compare_spectra,
    plot_histogram,
    plot_compressed_histogram,
)
from src.visuals.style import DEFAULT_STYLE
from src.ui.gui.models.workspace_item import WorkspaceItem
from src.ui.gui.widgets.metrics_table import MetricsTableWidget


class ResultsTab(QWidget):
    """
    Results and visualization tab.

    Owns visualization controls, metrics table, Matplotlib canvas,
    and display state.
    """

    show_rgb_requested = Signal()
    compare_selected_requested = Signal()
    plot_spectra_requested = Signal()
    plot_histogram_requested = Signal()

    def __init__(self):
        super().__init__()

        self.current_display_mode = None
        self.current_spectra_hsis: list[HSI] | None = None
        self.current_spectra_labels: list[str] | None = None

        self.spectra_update_timer = QTimer(self)
        self.spectra_update_timer.setSingleShot(True)
        self.spectra_update_timer.setInterval(200)
        self.spectra_update_timer.timeout.connect(
            self._refresh_current_spectra_plot
        )

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_visualization_controls_panel())
        layout.addWidget(self._build_metrics_panel())
        layout.addWidget(self._build_display_panel(), stretch=1)

    def _build_visualization_controls_panel(self) -> QGroupBox:
        box = QGroupBox("Visualization Controls")
        layout = QHBoxLayout(box)

        self.show_rgb_button = QPushButton("Show RGB")
        self.compare_selected_button = QPushButton("Compare Selected")
        self.compare_last_result_button = QPushButton("Compare Last Result")
        self.plot_spectra_button = QPushButton("Plot Spectra")
        self.plot_histogram_button = QPushButton("Histogram")
        self.clear_canvas_button = QPushButton("Clear Canvas")

        self.spectrum_x_spin = QSpinBox()
        self.spectrum_x_spin.setRange(0, 10000)
        self.spectrum_x_spin.setValue(100)
        self.spectrum_x_spin.valueChanged.connect(
            self._schedule_spectra_update
        )

        self.spectrum_y_spin = QSpinBox()
        self.spectrum_y_spin.setRange(0, 10000)
        self.spectrum_y_spin.setValue(100)
        self.spectrum_y_spin.valueChanged.connect(
            self._schedule_spectra_update
        )

        self.show_rgb_button.clicked.connect(self.show_rgb_requested.emit)
        self.compare_selected_button.clicked.connect(
            self.compare_selected_requested.emit
        )
        self.plot_spectra_button.clicked.connect(
            self.plot_spectra_requested.emit
        )
        self.plot_histogram_button.clicked.connect(
            self.plot_histogram_requested.emit
        )
        self.clear_canvas_button.clicked.connect(self.clear_canvas)

        layout.addWidget(self.show_rgb_button)
        layout.addWidget(self.compare_selected_button)
        layout.addWidget(self.compare_last_result_button)

        layout.addWidget(QLabel("x"))
        layout.addWidget(self.spectrum_x_spin)

        layout.addWidget(QLabel("y"))
        layout.addWidget(self.spectrum_y_spin)

        layout.addWidget(self.plot_spectra_button)
        layout.addWidget(self.plot_histogram_button)

        layout.addStretch()

        layout.addWidget(self.clear_canvas_button)

        self.set_action_availability()

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

    def set_action_availability(
        self,
        can_show_rgb: bool = False,
        can_compare_selected: bool = False,
        can_compare_last_result: bool = False,
        can_plot_spectra: bool = False,
        can_plot_histogram: bool = False,
    ):
        self.show_rgb_button.setEnabled(can_show_rgb)
        self.compare_selected_button.setEnabled(can_compare_selected)
        self.compare_last_result_button.setEnabled(can_compare_last_result)
        self.plot_spectra_button.setEnabled(can_plot_spectra)
        self.plot_histogram_button.setEnabled(can_plot_histogram)

    def show_item_metrics(self, item: WorkspaceItem):
        self.metrics_table.show_item_metrics(item)

    def show_metrics_comparison(self, items: list[WorkspaceItem]):
        self.metrics_table.show_metrics_comparison(items)

    def display_rgb(self, hsi: HSI, title: str | None = None):
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
        self.display_canvas.draw_idle()

    def display_rgb_comparison(
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
        self.display_canvas.draw_idle()

    def start_spectra_plot(
        self,
        hsis: list[HSI],
        labels: list[str],
    ):
        self.current_display_mode = "spectra"
        self.current_spectra_hsis = hsis
        self.current_spectra_labels = labels

        self._set_spectrum_pixel_limits(hsis[0])

        pixel = (
            self.spectrum_x_spin.value(),
            self.spectrum_y_spin.value(),
        )

        self._display_spectra(hsis, labels, pixel)

    def display_histograms(
        self,
        objs: list,
        labels: list[str],
        compressors: list
    ):
        self.current_display_mode = "histogram"
        self.current_spectra_hsis = None
        self.current_spectra_labels = None

        self.display_figure.clear()

        n_items = len(objs)
        n_cols = min(3, n_items)
        n_rows = math.ceil(n_items / n_cols)

        # Create subplots completely unlinked at first
        axes = self.display_figure.subplots(
            n_rows,
            n_cols,
            squeeze=False,
        ).ravel()

        first_hsi_ax = None  # We will use this to sync all subsequent HSI plots

        for i, (obj, label, compressor) in enumerate(zip(objs, labels, compressors)):
            ax = axes[i]
            
            if isinstance(obj, HSI):
                # If this is our first HSI, save its axis. 
                # If it's a subsequent HSI, link it to the first one!
                if first_hsi_ax is None:
                    first_hsi_ax = ax
                else:
                    ax.sharex(first_hsi_ax)
                    ax.sharey(first_hsi_ax)

                plot_histogram(
                    hsi=obj,
                    band=None,
                    bins=256,
                    style=DEFAULT_STYLE,
                    ax=ax,
                    title=label,
                )
            else:
                # Compressed items ignore the linking and scale independently
                try:
                    plot_compressed_histogram(
                        compressed=obj,
                        compressor=compressor,
                        bins=256,
                        style=DEFAULT_STYLE,
                        ax=ax,
                        title=label,
                    )
                except Exception as exc:
                    print(f"Error plotting {label}: {exc}")
                    continue

        # Hide any unused subplots
        for ax in axes[n_items:]:
            ax.set_axis_off()

        self.display_figure.tight_layout()
        self.display_canvas.draw_idle()

    def clear_canvas(self):
        self.current_display_mode = None
        self.current_spectra_hsis = None
        self.current_spectra_labels = None

        self.display_figure.clear()
        self.display_canvas.draw_idle()

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