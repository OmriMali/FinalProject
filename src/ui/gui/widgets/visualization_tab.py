from __future__ import annotations

import math

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from src.compressors.base import Compressor
from src.core.hsi import CompressedHSI, HSI
from src.visuals.hsi import (
    compare_spectra,
    plot_compressed_histogram,
    plot_histogram,
    plot_rgb,
    select_rgb_bands,
)
from src.visuals.style import DEFAULT_STYLE
from src.ui.gui.widgets.figure_popout import FigurePopoutWindow


class VisualizationTab(QWidget):
    """
    HSI visualization tab.

    Owns visualization controls, Matplotlib canvas, and display state.
    """

    show_rgb_requested = Signal()
    show_band_requested = Signal()
    plot_spectra_requested = Signal()
    plot_histogram_requested = Signal()

    def __init__(self):
        super().__init__()

        self._set_display_state(None)
        self.popout_windows: list[FigurePopoutWindow] = []

        self.spectra_update_timer = QTimer(self)
        self.spectra_update_timer.setSingleShot(True)
        self.spectra_update_timer.setInterval(100)
        self.spectra_update_timer.timeout.connect(
            self._refresh_current_spectra_plot
        )

        self.band_update_timer = QTimer(self)
        self.band_update_timer.setSingleShot(True)
        self.band_update_timer.setInterval(100)
        self.band_update_timer.timeout.connect(self._refresh_current_band_plot)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_visualization_controls_panel())
        layout.addWidget(self._build_display_panel(), stretch=1)

    def _build_visualization_controls_panel(self) -> QGroupBox:
        box = QGroupBox("Visualization Controls")
        layout = QHBoxLayout(box)

        self.show_rgb_button = QPushButton("Show RGB")
        self.show_band_button = QPushButton("Show Band")
        self.plot_spectra_button = QPushButton("Plot Spectra")
        self.plot_histogram_button = QPushButton("Histogram")
        self.clear_canvas_button = QPushButton("Clear Canvas")

        self.band_spin = QSpinBox()
        self.band_spin.setRange(0, 10000)
        self.band_spin.setValue(0)

        self.spectrum_x_spin = QSpinBox()
        self.spectrum_x_spin.setRange(0, 10000)
        self.spectrum_x_spin.setValue(100)

        self.spectrum_y_spin = QSpinBox()
        self.spectrum_y_spin.setRange(0, 10000)
        self.spectrum_y_spin.setValue(100)

        self.show_rgb_button.clicked.connect(self.show_rgb_requested.emit)
        self.show_band_button.clicked.connect(self.show_band_requested.emit)
        self.plot_spectra_button.clicked.connect(self.plot_spectra_requested.emit)
        self.plot_histogram_button.clicked.connect(self.plot_histogram_requested.emit)
        self.clear_canvas_button.clicked.connect(self.clear_canvas)

        self.band_spin.valueChanged.connect(self._schedule_band_update)
        self.spectrum_x_spin.valueChanged.connect(self._schedule_spectra_update)
        self.spectrum_y_spin.valueChanged.connect(self._schedule_spectra_update)

        layout.addWidget(self.show_rgb_button)

        layout.addWidget(QLabel("Band"))
        layout.addWidget(self.band_spin)
        layout.addWidget(self.show_band_button)

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

    def _build_display_panel(self) -> QGroupBox:
        box = QGroupBox("Display")
        layout = QVBoxLayout(box)
        toolbar_layout = QHBoxLayout()

        self.display_figure = Figure(figsize=(6, 5))
        self.display_canvas = FigureCanvas(self.display_figure)

        self.display_canvas.mpl_connect(
            "button_press_event",
            self._on_canvas_clicked,
        )

        self.display_toolbar = NavigationToolbar(
            self.display_canvas,
            self,
        )
        self.popout_canvas_button = QPushButton("Pop Out")
        self.popout_canvas_button.clicked.connect(self.popout_canvas)

        toolbar_layout.addWidget(self.display_toolbar)
        toolbar_layout.addWidget(self.popout_canvas_button)

        layout.addLayout(toolbar_layout)
        layout.addWidget(self.display_canvas, stretch=1)

        return box

    def set_action_availability(
        self,
        can_show_rgb: bool = False,
        can_show_band: bool = False,
        can_plot_spectra: bool = False,
        can_plot_histogram: bool = False,
    ):
        self.show_rgb_button.setEnabled(can_show_rgb)
        self.show_band_button.setEnabled(can_show_band)
        self.plot_spectra_button.setEnabled(can_plot_spectra)
        self.plot_histogram_button.setEnabled(can_plot_histogram)

    def display_rgb(self, hsi: HSI, title: str | None = None):
        self._set_display_state("rgb")

        self.display_figure.clear()
        ax = self.display_figure.add_subplot(1, 1, 1)

        self._set_spectrum_pixel_limits(hsi)

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
        self._set_display_state("rgb")

        self.display_figure.clear()

        self._set_spectrum_pixel_limits(hsis[0])

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
        self._set_display_state(
            "spectra",
            spectra_hsis=hsis,
            spectra_labels=labels,
        )

        self._set_spectrum_pixel_limits(hsis[0])

        pixel = (
            self.spectrum_x_spin.value(),
            self.spectrum_y_spin.value(),
        )

        self._display_spectra(hsis, labels, pixel)

    def display_hsi_histogram(
        self,
        hsi: HSI,
        title: str | None = None,
    ):
        self._set_display_state("histogram")

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

    def display_compressed_histogram(
        self,
        compressed: CompressedHSI,
        compressor: Compressor,
        title: str | None = None,
    ):
        self._set_display_state("compressed_histogram")

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

    def clear_canvas(self):
        self._set_display_state(None)

        self.display_figure.clear()
        self.display_canvas.draw_idle()

    def popout_canvas(self):
        if not self.display_figure.axes:
            QMessageBox.information(
                self,
                "Pop out plot",
                "There is no visualization to pop out yet.",
            )
            return

        window = FigurePopoutWindow(
            self.display_figure,
            "Visualization",
            self,
        )
        window.finished.connect(
            lambda _result, popout=window: self._forget_popout(popout)
        )
        self.popout_windows.append(window)
        window.show()

    def _forget_popout(self, window: FigurePopoutWindow):
        if window in self.popout_windows:
            self.popout_windows.remove(window)

    def _set_display_state(
        self,
        mode: str | None,
        spectra_hsis: list[HSI] | None = None,
        spectra_labels: list[str] | None = None,
        band_hsis: list[HSI] | None = None,
        band_labels: list[str] | None = None,
    ) -> None:
        self.current_display_mode = mode
        self.current_spectra_hsis = spectra_hsis
        self.current_spectra_labels = spectra_labels
        self.current_band_hsis = band_hsis
        self.current_band_labels = band_labels

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

        if self.current_spectra_labels is None:
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

    def current_band(self) -> int:
        return self.band_spin.value()

    def set_band_limits(self, hsi: HSI):
        n_bands = hsi.bands

        self.band_spin.blockSignals(True)
        self.band_spin.setRange(0, n_bands - 1)
        self.band_spin.blockSignals(False)

    def display_band(
        self,
        hsi: HSI,
        band: int,
        label: str | None = None,
        title: str | None = None,
    ):
        self._set_display_state(
            "band",
            band_hsis=[hsi],
            band_labels=[label or title or "HSI"],
        )

        if title is None:
            if label is not None:
                title = f"{label}_Band_{band}"
            else:
                title = f"Band {band}"

        if band < 0 or band >= hsi.data.shape[2]:
            raise ValueError(
                f"Band index {band} is outside valid range 0-{hsi.data.shape[2] - 1}"
            )

        self._set_spectrum_pixel_limits(hsi)
        self.set_band_limits(hsi)

        self.display_figure.clear()
        ax = self.display_figure.add_subplot(1, 1, 1)

        image = hsi.data[:, :, band]

        ax.imshow(image, cmap="gray")
        ax.set_axis_off()

        ax.set_title(title)

        self.display_figure.tight_layout()
        self.display_canvas.draw_idle()

    def display_band_comparison(
        self,
        hsis: list[HSI],
        labels: list[str],
        band: int,
    ):
        self._set_display_state(
            "band",
            band_hsis=hsis,
            band_labels=labels,
        )

        if band < 0 or band >= hsis[0].data.shape[2]:
            raise ValueError(
                f"Band index {band} is outside valid range 0-{hsis[0].data.shape[2] - 1}"
            )

        self._set_spectrum_pixel_limits(hsis[0])
        self.set_band_limits(hsis[0])

        self.display_figure.clear()

        n_images = len(hsis)
        n_cols = min(3, n_images)
        n_rows = math.ceil(n_images / n_cols)

        axes = self.display_figure.subplots(
            n_rows,
            n_cols,
            squeeze=False,
        ).ravel()

        for ax, hsi, label in zip(axes, hsis, labels):
            if band >= hsi.data.shape[2]:
                ax.set_title(f"{label}\nBand unavailable")
                ax.set_axis_off()
                continue

            ax.imshow(hsi.data[:, :, band], cmap="gray")
            ax.set_title(f"{label}\nBand {band}")
            ax.set_axis_off()

        for ax in axes[n_images:]:
            ax.set_axis_off()

        self.display_figure.tight_layout()
        self.display_canvas.draw_idle()

    def _on_canvas_clicked(self, event):
        if self.current_display_mode not in {"rgb", "band"}:
            return
        if event.inaxes is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if x < self.spectrum_x_spin.minimum() or x > self.spectrum_x_spin.maximum():
            return

        if y < self.spectrum_y_spin.minimum() or y > self.spectrum_y_spin.maximum():
            return

        self.spectrum_x_spin.setValue(x)
        self.spectrum_y_spin.setValue(y)

    def set_band_limits_for_hsis(self, hsis: list[HSI]):
        if not hsis:
            return

        n_bands = min(hsi.data.shape[2] for hsi in hsis)

        self.band_spin.blockSignals(True)
        self.band_spin.setRange(0, n_bands - 1)

        if self.band_spin.value() > n_bands - 1:
            self.band_spin.setValue(n_bands - 1)

        self.band_spin.blockSignals(False)

    def _schedule_band_update(self):
        if self.current_display_mode != "band":
            return

        if self.current_band_hsis is None:
            return

        self.band_update_timer.start()

    def _refresh_current_band_plot(self):
        if self.current_display_mode != "band":
            return

        if self.current_band_hsis is None:
            return

        if self.current_band_labels is None:
            return

        band = self.current_band()

        try:
            if len(self.current_band_hsis) == 1:
                self.display_band(
                    hsi=self.current_band_hsis[0],
                    band=band,
                    label=self.current_band_labels[0],
                )
                return

            self.display_band_comparison(
                hsis=self.current_band_hsis,
                labels=self.current_band_labels,
                band=band,
            )
        except ValueError:
            pass
