from __future__ import annotations

import io
import pickle

from PySide6.QtWidgets import QDialog, QVBoxLayout

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class FigurePopoutWindow(QDialog):
    """
    Standalone window showing a snapshot of an existing Matplotlib figure.
    """

    def __init__(
        self,
        source_figure: Figure,
        title: str,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.resize(1100, 800)

        layout = QVBoxLayout(self)

        self.figure = self._clone_figure(source_figure)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self.canvas.draw_idle()

    def _clone_figure(self, source_figure: Figure) -> Figure:
        buffer = io.BytesIO()
        pickle.dump(source_figure, buffer)
        buffer.seek(0)

        figure = pickle.load(buffer)
        figure.set_size_inches(11, 8, forward=True)

        return figure
