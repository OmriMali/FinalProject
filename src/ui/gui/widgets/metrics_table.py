from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem
from PySide6.QtCore import Qt

from src.ui.gui.models.workspace_item import WorkspaceItem


class MetricsTableWidget(QTableWidget):
    """
    Table widget for comparing metrics across workspace items.

    Rows are workspace items.
    Columns are metric names.
    """

    DEFAULT_METRIC_ORDER = (
        "RMSE",
        "PSNR",
        "SAM",
        "CR",
        "comp_time",
        "decomp_time",
    )

    def __init__(self):
        super().__init__()

        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)

        self.show_no_metrics()

    def show_no_metrics(self):
        self.clear()
        self.setRowCount(0)
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels(["#"])

    def show_item_metrics(self, item: WorkspaceItem):
        self.show_metrics_comparison([item])

    def show_metrics_comparison(self, items: list[WorkspaceItem]):
        items = [
            item
            for item in items
            if item.metrics is not None
        ]

        if not items:
            self.show_no_metrics()
            return

        metric_names = self._ordered_metric_names(items)

        headers = ["#"] + metric_names

        self.clear()
        self.setColumnCount(len(headers))
        self._set_headers(headers)
        self.setRowCount(len(items))

        for row, item in enumerate(items):
            self.setItem(
                row,
                0,
                self._make_item(self._item_label(item)),
            )

            for col, metric_name in enumerate(metric_names, start=1):
                metric = item.metrics.get(metric_name)

                if metric is None:
                    value_text = "-"
                else:
                    value_text = self._format_metric(metric)

                self.setItem(
                    row,
                    col,
                    QTableWidgetItem(value_text),
                )

        self.resizeColumnsToContents()

    def _ordered_metric_names(
        self,
        items: list[WorkspaceItem],
    ) -> list[str]:
        available = {
            metric_name
            for item in items
            for metric_name in item.metrics.keys()
        }

        ordered = [
            metric_name
            for metric_name in self.DEFAULT_METRIC_ORDER
            if metric_name in available
        ]

        extras = sorted(available - set(ordered))

        return ordered + extras

    def _item_label(self, item: WorkspaceItem) -> str:
        if item.number is not None:
            return str(item.number)

        return "-"

    def _format_metric(self, metric: Any) -> str:
        value = getattr(metric, "value", metric)
        unit = getattr(metric, "unit", "")

        value_text = self._format_metric_value(value)

        if unit:
            return f"{value_text} {unit}"

        return value_text

    def _format_metric_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"

        return str(value)
    
    def _set_headers(self, headers: list[str]):
        self.setColumnCount(len(headers))

        for col, header in enumerate(headers):
            item = QTableWidgetItem(header)
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.setHorizontalHeaderItem(col, item)

    def _make_item(self, value) -> QTableWidgetItem:
        item = QTableWidgetItem(str(value))
        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return item