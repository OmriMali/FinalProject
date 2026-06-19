from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.ui.gui.models.workspace_item import WorkspaceItem


class WorkspacePanel(QWidget):
    """
    Shared workspace panel.

    Contains file-control buttons and the loaded-items table.
    Owns the workspace item list and checked-item selection state.
    """

    load_requested = Signal()

    workspace_changed = Signal()
    selection_changed = Signal()
    cleared = Signal()

    def __init__(self):
        super().__init__()

        self.workspace_items: list[WorkspaceItem] = []
        self.next_workspace_number = 1

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_file_controls_panel())
        layout.addWidget(self._build_loaded_items_panel(), stretch=1)

    def _build_file_controls_panel(self) -> QGroupBox:
        box = QGroupBox("File Controls")
        layout = QHBoxLayout(box)

        self.load_button = QPushButton("Load")
        self.remove_selected_button = QPushButton("Remove Selected")
        self.clear_items_button = QPushButton("Clear")

        self.load_button.clicked.connect(self.load_requested.emit)

        self.remove_selected_button.clicked.connect(self.remove_checked_items)
        self.clear_items_button.clicked.connect(self.clear_workspace_items)

        layout.addWidget(self.load_button)
        layout.addWidget(self.remove_selected_button)
        layout.addWidget(self.clear_items_button)
        layout.addStretch()

        return box

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
        self.loaded_items_table.itemChanged.connect(
            self._on_loaded_item_changed
        )

        layout.addWidget(self.loaded_items_table)

        return box

    def add_workspace_item(self, item: WorkspaceItem):
        if item.number is None:
            item.number = self.next_workspace_number
            self.next_workspace_number += 1

        self.workspace_items.append(item)
        self._append_workspace_item_row(item)

        self.workspace_changed.emit()

    def clear_workspace_items(self):
        self.workspace_items.clear()
        self.loaded_items_table.setRowCount(0)
        self.next_workspace_number = 1

        self.cleared.emit()
        self.workspace_changed.emit()
        self.selection_changed.emit()

    def remove_checked_items(self):
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

        self.workspace_changed.emit()
        self.selection_changed.emit()

    def selected_workspace_items(self) -> list[WorkspaceItem]:
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

    def selected_workspace_item(self) -> WorkspaceItem | None:
        checked = self.selected_workspace_items()

        if len(checked) != 1:
            return None

        return checked[0]

    def set_controls_enabled(self, enabled: bool):
        self.load_button.setEnabled(enabled)
        self.remove_selected_button.setEnabled(enabled)
        self.clear_items_button.setEnabled(enabled)

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

    def _on_loaded_item_changed(self, item: QTableWidgetItem):
        if item.column() != 0:
            return

        self.selection_changed.emit()

    def _set_table_headers(self, table: QTableWidget, headers: list[str]):
        table.setColumnCount(len(headers))

        for col, header in enumerate(headers):
            item = QTableWidgetItem(header)
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setHorizontalHeaderItem(col, item)