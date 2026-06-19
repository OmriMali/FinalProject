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
    QCheckBox,
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

        layout.addWidget(self._build_workspace_panel(), stretch=1)

    def _build_workspace_panel(self) -> QGroupBox:
        box = QGroupBox("Workspace")
        layout = QVBoxLayout(box)

        toolbar = QHBoxLayout()

        self.load_button = QPushButton("Load")
        self.remove_selected_button = QPushButton("Remove Selected")
        self.clear_items_button = QPushButton("Clear")

        self.select_all_check = QCheckBox("Select all")
        self.select_all_check.setTristate(True)
        self.select_all_check.stateChanged.connect(self._on_select_all_changed)

        self.load_button.clicked.connect(self.load_requested.emit)
        self.remove_selected_button.clicked.connect(self.remove_checked_items)
        self.clear_items_button.clicked.connect(self.clear_workspace_items)

        toolbar.addWidget(self.load_button)
        toolbar.addWidget(self.remove_selected_button)
        toolbar.addWidget(self.clear_items_button)
        toolbar.addStretch()
        toolbar.addWidget(self.select_all_check)

        headers = [""] + WorkspaceItem.table_headers()

        self.loaded_items_table = QTableWidget(0, len(headers))
        self._set_table_headers(self.loaded_items_table, headers)

        header = self.loaded_items_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
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

        layout.addLayout(toolbar)
        layout.addWidget(self.loaded_items_table)

        return box

    def add_workspace_item(self, item: WorkspaceItem):
        if item.number is None:
            item.number = self.next_workspace_number
            self.next_workspace_number += 1

        self.workspace_items.append(item)
        self._append_workspace_item_row(item)

        self.workspace_changed.emit()
        self._sync_select_all_checkbox()

    def clear_workspace_items(self):
        self.workspace_items.clear()
        self.loaded_items_table.setRowCount(0)
        self.next_workspace_number = 1

        self.cleared.emit()
        self.workspace_changed.emit()
        self.selection_changed.emit()
        self._sync_select_all_checkbox()
        self._refresh_loaded_items_table_layout()

    def remove_checked_items(self):
        checked_rows_and_ids = self._checked_rows_and_ids()

        if not checked_rows_and_ids:
            return

        item_ids_to_remove = {
            item_id
            for _, item_id in checked_rows_and_ids
        }

        for row, _ in sorted(checked_rows_and_ids, reverse=True):
            self.loaded_items_table.removeRow(row)

        self.workspace_items = [
            item
            for item in self.workspace_items
            if item.item_id not in item_ids_to_remove
        ]

        self.workspace_changed.emit()
        self.selection_changed.emit()
        self._sync_select_all_checkbox()
        self._refresh_loaded_items_table_layout()

    def selected_workspace_items(self) -> list[WorkspaceItem]:
        checked_ids = {
            item_id
            for _, item_id in self._checked_rows_and_ids()
        }

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
        self.select_all_check.setEnabled(enabled)

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
            table_item.setToolTip(str(value))
            self.loaded_items_table.setItem(row, col, table_item)

        self.loaded_items_table.blockSignals(False)
        self._refresh_loaded_items_table_layout()

    def _on_loaded_item_changed(self, item: QTableWidgetItem):
        if item.column() != 0:
            return

        self._sync_select_all_checkbox()
        self.selection_changed.emit()

    def _set_table_headers(self, table: QTableWidget, headers: list[str]):
        table.setColumnCount(len(headers))

        for col, header in enumerate(headers):
            item = QTableWidgetItem(header)
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setHorizontalHeaderItem(col, item)

    def _on_select_all_changed(self, _state: int):
        check_all = not self._all_rows_checked()

        self.loaded_items_table.blockSignals(True)

        for row in range(self.loaded_items_table.rowCount()):
            check_item = self.loaded_items_table.item(row, 0)

            if check_item is None:
                continue

            check_item.setCheckState(
                Qt.CheckState.Checked
                if check_all
                else Qt.CheckState.Unchecked
            )

        self.loaded_items_table.blockSignals(False)

        self._sync_select_all_checkbox()
        self.selection_changed.emit()

    def _sync_select_all_checkbox(self):
        total = self.loaded_items_table.rowCount()

        if total == 0:
            self._set_select_all_check_state(Qt.CheckState.Unchecked)
            return

        checked = 0

        for row in range(total):
            check_item = self.loaded_items_table.item(row, 0)

            if check_item is None:
                continue

            if check_item.checkState() == Qt.CheckState.Checked:
                checked += 1

        if checked == 0:
            state = Qt.CheckState.Unchecked
        elif checked == total:
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked

        self._set_select_all_check_state(state)

    def _checked_rows_and_ids(self) -> list[tuple[int, str]]:
        checked_rows_and_ids = []

        for row in range(self.loaded_items_table.rowCount()):
            check_item = self.loaded_items_table.item(row, 0)

            if check_item is None:
                continue

            if check_item.checkState() != Qt.CheckState.Checked:
                continue

            checked_rows_and_ids.append(
                (row, check_item.data(Qt.ItemDataRole.UserRole))
            )

        return checked_rows_and_ids

    def _all_rows_checked(self) -> bool:
        total = self.loaded_items_table.rowCount()

        if total == 0:
            return False

        return len(self._checked_rows_and_ids()) == total

    def _set_select_all_check_state(self, state: Qt.CheckState) -> None:
        self.select_all_check.blockSignals(True)
        self.select_all_check.setCheckState(state)
        self.select_all_check.blockSignals(False)

    def _refresh_loaded_items_table_layout(self) -> None:
        self.loaded_items_table.resizeColumnsToContents()
        self.loaded_items_table.horizontalHeader().setStretchLastSection(True)

