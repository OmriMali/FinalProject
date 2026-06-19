from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from src.data_processing import (
    aggregate_mean_std,
    filter_compare,
    filter_in,
    filter_notna,
    load_compression_log,
)
from src.ui.gui.widgets.figure_popout import FigurePopoutWindow
from src.visuals.metrics import plot_metric_vs_metric, plot_runtime_comparison


class DataAnalysisTab(QWidget):
    """
    Experiment-log analysis tab.
    """

    FILTER_OPERATORS = [
        "contains",
        "in",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
    ]

    def __init__(self):
        super().__init__()

        self.log_paths: list[Path] = []
        self.log_df: pd.DataFrame | None = None
        self.filtered_df: pd.DataFrame | None = None
        self.filters: list[tuple[str, str, str]] = []
        self.popout_windows: list[FigurePopoutWindow] = []

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_log_panel(), stretch=2)
        layout.addWidget(self._build_filter_panel())
        layout.addWidget(self._build_plot_panel(), stretch=1)

    def _build_log_panel(self) -> QGroupBox:
        box = QGroupBox("Experiment Logs")
        layout = QVBoxLayout(box)

        controls = QHBoxLayout()

        self.add_logs_button = QPushButton("Add Logs")
        self.remove_logs_button = QPushButton("Remove Selected")
        self.clear_logs_button = QPushButton("Clear Logs")
        self.add_logs_button.clicked.connect(self._on_add_logs)
        self.remove_logs_button.clicked.connect(self._remove_selected_logs)
        self.clear_logs_button.clicked.connect(self._clear_logs)

        self.log_status_label = QLabel("No logs loaded")
        self.log_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        controls.addWidget(self.add_logs_button)
        controls.addWidget(self.remove_logs_button)
        controls.addWidget(self.clear_logs_button)
        controls.addWidget(self.log_status_label)
        controls.addStretch()

        self.loaded_logs_table = QTableWidget(0, 1)
        self.loaded_logs_table.setHorizontalHeaderLabels(["Loaded log files"])
        self.loaded_logs_table.verticalHeader().setVisible(False)
        self.loaded_logs_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.loaded_logs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.loaded_logs_table.setMaximumHeight(120)
        self.loaded_logs_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Stretch,
        )

        self.log_table = QTableWidget()
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log_table.setSortingEnabled(True)

        layout.addLayout(controls)
        layout.addWidget(self.loaded_logs_table)
        layout.addWidget(self.log_table)

        return box

    def _build_filter_panel(self) -> QGroupBox:
        box = QGroupBox("Filters")
        layout = QVBoxLayout(box)

        controls = QHBoxLayout()

        self.filter_column_combo = QComboBox()
        self.filter_operator_combo = QComboBox()
        self.filter_operator_combo.addItems(self.FILTER_OPERATORS)
        self.filter_value_edit = QLineEdit()
        self.filter_value_edit.setPlaceholderText("value")

        self.add_filter_button = QPushButton("Add Filter")
        self.remove_filter_button = QPushButton("Remove Selected")
        self.clear_filters_button = QPushButton("Clear Filters")
        self.add_filter_button.clicked.connect(self._add_filter)
        self.remove_filter_button.clicked.connect(self._remove_selected_filters)
        self.clear_filters_button.clicked.connect(self._clear_filters)

        controls.addWidget(QLabel("Column"))
        controls.addWidget(self.filter_column_combo)
        controls.addWidget(QLabel("Op"))
        controls.addWidget(self.filter_operator_combo)
        controls.addWidget(QLabel("Value"))
        controls.addWidget(self.filter_value_edit)
        controls.addWidget(self.add_filter_button)
        controls.addWidget(self.remove_filter_button)
        controls.addWidget(self.clear_filters_button)

        self.filters_table = QTableWidget(0, 3)
        self.filters_table.setHorizontalHeaderLabels(["Column", "Op", "Value"])
        self.filters_table.verticalHeader().setVisible(False)
        self.filters_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.filters_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.filters_table.setMaximumHeight(120)
        self.filters_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        layout.addLayout(controls)
        layout.addWidget(self.filters_table)

        self._set_filter_controls_enabled(False)

        return box

    def _build_plot_panel(self) -> QGroupBox:
        box = QGroupBox("Metric Plots")
        layout = QVBoxLayout(box)

        controls = QHBoxLayout()

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.method_combo = QComboBox()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["line", "bar"])

        self.aggregate_check = QCheckBox("Aggregate mean/std")
        self.aggregate_check.setChecked(True)

        self.plot_metric_button = QPushButton("Plot Metric")
        self.plot_runtime_button = QPushButton("Runtime")
        self.popout_plot_button = QPushButton("Pop Out")
        self.plot_metric_button.clicked.connect(self._plot_metric)
        self.plot_runtime_button.clicked.connect(self._plot_runtime)
        self.popout_plot_button.clicked.connect(self._popout_plot)

        controls.addWidget(QLabel("x"))
        controls.addWidget(self.x_combo)
        controls.addWidget(QLabel("y"))
        controls.addWidget(self.y_combo)
        controls.addWidget(QLabel("method"))
        controls.addWidget(self.method_combo)
        controls.addWidget(QLabel("type"))
        controls.addWidget(self.plot_type_combo)
        controls.addWidget(self.aggregate_check)
        controls.addWidget(self.plot_metric_button)
        controls.addWidget(self.plot_runtime_button)
        controls.addWidget(self.popout_plot_button)

        self.analysis_figure = Figure(figsize=(6, 4))
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        self.analysis_toolbar = NavigationToolbar(
            self.analysis_canvas,
            self,
        )

        layout.addLayout(controls)
        layout.addWidget(self.analysis_toolbar)
        layout.addWidget(self.analysis_canvas, stretch=1)

        self._set_analysis_controls_enabled(False)

        return box

    def _on_add_logs(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load compression log files",
            "",
            "CSV files (*.csv);;All files (*.*)",
        )

        if not paths:
            return

        existing = {path.resolve() for path in self.log_paths}

        for path_text in paths:
            path = Path(path_text).resolve()

            if path in existing:
                continue

            self.log_paths.append(path)
            existing.add(path)

        self._reload_logs()

    def _remove_selected_logs(self):
        selected_rows = {
            index.row()
            for index in self.loaded_logs_table.selectedIndexes()
        }

        if not selected_rows:
            return

        self.log_paths = [
            path
            for index, path in enumerate(self.log_paths)
            if index not in selected_rows
        ]

        if self.log_paths:
            self._reload_logs()
        else:
            self._clear_logs()

    def _clear_logs(self):
        self.log_paths.clear()
        self.log_df = None
        self.filtered_df = None
        self.filters.clear()
        self._show_loaded_log_paths()
        self._render_filters()
        self._show_dataframe(pd.DataFrame())
        self._populate_controls(pd.DataFrame())
        self._set_filter_controls_enabled(False)
        self._set_analysis_controls_enabled(False)
        self.log_status_label.setText("No logs loaded")
        self.analysis_figure.clear()
        self.analysis_canvas.draw_idle()

    def _reload_logs(self):
        if not self.log_paths:
            self._clear_logs()
            return

        try:
            self.log_df = self._load_log_dataframe()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Load logs",
                f"Could not load logs:\n{exc}",
            )
            return

        self._show_loaded_log_paths()
        self._populate_controls(self.log_df)
        self._set_filter_controls_enabled(True)
        self._set_analysis_controls_enabled(True)
        self._apply_filters()

    def _load_log_dataframe(self) -> pd.DataFrame:
        frames = []

        for path in self.log_paths:
            frame = load_compression_log(path)
            frame.insert(0, "log_file", path.name)
            frame.insert(1, "log_path", str(path))
            frames.append(frame)

        return pd.concat(frames, ignore_index=True)

    def _show_loaded_log_paths(self):
        self.loaded_logs_table.clearContents()
        self.loaded_logs_table.setRowCount(len(self.log_paths))

        for row, path in enumerate(self.log_paths):
            item = QTableWidgetItem(str(path))
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            item.setToolTip(str(path))
            self.loaded_logs_table.setItem(row, 0, item)

    def _add_filter(self):
        column = self.filter_column_combo.currentText()
        operator = self.filter_operator_combo.currentText()
        value = self.filter_value_edit.text().strip()

        if not column or not operator:
            return

        if value == "":
            QMessageBox.warning(
                self,
                "Add filter",
                "Filter value cannot be empty.",
            )
            return

        self.filters.append((column, operator, value))
        self.filter_value_edit.clear()
        self._render_filters()
        self._apply_filters()

    def _remove_selected_filters(self):
        selected_rows = {
            index.row()
            for index in self.filters_table.selectedIndexes()
        }

        if not selected_rows:
            return

        self.filters = [
            item
            for index, item in enumerate(self.filters)
            if index not in selected_rows
        ]
        self._render_filters()
        self._apply_filters()

    def _clear_filters(self):
        self.filters.clear()
        self._render_filters()
        self._apply_filters()

    def _render_filters(self):
        self.filters_table.clearContents()
        self.filters_table.setRowCount(len(self.filters))

        for row, (column, operator, value) in enumerate(self.filters):
            for col, text in enumerate([column, operator, value]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.filters_table.setItem(row, col, item)

    def _apply_filters(self):
        if self.log_df is None:
            self.filtered_df = None
            return

        df = self.log_df.copy()

        try:
            for column, operator, value in self.filters:
                df = self._apply_filter(df, column, operator, value)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Apply filters",
                str(exc),
            )
            return

        self.filtered_df = df
        self._show_dataframe(df)
        self.log_status_label.setText(
            f"{len(df)} / {len(self.log_df)} rows shown from {len(self.log_paths)} logs"
        )

    def _apply_filter(
        self,
        df: pd.DataFrame,
        column: str,
        operator: str,
        value: str,
    ) -> pd.DataFrame:
        if column not in df.columns:
            raise ValueError(f"Unknown column: {column}")

        if operator == "contains":
            return df[
                df[column].astype(str).str.contains(value, case=False, na=False)
            ].copy()

        if operator == "in":
            values = [
                self._parse_filter_value(item.strip(), df[column])
                for item in value.split(",")
                if item.strip()
            ]
            return filter_in(df, column, values)

        parsed_value = self._parse_filter_value(value, df[column])
        return filter_compare(df, column, operator, parsed_value)

    def _parse_filter_value(self, value: str, series: pd.Series) -> Any:
        if pd.api.types.is_integer_dtype(series):
            return int(value)

        if pd.api.types.is_float_dtype(series):
            return float(value)

        return value

    def _show_dataframe(self, df: pd.DataFrame):
        self.log_table.setSortingEnabled(False)
        self.log_table.clear()
        self.log_table.setRowCount(len(df))
        self.log_table.setColumnCount(len(df.columns))
        self.log_table.setHorizontalHeaderLabels([str(col) for col in df.columns])

        for row, (_, series) in enumerate(df.iterrows()):
            for col, value in enumerate(series):
                item = QTableWidgetItem(self._format_value(value))
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.log_table.setItem(row, col, item)

        self.log_table.resizeColumnsToContents()
        self.log_table.setSortingEnabled(True)

    def _populate_controls(self, df: pd.DataFrame):
        columns = [str(column) for column in df.columns]
        numeric_columns = [
            str(column)
            for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]

        self._set_combo_items(self.filter_column_combo, columns)
        self._set_combo_items(self.x_combo, columns)
        self._set_combo_items(self.y_combo, numeric_columns)
        self._set_combo_items(self.method_combo, columns)

        self._set_combo_value(self.method_combo, "method")
        self._set_combo_value(self.x_combo, self._first_existing(columns, ["sr", "sr_b", "k"]))
        self._set_combo_value(self.y_combo, self._first_existing(numeric_columns, ["psnr", "rmse", "sam"]))

    def _plot_metric(self):
        df = self.filtered_df

        if df is None:
            return

        x = self.x_combo.currentText()
        y = self.y_combo.currentText()
        method_col = self.method_combo.currentText()

        if not x or not y or not method_col:
            return

        try:
            plot_df = filter_notna(df, [x, y, method_col])

            yerr = None
            if self.aggregate_check.isChecked():
                plot_df = aggregate_mean_std(
                    plot_df,
                    group_cols=[method_col, x],
                    value_cols=[y],
                )
                yerr = f"{y}_std"
                y = f"{y}_mean"

            self.analysis_figure.clear()
            ax = self.analysis_figure.add_subplot(1, 1, 1)

            plot_metric_vs_metric(
                df=plot_df,
                x=x,
                y=y,
                method_col=method_col,
                yerr=yerr,
                ax=ax,
                title=f"{y} vs {x}",
                plot_type=self.plot_type_combo.currentText(),
            )

            self.analysis_canvas.draw_idle()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Plot metric",
                str(exc),
            )

    def _plot_runtime(self):
        df = self.filtered_df

        if df is None:
            return

        method_col = self.method_combo.currentText() or "method"

        try:
            plot_df = filter_notna(df, [method_col, "comp_time", "decomp_time"])
            plot_df = aggregate_mean_std(
                plot_df,
                group_cols=[method_col],
                value_cols=["comp_time", "decomp_time"],
            )

            self.analysis_figure.clear()
            ax = self.analysis_figure.add_subplot(1, 1, 1)

            plot_runtime_comparison(
                df=plot_df,
                method_col=method_col,
                compression_error_col="comp_time_std",
                decompression_error_col="decomp_time_std",
                ax=ax,
                title="Runtime Comparison",
            )

            self.analysis_canvas.draw_idle()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Runtime plot",
                str(exc),
            )

    def _set_filter_controls_enabled(self, enabled: bool):
        widgets = [
            self.filter_column_combo,
            self.filter_operator_combo,
            self.filter_value_edit,
            self.add_filter_button,
            self.remove_filter_button,
            self.clear_filters_button,
        ]

        for widget in widgets:
            widget.setEnabled(enabled)

    def _set_analysis_controls_enabled(self, enabled: bool):
        widgets = [
            self.x_combo,
            self.y_combo,
            self.method_combo,
            self.plot_type_combo,
            self.aggregate_check,
            self.plot_metric_button,
            self.plot_runtime_button,
            self.popout_plot_button,
        ]

        for widget in widgets:
            widget.setEnabled(enabled)

    def _popout_plot(self):
        if not self.analysis_figure.axes:
            QMessageBox.information(
                self,
                "Pop out plot",
                "There is no analysis plot to pop out yet.",
            )
            return

        window = FigurePopoutWindow(
            self.analysis_figure,
            "Data Analysis Plot",
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

    def _set_combo_items(self, combo: QComboBox, values: list[str]):
        current = combo.currentText()

        combo.blockSignals(True)
        combo.clear()
        combo.addItems(values)

        if current in values:
            combo.setCurrentText(current)

        combo.blockSignals(False)

    def _set_combo_value(self, combo: QComboBox, value: str | None):
        if not value:
            return

        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _first_existing(
        self,
        values: list[str],
        preferred: list[str],
    ) -> str | None:
        for value in preferred:
            if value in values:
                return value

        return values[0] if values else None

    def _format_value(self, value: Any) -> str:
        if pd.isna(value):
            return ""

        return str(value)
