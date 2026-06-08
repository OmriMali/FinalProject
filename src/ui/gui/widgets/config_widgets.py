from dataclasses import Field
from typing import Any, get_args, get_origin, Tuple

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QSpinBox,
    QWidget,
)

from src.core.dictionary import Axis
from src.ui.gui.models.config_options import (
    AXIS_OPTIONS,
    LOCAL_SUM_MODE_OPTIONS,
    PHI_OPTIONS,
    PSI_OPTIONS,
)


class TupleWidget(QWidget):
    """
    Composite widget for tuple-valued config fields.
    """

    def __init__(self, widgets: list[QWidget]):
        super().__init__()

        self.widgets = widgets

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        for widget in widgets:
            layout.addWidget(widget)

    def value(self) -> tuple:
        return tuple(read_widget_value(widget) for widget in self.widgets)


def create_config_widget(field: Field, default: Any) -> QWidget:
    """
    Create a Qt widget suitable for editing a compressor config field.
    """
    name = field.name
    field_type = field.type

    if field_type is bool:
        widget = QCheckBox()
        widget.setChecked(bool(default))
        return widget

    if field_type is int:
        widget = QSpinBox()
        widget.setRange(0, 1_000_000)
        widget.setValue(int(default))
        return widget

    if field_type is float:
        widget = QDoubleSpinBox()
        widget.setRange(0.0, 1.0)
        widget.setSingleStep(0.05)
        widget.setDecimals(4)
        widget.setValue(float(default))
        return widget

    if field_type is Axis:
        widget = QComboBox()

        for label, axis in AXIS_OPTIONS.items():
            widget.addItem(label, axis)

        index = widget.findData(default)
        if index >= 0:
            widget.setCurrentIndex(index)

        return widget

    if name in {"Phi", "Phis"}:
        return _create_string_or_tuple_combo(field_type, default, PHI_OPTIONS)

    if name in {"Psi", "Psis"}:
        return _create_string_or_tuple_combo(field_type, default, PSI_OPTIONS)

    if name == "local_sum_mode":
        widget = QComboBox()
        widget.addItems(LOCAL_SUM_MODE_OPTIONS)

        index = widget.findText(str(default))
        if index >= 0:
            widget.setCurrentIndex(index)

        return widget

    if _is_tuple_type(field_type):
        return _create_tuple_widget(field_type, default)

    # Fallback: string combo/line-edit alternative.
    # For now, use a combo only where options are known.
    widget = QComboBox()
    widget.setEditable(True)
    widget.addItem(str(default))
    return widget


def read_widget_value(widget: QWidget) -> Any:
    """
    Read a typed Python value from a config widget.
    """
    if isinstance(widget, QCheckBox):
        return widget.isChecked()

    if isinstance(widget, QSpinBox):
        return widget.value()

    if isinstance(widget, QDoubleSpinBox):
        return widget.value()

    if isinstance(widget, QComboBox):
        data = widget.currentData()
        if data is not None:
            return data
        return widget.currentText()

    if isinstance(widget, TupleWidget):
        return widget.value()

    raise TypeError(f"Unsupported config widget: {type(widget)}")


def _create_string_or_tuple_combo(
    field_type,
    default,
    options: list[str],
) -> QWidget:
    if _is_tuple_type(field_type):
        widgets = []

        for value in default:
            combo = QComboBox()
            combo.addItems(options)

            index = combo.findText(str(value))
            if index >= 0:
                combo.setCurrentIndex(index)

            widgets.append(combo)

        return TupleWidget(widgets)

    combo = QComboBox()
    combo.addItems(options)

    index = combo.findText(str(default))
    if index >= 0:
        combo.setCurrentIndex(index)

    return combo


def _create_tuple_widget(field_type, default) -> TupleWidget:
    args = get_args(field_type)
    widgets = []

    for i, value in enumerate(default):
        item_type = args[i] if i < len(args) else type(value)

        if item_type is float:
            widget = QDoubleSpinBox()
            widget.setRange(0.0, 1.0)
            widget.setSingleStep(0.05)
            widget.setDecimals(4)
            widget.setValue(float(value))

        elif item_type is int:
            widget = QSpinBox()
            widget.setRange(0, 1_000_000)
            widget.setValue(int(value))

        else:
            widget = QComboBox()
            widget.setEditable(True)
            widget.addItem(str(value))

        widgets.append(widget)

    return TupleWidget(widgets)


def _is_tuple_type(field_type) -> bool:
    return get_origin(field_type) in {tuple, Tuple}