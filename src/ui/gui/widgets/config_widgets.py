from dataclasses import Field
from pathlib import Path
from typing import Any, get_args, get_origin, Tuple

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QSpinBox,
    QWidget,
    QFileDialog,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
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


class BasisSelectionWidget(QWidget):
    """
    Select a transform/basis.

    If LEARNED is selected, a dictionary path field is shown.
    """

    def __init__(
        self,
        options: list[str],
        default: str,
        allow_learned_path: bool = False,
    ):
        super().__init__()

        self.allow_learned_path = allow_learned_path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        basis_name, dictionary_path = self._parse_default(default)

        self.combo = QComboBox()
        self.combo.addItems(options)

        if basis_name in options:
            self.combo.setCurrentText(basis_name)

        self.path_container = QWidget()
        path_layout = QHBoxLayout(self.path_container)
        path_layout.setContentsMargins(0, 0, 0, 0)

        self.path_edit = QLineEdit(dictionary_path or "")
        self.path_edit.setPlaceholderText("Dictionary path")

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self._browse_dictionary)

        path_layout.addWidget(self.path_edit, stretch=1)
        path_layout.addWidget(self.browse_button)

        layout.addWidget(self.combo)
        layout.addWidget(self.path_container)

        self.combo.currentTextChanged.connect(self._update_path_visibility)
        self._update_path_visibility(self.combo.currentText())

    def value(self) -> str:
        basis = self.combo.currentText()

        if basis != "LEARNED":
            return basis

        if not self.allow_learned_path:
            return basis

        path_text = self.path_edit.text().strip()

        if not path_text:
            raise ValueError("A dictionary file must be selected for LEARNED Psi")

        path = Path(path_text)

        dict_dir = path.parent
        dict_name = path.stem

        return f"LEARNED:directory={dict_dir},name={dict_name}"

    def _browse_dictionary(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select learned dictionary",
            "",
            "NumPy files (*.npz);;All files (*.*)",
        )

        if path:
            self.path_edit.setText(path)

    def _update_path_visibility(self, basis: str):
        visible = self.allow_learned_path and basis == "LEARNED"
        self.path_container.setVisible(visible)

    def _parse_default(self, default: str) -> tuple[str, str | None]:
        if not isinstance(default, str):
            return str(default), None

        if default.startswith("LEARNED:directory="):
            payload = default.removeprefix("LEARNED:directory=")

            parts = {}
            for item in payload.split(","):
                if "=" not in item:
                    continue

                key, value = item.split("=", 1)
                parts[key.strip()] = value.strip()

            directory = parts.get("directory")
            name = parts.get("name")

            if directory and name:
                return "LEARNED", str(Path(directory) / f"{name}.npz")

            return "LEARNED", None

        # Optional backward compatibility
        if default.startswith("LEARNED:path="):
            path = default.removeprefix("LEARNED:path=")
            return "LEARNED", path

        return default, None


def create_config_widget(field: Field, default: Any) -> QWidget:
    """
    Create a Qt widget suitable for editing a compressor config field.
    """
    name = field.name
    field_name = name.lower()
    field_type = field.type

    # ------------------------------------------------------------
    # Basis / sensing fields
    # ------------------------------------------------------------
    if field_name in {"psi", "psi_name"}:
        return BasisSelectionWidget(
            options=PSI_OPTIONS,
            default=str(default),
            allow_learned_path=True,
        )

    if field_name in {"psis", "psi_names"}:
        return _create_basis_tuple_widget(
            default=default,
            options=PSI_OPTIONS,
            allow_learned_path=True,
        )

    if field_name in {"phi", "phi_name"}:
        return BasisSelectionWidget(
            options=PHI_OPTIONS,
            default=str(default),
            allow_learned_path=False,
        )

    if field_name in {"phis", "phi_names"}:
        return _create_basis_tuple_widget(
            default=default,
            options=PHI_OPTIONS,
            allow_learned_path=False,
        )

    # ------------------------------------------------------------
    # Known enum/string options
    # ------------------------------------------------------------
    if field_name == "local_sum_mode":
        widget = QComboBox()
        widget.addItems(LOCAL_SUM_MODE_OPTIONS)

        index = widget.findText(str(default))
        if index >= 0:
            widget.setCurrentIndex(index)

        return widget

    if field_type is Axis:
        widget = QComboBox()

        for label, axis in AXIS_OPTIONS.items():
            widget.addItem(label, axis)

        index = widget.findData(default)
        if index >= 0:
            widget.setCurrentIndex(index)

        return widget

    # ------------------------------------------------------------
    # Primitive fields
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Generic tuple fallback, e.g. sr=(0.5, 0.5, 0.5)
    # ------------------------------------------------------------
    if _is_tuple_type(field_type) or isinstance(default, tuple):
        return _create_tuple_widget(field_type, default)

    widget = QComboBox()
    widget.setEditable(True)
    widget.addItem(str(default))
    return widget


def read_widget_value(widget: QWidget) -> Any:
    """
    Read a typed Python value from a config widget.
    """
    if hasattr(widget, "value"):
        return widget.value()

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


def _create_basis_tuple_widget(
    default,
    options: list[str],
    allow_learned_path: bool,
) -> TupleWidget:
    widgets = []

    for value in default:
        widgets.append(
            BasisSelectionWidget(
                options=options,
                default=str(value),
                allow_learned_path=allow_learned_path,
            )
        )

    return TupleWidget(widgets)


def _tuple_item_type(args, index: int, value: Any):
    if not args:
        return type(value)

    if len(args) == 2 and args[1] is Ellipsis:
        return args[0]

    if index < len(args):
        return args[index]

    return type(value)


def _create_tuple_widget(field_type, default) -> TupleWidget:
    args = get_args(field_type)
    widgets = []

    for i, value in enumerate(default):
        item_type = _tuple_item_type(args, i, value)

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