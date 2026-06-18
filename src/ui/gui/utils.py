from PySide6.QtWidgets import QMessageBox, QWidget


def show_warning(
    parent: QWidget | None,
    title: str,
    message: str,
):
    QMessageBox.warning(parent, title, message)


def show_error(
    parent: QWidget | None,
    title: str,
    message: str,
):
    QMessageBox.critical(parent, title, message)


def show_info(
    parent: QWidget | None,
    title: str,
    message: str,
):
    QMessageBox.information(parent, title, message)