"""GUI entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")  # for pyvista / qtpy

APP_NAME = "SolidOcean"
ICON = Path(__file__).parent / "resources" / "icon.png"


def run(path: str | None = None) -> int:
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from . import theme
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(str(ICON)))  # also the Dock icon on macOS
    theme.apply_accent(app)
    app.setStyleSheet(theme.stylesheet())
    win = MainWindow(path)
    win.show()
    return app.exec()
