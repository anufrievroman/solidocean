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

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(str(ICON)))  # also the Dock icon on macOS

    from . import theme
    from .main_window import MainWindow

    theme.apply_accent(app)
    app.setStyleSheet(theme.stylesheet())
    win = MainWindow(path)
    win.remember_geometry = True
    if win.restore_geometry():
        win.show()
    else:
        win.showMaximized()   # first start; un-maximising gives the default size
    if not path:   # started without a model: offer the New dialog (Cancel keeps the empty model)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, win.new)
    return app.exec()
