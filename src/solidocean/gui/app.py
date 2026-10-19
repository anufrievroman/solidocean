"""GUI entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")  # for pyvista / qtpy

APP_NAME = "SolidOcean"
ICON = Path(__file__).parent / "resources" / "icon.png"


def file_open_filter(win, parent):
    """macOS hands a document double-clicked in Finder (the bundle declares .toml, macapp.py) to the running
    app as a QFileOpenEvent, both at start-up and while it runs, never in argv."""
    from PySide6.QtCore import QEvent, QObject

    class Filter(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.FileOpen and event.file():
                win.open_requested(event.file())
                return True
            return super().eventFilter(obj, event)

    return Filter(parent)


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
    app.installEventFilter(file_open_filter(win, app))
    if not path:   # started without a model: offer the New dialog (Cancel keeps the empty model)
        from PySide6.QtCore import QTimer
        # unless a file opened from Finder came first (its event arrives once the loop runs)
        QTimer.singleShot(0, lambda: win.project.path is None and win.new())
    return app.exec()
