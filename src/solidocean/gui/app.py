"""GUI entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")  # for pyvista / qtpy

APP_NAME = "SolidOcean"
ICON = Path(__file__).parent / "resources" / "icon.png"


def _splash(app):
    """A splash screen (icon, name, "Loading...") shown while the heavy modules load.

    The first start after installing can take a minute: macOS scans every new Qt/VTK library."""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QFont, QPainter, QPainterPath, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    from .. import __version__

    w, h, r = 300, 300, 18
    dpr = app.primaryScreen().devicePixelRatio() if app.primaryScreen() else 1.0
    pix = QPixmap(int(w * dpr), int(h * dpr))
    pix.setDevicePixelRatio(dpr)
    pix.fill(Qt.transparent)
    pal = app.palette()
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    outline = QPainterPath()
    outline.addRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), r, r)
    p.fillPath(outline, pal.window())
    p.setPen(pal.mid().color())
    p.drawPath(outline)
    p.drawPixmap(QRectF((w - 150) / 2, 32, 150, 150), QPixmap(str(ICON)), QRectF())
    font = QFont(app.font())
    font.setPointSizeF(font.pointSizeF() * 1.6)
    font.setBold(True)
    p.setFont(font)
    p.setPen(pal.windowText().color())
    p.drawText(QRectF(0, 196, w, 34), Qt.AlignCenter, APP_NAME)
    font.setBold(False)
    font.setPointSizeF(app.font().pointSizeF())
    p.setFont(font)
    p.setPen(pal.placeholderText().color())
    p.drawText(QRectF(0, 234, w, 24), Qt.AlignCenter, f"{__version__} · Loading…")
    p.end()

    splash = QSplashScreen(pix)
    splash.setAttribute(Qt.WA_TranslucentBackground)
    splash.show()
    for _ in range(3):  # let the window server draw it before the imports block the event loop
        app.processEvents()
    return splash


def run(path: str | None = None) -> int:
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(str(ICON)))  # also the Dock icon on macOS
    splash = _splash(app)

    from . import theme
    from .main_window import MainWindow

    theme.apply_accent(app)
    app.setStyleSheet(theme.stylesheet())
    win = MainWindow(path)
    win.show()
    splash.finish(win)
    return app.exec()
