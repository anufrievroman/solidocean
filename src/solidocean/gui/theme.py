"""Light/dark-aware colours derived from the system palette."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from ..materials import DEFAULT_COLORS

ACCENT = DEFAULT_COLORS[0]  # the first material colour: selections, chosen buttons
ACCENT_TEXT = "#04302b"     # text on the accent (too bright for white text)


def is_dark() -> bool:
    return QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128


def muted() -> str:
    """Secondary text (hints, computed values)."""
    return "#a8a8a8" if is_dark() else "#666666"


def disabled_text() -> str:
    return "#7a7a7a" if is_dark() else "#9a9a9a"


def error() -> str:
    return "#ff6b6b" if is_dark() else "#c62828"


def apply_accent(app: QApplication):
    """Our accent instead of the system's for everything drawn from the palette (selected sidebar items, lists,
    selected text). Native macOS controls (checkboxes, progress bars, focus rings) keep the system accent."""
    pal = app.palette()
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        pal.setColor(group, QPalette.ColorRole.Highlight, QColor(ACCENT))
        pal.setColor(group, QPalette.ColorRole.HighlightedText, QColor(ACCENT_TEXT))
        pal.setColor(group, QPalette.ColorRole.Accent, QColor(ACCENT))
    app.setPalette(pal)


def stylesheet() -> str:
    grid = "#6e6e6e" if is_dark() else "#c4c4c4"
    box_border = "#4a4a4a" if is_dark() else "#d2d2d2"
    box_fill = "rgba(255, 255, 255, 0.035)" if is_dark() else "rgba(0, 0, 0, 0.03)"
    hover = "rgba(255, 255, 255, 0.07)" if is_dark() else "rgba(0, 0, 0, 0.06)"
    return f"""
        QTreeWidget#sidebar {{ background: palette(window); border: none; outline: 0; padding: 2px 4px;
                               show-decoration-selected: 0; }}
        QTreeWidget#sidebar::item {{ padding: 2px 6px; border-radius: 5px; }}
        QTreeWidget#sidebar::item:hover:!selected {{ background: {hover}; }}
        QTreeWidget#sidebar::item:selected {{ background: palette(highlight); color: palette(highlighted-text); }}
        QTreeWidget#sidebar::branch, QTreeWidget#sidebar::branch:selected, QTreeWidget#sidebar::branch:hover {{
            background: palette(window); image: none; border-image: none; }}
        QGroupBox {{ border: 1px solid {box_border}; border-radius: 6px; background: {box_fill};
                     margin-top: 30px; padding: 8px 6px 6px 6px; }}
        QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top center;
                            padding: 6px 6px 0px 6px; }}
        QTableView {{ gridline-color: {grid}; }}
        QHeaderView::section {{ border: 0px; border-right: 1px solid {grid}; border-bottom: 1px solid {grid};
                                padding: 3px; }}
        QToolBar#mainToolbar QToolButton {{ font-size: 14px; padding: 4px 8px; }}
        QToolBar#viewToolbar {{ border: none; spacing: 2px; }}
        QToolButton#cancelButton {{ font-size: 14px; padding: 0px 6px; border: none; background: transparent; }}
        QToolButton#cancelButton:hover {{ color: {error()}; }}
        QToolButton#logButton {{ font-size: 12px; padding: 0px 6px; border: none; background: transparent;
                                 color: {muted()}; }}
        QToolButton#logButton:hover {{ color: palette(text); }}
        QToolBar#viewToolbar QToolButton {{ font-size: 13px; padding: 2px 8px; }}
        QPlainTextEdit#messages {{ background: palette(window); border: none; }}
        QDockWidget {{ border: none; }}
        QDockWidget::title {{ background: palette(window); border: none; padding: 4px 6px; text-align: left; }}
        QMainWindow::separator {{ background: palette(window); border: none; width: 4px; height: 4px; }}
        QStatusBar {{ border: none; }}
        QStatusBar::item {{ border: none; }}
    """
