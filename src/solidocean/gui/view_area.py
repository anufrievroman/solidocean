"""The Structure and Plot views: one at a time behind tabs, or both at once in a split.

Both widgets live in the splitter from the start and only their visibility and the splitter's orientation
change, so the 3D view is never reparented — moving a live render window between a tab widget and a
splitter is what loses its context. The choice lasts for the session; it belongs to no model.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QTabBar, QVBoxLayout, QWidget

# named for the divider between the views, as editors name them: a horizontal split is a horizontal line
# with a view above and below it, a vertical one puts them side by side
MODES = ("tabs", "stacked", "side")
MODE_NAMES = {"tabs": "Single", "stacked": "Horizontal split", "side": "Vertical split"}
MODE_HINTS = {"tabs": "One view at a time, chosen by the tabs on the left",
              "stacked": "Both views, one above the other", "side": "Both views, side by side"}


class ViewArea(QWidget):
    """Holds the two result views with a tab bar and the layout buttons above them."""

    def __init__(self, viewer: QWidget, plot: QWidget, parent=None):
        super().__init__(parent)
        self._views = [viewer, plot]
        self._current = viewer
        self._mode = "tabs"

        self.bar = QTabBar()
        self.bar.setDrawBase(False)
        for name in ("Structure", "Plot"):
            self.bar.addTab(name)
        self.bar.currentChanged.connect(lambda i: self.setCurrentWidget(self._views[i]))

        self.modes = QTabBar()      # the layout, in the same shape as the tabs that choose a view
        self.modes.setDrawBase(False)
        for n, mode in enumerate(MODES):
            self.modes.addTab(MODE_NAMES[mode])
            self.modes.setTabToolTip(n, MODE_HINTS[mode])
        self.modes.setCurrentIndex(MODES.index(self._mode))
        self.modes.currentChanged.connect(lambda n: self.set_mode(MODES[n]))

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 4, 0)
        row.setSpacing(2)
        row.addWidget(self.bar)
        row.addStretch(1)
        row.addWidget(self.modes)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        for view in self._views:
            self.splitter.addWidget(view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(row)
        layout.addWidget(self.splitter, 1)
        self._apply()

    # ---- the API the window uses, as a tab widget would ----------------------------
    def currentWidget(self) -> QWidget:
        return self._current

    def setCurrentWidget(self, view: QWidget) -> None:
        """Bring a view forward; with both on show there is nothing to raise, so it takes the focus."""
        if view not in self._views:
            return
        self._current = view
        self._apply()
        if self._mode != "tabs":
            view.setFocus(Qt.FocusReason.OtherFocusReason)

    # ---- the layout ----------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self._mode:
            return
        self._mode = mode
        self._apply()
        if mode != "tabs":
            self._halve()

    def _halve(self) -> None:
        """Put the handle in the middle, for the user to drag from. Asking for equal halves rather than
        [1, 1], which falls back to the views' size hints and lands off centre."""
        extent = (self.splitter.width() if self.splitter.orientation() == Qt.Orientation.Horizontal
                  else self.splitter.height())
        if extent > 0:
            self.splitter.setSizes([extent // 2, extent - extent // 2])

    def _apply(self) -> None:
        tabs = self._mode == "tabs"
        self.bar.setVisible(tabs)
        self.splitter.setOrientation(Qt.Orientation.Horizontal if self._mode == "side"
                                     else Qt.Orientation.Vertical)
        for view in self._views:
            view.setVisible(not tabs or view is self._current)
        index = self._views.index(self._current)
        if self.bar.currentIndex() != index:
            self.bar.blockSignals(True)
            self.bar.setCurrentIndex(index)
            self.bar.blockSignals(False)
        index = MODES.index(self._mode)
        if self.modes.currentIndex() != index:
            self.modes.blockSignals(True)
            self.modes.setCurrentIndex(index)
            self.modes.blockSignals(False)
