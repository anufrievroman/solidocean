"""Matplotlib canvas for dispersion / sweep plots with point picking, and for the 3D band surfaces (a
rotatable 3D axes, which has nothing to pick)."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..results.modal import draw
from ..results.style import use_plot_style


TOOLBAR_SCALE = 0.7      # of matplotlib's default icon size


class PlotCanvas(QWidget):
    # (outer index, inner index, band index)
    picked = Signal(int, int, int)
    deselected = Signal()   # the selected point was clicked again

    def __init__(self, parent=None):
        super().__init__(parent)
        use_plot_style()
        self.figure = Figure(figsize=(5, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        icons = toolbar.iconSize()      # matplotlib's own are chunky beside the rest of the interface
        toolbar.setIconSize(icons * TOOLBAR_SCALE)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot()
        self.results = None
        self.table = np.zeros((0, 5))
        self._marker = None
        self._marked_row = None
        self._flashing = False
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.show_message("No results yet")

    # plot kinds drawn in a 3D axes (kx, ky, frequency), rotated with the mouse
    PROJECTED_3D = ("surfaces",)

    MIN_INCHES = 1.0   # a figure with a zero side has singular transforms and cannot be drawn on

    def _fresh_axes(self, projection: str | None = None):
        """Start from an empty figure (removes colour bars of the previous plot); its selection circle goes too."""
        if min(self.figure.get_size_inches()) < self.MIN_INCHES:
            # drawn while hidden (the other view has the room): keep a workable size until it is laid out,
            # which resizes the figure and redraws it at the real one
            self.figure.set_size_inches(5, 4, forward=False)
        self.figure.clear()
        self.ax = self.figure.add_subplot(projection=projection)
        self._marker = self._marked_row = None

    def show_message(self, text: str):
        self._fresh_axes()
        self.table = np.zeros((0, 5))   # nothing to pick or mark
        self.ax.text(0.5, 0.5, text, ha="center", va="center", color="0.5", transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def show_results(self, results, project, pdef, values=None, colors=None, drawer=None):
        self.results = results
        self._fresh_axes("3d" if pdef.kind in self.PROJECTED_3D else None)
        if drawer is not None:  # e.g. plots of field results
            self.table = drawer(results, self.ax, project, pdef, values)
        else:
            self.table = draw(results, self.ax, project, pdef, values, colors)  # x, y, i, j, b per point
        self.canvas.draw_idle()

    def _on_click(self, event):
        if self.results is None or event.inaxes is not self.ax or event.button != 1 or not len(self.table):
            return
        if self.canvas.toolbar is not None and self.canvas.toolbar.mode:
            return  # zoom/pan active
        t = self.table
        xr = np.ptp(self.ax.get_xlim()) or 1
        yr = np.ptp(self.ax.get_ylim()) or 1
        d = ((t[:, 0] - event.xdata) / xr) ** 2 + ((t[:, 1] - event.ydata) / yr) ** 2
        d = np.where(np.isnan(d), np.inf, d)
        n = int(np.argmin(d))
        if self._flashing:
            return
        if self._marker is not None and n == self._marked_row:  # the selected point again: deselect
            self._unmark()
            self.deselected.emit()
            return
        i, j, b = int(t[n, 2]), int(t[n, 3]), int(t[n, 4])
        self._flash(n, lambda: self.picked.emit(i, j, b))

    FLASH = ((20, 0.45), (14, 0.25), (9, 0.0))   # (ring size, fill opacity) per frame, FRAME_MS apart
    FRAME_MS = 40

    def _flash(self, n, done):
        """Click confirmation: the ring shrinks from large and filled to its normal size (~0.1 s), then
        `done` runs (which typically switches to the field view)."""
        if self._flashing:
            return  # one pick at a time
        self._flashing = True
        self._mark_row(n)
        marker = self._marker

        def frame(k=0):
            if k < len(self.FLASH) and self._marker is marker:
                size, fill = self.FLASH[k]
                marker.set_markersize(size)
                marker.set_markerfacecolor((0.86, 0.08, 0.24, fill))
                self.canvas.draw_idle()
                QTimer.singleShot(self.FRAME_MS if k + 1 < len(self.FLASH) else 20, lambda: frame(k + 1))
            else:
                self._flashing = False
                done()

        frame()

    def _unmark(self):
        if self._marker is not None:
            self._marker.remove()
            self.canvas.draw_idle()
        self._marker = self._marked_row = None

    def _mark_row(self, n):
        self._marked_row = n
        if self._marker is not None:
            self._marker.remove()
        (self._marker,) = self.ax.plot([self.table[n, 0]], [self.table[n, 1]], "o", ms=9, mfc="none",
                                       mec="crimson", mew=2)
        self.canvas.draw_idle()

    def mark(self, i, j, b):
        """Circle the point of solution (i, j, b), or remove the circle if this plot does not show it."""
        rows = np.nonzero((self.table[:, 2] == i) & (self.table[:, 3] == j) & (self.table[:, 4] == b))[0]
        if len(rows):
            self._mark_row(int(rows[0]))
        else:
            self._unmark()
