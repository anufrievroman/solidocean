"""Results plots of heat transfer: kinds, titles and constructors."""

from __future__ import annotations

from ...plots import PlotStyle, ResultPlot

PLOT_KINDS = ("field", "section", "quiver", "stream", "time", "quantity", "line")
PLOT_TITLES = {"field": "Field", "section": "Cross-section", "quiver": "Heat flux arrows",
               "stream": "Heat flux streamlines", "time": "Time plot", "quantity": "Sweep plot",
               "line": "Profile along line"}


def field_plot(name: str = "Temperature field", quantity: str = "T") -> ResultPlot:
    """A 3D view of the structure coloured by a quantity (default: the temperature)."""
    plot = ResultPlot.field_view(name, "magma")
    plot.quantity = quantity
    return plot


def time_plot(name: str = "Average temperature", quantity: str = "mean(T)",
              y_label: str = "Average temperature (K)") -> ResultPlot:
    return ResultPlot(name=name, kind="time", quantity=quantity, y_label=y_label, style=PlotStyle(markers=False))


def default_plots(transient: bool = False) -> list[ResultPlot]:
    """The temperature field, plus the average temperature vs time for transient problems."""
    return [field_plot()] + ([time_plot()] if transient else [])
