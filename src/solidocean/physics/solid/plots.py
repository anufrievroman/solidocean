"""Results plots of solid mechanics: kinds, titles and constructors."""

from __future__ import annotations

from ...plots import PlotStyle, ResultPlot

PLOT_KINDS = ("dispersion", "colored", "quantity", "field", "section", "line")
PLOT_TITLES = {"dispersion": "Dispersion", "colored": "Dispersion (colored)", "quantity": "Quantity vs sweep",
               "field": "Displacement field", "section": "Cross-section", "line": "Profile along line"}


def field_plot(name: str = "") -> ResultPlot:
    return ResultPlot.field_view(name or PLOT_TITLES["field"], "viridis")


def colored_plot(name: str = "Dispersion (colored)", color_by: str = "mean(abs(uz)^2)") -> ResultPlot:
    """Dispersion with every point coloured by an expression (default: the out-of-plane fraction)."""
    return ResultPlot(name=name, kind="colored", color_by=color_by,
                      style=PlotStyle(colormap="Blues", line_style="none", marker_size=2.0, title=False))


def default_plots(stationary: bool = False) -> list[ResultPlot]:
    """A dispersion plot and the displacement field (stationary: the displacement field)."""
    return ([] if stationary else [ResultPlot()]) + [field_plot()]
