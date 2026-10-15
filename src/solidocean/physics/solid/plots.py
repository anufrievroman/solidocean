"""Results plots of solid mechanics: kinds, titles and constructors."""

from __future__ import annotations

from ...plots import PlotStyle, ResultPlot

PLOT_KINDS = ("dispersion", "colored", "quantity", "kmap", "field", "section", "line")
PLOT_TITLES = {"dispersion": "Dispersion", "colored": "Dispersion (colored)", "quantity": "Quantity vs sweep",
               "kmap": "Band map over the zone", "field": "Displacement field", "section": "Cross-section",
               "line": "Profile along line"}


def field_plot(name: str = "") -> ResultPlot:
    return ResultPlot.field_view(name or PLOT_TITLES["field"], "viridis")


def colored_plot(name: str = "Dispersion (colored)", color_by: str = "mean(abs(uz)^2)") -> ResultPlot:
    """Dispersion with every point coloured by an expression (default: the out-of-plane fraction)."""
    return ResultPlot(name=name, kind="colored", color_by=color_by,
                      style=PlotStyle(colormap="Blues", line_style="none", marker_size=2.0, title=False))


def band_map(name: str = "", quantity: str = "freq", band: int = 0) -> ResultPlot:
    """One band of a wavevector mesh as a map over the Brillouin zone."""
    return ResultPlot(name=name or PLOT_TITLES["kmap"], kind="kmap", quantity=quantity, band=band, at=0,
                      style=PlotStyle(colormap="viridis", markers=False, title=False))


def default_plots(stationary: bool = False, coherent: bool = False) -> list[ResultPlot]:
    """A dispersion plot and the displacement field (stationary: the displacement field; coherent heat
    conduction: a map of the lowest band over the zone, there being no path to plot along)."""
    if coherent:
        return [band_map(), field_plot()]
    return ([] if stationary else [ResultPlot()]) + [field_plot()]
