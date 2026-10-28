"""Results plots of solid mechanics: kinds, titles and constructors."""

from __future__ import annotations

from ...plots import PlotStyle, ResultPlot

PLOT_KINDS = ("dispersion", "colored", "quantity", "kmap", "surfaces", "spectrum", "conductance", "field",
              "section", "quiver", "stream", "line")
PLOT_TITLES = {"dispersion": "Dispersion", "colored": "Dispersion (colored)", "quantity": "Quantity vs sweep",
               "kmap": "Band map", "surfaces": "Band surfaces", "spectrum": "Spectrum", "conductance": "Conductance", "field": "Displacement field",
               "section": "Cross-section", "quiver": "Displacement arrows",
               "stream": "Displacement streamlines", "line": "Profile along line"}


def field_plot(name: str = "") -> ResultPlot:
    return ResultPlot.field_view(name or PLOT_TITLES["field"], "viridis")


def colored_plot(name: str = "Dispersion (colored)", color_by: str = "mean(abs(uz)^2)") -> ResultPlot:
    """Dispersion with every point coloured by an expression (default: the out-of-plane fraction)."""
    return ResultPlot(name=name, kind="colored", color_by=color_by,
                      style=PlotStyle(colormap="Blues", line_style="none", marker_size=2.0, title=False))


def band_map(name: str = "", quantity: str = "freq", band: int = 0) -> ResultPlot:
    """One band of a wavevector mesh as a map over the Brillouin zone."""
    return ResultPlot(name=name or PLOT_TITLES["kmap"], kind="kmap", quantity=quantity, band=band, at=0,
                      style=PlotStyle(colormap="viridis", markers=False, title=False, k_unit="1/um"))


def band_surfaces(name: str = "", quantity: str = "freq", band: int = 0, bands: int = 6) -> ResultPlot:
    """Bands of a wavevector mesh as surfaces over the zone, in a 3D kx, ky, frequency view."""
    return ResultPlot(name=name or PLOT_TITLES["surfaces"], kind="surfaces", quantity=quantity, band=band,
                      bands=bands, at=0, style=PlotStyle(title=False, k_unit="1/um"))


def spectrum_plot(name: str = "", spectrum: str = "dos") -> ResultPlot:
    """A spectrum of the wavevector mesh against frequency: density of states, group velocity or heat flux
    (physics/solid/spectra.py). All the bands take part, so it starts at the lowest with no limit."""
    from .spectra import SPECTRUM_TITLES
    return ResultPlot(name=name or SPECTRUM_TITLES[spectrum], kind="spectrum", spectrum=spectrum,
                      band=0, bands=0, at=0, style=PlotStyle(markers=False))


def conductance_plot(name: str = "", transport: str = "conductance") -> ResultPlot:
    """The heat flux integrated over the bands, against temperature: the conductance G(T) or the radiated
    power P(T). Drawn log-log, where a ballistic power law is a straight line."""
    from .spectra import TRANSPORT_TITLES
    return ResultPlot(name=name or TRANSPORT_TITLES[transport], kind="conductance", transport=transport,
                      band=0, bands=0, at=0, style=PlotStyle(markers=False, x_log=True, y_log=True))


def default_plots(stationary: bool = False, coherent: bool = False) -> list[ResultPlot]:
    """A dispersion plot and the displacement field (stationary: the displacement field; coherent heat
    conduction: a map of the lowest band over the zone, there being no path to plot along)."""
    if coherent:
        return [band_map(), band_surfaces(), spectrum_plot(), conductance_plot(), field_plot()]
    return ([] if stationary else [ResultPlot()]) + [field_plot()]
