"""Plots of modal results (eigenfrequencies per sweep point and mode): dispersion, coloured dispersion and
quantities per mode against a k-path or a variable sweep, and — over a wavevector mesh of the Brillouin
zone — the map of one band, several bands as surfaces in a 3D view, the spectra summed over the bands
(density of states, group velocity, heat flux) and those integrated against temperature (conductance)."""

from __future__ import annotations

import numpy as np

from ..expressions import UNITS
from .style import LINESTYLES, outer_axis, plot_ranges, set_scales, use_plot_style

def k_axis_label(unit: str, mathtext: bool = False) -> str:
    """'cm⁻¹' for the GUI; with mathtext=True 'cm$^{-1}$' for matplotlib (Arial lacks ⁻)."""
    base = {"1/m": "m", "1/cm": "cm", "1/mm": "mm", "1/um": "µm", "1/nm": "nm"}[unit]
    return f"{base}$^{{-1}}$" if mathtext else f"{base}⁻¹"


def inner_axis(results, style=None) -> tuple[np.ndarray, str]:
    """x values and label of a k-path plot: path coordinate, or wavevector in style.k_unit."""
    from ..plots import K_UNITS
    if style is not None and style.x_axis == "k":
        return results.x / K_UNITS[style.k_unit], f"Wavevector ({k_axis_label(style.k_unit, mathtext=True)})"
    return results.x, "Wave vector"


def sweep_of(results, pdef) -> str:
    """Resolve the plot's x-axis sweep: 'kpath', 'variable' or 'modes' (single solve)."""
    if pdef.sweep == "kpath" and results.is_kpath:
        return "kpath"
    if pdef.sweep == "variable" and results.outer_name:
        return "variable"
    if results.is_kpath:
        return "kpath"
    return "variable" if results.outer_name else "modes"


def quantity_label(project, pdef) -> tuple[float, str]:
    """(scale to divide values by, y-axis label) for a plot's quantity."""
    q = pdef.quantity.strip()
    if q == "freq":
        unit = project.frequency_unit
        return UNITS[unit], pdef.y_label or f"Frequency ({unit})"
    return 1.0, pdef.y_label or q


def draw(results, ax, project, pdef, values: np.ndarray | None = None,
         colors: np.ndarray | None = None) -> np.ndarray:
    """Draw a results plot; returns the pick table (P, 5): x, y, i, j, b for every plotted point.

    For a colored dispersion, `colors` (same shape as values) sets each point's colour.
    """
    import matplotlib
    use_plot_style()
    st = pdef.style
    colored = pdef.kind == "colored"
    if values is None:
        values = results.values(pdef.quantity)
    if pdef.kind == "kmap":
        return draw_map(results, ax, project, pdef, values)
    if pdef.kind == "surfaces":
        return draw_surfaces(results, ax, project, pdef, values)
    if pdef.kind == "spectrum":
        return draw_spectrum(results, ax, project, pdef)
    if pdef.kind == "conductance":
        return draw_conductance(results, ax, project, pdef)
    if colored and colors is None:
        colors = results.values(pdef.color_by)
    scatter = []  # colored: (x, y, colour value) per point
    y_scale, y_label = quantity_label(project, pdef)
    vals = values / y_scale
    n_outer, n_inner, n_bands = vals.shape
    sweep = sweep_of(results, pdef)
    cmap = matplotlib.colormaps[st.colormap]
    palette = getattr(cmap, "colors", None)  # qualitative maps: a list of distinct colours

    def pick(i, n):
        if palette is not None:
            return palette[i % len(palette)]
        return cmap(i / max(n - 1, 1))

    line = dict(ls=LINESTYLES.get(st.line_style, "-"), lw=st.line_width,
                marker="o" if st.markers else None, ms=st.marker_size, mew=0)
    table = []

    def curves(xs, series, idx, label=None, base=None):
        """series: (n_points, n_bands); idx(point) -> (i, j)."""
        if colored:
            for b in range(n_bands):
                if st.line_style != "none":  # thin neutral guide lines under the coloured points
                    ax.plot(xs, series[:, b], color="0.75", lw=st.line_width, ls=line["ls"], zorder=1)
                for n, xv in enumerate(xs):
                    i, j = idx(n)
                    table.append((xv, series[n, b], i, j, b))
                    scatter.append((xv, series[n, b], colors[i, j, b]))
            return
        for b in range(n_bands):
            color = base if base is not None else (pick(b, n_bands) if st.colors == "bands" else st.color)
            ax.plot(xs, series[:, b], color=color, label=label if b == 0 else None, **line)
            for n, xv in enumerate(xs):
                i, j = idx(n)
                table.append((xv, series[n, b], i, j, b))

    if sweep == "kpath":
        kx, k_label = inner_axis(results, st)
        k_scale = kx[-1] / results.x[-1] if results.x[-1] else 1.0
        px, p_label = outer_axis(results, project.display_units)
        unit = p_label.split("(")[-1].rstrip(")") if "(" in p_label else ""
        overlay = n_outer > 1 and pdef.at < 0
        for i in (range(n_outer) if overlay else [min(max(pdef.at, 0), n_outer - 1)]):
            label = f"{results.outer_name} = {px[i]:.4g} {unit}".rstrip() if overlay else None
            curves(kx, vals[i], lambda n, i=i: (i, n), label, pick(i, n_outer) if overlay else None)
        positions = [p * k_scale for p, _ in results.ticks]
        names = [_pretty(lbl) for _, lbl in results.ticks]
        for pos in positions:
            ax.axvline(pos, color="0.6", lw=0.6)
        if st.x_axis == "k":
            top = ax.secondary_xaxis("top")  # keep the symmetry-point names along the top
            top.set_xticks(positions, names)
            top.tick_params(labelsize=11, length=0)
        else:
            ax.set_xticks(positions, names)
        ax.set_xlim(kx[0], kx[-1])
        ax.set_xlabel(k_label)
        if overlay and st.legend:
            ax.legend(fontsize="small")
    elif sweep == "variable":
        px, p_label = outer_axis(results, project.display_units)
        overlay = n_inner > 1 and pdef.at < 0
        for j in (range(n_inner) if overlay else [min(max(pdef.at, 0), n_inner - 1)]):
            label = f"k point {j + 1}" if overlay else None
            curves(px, vals[:, j], lambda n, j=j: (n, j), label, pick(j, n_inner) if overlay else None)
        ax.set_xlabel(p_label)
        if overlay and n_inner <= 12 and st.legend:
            ax.legend(fontsize="small")
    else:  # single solve: quantity per mode
        xs = np.arange(1, n_bands + 1)
        if colored:
            scatter.extend((x, vals[0, 0, b], colors[0, 0, b]) for b, x in enumerate(xs))
        else:
            ax.plot(xs, vals[0, 0], color=st.color, **line)
        table = [(x, vals[0, 0, b], 0, 0, b) for b, x in enumerate(xs)]
        ax.set_xlabel("Mode")
    if colored and scatter:
        pts = np.array(scatter, dtype=float)
        lo, hi = _color_range(project, st)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap=st.colormap, vmin=lo, vmax=hi,
                        s=(2 * st.marker_size) ** 2, edgecolors="none", zorder=3)
        if st.colorbar:
            cb = ax.figure.colorbar(sc, ax=ax, pad=0.02, aspect=40)  # half the default thickness
            cb.set_label(pdef.color_label or " ".join(pdef.color_by.split()), fontsize=13)
            cb.ax.tick_params(labelsize=11)
    ax.set_ylabel(y_label, fontsize=13)
    if pdef.x_label:
        ax.set_xlabel(pdef.x_label)
    ax.xaxis.label.set_size(13)
    ax.tick_params(labelsize=11)
    (ylo, yhi), (xlo, xhi) = plot_ranges(project, pdef, y_scale)
    set_scales(ax, st, x=sweep != "kpath")
    if pdef.quantity.strip() == "freq" and ylo is None and not st.y_log:
        ylo = 0
    ax.set_ylim(bottom=ylo, top=yhi)
    if xlo is not None or xhi is not None:
        ax.set_xlim(left=xlo, right=xhi)
    if st.grid:
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(False)  # passing line properties would silently re-enable it
    if st.title:
        title = project.name if pdef.quantity.strip() == "freq" else f"{project.name}: {pdef.name}"
        ax.set_title(title, pad=22 if sweep == "kpath" and st.x_axis == "k" else 6)
    return np.array(table, dtype=float).reshape(-1, 5)


def _color_range(project, st) -> tuple:
    ev = project.evaluator()
    out = []
    for text in (st.color_min, st.color_max):
        text = str(text).strip()
        try:
            out.append(ev(text) if text else None)
        except Exception:
            out.append(None)
    return tuple(out)


def _pretty(label: str) -> str:
    return {"G": "Γ", "Gamma": "Γ"}.get(label, label)


def draw_map(results, ax, project, pdef, values: np.ndarray | None = None) -> np.ndarray:
    """One band of a wavevector mesh as a map over the Brillouin zone: filled contours of the quantity on the
    triangulated k points, with the zone outlined. Returns the pick table (P, 5) like `draw`."""
    import matplotlib
    from matplotlib.tri import Triangulation

    from ..plots import K_UNITS
    use_plot_style()
    st = pdef.style
    if values is None:
        values = results.values(pdef.quantity)
    n_outer, _, n_bands = values.shape
    i = min(max(pdef.at, 0), n_outer - 1)          # one value of the variable sweep at a time
    b = min(max(pdef.band, 0), n_bands - 1)
    scale, label = quantity_label(project, pdef)
    k_scale = K_UNITS[st.k_unit]
    kx, ky = results.k[i][:, 0] / k_scale, results.k[i][:, 1] / k_scale
    z = values[i, :, b] / scale
    tri = Triangulation(kx, ky, results.ktris)
    bad = np.isnan(z)
    if bad.any():                                   # bands the eigensolver did not reach at every point
        tri.set_mask(bad[results.ktris].any(axis=1))
    levels = max(int(st.levels), 2)
    lo, hi = plot_ranges(project, pdef, scale)[0]      # the quantity range is the colour range of a map
    if np.all(bad):
        ax.text(0.5, 0.5, f"Band {b + 1} has no values", ha="center", va="center", color="0.5",
                transform=ax.transAxes)
    else:
        filled = ax.tricontourf(tri, np.nan_to_num(z, nan=float(np.nanmin(z))), levels=levels,
                                cmap=st.colormap, vmin=lo, vmax=hi)
        if st.colorbar:
            cb = ax.figure.colorbar(filled, ax=ax, pad=0.02, aspect=40)
            cb.set_label(label, fontsize=13)
            cb.ax.tick_params(labelsize=11)
    if st.markers:
        ax.plot(kx, ky, ".", color="0.25", ms=st.marker_size, zorder=3)
    _zone_outline(ax, project, results, i, k_scale)
    axis_label = k_axis_label(st.k_unit, mathtext=True)
    ax.set_xlabel(f"Wavevector $k_x$ ({axis_label})", fontsize=13)
    ax.set_ylabel(f"Wavevector $k_y$ ({axis_label})", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_aspect("equal")
    if st.grid:
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(False)      # passing line properties would silently re-enable it
    _zone_limits(ax, project, st, k_scale)
    if st.title:
        ax.set_title(f"{project.name}: {pdef.name} (band {b + 1})")
    matplotlib.pyplot.setp(ax.spines.values(), visible=True)
    table = np.column_stack([kx, ky, np.full(len(kx), i), np.arange(len(kx)), np.full(len(kx), b)])
    return table.astype(float)


def _zone_outline(ax, project, results, i: int, k_scale: float) -> None:
    """Outline the first Brillouin zone behind the map (skipped if the lattice cannot be evaluated here)."""
    from .. import lattice as lat
    try:
        ev = project.evaluator()
        if results.outer_name:
            ev = ev.with_overrides({results.outer_name: float(results.outer_values[i])})
        recip = [v for v in lat.reciprocal(lat.slots(project.lattice, ev)) if v is not None]
        if len(recip) != 2:
            return
        poly = lat.brillouin_zone_2d(*recip) / k_scale
    except Exception:
        return
    ax.plot(*np.vstack([poly, poly[:1]]).T, color="0.5", lw=0.8, zorder=2)


def draw_surfaces(results, ax, project, pdef, values: np.ndarray | None = None) -> np.ndarray:
    """Bands of a wavevector mesh as surfaces ω(kx, ky) in a 3D view: `bands` of them from band `band` up,
    each a surface over the triangulated k points. `ax` must be a 3D axes. There is nothing to pick in a
    rotatable 3D view, so the pick table comes back empty."""
    import matplotlib
    from matplotlib.tri import Triangulation

    from ..plots import K_UNITS
    use_plot_style()
    st = pdef.style
    if values is None:
        values = results.values(pdef.quantity)
    n_outer, _, n_bands = values.shape
    i = min(max(pdef.at, 0), n_outer - 1)
    first = min(max(pdef.band, 0), n_bands - 1)
    last = n_bands if pdef.bands < 1 else min(first + pdef.bands, n_bands)
    scale, label = quantity_label(project, pdef)
    k_scale = K_UNITS[st.k_unit]
    kx, ky = results.k[i][:, 0] / k_scale, results.k[i][:, 1] / k_scale
    cmap = matplotlib.colormaps[st.colormap]
    palette = getattr(cmap, "colors", None)
    drawn = 0
    for b in range(first, last):
        z = values[i, :, b] / scale
        if np.all(np.isnan(z)):
            continue
        tri = Triangulation(kx, ky, results.ktris)
        bad = np.isnan(z)
        if bad.any():
            tri.set_mask(bad[results.ktris].any(axis=1))
        colour = palette[b % len(palette)] if palette else cmap((b - first) / max(last - first - 1, 1))
        ax.plot_trisurf(tri, np.nan_to_num(z, nan=float(np.nanmin(z))), color=colour, linewidth=0,
                        antialiased=True, shade=True)
        drawn += 1
    if not drawn:
        ax.text2D(0.5, 0.5, "No values in these bands", ha="center", va="center", color="0.5",
                  transform=ax.transAxes)
    axis_label = k_axis_label(st.k_unit, mathtext=True)
    ax.set_xlabel(f"$k_x$ ({axis_label})", fontsize=11, labelpad=6)
    ax.set_ylabel(f"$k_y$ ({axis_label})", fontsize=11, labelpad=6)
    ax.set_zlabel(label, fontsize=11, labelpad=6)
    ax.tick_params(labelsize=9)
    ax.view_init(elev=22, azim=-128)
    (zlo, zhi), _ = plot_ranges(project, pdef, scale)   # the quantity range is the vertical axis here
    if pdef.quantity.strip() == "freq" and zlo is None:
        zlo = 0
    ax.set_zlim(bottom=zlo, top=zhi)
    _zone_limits(ax, project, st, k_scale)
    ax.grid(st.grid)
    if st.title:
        span = f"band {first + 1}" if last - first == 1 else f"bands {first + 1}–{last}"
        ax.set_title(f"{project.name}: {pdef.name} ({span})")
    return np.zeros((0, 5))


def _scaled_range(project, style, attrs, scale: float) -> tuple:
    """(low, high) of a pair of range fields: plain numbers are in the displayed unit, expressions carrying
    units (e.g. 5[1/um]) are SI, as everywhere else."""
    ev = project.evaluator()
    out = []
    for attr in attrs:
        text = str(getattr(style, attr)).strip()
        try:
            value = ev(text) if text else None
        except Exception:
            value = None
        out.append(value / scale if value is not None and "[" in text else value)
    return tuple(out)


def _zone_limits(ax, project, style, k_scale: float) -> None:
    """Apply the kx (x_min, x_max) and ky (ky_min, ky_max) ranges of a plot over the zone."""
    for attrs, setter in ((("x_min", "x_max"), ax.set_xlim), (("ky_min", "ky_max"), ax.set_ylim)):
        lo, hi = _scaled_range(project, style, attrs, k_scale)
        if lo is not None or hi is not None:
            setter(lo, hi)


def draw_spectrum(results, ax, project, pdef) -> np.ndarray:
    """A spectrum of the wavevector mesh against frequency: the density of states, the average group
    velocity or the heat flux, each summed over the bands (physics/solid/spectra.py). Nothing on the curve
    belongs to a single mode, so the pick table comes back empty."""
    from ..physics.solid import spectra
    use_plot_style()
    st = pdef.style
    i = min(max(pdef.at, 0), results.freqs.shape[0] - 1)
    try:
        temperature = project.evaluator()(pdef.temperature)
    except Exception as exc:
        raise ValueError(f"temperature '{pdef.temperature}': {exc}") from None
    try:
        freq, values = spectra.spectrum(results, pdef.spectrum, bins=pdef.bins, temperature=temperature,
                                        outer=i, band=pdef.band, bands=pdef.bands,
                                        incomplete=pdef.show_incomplete)
    except spectra.NoCompleteRange as exc:
        ax.text(0.5, 0.5, "\n".join(_wrap(str(exc))), ha="center", va="center", color="0.5",
                transform=ax.transAxes)
        return np.zeros((0, 5))
    scale = UNITS[project.frequency_unit]
    ax.plot(freq / scale, values, color=st.color, ls=LINESTYLES.get(st.line_style, "-"), lw=st.line_width,
            marker="o" if st.markers else None, ms=st.marker_size, mew=0)
    if pdef.show_incomplete:
        complete = spectra.complete_to(results, i, pdef.band, pdef.bands)
        ax.axvline(complete / scale, color="0.55", ls="--", lw=1.0, zorder=1)
        ax.annotate("bands complete to here", (complete / scale, 0.97), xycoords=("data", "axes fraction"),
                    ha="right", va="top", rotation=90, fontsize=9, color="0.45",
                    textcoords="offset points", xytext=(-4, 0))
    ax.set_xlabel(pdef.x_label or f"Frequency ({project.frequency_unit})", fontsize=13)
    ax.set_ylabel(pdef.y_label or spectra.SPECTRUM_LABELS[pdef.spectrum], fontsize=13)
    ax.tick_params(labelsize=11)
    set_scales(ax, st)
    lo, hi = _scaled_range(project, st, ("y_min", "y_max"), 1.0)
    ax.set_ylim(bottom=0 if lo is None and not st.y_log else lo, top=hi)
    lo, hi = _scaled_range(project, st, ("x_min", "x_max"), scale)
    ax.set_xlim(left=lo, right=hi)
    if st.grid:
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(False)
    if st.title:
        ax.set_title(f"{project.name}: {pdef.name}")
    return np.zeros((0, 5))


def temperatures_of(project, pdef) -> np.ndarray:
    """The temperatures (K) a conductance plot is drawn at: spaced logarithmically when the axis is."""
    ev = project.evaluator()
    low, high = ev(pdef.t_min), ev(pdef.t_max)
    if not (high > low > 0):
        raise ValueError(f"temperature range {low:g} to {high:g} K: both must be positive and increasing")
    points = max(int(pdef.t_points), 2)
    space = np.geomspace if pdef.style.x_log else np.linspace
    return space(low, high, points)


def draw_conductance(results, ax, project, pdef) -> np.ndarray:
    """The heat flux integrated over the computed bands, against temperature: the ballistic thermal
    conductance G(T) or the radiated power P(T). Nothing on the curve belongs to one mode, so the pick
    table comes back empty."""
    from ..physics.solid import spectra
    use_plot_style()
    st = pdef.style
    i = min(max(pdef.at, 0), results.freqs.shape[0] - 1)
    wanted = temperatures_of(project, pdef)
    try:
        temperatures, values = spectra.transport(results, pdef.transport, wanted, bins=pdef.bins, outer=i,
                                                 band=pdef.band, bands=pdef.bands,
                                                 incomplete=pdef.show_incomplete)
    except spectra.NoCompleteRange as exc:
        ax.text(0.5, 0.5, "\n".join(_wrap(str(exc))), ha="center", va="center", color="0.5",
                transform=ax.transAxes)
        return np.zeros((0, 5))
    ax.plot(temperatures, values, color=st.color, ls=LINESTYLES.get(st.line_style, "-"), lw=st.line_width,
            marker="o" if st.markers else None, ms=st.marker_size, mew=0)
    ax.set_xlabel(pdef.x_label or "Temperature (K)", fontsize=13)
    ax.set_ylabel(pdef.y_label or spectra.TRANSPORT_LABELS[pdef.transport], fontsize=13)
    ax.tick_params(labelsize=11)
    set_scales(ax, st)
    ax.set_xlim(temperatures[0], temperatures[-1])      # the axis is the range that was computed
    lo, hi = _scaled_range(project, st, ("y_min", "y_max"), 1.0)
    if lo is not None or hi is not None:
        ax.set_ylim(lo, hi)
    if st.grid:
        ax.grid(True, alpha=0.3, which="both")
    else:
        ax.grid(False)
    if st.title:
        ax.set_title(f"{project.name}: {pdef.name}")
    return np.zeros((0, 5))


def _wrap(text: str, width: int = 46) -> list[str]:
    """Break a message into lines short enough for a plot area."""
    import textwrap
    return textwrap.wrap(text, width)
