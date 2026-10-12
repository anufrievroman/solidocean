"""Plots of modal results (eigenfrequencies per sweep point and mode): dispersion, coloured dispersion and
quantities per mode against a k-path or a variable sweep."""

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
