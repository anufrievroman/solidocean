"""Field results: one solution field per variable value and time step (heat transfer, stationary solid
mechanics, ...), and their plots: quantity against time or the swept variable, and profiles along a line."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import fields, outputs
from ..plots import VECTOR_KINDS
from .style import LINESTYLES, outer_axis, outer_text, plot_ranges, set_scales, time_axis, use_plot_style


@dataclass
class FieldResults:
    """Base of the results of field problems. Subclasses hold the solution fields and implement
    point_values (line plots) and save; `quantities` maps an expression to its values (n_outer, n_times)."""
    outer_name: str = ""
    outer_values: np.ndarray = field(default_factory=lambda: np.zeros(1))
    times: np.ndarray = field(default_factory=lambda: np.zeros(1))     # (n_t,) s; one entry if stationary
    quantities: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    meshes: dict = field(default_factory=dict, repr=False)               # variable index -> mesh
    samplers: dict = field(default_factory=dict, repr=False)             # point locations of line plots (cache)
    kind: str = "stationary"                                             # stationary | transient
    timed: bool = True        # tables have a time column
    columns: list = field(default_factory=list, repr=False)   # (name, expression) of quantities.csv, by plot

    def values(self, quantity: str):
        return self.quantities.get(quantity.strip())

    def point_values(self, i: int, j: int, expr: str, scalars, sampler) -> np.ndarray:
        """An expression at the sampler's points for variable value i, time step j."""
        raise NotImplementedError

    def compute(self, project, quantity: str) -> np.ndarray:
        """Evaluate a quantity on the stored fields (n_outer, n_times) and keep it in `quantities`."""
        raise NotImplementedError

    def save(self, directory: Path, contents=outputs.CONTENTS) -> list[Path]:
        """Write the results files (quantities.csv and the fields) into a directory."""
        raise NotImplementedError

    def write_outputs(self, project, directory: Path, plots) -> list[Path]:
        """Everything a run writes, as [output] contents says: the results files, a CSV per line or
        section plot, and the model with its run record. Plots stay in the GUI ("Save plot…")."""
        from .style import slug
        out = project.output
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.columns = quantity_columns(plots, self.quantities)
        written = self.save(directory, out.contents)
        if out.wants("tables"):
            for pl in plots:
                if pl.kind == "line":  # profile along the line: one column per variable value
                    written.append(save_line_table(self, project, pl, directory / f"{slug(pl.name)}.csv"))
                elif pl.kind == "section":
                    written.append(save_section_table(self, project, pl, directory / f"{slug(pl.name)}.csv"))
                elif pl.kind in VECTOR_KINDS:
                    written.append(save_vector_table(self, project, pl, directory / f"{slug(pl.name)}.csv"))
        if out.wants("model"):
            written += outputs.write_record(project, directory, self)
        return written

    def save_table(self, path: Path, quantity: str) -> Path:
        vals = self.values(quantity)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(([self.outer_name] if self.outer_name else []) + (["time [s]"] if self.timed else [])
                       + [quantity])
            for i in range(vals.shape[0]):
                for j, t in enumerate(self.times):
                    w.writerow(([f"{self.outer_values[i]:.10g}"] if self.outer_name else []) +
                               ([f"{t:.10g}"] if self.timed else []) + [f"{vals[i, j]:.10g}"])
        return Path(path)

    def save_quantities(self, directory: Path) -> list[Path]:
        """quantities.csv: every evaluated quantity per variable value (and time), one column per plot headed
        by the plot's name, and quantities.json: column name -> expression, in the order of the columns."""
        columns = self.columns or quantity_columns([], self.quantities)
        if not columns:
            return []
        path = directory / "quantities.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(([self.outer_name] if self.outer_name else []) + (["time [s]"] if self.timed else []) +
                       [name for name, _ in columns])
            for i in range(len(self.outer_values)):
                for j, t in enumerate(self.times):
                    w.writerow(([f"{self.outer_values[i]:.10g}"] if self.outer_name else []) +
                               ([f"{t:.10g}"] if self.timed else []) +
                               [f"{self.quantities[q][i, j]:.10g}" for _, q in columns])
        index = directory / "quantities.json"
        index.write_text(json.dumps(dict(columns), indent=2, ensure_ascii=False), encoding="utf-8")
        return [path, index]


def quantity_columns(plots, quantities) -> list[tuple[str, str]]:
    """The columns of quantities.csv: (name, expression) for every plot whose quantity was evaluated, named
    after the plot (a repeated name gets " (2)", ...), then the other evaluated quantities under their
    expression."""
    columns, names = [], set()

    def add(name, q):
        unique, n = name, 2
        while unique in names:
            unique, n = f"{name} ({n})", n + 1
        columns.append((unique, q))
        names.add(unique)
    for pl in plots:
        for q in pl.expressions():
            if q.strip() in quantities:
                add(pl.name.strip() or q.strip(), q.strip())
    plotted = {q for _, q in columns}
    for q in quantities:
        if q not in plotted:
            add(q, q)
    return columns


def line_profile(results, project, pdef):
    """A line plot's data: x values (n_outer, n) in SI (the line may depend on the swept variable), values
    (n_outer, n) at time step pdef.at_time, the time index used, and the x-axis label. Points outside the
    structure give NaN (gaps in the curve). Works for heat and stationary solid results (both provide
    point_values)."""
    from ..units import LENGTH, axis_scale
    base = project.evaluator()
    j = len(results.times) - 1 if pdef.at_time < 0 else min(pdef.at_time, len(results.times) - 1)
    out, xs = [], []
    for i in range(len(results.outer_values)):
        ev = base.with_overrides({results.outer_name: results.outer_values[i]}) if results.outer_name else base
        a, b = ev.vector(pdef.start), ev.vector(pdef.end)
        t = np.linspace(0.0, 1.0, max(int(pdef.points), 2))
        pts = a + t[:, None] * (b - a)
        mesh = results.meshes[i]
        key = (id(mesh), pts.round(15).tobytes())
        sampler = results.samplers.get(key)
        if sampler is None:
            sampler = results.samplers[key] = fields.PointSampler(mesh, pts)
        out.append(results.point_values(i, j, pdef.quantity, {**ev.values, "time": results.times[j]}, sampler))
        xs.append(t * np.linalg.norm(b - a) if pdef.line_axis == "s" else pts[:, "xyz".index(pdef.line_axis)])
    _, unit = axis_scale(LENGTH, project.length_unit)
    label = f"Distance along line ({unit})" if pdef.line_axis == "s" else f"{pdef.line_axis} ({unit})"
    return np.array(xs), np.array(out), j, label


def save_line_table(results, project, pdef, path: Path) -> Path:
    xs, vals, j, label = line_profile(results, project, pdef)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        xname = "s [m]" if pdef.line_axis == "s" else f"{pdef.line_axis} [m]"
        names = [f"{results.outer_name}={v:.6g}" for v in results.outer_values] if results.outer_name \
            else [pdef.quantity]
        same = np.allclose(xs, xs[0], rtol=1e-12, atol=0)     # one x column, else an x column per value
        w.writerow([xname] + names if same else [h for nm in names for h in (f"{xname} ({nm})", nm)])
        for n in range(xs.shape[1]):
            if same:
                w.writerow([f"{xs[0, n]:.10g}"] + [f"{vals[i, n]:.10g}" for i in range(len(vals))])
            else:
                w.writerow([v for i in range(len(vals)) for v in (f"{xs[i, n]:.10g}", f"{vals[i, n]:.10g}")])
    return Path(path)


def draw(results, ax, project, pdef, values: np.ndarray | None = None) -> np.ndarray:
    """Plots of field results. Time plot (kind "time"): the quantity against time, one curve per variable value (or the one
    selected by `at`); sweep plot (kind "quantity"): against the swept variable at time step `at`
    (-1 = the last); line plot (kind "line"): along a segment at time step `at_time`, one curve per variable
    value (or the one selected by `at`). Returns the pick table x, y, i, j, 0."""
    import matplotlib
    use_plot_style()
    if pdef.kind == "section":
        return draw_section(results, ax, project, pdef)
    if pdef.kind in VECTOR_KINDS:
        return draw_vectors(results, ax, project, pdef)
    st = pdef.style
    vals = results.values(pdef.quantity) if values is None else values
    cmap = matplotlib.colormaps[st.colormap]
    palette = getattr(cmap, "colors", None)

    def pick(i, n):
        return palette[i % len(palette)] if palette is not None else cmap(i / max(n - 1, 1))

    line = dict(ls=LINESTYLES.get(st.line_style, "-"), lw=st.line_width,
                marker="o" if st.markers else None, ms=st.marker_size, mew=0)
    table = []
    n_outer, n_t = (len(results.outer_values), len(results.times)) if vals is None else vals.shape
    px, p_label = outer_axis(results, project.display_units) if results.outer_name else (np.arange(n_outer), "")
    if pdef.kind == "line":  # profile along a segment, one curve per variable value (or the chosen one)
        from ..units import LENGTH, axis_scale
        xs, lvals, j, xlabel = line_profile(results, project, pdef)
        scale, _ = axis_scale(LENGTH, project.length_unit)
        n_outer = len(lvals)
        overlay = n_outer > 1 and pdef.at < 0
        for i in (range(n_outer) if overlay else [min(max(pdef.at, 0), n_outer - 1)]):
            label = f"{results.outer_name} = {outer_text(results, project, i)}" if overlay else None
            ax.plot(xs[i] * scale, lvals[i], color=pick(i, n_outer) if overlay else st.color, label=label, **line)
            table += [(x, y, i, j, 0) for x, y in zip(xs[i] * scale, lvals[i]) if np.isfinite(y)]
        if overlay and st.legend:
            ax.legend(fontsize="small")
        ax.set_xlabel(xlabel)
        if len(results.times) > 1:
            tscale, tunit = time_axis(project)
            ax.set_title(f"t = {results.times[j] / tscale:.4g} {tunit}", fontsize=11, loc="right")
    elif pdef.kind == "time":
        scale, unit = time_axis(project)
        tx = results.times / scale
        overlay = n_outer > 1 and pdef.at < 0
        for i in (range(n_outer) if overlay else [min(max(pdef.at, 0), n_outer - 1)]):
            label = f"{results.outer_name} = {outer_text(results, project, i)}" if overlay else None
            ax.plot(tx, vals[i], color=pick(i, n_outer) if overlay else st.color,
                    label=label, **line)
            table += [(x, y, i, j, 0) for j, (x, y) in enumerate(zip(tx, vals[i]))]
        ax.set_xlabel(f"Time ({unit})")
        if overlay and st.legend:
            ax.legend(fontsize="small")
    else:
        j = n_t - 1 if pdef.at < 0 else min(pdef.at, n_t - 1)
        ax.plot(px, vals[:, j], color=st.color, **line)
        table = [(x, y, i, j, 0) for i, (x, y) in enumerate(zip(px, vals[:, j]))]
        ax.set_xlabel(p_label or "step")
    ax.set_ylabel(pdef.y_label or named_quantity(project, pdef.quantity), fontsize=13)
    if pdef.x_label:
        ax.set_xlabel(pdef.x_label)
    ax.xaxis.label.set_size(13)
    ax.tick_params(labelsize=11)
    (ylo, yhi), (xlo, xhi) = plot_ranges(project, pdef)
    set_scales(ax, st)
    if ylo is not None or yhi is not None:
        ax.set_ylim(bottom=ylo, top=yhi)
    if xlo is not None or xhi is not None:
        ax.set_xlim(left=xlo, right=xhi)
    if st.grid:
        ax.grid(True, alpha=0.3)
    if st.title:
        ax.set_title(f"{project.name}: {pdef.name}")
    return np.array(table, dtype=float).reshape(-1, 5)


def named_quantity(project, expr: str) -> str:
    """The label of a named quantity of the physics (e.g. "Temperature (K)" for T), else the expression."""
    from .. import physics
    text = " ".join(expr.split())
    for label, e in physics.of(project).quantities(project):
        if " ".join(e.split()) == text:
            return label
    return text


# ---- cross-sections ------------------------------------------------------------------------------------
_PLANE_AXES = {"xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}


def section_plane(project, pdef, ev, mesh=None, bounds=None):
    """Origin and in-plane unit vectors (e1, e2) of a cross-section, and the (u, v) ranges that cover the
    model's bounding box (of the mesh, or `bounds` = (lo, hi)), in SI."""
    from ..materials import rotation_matrix
    a, b = _PLANE_AXES[pdef.plane]
    R = rotation_matrix(ev.vector(pdef.rotation))
    e1, e2 = R @ np.eye(3)[a], R @ np.eye(3)[b]
    origin = ev.vector(pdef.origin)
    lo, hi = mesh.bounds() if mesh is not None else (np.asarray(bounds[0]), np.asarray(bounds[1]))
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    u, v = (corners - origin) @ e1, (corners - origin) @ e2
    return origin, e1, e2, (u.min(), u.max()), (v.min(), v.max())


def section_grid(results, project, pdef, n: int | None = None, centered: bool = False):
    """The sample points of a cross-section: a namespace with the axes u (nu,) and v (nv,) in SI (measured from
    the plane's origin), the point sampler of the grid, the scalars of the evaluation, the variable and time indices
    used, the plane's unit vectors e1, e2 and origin, the displayed axes ud, vd and their names, and the size and
    the displayed box (u0, u1, v0, v1) of the whole plane. `n` samples
    along the longer side (default: the plot's resolution); `centered`: at the centres of n cells rather than
    on the edges. An unrotated plane has the model's coordinates on its axes (x, y, z); a rotated one in-plane
    coordinates u, v measured from its origin."""
    from types import SimpleNamespace
    i = min(max(pdef.at, 0), len(results.outer_values) - 1)
    j = len(results.times) - 1 if pdef.at_time < 0 else min(pdef.at_time, len(results.times) - 1)
    base = project.evaluator()
    ev = base.with_overrides({results.outer_name: results.outer_values[i]}) if results.outer_name else base
    mesh = results.meshes[i]
    origin, e1, e2, (u0, u1), (v0, v1) = section_plane(project, pdef, ev, mesh)
    n = int(pdef.resolution if n is None else n)
    # samples: square cells on the data, but for arrows square on the screen, where a slender section is stretched
    w, h = u1 - u0, (v1 - v0) * (stretch_aspect(u1 - u0, v1 - v0) if centered else 1.0)
    span = max(w, h)
    floor = min(n, 3) if centered else min(n, 40)
    nu = max(floor, int(round(n * w / span)))            # at least a few samples across thin sections
    nv = max(floor, int(round(n * h / span)))
    if centered:
        u = u0 + (np.arange(nu) + 0.5) * (u1 - u0) / nu
        v = v0 + (np.arange(nv) + 0.5) * (v1 - v0) / nv
    else:
        u, v = np.linspace(u0, u1, nu), np.linspace(v0, v1, nv)
    U, V = np.meshgrid(u, v)
    pts = origin + U.reshape(-1, 1) * e1 + V.reshape(-1, 1) * e2
    key = (id(mesh), pts.round(15).tobytes())
    sampler = results.samplers.get(key)
    if sampler is None:
        sampler = results.samplers[key] = fields.PointSampler(mesh, pts)
    rotated = any(abs(float(x)) > 0 for x in ev.vector(pdef.rotation))
    a, b = _PLANE_AXES[pdef.plane]
    ud, vd, names = (u, v, ("u", "v")) if rotated else (u + origin[a], v + origin[b], ("xyz"[a], "xyz"[b]))
    du, dv = ud[0] - u[0], vd[0] - v[0]
    return SimpleNamespace(u=u, v=v, nu=nu, nv=nv, sampler=sampler, i=i, j=j, e1=e1, e2=e2, origin=origin,
                           scalars={**ev.values, "time": results.times[j]}, ud=ud, vd=vd, names=names,
                           size=(u1 - u0, v1 - v0), box=(u0 + du, u1 + du, v0 + dv, v1 + dv))


def section_values(results, project, pdef):
    """A cross-section's data: the axes u (nu,) and v (nv,) in SI, values (nv, nu) (NaN outside the structure),
    the variable and time indices used, and the axis names (see section_grid)."""
    from .. import physics
    g = section_grid(results, project, pdef)
    expr = pdef.quantity.strip() or physics.of(project).field_quantity(project)
    vals = results.point_values(g.i, g.j, expr, g.scalars, g.sampler).reshape(g.nv, g.nu)
    return g.ud, g.vd, vals, g.i, g.j, g.names


def _grid_values(results, g, expr: str) -> np.ndarray:
    """An expression on a section grid, (nv, nu); a constant is spread over the grid."""
    x = np.asarray(results.point_values(g.i, g.j, expr, g.scalars, g.sampler), dtype=float)
    return np.full(g.nu * g.nv, x.item()).reshape(g.nv, g.nu) if x.size == 1 else x.reshape(g.nv, g.nu)


def vector_values(results, project, pdef, n: int | None = None, centered: bool = False):
    """A vector quantity on a cross-section: the grid (see section_grid) and the vector's components along the
    plane's axes, Vu and Vv (nv, nu), in the units of the quantity (NaN outside the structure)."""
    g = section_grid(results, project, pdef, n, centered)
    V = np.stack([_grid_values(results, g, e) for e in pdef.vector], axis=-1)       # (nv, nu, 3)
    return g, V @ g.e1, V @ g.e2


def stretch_aspect(width: float, height: float) -> float:
    """Vertical data units per horizontal one on the screen of a map of the given size: 1, but a slender section at
    true proportions would be a thin line, so it is stretched to fill the plot."""
    stretch = max(width, height) / max(min(width, height), 1e-300)
    if stretch <= 4:
        return 1.0
    return 0.6 * width / height if width > height else 1.6 * width / height


def _finish_map(ax, project, pdef, u, v, names, g_i, g_j, results, mappable=None, label=""):
    """Everything around a map of the plane: aspect ratio, axis labels, colour bar (of `mappable`) and titles."""
    from ..units import LENGTH, axis_scale
    st = pdef.style
    scale, unit = axis_scale(LENGTH, project.length_unit)
    width, height = u[-1] - u[0], v[-1] - v[0]
    stretch = max(width, height) / max(min(width, height), 1e-300)
    labels = [f"{names[0]} ({unit})", f"{names[1]} ({unit})"]
    aspect = stretch_aspect(width, height)
    if aspect != 1.0:
        k = 1 if width > height else 0
        labels[k] += f", stretched ×{stretch * (0.6 if k else 1 / 1.6):.0f}"
    ax.set_aspect(aspect)
    ax.set_xlabel(labels[0], fontsize=13)
    ax.set_ylabel(labels[1], fontsize=13)
    ax.tick_params(labelsize=11)
    if mappable is not None and st.colorbar:   # about as tall as the map, which is less tall than the axes box when wide
        fw, fh = ax.figure.get_size_inches()
        shrink = float(np.clip(fw * 0.8 / (fh * width / (height * aspect)), 0.25, 1.0))
        cb = ax.figure.colorbar(mappable, ax=ax, shrink=shrink, pad=0.02, aspect=25 * shrink)
        cb.set_label(label, fontsize=13)
        cb.ax.tick_params(labelsize=11)
    parts = []
    if results.outer_name:
        parts.append(f"{results.outer_name} = {outer_text(results, project, g_i)}")
    if len(results.times) > 1:
        tscale, tunit = time_axis(project)
        parts.append(f"t = {results.times[g_j] / tscale:.4g} {tunit}")
    if parts:
        ax.set_title(", ".join(parts), fontsize=11, loc="right")
    if st.title:
        ax.set_title(f"{project.name}: {pdef.name}", loc="left")
    return aspect


def _colour_scale(st):
    """The colormap of a map: a qualitative palette (for curves) is replaced by a colour scale."""
    import matplotlib
    cmap = matplotlib.colormaps[st.colormap]
    return matplotlib.colormaps["viridis"] if cmap.N <= 20 else cmap


def draw_section(results, ax, project, pdef) -> np.ndarray:
    """A 2D colour map of the quantity on the cross-section; returns no pick points."""
    from .. import physics
    from ..units import LENGTH, axis_scale
    st = pdef.style
    u, v, vals, i, j, names = section_values(results, project, pdef)
    scale, unit = axis_scale(LENGTH, project.length_unit)
    lo, hi = _section_range(project, st, vals)
    im = ax.imshow(np.ma.masked_invalid(vals), origin="lower", cmap=_colour_scale(st), vmin=lo, vmax=hi,
                   extent=(u[0] * scale, u[-1] * scale, v[0] * scale, v[-1] * scale), interpolation="nearest")
    expr = pdef.quantity.strip() or physics.of(project).field_quantity(project)
    _finish_map(ax, project, pdef, u, v, names, i, j, results, im, pdef.color_label or named_quantity(project, expr))
    return np.zeros((0, 5))


def draw_vectors(results, ax, project, pdef) -> np.ndarray:
    """A vector quantity on the cross-section: its in-plane part as arrows (kind "quiver", on a grid of
    `arrows` cells across the longer side) or as streamlines (kind "stream"), over a colour map of `quantity`
    if there is one. Without a map the arrows are coloured by their length. Returns no pick points."""
    import matplotlib
    from ..units import LENGTH, axis_scale
    st = pdef.style
    scale, unit = axis_scale(LENGTH, project.length_unit)
    quiver = pdef.kind == "quiver"
    # the map and the streamlines live on the plot's resolution grid, the arrows on a coarser one
    g, Vu, Vv = vector_values(results, project, pdef, int(pdef.arrows) if quiver else None, centered=quiver)
    u, v = np.array(g.box[:2]), np.array(g.box[2:])      # the whole plane (the arrows' grid is inside it)
    mappable, label = None, ""
    if pdef.quantity.strip():
        bu, bv, vals, _, _, _ = section_values(results, project, pdef)
        lo, hi = _section_range(project, st, vals)
        mappable = ax.imshow(np.ma.masked_invalid(vals), origin="lower", cmap=_colour_scale(st), vmin=lo, vmax=hi,
                             extent=(bu[0] * scale, bu[-1] * scale, bv[0] * scale, bv[-1] * scale),
                             interpolation="nearest")
        label = pdef.color_label or named_quantity(project, pdef.quantity.strip())
        arrow_color, norm, cmap, mappable_arrows = st.color, None, None, False
    else:
        cmap, mappable_arrows = _colour_scale(st), True
        mag = np.hypot(Vu, Vv)
        lo, hi = _section_range(project, st, mag)
        norm = matplotlib.colors.Normalize(lo, hi)
        label = pdef.color_label or "In-plane magnitude"
    # lengths on the screen: a stretched section shows the vertical axis shorter or longer than the data
    # extent, so the arrows are given in screen units
    aspect = stretch_aspect(u[-1] - u[0], v[-1] - v[0])
    ok = np.isfinite(Vu) & np.isfinite(Vv)
    X, Y = np.meshgrid(g.ud * scale, g.vd * scale)
    artist = None
    if ok.any() and quiver:
        Us, Vs = Vu[ok], Vv[ok] * aspect
        longest = float(np.hypot(Us, Vs).max())
        pitch = min(g.size[0] / g.nu, g.size[1] / g.nv * aspect) * scale      # spacing on the screen (x units)
        kw = dict(angles="uv", scale_units="x", pivot="mid", width=0.0035 * pdef.arrow_width,
                  scale=max(longest, 1e-300) / (pitch * pdef.arrow_scale))
        if mappable_arrows:
            artist = ax.quiver(X[ok], Y[ok], Us, Vs, np.hypot(Vu, Vv)[ok], cmap=cmap, norm=norm, **kw)
        else:
            artist = ax.quiver(X[ok], Y[ok], Us, Vs, color=arrow_color, **kw)
    elif ok.any():
        sx = np.ma.masked_invalid(np.where(ok, Vu, np.nan))
        sy = np.ma.masked_invalid(np.where(ok, Vv, np.nan))
        kw = dict(density=pdef.stream_density, linewidth=st.line_width, arrowsize=pdef.arrow_scale)
        if mappable_arrows:
            artist = ax.streamplot(X[0], Y[:, 0], sx, sy, color=np.hypot(Vu, Vv), cmap=cmap, norm=norm,
                                   **kw).lines
        else:
            ax.streamplot(X[0], Y[:, 0], sx, sy, color=arrow_color, **kw)
    ax.set_xlim(u[0] * scale, u[-1] * scale)
    ax.set_ylim(v[0] * scale, v[-1] * scale)
    _finish_map(ax, project, pdef, u, v, g.names, g.i, g.j, results, mappable if mappable is not None else artist,
                label)
    return np.zeros((0, 5))


def _section_range(project, st, vals):
    """Colour range: the style's, else the data's; a (nearly) constant field gets a range around its value
    instead of stretching round-off noise over the whole colour scale."""
    ok = np.isfinite(vals).any()
    lo, hi = (float(np.nanmin(vals)), float(np.nanmax(vals))) if ok else (0.0, 1.0)
    if hi - lo <= 1e-9 * max(abs(lo), abs(hi)):
        pad = 0.5 if lo == hi == 0 else 1e-3 * max(abs(lo), abs(hi))
        lo, hi = lo - pad, hi + pad
    ev = project.evaluator()
    out = []
    for text, fallback in ((st.color_min, lo), (st.color_max, hi)):
        try:
            out.append(ev(text) if str(text).strip() else fallback)
        except Exception:
            out.append(fallback)
    return out


def save_section_table(results, project, pdef, path: Path) -> Path:
    """The cross-section as a table, one row per sample: the point x, y, z (m) and the value; a rotated plane
    also has its in-plane coordinates u, v (m) in front."""
    from .. import physics
    u, v, vals, i, j, names = section_values(results, project, pdef)
    ev = project.evaluator()
    if results.outer_name:
        ev = ev.with_overrides({results.outer_name: results.outer_values[i]})
    origin, e1, e2, (u0, _), (v0, _) = section_plane(project, pdef, ev, results.meshes[i])
    du, dv = u[0] - u0, v[0] - v0          # the displayed axes may be shifted by the origin
    in_plane = names == ("u", "v")         # an unrotated plane's axes are already among x, y, z
    value = pdef.quantity.strip() or physics.of(project).field_quantity(project)
    at = [f"{results.outer_name}={results.outer_values[i]:.6g}"] if results.outer_name else []
    if len(results.times) > 1:
        at.append(f"t={results.times[j]:.6g} s")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow((["u [m]", "v [m]"] if in_plane else []) + ["x [m]", "y [m]", "z [m]"]
                   + [f"{value} ({', '.join(at)})" if at else value])
        for b, vv in enumerate(v):
            for a, uu in enumerate(u):
                p = origin + (uu - du) * e1 + (vv - dv) * e2
                w.writerow(([f"{uu:.10g}", f"{vv:.10g}"] if in_plane else []) + [f"{c:.10g}" for c in p]
                           + [f"{vals[b, a]:.10g}"])
    return Path(path)


def save_vector_table(results, project, pdef, path: Path) -> Path:
    """A vector plot as a table, one row per sample of the plot's resolution grid: the point x, y, z (m), the
    three components of the vector, and the quantity behind it if there is one."""
    g, _, _ = vector_values(results, project, pdef)
    V = [_grid_values(results, g, e) for e in pdef.vector]
    bg = pdef.quantity.strip()
    extra = [_grid_values(results, g, bg)] if bg else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x [m]", "y [m]", "z [m]"] + [" ".join(e.split()) for e in pdef.vector] + ([bg] if bg else []))
        for b, vv in enumerate(g.v):
            for a, uu in enumerate(g.u):
                p = g.origin + uu * g.e1 + vv * g.e2
                w.writerow([f"{c:.10g}" for c in p] + [f"{c[b, a]:.10g}" for c in V + extra])
    return Path(path)


def section_corners(project, pdef, bounds) -> np.ndarray:
    """The four corners (4, 3) of a cross-section over the model's bounding box (to show the plane)."""
    origin, e1, e2, (u0, u1), (v0, v1) = section_plane(project, pdef, project.evaluator(), bounds=bounds)
    return np.array([origin + u * e1 + v * e2 for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])
