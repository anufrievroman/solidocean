"""Results plot definitions (saved in the model file): 1D plots and 3D field views, with their style.

Which kinds a physics offers, their titles and defaults are defined by the physics (physics/*/spec);
these classes are shared by all physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tomlio import without_defaults

Expr = str | float | int


LINE_STYLES = ("solid", "dashed", "dotted", "none")


X_AXES = ("points", "k")


K_UNITS = {"1/m": 1.0, "1/cm": 1e2, "1/mm": 1e3, "1/um": 1e6, "1/nm": 1e9}  # rad/m per display unit


COLOR_MODES = ("single", "bands")


@dataclass
class PlotStyle:
    """Appearance of the dispersion plot (GUI and the CLI's dispersion.png)."""
    markers: bool = True
    marker_size: float = 3.0
    line_style: str = "solid"   # solid | dashed | dotted | none
    line_width: float = 1.2
    colors: str = "bands"       # one colour per band from the colormap, or a single colour
    color: str = "#1f77b4"
    colormap: str = "Paired"    # qualitative palette: per-band colours and overlaid variable values
    grid: bool = True
    title: bool = False
    legend: bool = True         # a legend of the overlaid curves (variable values, k points)
    # ranges: empty = automatic; plain numbers are in the displayed unit, or give units: "10[GHz]"
    y_min: Expr = ""
    y_max: Expr = ""
    x_axis: str = "points"      # k-path x axis: "points" (Γ, X, M ticks) or "k" (wavevector values)
    k_unit: str = "1/cm"        # unit of the wavevector axis: 1/m, 1/cm, 1/mm, 1/um, 1/nm
    x_min: Expr = ""            # x range in the displayed x unit
    x_max: Expr = ""
    y_log: bool = False         # logarithmic axes (not for the x axis of a k-path)
    x_log: bool = False
    # colored dispersion: colour scale range (empty = data range), colour bar and its label
    color_min: Expr = ""
    color_max: Expr = ""
    colorbar: bool = True
    # field plots (3D view)
    outline: bool = True        # displacement: undeformed outline
    deform_scale: float = 0.05  # displacement: largest deformation as a fraction of the model size

    FIELD_KEYS = ("colormap", "colorbar", "title", "outline", "deform_scale")

    def to_dict(self, kind: str = ""):
        """The style keys of a plot kind; 1D plots without their defaults. Field views keep all their keys: on
        loading they are completed from the physics' field defaults, not from these."""
        d = dict(self.__dict__)
        if kind == "field":
            return {k: d[k] for k in self.FIELD_KEYS}
        for k in ("outline", "deform_scale"):
            d.pop(k)
        return without_defaults(self, d)

    @classmethod
    def from_dict(cls, d: dict) -> "PlotStyle":
        return cls(**d)


SWEEP_AXES = ("auto", "kpath", "variable")


# dispersion: frequency vs sweep; quantity: any expression vs sweep (heat: vs the swept variable);
# colored: frequency vs sweep with every point coloured by an expression; field: 3D view of the solution
# (displacement / temperature); time: heat, any expression vs time (transient problems); line: heat, an
# expression evaluated point by point along a straight line (a cross-section profile)
PLOT_KINDS = ("dispersion", "quantity", "colored", "field", "time", "line", "section")
# the type of each kind, shown with plot names: 3D views of the structure, 2D maps (cross-sections), XY plots, and
# XY plots whose points are coloured by a third value
PLOT_TYPES = {"field": "3D", "section": "2D", "dispersion": "XY", "quantity": "XY", "time": "XY", "line": "XY",
              "colored": "XYz"}


def plot_label(kind: str, text: str) -> str:
    """A plot (or plot kind) with its type, e.g. "3D: Temperature field", "XY: Dispersion"."""
    return f"{PLOT_TYPES.get(kind, '?')}: {text}"


LINE_AXES = ("s", "x", "y", "z")   # x axis of a line plot: distance from the start, or a coordinate
SECTION_PLANES = ("xy", "yz", "xz")  # coordinate plane a cross-section starts from (before its rotation)


FIELD_COMPONENTS = ("|u|", "ux", "uy", "uz", "mises")   # solid mechanics; mises: von Mises stress (stationary)


@dataclass
class ResultPlot:
    """A results plot: a 1D plot of one value per mode / time step against a sweep, or a 3D field view."""
    name: str = "Dispersion"
    quantity: str = "freq"      # y values (expression, see fields.HELP); always freq for dispersion/colored
    kind: str = ""              # empty = inferred: dispersion for freq, quantity otherwise
    color_by: str = ""          # colored: colour expression
    sweep: str = "auto"         # x axis: kpath | variable | auto (k-path if there is one)
    at: int = -1                # index into the other sweep; -1 = all values overlaid
    y_label: str = ""           # axis label; empty = derived from the quantity
    x_label: str = ""           # x axis label; empty = derived from the sweep
    color_label: str = ""       # colored: colour-bar label; empty = the expression
    style: PlotStyle = field(default_factory=PlotStyle)
    component: str = "|u|"      # field, solid mechanics: |u|, ux, uy, uz or mises
    # line plots: the segment start -> end (expressions), number of points, x axis, time step (-1 = last)
    start: list[str] = field(default_factory=lambda: ["0", "0", "0"])
    end: list[str] = field(default_factory=lambda: ["1[um]", "0", "0"])
    points: int = 200
    line_axis: str = "s"
    at_time: int = -1
    # cross-sections: the plane through `origin` (expressions), a coordinate plane rotated about x, then y,
    # then z (degrees), sampled on resolution x resolution points over the model
    plane: str = "xy"
    origin: list[str] = field(default_factory=lambda: ["0", "0", "0"])
    rotation: list[str] = field(default_factory=lambda: ["0", "0", "0"])
    resolution: int = 150

    def __post_init__(self):
        if not self.kind:
            self.kind = "dispersion" if self.quantity.strip() == "freq" else "quantity"
        if self.kind in ("dispersion", "colored"):
            self.quantity = "freq"
        elif self.kind == "field" and self.quantity.strip() == "freq":
            self.quantity = ""       # the field of the physics' default quantity

    @classmethod
    def field_view(cls, name: str = "Field", colormap: str = "viridis") -> "ResultPlot":
        """A 3D field view (each physics chooses its name and colour scale, see physics/*/spec)."""
        return cls(name=name, kind="field", quantity="", style=PlotStyle(colormap=colormap, title=True))

    def expressions(self) -> list[str]:
        """Quantities this plot needs evaluated (one number per solution; field, line and section plots
        evaluate their own, point by point)."""
        if self.kind in ("field", "line", "section"):
            return []
        return [self.quantity] + ([self.color_by] if self.kind == "colored" and self.color_by.strip() else [])

    def to_dict(self):
        d = {"name": self.name, "kind": self.kind}
        if self.kind == "field":
            if self.quantity.strip():
                d["quantity"] = self.quantity
            if self.component != "|u|":
                d["component"] = self.component
            d["style"] = self.style.to_dict("field")
            return d
        if self.kind == "section":
            d.update(quantity=self.quantity, plane=self.plane, origin=list(self.origin), rotation=list(self.rotation),
                     resolution=self.resolution, at=self.at, at_time=self.at_time, style=self.style.to_dict())
            return self._compact(d)
        if self.kind == "line":
            d.update(quantity=self.quantity, start=list(self.start), end=list(self.end), points=self.points,
                     line_axis=self.line_axis, at=self.at, at_time=self.at_time, y_label=self.y_label,
                     x_label=self.x_label, style=self.style.to_dict())
            return self._compact(d)
        if self.kind in ("quantity", "time"):
            d["quantity"] = self.quantity
        if self.kind == "colored":
            d["color_by"] = self.color_by
            d["color_label"] = self.color_label
        d.update({"sweep": self.sweep, "at": self.at, "y_label": self.y_label, "x_label": self.x_label})
        d["style"] = self.style.to_dict(self.kind)
        return self._compact(d)

    def _compact(self, d: dict) -> dict:
        d = without_defaults(self, d, keep=("name", "kind", "quantity", "start", "end", "plane"))
        if not d.get("style"):
            d.pop("style", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ResultPlot":
        d = dict(d)
        style = PlotStyle.from_dict(d.pop("style", {}))
        return cls(**d, style=style)  # older files without "kind": inferred from the quantity
