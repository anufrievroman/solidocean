"""Project model: the complete, serialisable description of a simulation.

A project is stored as a human-editable TOML file. Both the CLI and the GUI operate on this same structure;
the GUI is only an editor/viewer for it. This module holds what every physics shares (variables,
materials, geometry, mesh, sweeps, solver, output, plots); the settings of each physics are defined by the
physics itself (physics/<name>/model.py) and stored under its section.
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import physics, tomlio
from .expressions import Evaluator
from .tomlio import without_defaults
from .lattice import Lattice
from .physics.heat.model import HeatPhysics
from .physics.solid.model import SolidPhysics
from .plots import (COLOR_MODES, FIELD_COMPONENTS, K_UNITS, LINE_AXES, LINE_STYLES, SECTION_PLANES, SWEEP_AXES,
                    X_AXES, ResultPlot)

Expr = str | float | int
# Geometry primitives: parameter name -> default value.
# Vectors are 3-element lists of expressions.
FEATURE_TYPES: dict[str, dict[str, Any]] = {
    # unit cell of the lattice ([lattice] a1, a2): parallelogram prism along z
    "cell": {"center": ["0", "0", "0"], "height": "1"},
    "block": {"center": ["0", "0", "0"], "size": ["1", "1", "1"]},
    "cylinder": {"center": ["0", "0", "0"], "axis": ["0", "0", "1"], "radius": "0.5", "height": "1"},
    "cone": {"center": ["0", "0", "0"], "axis": ["0", "0", "1"], "radius1": "0.5", "radius2": "0.25", "height": "1"},
    "sphere": {"center": ["0", "0", "0"], "radius": "0.5"},
    "prism": {"points": [["0", "0"], ["1", "0"], ["0", "1"]], "z0": "0", "height": "1"},
    # isosceles triangle in the x-y plane (base along x, apex towards +y), extruded along z; centred on its box
    "triangle": {"center": ["0", "0", "0"], "base": "1", "height": "1", "thickness": "1"},
}
FEATURE_TITLES = {"cell": "Lattice cell", "block": "Block", "cylinder": "Cylinder", "cone": "Cone",
                  "sphere": "Sphere", "prism": "Prism", "triangle": "Triangular block"}
# every feature except the lattice cell (which must match the lattice) can be rotated: angles in degrees about
# the global x, then y, then z axis, around the feature's reference point (center; base center for cylinders/cones)
ROTATION_KEY = "rotation"
NO_ROTATION = ["0", "0", "0"]
# every feature except the lattice cell can be repeated as an array: `array` copies along x, y, z (counting the
# feature itself), `spacing` apart (the shift between neighbouring copies, may be negative). The copies form one
# object (one name, material and selection); they are merged where they touch or overlap
ARRAY_KEY, SPACING_KEY = "array", "spacing"
NO_ARRAY, NO_SPACING = ["1", "1", "1"], ["0", "0", "0"]
MAX_ARRAY = 10000   # copies of one feature


def array_counts(values, name: str) -> list[int]:
    """Copies along x, y, z from evaluated `array` values: whole numbers >= 1 (raises ValueError otherwise)."""
    counts = [float(n) for n in values]
    if len(counts) != 3 or any(n < 1 or abs(n - round(n)) > 1e-9 for n in counts):
        raise ValueError(f"geometry '{name}': array needs 3 whole numbers >= 1 (copies along x, y, z)")
    counts = [int(round(n)) for n in counts]
    if counts[0] * counts[1] * counts[2] > MAX_ARRAY:
        raise ValueError(f"geometry '{name}': array of more than {MAX_ARRAY} copies")
    return counts
OPS = ("add", "subtract")
AXES = ("x", "y", "z")


@dataclass
class Material:
    name: str
    props: dict[str, Expr] = field(default_factory=dict)  # may include 'model', 'library'
    color: str = ""   # display colour (#rrggbb); "" = automatic, by the order of the materials

    def to_dict(self) -> dict:
        return {"name": self.name, **({"color": self.color} if self.color else {}), **self.props}

    @classmethod
    def from_dict(cls, d: dict) -> "Material":
        d = dict(d)
        return cls(name=d.pop("name"), color=d.pop("color", ""), props=d)


@dataclass
class Feature:
    type: str
    name: str
    op: str = "add"
    material: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True  # disabled ("bypassed") features are kept but ignored

    def __post_init__(self):
        if self.type not in FEATURE_TYPES:
            raise ValueError(f"unknown geometry type '{self.type}'")
        for k, v in FEATURE_TYPES[self.type].items():
            self.params.setdefault(k, copy.deepcopy(v))
        if self.type != "cell":
            self.params.setdefault(ROTATION_KEY, list(NO_ROTATION))
            self.params.setdefault(ARRAY_KEY, list(NO_ARRAY))
            self.params.setdefault(SPACING_KEY, list(NO_SPACING))

    @property
    def rotated(self) -> bool:
        return any(str(a).strip() not in ("0", "0.0", "") for a in self.params.get(ROTATION_KEY, NO_ROTATION))

    @property
    def arrayed(self) -> bool:
        return any(str(n).strip() not in ("1", "1.0", "") for n in self.params.get(ARRAY_KEY, NO_ARRAY))

    @property
    def array_label(self) -> str:
        """"3×3×1" for an array, "" otherwise."""
        return "×".join(str(n).strip() for n in self.params[ARRAY_KEY]) if self.arrayed else ""

    def to_dict(self) -> dict:
        d = {"type": self.type, "name": self.name, "op": self.op}
        if self.op == "add":
            d["material"] = self.material
        if not self.enabled:
            d["enabled"] = False
        d.update(self.params)
        if not self.rotated:
            d.pop(ROTATION_KEY, None)  # files list a rotation only when there is one
        if not self.arrayed:
            d.pop(ARRAY_KEY, None)    # ... and an array only when there is one
            d.pop(SPACING_KEY, None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Feature":
        d = dict(d)
        return cls(type=d.pop("type"), name=d.pop("name", d.get("type", "")),
                   op=d.pop("op", "add"), material=d.pop("material", ""),
                   enabled=d.pop("enabled", True), params=d)


@dataclass
class MeshRegion:
    """A different maximum element size for one object (a geometry feature, or all domains of a material).
    Regions apply in order (a later one wins where they overlap); the rest of the model keeps the mesh's
    max_size. The mesh stays conforming across interfaces: their elements are graded."""
    on: str = ""
    max_size: Expr = "1"
    name: str = ""

    def to_dict(self):
        d = {"name": self.name} if self.name else {}
        d.update(on=self.on, max_size=self.max_size)
        return d


@dataclass
class MeshSettings:
    max_size: Expr = "1"               # everywhere but in the regions
    min_size: Expr = "0"
    order: int = 2
    optimize: bool = True
    regions: list[MeshRegion] = field(default_factory=list)

    def __post_init__(self):
        self.regions = [r if isinstance(r, MeshRegion) else MeshRegion(**r) for r in self.regions]

    def to_dict(self):
        d = without_defaults(self, {"max_size": self.max_size, "min_size": self.min_size, "order": self.order,
                                    "optimize": self.optimize}, keep=("max_size",))
        if self.regions:
            d["regions"] = [r.to_dict() for r in self.regions]
        return d

    def key(self, ev) -> tuple:
        """What determines the mesh (for caching): sizes, order, regions."""
        return (ev(self.max_size), ev(self.min_size), self.order, self.optimize,
                tuple((r.on, ev(r.max_size)) for r in self.regions))


@dataclass
class VariableSweep:
    variable: str = ""
    values: str = ""
    enabled: bool = True

    def to_dict(self):
        d = {"variable": self.variable, "values": self.values}
        if not self.enabled:
            d["enabled"] = False
        return d


SOLVER_BACKENDS = ("auto", "mumps", "superlu")


@dataclass
class Study:
    """What every physics' study shares ([study]): the variable sweep (outer loop) and the sparse direct
    solver. The study type and its own settings belong to the physics ([solid.study], [heat.study])."""
    sweep: VariableSweep | None = None
    solver: str = "auto"            # sparse direct solver: auto, mumps or superlu

    def __post_init__(self):
        if isinstance(self.sweep, dict):
            self.sweep = VariableSweep(**self.sweep)

    @property
    def active_sweep(self) -> VariableSweep | None:
        sw = self.sweep
        return sw if sw and sw.enabled and sw.variable else None

    def to_dict(self):
        d: dict[str, Any] = {"solver": self.solver} if self.solver != "auto" else {}
        if self.sweep:
            d["sweep"] = self.sweep.to_dict()
        return d


@dataclass
class Output:
    dir: str = ""  # default: <input stem>_results next to the input file
    plot: bool = True  # write a PNG per results plot

    def to_dict(self):
        return without_defaults(self, {"dir": self.dir, "plot": self.plot})

    @classmethod
    def from_dict(cls, d: dict) -> "Output":
        return cls(**d)


@dataclass
class Project:
    name: str = "Untitled"
    description: str = ""
    # preferred units, for display only (GUI, plots, result tables); all values stay SI
    length_unit: str = "um"
    frequency_unit: str = "GHz"
    time_unit: str = "s"
    physics_kind: str = "solid"  # the physics solved, a key of physics.REGISTRY
    variables: dict[str, Expr] = field(default_factory=dict)
    materials: list[Material] = field(default_factory=list)
    geometry: list[Feature] = field(default_factory=list)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    lattice: Lattice = field(default_factory=Lattice)
    study: Study = field(default_factory=Study)
    output: Output = field(default_factory=Output)
    plots: list[ResultPlot] | None = None     # None: the default plots of the physics
    # settings of each physics: attribute = PhysicsSpec.section = TOML table = the physics' kind. Only the
    # physics solved is saved.
    solid: SolidPhysics = field(default_factory=SolidPhysics)
    heat: HeatPhysics = field(default_factory=HeatPhysics)
    path: Path | None = None  # file this project was loaded from / saved to

    SECTIONS = ("project", "variables", "materials", "geometry", "lattice", "mesh", "study", "output", "plots")

    def __post_init__(self):
        if self.plots is None:
            self.plots = physics.of(self).default_plots(self) if self.physics_kind in physics.REGISTRY else []

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        d = {
            "project": {"name": self.name, "description": self.description, "length_unit": self.length_unit,
                        "frequency_unit": self.frequency_unit, "time_unit": self.time_unit,
                        "physics": self.physics_kind},
            "variables": dict(self.variables),
            "materials": [m.to_dict() for m in self.materials],
            "geometry": [f.to_dict() for f in self.geometry],
        }
        if self.lattice.is_periodic:
            d["lattice"] = self.lattice.to_dict()
        d["mesh"] = self.mesh.to_dict()
        if self.study.to_dict():
            d["study"] = self.study.to_dict()
        if self.physics_kind in physics.REGISTRY:
            spec = physics.of(self)
            d[spec.section] = spec.settings(self).to_dict()
        if self.output.to_dict():
            d["output"] = self.output.to_dict()
        d["plots"] = [pl.to_dict() for pl in self.plots]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        sections = {spec.section: spec for spec in physics.REGISTRY.values()}
        unknown = set(d) - set(cls.SECTIONS) - set(sections)
        if unknown:
            raise ValueError(f"unknown section(s): {', '.join(sorted(unknown))}")
        info = d.get("project", {})
        physics_kind = info.get("physics", "solid")
        spec = physics.get(physics_kind)
        plots = []
        for pl in d.get("plots", []):
            if pl.get("kind") == "field":  # field defaults of the physics (colour scale, title on)
                base = spec.field_plot().style.to_dict("field")
                pl = {**pl, "style": {**base, **pl.get("style", {})}}
            plots.append(ResultPlot.from_dict(pl))
        return cls(
            name=info.get("name", "Untitled"),
            description=info.get("description", ""),
            length_unit=info.get("length_unit", "um"),
            frequency_unit=info.get("frequency_unit", "GHz"),
            time_unit=info.get("time_unit", "s"),
            physics_kind=physics_kind,
            variables=dict(d.get("variables", {})),
            materials=[Material.from_dict(m) for m in d.get("materials", [])],
            geometry=[Feature.from_dict(f) for f in d.get("geometry", [])],
            lattice=Lattice.from_dict(d.get("lattice", {})),
            mesh=MeshSettings(**d.get("mesh", {})),
            study=Study(**d.get("study", {})),
            output=Output.from_dict(d.get("output", {})),
            plots=plots or None,       # none in the file: the physics' defaults
            **{name: s.settings_from_dict(d.get(name, {})) for name, s in sections.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        with open(path, "rb") as fh:
            proj = cls.from_dict(tomllib.load(fh))
        proj.path = path
        return proj

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.path)
        text = "# SolidOcean model file — see docs/input-format.md\n\n" + tomlio.dumps(self.to_dict())
        path.write_text(text)
        self.path = path
        return path

    # ---- helpers -------------------------------------------------------
    @property
    def display_units(self):
        """The preferred units (units.DisplayUnits) for displaying values."""
        from .units import DisplayUnits
        return DisplayUnits(self.length_unit, self.frequency_unit, self.time_unit)

    def evaluator(self, overrides: dict[str, float] | None = None) -> Evaluator:
        return Evaluator(self.variables, overrides)

    def material(self, name: str) -> Material:
        for m in self.materials:
            if m.name == name:
                return m
        raise KeyError(f"material '{name}' not defined")

    def material_colors(self) -> dict[str, str]:
        """Display colour of every material: its own, or the next default colour in the order of the list."""
        from .materials import DEFAULT_COLORS
        return {m.name: m.color or DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, m in enumerate(self.materials)}

    def results_dir(self) -> Path:
        base = self.path.parent if self.path else Path.cwd()
        if self.output.dir:
            return base / self.output.dir
        stem = self.path.stem if self.path else "solidocean"
        return base / f"{stem}_results"

    def validate_geometry(self) -> list[str]:
        """Problems that prevent building the geometry or the mesh (variables, features, lattice, mesh).
        Physics, boundary conditions, studies and plots are not needed for that."""
        errors: list[str] = []
        try:
            ev = self.evaluator()
        except Exception as exc:
            return [f"variables: {exc}"]
        names = [m.name for m in self.materials]
        if len(set(names)) != len(names):
            errors.append("materials: duplicate names")
        from .materials import is_color
        for m in self.materials:
            if m.color and not is_color(m.color):
                errors.append(f"material '{m.name}': color must be #rrggbb (or empty for automatic)")
        active = [f for f in self.geometry if f.enabled]
        if not any(f.op == "add" for f in active):
            errors.append("geometry: no enabled solid ('add') features")
        for f in active:
            if f.op not in OPS:
                errors.append(f"geometry '{f.name}': op must be one of {OPS}")
            if f.op == "add" and f.material not in names:
                errors.append(f"geometry '{f.name}': material '{f.material}' not defined")
            if f.arrayed:
                try:
                    array_counts(ev.vector(f.params[ARRAY_KEY]), f.name)
                    ev.vector(f.params.get(SPACING_KEY, NO_SPACING))
                except Exception as exc:
                    errors.append(str(exc) if str(exc).startswith("geometry") else f"geometry '{f.name}': array: {exc}")
        lt = self.lattice
        for ax in lt.periodic:
            if ax not in AXES:
                errors.append(f"lattice.periodic: bad axis '{ax}'")
        if lt.vectors:
            from . import lattice as lat
            if lt.periodic:
                errors.append("lattice: give either `periodic` or `vectors`, not both")
            if not 1 <= len(lt.vectors) <= 3 or any(len(v) != 3 for v in lt.vectors):
                errors.append("lattice.vectors: 1 to 3 vectors of 3 components")
            else:
                try:
                    msg = lat.check(lat.slots(lt, ev))
                    if msg:
                        errors.append(f"lattice.vectors: {msg}")
                except Exception as exc:
                    errors.append(f"lattice.vectors: {exc}")
        if any(f.type == "cell" and f.enabled for f in self.geometry):
            from . import lattice as lat
            try:
                inplane = [v for v in lat.slots(self.lattice, ev) if v is not None and abs(v[2]) == 0]
            except Exception:
                inplane = []
            if len(inplane) < 2:
                errors.append("geometry: a 'Lattice cell' needs a lattice periodic in two in-plane directions "
                              "(e.g. periodic in x and y, or a hexagonal lattice)")
        if self.mesh.order not in (1, 2):
            errors.append("mesh.order must be 1 or 2")
        try:
            ev(self.mesh.max_size), ev(self.mesh.min_size)
        except Exception as exc:
            errors.append(f"mesh size: {exc}")
        objects = {f.name for f in self.geometry} | {m.name for m in self.materials}
        for r in self.mesh.regions:
            where = f"mesh region '{r.name or r.on}'"
            if r.on not in objects:
                errors.append(f"{where}: unknown object '{r.on}' (a geometry feature or material name)")
            try:
                if ev(r.max_size) <= 0:
                    errors.append(f"{where}: max_size must be positive")
            except Exception as exc:
                errors.append(f"{where}: max_size: {exc}")
        return errors

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty if OK): everything needed to run the study."""
        errors = self.validate_geometry()
        from .units import FREQUENCY_UNITS, LENGTH_UNITS, TIME_UNITS
        for key, allowed in (("length_unit", LENGTH_UNITS), ("frequency_unit", FREQUENCY_UNITS),
                             ("time_unit", TIME_UNITS)):
            if getattr(self, key) not in allowed:
                errors.append(f"project.{key} must be one of {', '.join(allowed)}")
        try:
            ev = self.evaluator()
        except Exception:
            return errors
        if self.physics_kind not in physics.REGISTRY:
            return errors + [f"project.physics must be one of {', '.join(physics.REGISTRY)}"]
        spec = physics.of(self)
        errors += spec.validate(self, ev)
        sw = self.study.active_sweep
        if sw:
            if sw.variable not in self.variables:
                errors.append(f"study.sweep: unknown variable '{sw.variable}'")
            try:
                ev.array(sw.values)
            except Exception as exc:
                errors.append(f"study.sweep.values: {exc}")
        from . import fields
        names = [pl.name for pl in self.plots]
        if len(set(names)) != len(names):
            errors.append("plots: duplicate names")
        for pl in self.plots:
            where = f"plot '{pl.name}'"
            if pl.kind not in spec.plot_kinds:
                errors.append(f"{where}: kind must be one of {', '.join(spec.plot_kinds)} for {spec.title.lower()}")
                continue
            if pl.kind == "line":
                try:
                    a, b = ev.vector(pl.start), ev.vector(pl.end)
                    if len(a) != 3 or len(b) != 3:
                        errors.append(f"{where}: start and end need 3 coordinates")
                    elif all(abs(x - y) <= 1e-30 for x, y in zip(a, b)):
                        errors.append(f"{where}: start and end are the same point")
                except Exception as exc:
                    errors.append(f"{where}: line: {exc}")
                if pl.points < 2:
                    errors.append(f"{where}: points must be at least 2")
                if pl.line_axis not in LINE_AXES:
                    errors.append(f"{where}: line_axis must be one of {', '.join(LINE_AXES)}")
            if pl.kind == "section":
                if pl.plane not in SECTION_PLANES:
                    errors.append(f"{where}: plane must be one of {', '.join(SECTION_PLANES)}")
                try:
                    if len(ev.vector(pl.origin)) != 3 or len(ev.vector(pl.rotation)) != 3:
                        errors.append(f"{where}: origin and rotation need 3 values")
                except Exception as exc:
                    errors.append(f"{where}: origin / rotation: {exc}")
                if not 10 <= pl.resolution <= 1000:
                    errors.append(f"{where}: resolution must be between 10 and 1000")
            if pl.kind in ("field", "section") and pl.quantity.strip():
                try:
                    fields.check(pl.quantity, ev.values, spec.field_names, spec.scalar_names)
                except Exception as exc:
                    errors.append(f"{where}: quantity: {exc}")
            if pl.kind == "section":
                continue
            if pl.kind == "field":
                if pl.component not in FIELD_COMPONENTS:
                    errors.append(f"{where}: component must be one of {', '.join(FIELD_COMPONENTS)}")
                continue
            checks = [("quantity", pl.quantity)] + ([("color_by", pl.color_by)] if pl.kind == "colored" else [])
            for label, expr in checks:
                if not expr.strip():
                    errors.append(f"{where}: {label} is empty")
                    continue
                try:
                    fields.check(expr, ev.values, spec.field_names, spec.scalar_names)
                except Exception as exc:
                    errors.append(f"{where}: {label}: {exc}")
            if pl.sweep not in SWEEP_AXES:
                errors.append(f"{where}: sweep must be one of {', '.join(SWEEP_AXES)}")
            ps = pl.style
            if ps.line_style not in LINE_STYLES:
                errors.append(f"{where}: line_style must be one of {', '.join(LINE_STYLES)}")
            if ps.colors not in COLOR_MODES:
                errors.append(f"{where}: colors must be one of {', '.join(COLOR_MODES)}")
            if ps.x_axis not in X_AXES:
                errors.append(f"{where}: x_axis must be one of {', '.join(X_AXES)}")
            if ps.k_unit not in K_UNITS:
                errors.append(f"{where}: k_unit must be one of {', '.join(K_UNITS)}")
            for key in ("y_min", "y_max", "x_min", "x_max", "color_min", "color_max"):
                if str(getattr(ps, key)).strip():
                    try:
                        ev(getattr(ps, key))
                    except Exception as exc:
                        errors.append(f"{where}: {key}: {exc}")
        if self.study.solver not in SOLVER_BACKENDS:
            errors.append(f"study.solver must be one of {', '.join(SOLVER_BACKENDS)}")
        return errors
