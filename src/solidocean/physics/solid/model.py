"""Solid mechanics settings ([solid] in the model file): symmetry planes, boundary conditions on selections
and the problem (eigenfrequency or stationary)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...selection import FACES, check_selection, selection_to_dict
from ...tomlio import without_defaults

Expr = str | float | int

SOLID_CONDITION_TYPES = ("fixed", "displacement", "load")
SOLID_CONDITION_TITLES = {"fixed": "Fixed constraint", "displacement": "Prescribed displacement",
                          "load": "Boundary load"}
LOAD_TYPES = ("area", "total", "pressure")   # force per area (Pa), total force (N), pressure (Pa, along -n)


@dataclass
class SolidCondition:
    """Boundary condition on a selection of boundary faces (bounding-box faces or all faces of the model or
    of one object, optionally only inside a box): fixed constraint, prescribed displacement (per component;
    an empty component is free) or boundary load (stationary problems only)."""
    type: str = "fixed"                # fixed | displacement | load
    faces: list[str] = field(default_factory=list)
    name: str = ""
    on: str = ""                       # "" = the whole model, else a feature or material name
    box: dict = field(default_factory=dict)
    displacement: list[Expr] = field(default_factory=lambda: ["0", "0", "0"])   # "" = free component
    load_type: str = "area"            # area | total | pressure
    load: list[Expr] = field(default_factory=lambda: ["0", "0", "0"])           # Pa (area) or N (total)
    pressure: Expr = "0[Pa]"           # pressure: acts against the outward normal (pushes on the surface)

    def to_dict(self):
        d = selection_to_dict(self, {"type": self.type, "name": self.name} if self.name else {"type": self.type})
        if self.type == "displacement":
            d["displacement"] = list(self.displacement)
        elif self.type == "load":
            d["load_type"] = self.load_type
            if self.load_type == "pressure":
                d["pressure"] = self.pressure
            else:
                d["load"] = list(self.load)
        return d

    def free_components(self) -> tuple[bool, bool, bool]:
        """Components a constraint leaves free (prescribed displacement with an empty component)."""
        if self.type == "displacement":
            return tuple(not str(v).strip() for v in self.displacement)
        return (False, False, False) if self.type == "fixed" else (True, True, True)


@dataclass
class KPath:
    # named points in reduced coordinates (fractions of reciprocal lattice vectors)
    points: dict[str, list[float]] = field(default_factory=lambda: {"G": [0, 0, 0], "X": [0.5, 0, 0]})
    path: list[str] = field(default_factory=lambda: ["G", "X"])
    n: int = 20  # points per segment
    enabled: bool = True  # disabled sweeps are kept in the file but not run

    def to_dict(self):
        d = {"points": dict(self.points), "path": list(self.path), "n": self.n}
        if not self.enabled:
            d["enabled"] = False
        return d


@dataclass
class SolidProblem:
    """Problem of solid mechanics: eigenfrequencies (modes, phonon dispersion along a k-path), the stationary
    response to loads, or the harmonic (frequency-domain) response to loads and displacements oscillating at a
    frequency ([solid.problem])."""
    type: str = "eigenfrequency"     # eigenfrequency | stationary | frequency
    num_eigs: int = 12               # eigenfrequencies per solve
    search_freq: Expr = "0[GHz]"     # find the eigenfrequencies nearest this frequency
    k: list[Expr] = field(default_factory=lambda: ["0", "0", "0"])   # Bloch wave vector (rad/m) without k-path
    kpath: KPath | None = None       # wavevector sweep (inner loop of a variable sweep)
    tol: float = 0.0                 # relative eigenvalue tolerance (0 = machine precision)
    maxiter: int = 0                 # Lanczos iterations (0 = ARPACK default)
    frequency: Expr = "f"            # frequency domain: the drive frequency (sweep it with a variable)
    loss: Expr = "0"                 # frequency domain: loss factor of every material (added to its own `eta`)

    def __post_init__(self):
        if isinstance(self.kpath, dict):
            self.kpath = KPath(**self.kpath)

    @property
    def stationary(self) -> bool:
        return self.type == "stationary"

    @property
    def harmonic(self) -> bool:
        return self.type == "frequency"

    @property
    def field_problem(self) -> bool:
        """A problem with displacement fields as results (stationary or frequency domain), not modes."""
        return self.type in ("stationary", "frequency")

    @property
    def active_kpath(self) -> KPath | None:
        kp = self.kpath
        return kp if kp and kp.enabled and not self.field_problem else None

    def to_dict(self):
        if self.harmonic:
            return without_defaults(self, {"type": self.type, "frequency": self.frequency, "loss": self.loss},
                                    keep=("type", "frequency"))
        d = without_defaults(self, {"type": self.type, "num_eigs": self.num_eigs, "search_freq": self.search_freq,
                                    "k": list(self.k), "tol": self.tol, "maxiter": self.maxiter},
                             keep=("type",))
        if self.kpath:
            d["kpath"] = self.kpath.to_dict()
        return d


@dataclass
class SolidPhysics:
    """Settings of solid mechanics ([solid]): symmetry planes, boundary conditions on selections, problem. The
    periodicity is the model's lattice (project.lattice)."""
    # bounding-box faces with zero normal displacement (symmetry plane)
    symmetry: list[str] = field(default_factory=list)
    # bounding-box faces with zero tangential displacement
    antisymmetry: list[str] = field(default_factory=list)
    # fixed constraints, prescribed displacements and loads on selections of faces
    conditions: list[SolidCondition] = field(default_factory=list)
    problem: SolidProblem = field(default_factory=SolidProblem)

    def __post_init__(self):
        self.conditions = [c if isinstance(c, SolidCondition) else SolidCondition(**c) for c in self.conditions]
        if isinstance(self.problem, dict):
            self.problem = SolidProblem(**self.problem)

    @classmethod
    def from_dict(cls, d: dict) -> "SolidPhysics":
        return cls(**d)

    def to_dict(self):
        d = {key: list(getattr(self, key)) for key in ("symmetry", "antisymmetry") if getattr(self, key)}
        if self.conditions:
            d["conditions"] = [c.to_dict() for c in self.conditions]
        d["problem"] = self.problem.to_dict()
        return d

    def face_conditions(self, lattice) -> dict[str, str]:
        """Face -> condition name for the bounding-box faces with periodicity (axis shorthand of the lattice) or
        (anti)symmetry. Conditions on selections are listed separately (`conditions`)."""
        out = {}
        for ax in ([] if lattice.vectors else lattice.periodic):
            out[f"{ax}min"] = out[f"{ax}max"] = f"periodic {ax}"
        for kind in ("antisymmetry", "symmetry"):
            for face in getattr(self, kind):
                out[face] = kind
        return out


PROBLEM_TYPES = ("eigenfrequency", "stationary", "frequency")
PROBLEM_TITLES = {"eigenfrequency": "Eigenfrequency", "stationary": "Stationary", "frequency": "Frequency domain"}


def validate_settings(project, ev, material_names) -> list[str]:
    """Problems with the solid-mechanics settings: symmetry faces, conditions, problem type."""
    errors, seen = [], {}
    ph = project.solid
    periodic = project.lattice.periodic_axes()
    for kind in ("symmetry", "antisymmetry"):
        for face in getattr(ph, kind):
            if face not in FACES:
                errors.append(f"solid.{kind}: bad face '{face}' (use {', '.join(FACES)})")
                continue
            if face[0] in periodic:
                errors.append(f"solid.{kind}: face {face} is periodic (along {face[0]}) and cannot also be {kind}")
            if face in seen:
                errors.append(f"solid: face {face} is both {seen[face]} and {kind}")
            seen[face] = kind
    objects = {f.name for f in project.objects} | set(material_names)
    for c in ph.conditions:
        where = f"condition '{c.name or SOLID_CONDITION_TITLES.get(c.type, c.type)}'"
        if c.type not in SOLID_CONDITION_TYPES:
            errors.append(f"{where}: type must be one of {', '.join(SOLID_CONDITION_TYPES)}")
            continue
        errors += check_selection(where, c, ev, objects)
        if not c.on:
            for face in c.faces:
                if face in FACES and face[0] in periodic:
                    errors.append(f"{where}: face {face} is periodic (along {face[0]})")
        if c.type == "displacement":
            if len(c.displacement) != 3:
                errors.append(f"{where}: displacement needs 3 components (empty = free)")
            elif all(c.free_components()):
                errors.append(f"{where}: no displacement component given (empty components are free)")
            for v in c.displacement:
                if str(v).strip():
                    try:
                        ev(v)
                    except Exception as exc:
                        errors.append(f"{where}: displacement: {exc}")
        elif c.type == "load":
            if c.load_type not in LOAD_TYPES:
                errors.append(f"{where}: load_type must be one of {', '.join(LOAD_TYPES)}")
            try:
                if c.load_type == "pressure":
                    ev(c.pressure)
                elif len(ev.vector(c.load)) != 3:
                    errors.append(f"{where}: load needs 3 components")
            except Exception as exc:
                errors.append(f"{where}: load: {exc}")
    st = ph.problem
    if st.type not in PROBLEM_TYPES:
        errors.append(f"solid.problem.type must be one of {', '.join(PROBLEM_TYPES)}")
    if st.harmonic:
        try:
            if ev(st.frequency) <= 0:
                errors.append("solid.problem.frequency must be positive")
            if ev(st.loss) < 0:
                errors.append("solid.problem.loss must not be negative")
        except Exception as exc:
            errors.append(f"solid.problem: {exc}")
    from ... import fields, materials as mat
    for m in project.materials:
        eta = mat.loss_factor(m.props) if m.name in material_names else ""
        if eta:
            try:
                fields.check(eta, ev.values, fields=(), scalars=())
            except Exception as exc:
                errors.append(f"material '{m.name}': eta: {exc}")
    if st.num_eigs < 1:
        errors.append("solid.problem.num_eigs must be at least 1")
    if st.tol < 0 or st.maxiter < 0:
        errors.append("solid.problem: tol and maxiter must be >= 0")
    try:
        ev(st.search_freq)
        if len(ev.vector(st.k)) != 3:
            errors.append("solid.problem.k needs 3 components")
    except Exception as exc:
        errors.append(f"solid.problem: {exc}")
    kp = st.active_kpath
    if kp:
        if not project.lattice.is_periodic:
            errors.append("solid.problem.kpath needs a periodic lattice ([lattice])")
        for p in kp.path:
            if p not in kp.points:
                errors.append(f"solid.problem.kpath: point '{p}' not defined")
    return errors
