"""Heat transfer settings ([heat] in the model file): initial values, boundary conditions on selections,
volume sources with optional pulses, thermal contact at interfaces, and the problem (stationary or transient)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...selection import FACES, check_selection, selection_to_dict
from ...tomlio import check_keys, label, table, without_defaults

Expr = str | float | int

HEAT_CONDITION_TYPES = ("temperature", "flux", "convection", "radiation")
ADDITIVE = ("flux", "radiation")       # may share a face with another condition: their heat flows add
HEAT_PROBLEM_TYPES = ("stationary", "transient", "frequency")
INTERFACE_TYPES = ("resistance", "layer")
PULSE_SHAPES = ("", "rect", "gauss", "sine")   # "" = constant in time
PROFILE_SHAPES = ("", "gauss", "expr")         # "" = uniform in space


@dataclass
class Pulse:
    """Time profile of a heat flux or source. Pulses multiply the condition's value (their peak): rectangular
    (on from `start` for `duration`) or Gaussian (peak at `start`, full width at half maximum `duration`),
    repeated every `period` if given. A sine oscillates about the value (its mean):
    value + amplitude sin(2 pi (t - start) / period)."""
    shape: str = ""                    # "" (constant) | rect | gauss | sine
    start: Expr = "0[s]"
    duration: Expr = "10[ns]"
    period: Expr = ""
    amplitude: Expr = ""               # sine: in the units of the value

    def to_dict(self):
        d = {"shape": self.shape, "start": self.start, "period": self.period}
        if self.shape == "sine":
            d["amplitude"] = self.amplitude
        else:
            d["duration"] = self.duration
        return without_defaults(self, d, keep=("shape", "amplitude"))

    @classmethod
    def from_dict(cls, d) -> "Pulse":
        return table(cls, d, "pulse") if isinstance(d, dict) else cls()


@dataclass
class Profile:
    """Spatial profile of a heat flux or source: a dimensionless factor multiplying the value, which is its
    peak. 'gauss': exp(-4 ln2 r^2 / fwhm^2) with r the distance from `center` in the x-y plane (a beam along z;
    a beam of power P has the peak flux 4 ln2 P / (pi fwhm^2)); 'expr': any expression of x, y, z (m), e.g.
    exp((z - z_top) / delta) for absorption over a depth delta."""
    shape: str = ""                    # "" (uniform) | gauss | expr
    center: list = field(default_factory=lambda: ["0", "0"])
    fwhm: Expr = "3[um]"
    expr: Expr = "1"

    def to_dict(self):
        if self.shape == "expr":
            return {"shape": self.shape, "expr": self.expr}
        return {"shape": self.shape, "center": list(self.center), "fwhm": self.fwhm}

    @classmethod
    def from_dict(cls, d) -> "Profile":
        return table(cls, d, "profile") if isinstance(d, dict) else cls()

    def factor(self, ev, points):
        """The factor at points (..., 3) (SI)."""
        from ... import fields
        pts = np.asarray(points, float)
        if self.shape == "gauss":
            x0, y0 = (ev(v) for v in self.center)
            r2 = (pts[..., 0] - x0) ** 2 + (pts[..., 1] - y0) ** 2
            return np.exp(-4 * np.log(2) * r2 / ev(self.fwhm) ** 2)
        if self.shape == "expr":
            flat = pts.reshape(-1, 3)
            return fields.evaluate_nodal(str(self.expr), flat, ev.values).reshape(pts.shape[:-1])
        return np.ones(pts.shape[:-1])


@dataclass
class HeatCondition:
    """Boundary condition on bounding-box faces of the model, or of one object (`on`: one of the objects
    or a material; e.g. the top face of a pad). Faces without a condition are insulated."""
    type: str = "temperature"          # temperature | flux | convection | radiation
    faces: list[str] = field(default_factory=list)
    value: Expr = "300[K]"             # fixed temperature (K) or inward heat flux (W/m^2); may use `time`
    h: Expr = "10[W/(m^2*K)]"          # convection: heat-transfer coefficient
    T_ext: Expr = "293.15[K]"          # convection, radiation: ambient temperature
    emissivity: Expr = "1"             # radiation: q = emissivity sigma (T_ext^4 - T^4)
    name: str = ""
    on: str = ""                       # "" = the whole model, else a feature or material name
    pulse: Pulse = field(default_factory=Pulse)   # flux: time profile (transient problems)
    box: dict = field(default_factory=dict)       # optional {"min": [x, y, z], "max": [...]}: faces inside it
    profile: Profile = field(default_factory=Profile)   # flux: spatial profile

    def __post_init__(self):
        if isinstance(self.pulse, dict):
            self.pulse = Pulse.from_dict(self.pulse)
        if isinstance(self.profile, dict):
            self.profile = Profile.from_dict(self.profile)

    def to_dict(self):
        d = selection_to_dict(self, {"type": self.type, "name": self.name} if self.name else {"type": self.type})
        if self.type == "convection":
            d.update(h=self.h, T_ext=self.T_ext)
        elif self.type == "radiation":
            d.update(emissivity=self.emissivity, T_ext=self.T_ext)
        else:
            d["value"] = self.value
        if self.type == "flux" and self.pulse.shape:
            d["pulse"] = self.pulse.to_dict()
        if self.type == "flux" and self.profile.shape:
            d["profile"] = self.profile.to_dict()
        return d


@dataclass
class HeatSource:
    """Volumetric heat source (W/m^3) in every domain of a material."""
    material: str = ""
    value: Expr = "1e12[W/m^3]"        # may use `time`
    name: str = ""
    pulse: Pulse = field(default_factory=Pulse)
    profile: Profile = field(default_factory=Profile)

    def __post_init__(self):
        if isinstance(self.pulse, dict):
            self.pulse = Pulse.from_dict(self.pulse)
        if isinstance(self.profile, dict):
            self.profile = Profile.from_dict(self.profile)

    def to_dict(self):
        d = {"name": self.name, "material": self.material, "value": self.value} if self.name else \
            {"material": self.material, "value": self.value}
        if self.pulse.shape:
            d["pulse"] = self.pulse.to_dict()
        if self.profile.shape:
            d["profile"] = self.profile.to_dict()
        return d


@dataclass
class HeatInterface:
    """Thermal contact at the interfaces between two objects (objects or materials): the temperature
    jumps across them, q = (T1 - T2) / R. `between` = [a, b]: the faces a and b share; [a]: a's interfaces with
    any other object; []: every interface between different materials.

    resistance: a thermal boundary (Kapitza) resistance R (m^2 K/W), may depend on T (the mean of both sides);
    layer: a thin layer of `material`, not meshed, of thickness d: R = d / k (k across the layer) and heat
    capacity rho cp d (half to each side); no heat flow along the layer."""
    type: str = "resistance"           # resistance | layer
    between: list[str] = field(default_factory=list)
    R: Expr = "1e-8[m^2*K/W]"
    material: str = ""
    thickness: Expr = "5[nm]"
    name: str = ""

    def to_dict(self):
        d = {"type": self.type, "name": self.name} if self.name else {"type": self.type}
        d["between"] = list(self.between)
        if self.type == "layer":
            d.update(material=self.material, thickness=self.thickness)
        else:
            d["R"] = self.R
        return d


@dataclass
class HeatProblem:
    type: str = "stationary"           # stationary | transient | frequency (small-signal harmonic response)
    t_end: Expr = "1[us]"
    t_start: Expr = "0[s]"             # may be negative, e.g. to settle before a pulse at t = 0
    steps: int = 100
    theta: float = 1.0                 # 1 = backward Euler (robust), 0.5 = Crank-Nicolson
    save_every: int = 1
    # nonlinear models (radiation, temperature-dependent properties): iterate until the largest temperature
    # change is below tol times the largest temperature (every time step of a transient problem)
    tol: float = 1e-6
    max_iter: int = 50
    # frequency domain: every flux and source oscillates at this frequency (sweep it with a variable)
    frequency: Expr = "f"

    def to_dict(self):
        d = {"type": self.type, "t_start": self.t_start, "t_end": self.t_end, "steps": self.steps,
             "theta": self.theta, "save_every": self.save_every, "tol": self.tol, "max_iter": self.max_iter}
        if self.type == "frequency":
            d = {"type": self.type, "frequency": self.frequency}
        return without_defaults(self, d, keep=("type", "frequency"))


@dataclass
class HeatPhysics:
    initial: Expr = "293.15[K]"        # initial temperature; may use x, y, z (m)
    conditions: list[HeatCondition] = field(default_factory=list)
    sources: list[HeatSource] = field(default_factory=list)
    problem: HeatProblem = field(default_factory=HeatProblem)
    interfaces: list[HeatInterface] = field(default_factory=list)
    # bounding-box faces of the model that are mirror planes: insulated, and left out of every selection
    # (a condition on "all" faces stops at them)
    symmetry: list[str] = field(default_factory=list)

    def to_dict(self):
        d = {"initial": self.initial}
        if self.symmetry:
            d["symmetry"] = list(self.symmetry)
        d["conditions"] = [c.to_dict() for c in self.conditions]
        if self.sources:
            d["sources"] = [q.to_dict() for q in self.sources]
        if self.interfaces:
            d["interfaces"] = [c.to_dict() for c in self.interfaces]
        d["problem"] = self.problem.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HeatPhysics":
        check_keys(d, ["initial", "conditions", "sources", "problem", "interfaces", "symmetry"], "[heat]")
        return cls(initial=d.get("initial", "293.15[K]"),
                   conditions=[table(HeatCondition, c, f"heat condition {label(c, 'temperature')}")
                               for c in d.get("conditions", [])],
                   sources=[table(HeatSource, q, f"heat source {label(q, q.get('material', ''))}")
                            for q in d.get("sources", [])],
                   problem=table(HeatProblem, d.get("problem", {}), "[heat.problem]"),
                   interfaces=[table(HeatInterface, c, f"heat interface {label(c, 'resistance')}")
                               for c in d.get("interfaces", [])],
                   symmetry=list(d.get("symmetry", [])))

    def face_conditions(self) -> dict[str, str]:
        """Conditions on the faces of the model's bounding box (for display)."""
        out = {face: "symmetry" for face in self.symmetry}
        for c in self.conditions:
            if c.on or c.box:
                continue
            for face in c.faces:
                out[face] = c.type
        return out


def validate_settings(project, ev, material_names) -> list[str]:
    """Problems with the heat-transfer settings: conditions, sources, problem, initial values."""
    errors, seen = [], {}
    h = project.heat
    evt = ev.with_overrides({"time": 0.0})      # values may depend on time
    objects = {f.name for f in project.objects} | set(material_names)
    transient = h.problem.type == "transient"
    harmonic = h.problem.type == "frequency"     # time profiles: a sine gives its amplitude, pulses are ignored
    for face in h.symmetry:
        if face not in FACES:
            errors.append(f"heat.symmetry: bad face '{face}' (use {', '.join(FACES)})")
    if len(set(h.symmetry)) != len(h.symmetry):
        errors.append("heat.symmetry: a face is listed twice")

    def check_pulse(where, pulse):
        if pulse.shape not in PULSE_SHAPES:
            errors.append(f"{where}: pulse shape must be 'rect', 'gauss' or 'sine'")
        elif pulse.shape == "sine":
            if not (transient or harmonic):
                errors.append(f"{where}: an oscillation needs a transient or frequency-domain problem")
            if not str(pulse.amplitude).strip() or (transient and not str(pulse.period).strip()):
                errors.append(f"{where}: an oscillation needs a period and an amplitude")
            for key in ("start", "period", "amplitude"):
                if not str(getattr(pulse, key)).strip():
                    continue
                try:
                    v = ev(getattr(pulse, key))
                    if key == "period" and v <= 0:
                        errors.append(f"{where}: oscillation period must be positive")
                except Exception as exc:
                    errors.append(f"{where}: oscillation {key}: {exc}")
        elif pulse.shape and not harmonic:
            if not transient:
                errors.append(f"{where}: a pulse needs a transient problem")
            for key in ("start", "duration") + (("period",) if pulse.period else ()):
                try:
                    v = ev(getattr(pulse, key))
                    if key != "start" and v <= 0:
                        errors.append(f"{where}: pulse {key} must be positive")
                except Exception as exc:
                    errors.append(f"{where}: pulse {key}: {exc}")

    def check_profile(where, prof):
        if prof.shape not in PROFILE_SHAPES:
            errors.append(f"{where}: profile shape must be 'gauss' or 'expr'")
        elif prof.shape == "gauss":
            try:
                if len(prof.center) != 2:
                    errors.append(f"{where}: profile center needs x and y")
                [ev(v) for v in prof.center]
                if ev(prof.fwhm) <= 0:
                    errors.append(f"{where}: profile fwhm must be positive")
            except Exception as exc:
                errors.append(f"{where}: profile: {exc}")
        elif prof.shape == "expr":
            try:
                from ... import fields
                fields.check(str(prof.expr), ev.values, fields=(), scalars=())
            except Exception as exc:
                errors.append(f"{where}: profile: {exc}")

    for c in h.conditions:
        where = f"heat condition '{c.name or c.type}'"
        if c.type not in HEAT_CONDITION_TYPES:
            errors.append(f"{where}: type must be one of {', '.join(HEAT_CONDITION_TYPES)}")
        errors += check_selection(where, c, ev, objects)
        if not c.on:
            for face in c.faces:
                if face in h.symmetry:
                    errors.append(f"{where}: face {face} is a symmetry plane (heat.symmetry) and cannot carry "
                                  f"a condition")
        for face in ([] if c.box or c.type in ADDITIVE else c.faces):   # boxes, "all" and added heat flows may
            # overlap other selections on purpose
            key = (c.on, face)
            if face in FACES and key in seen:
                errors.append(f"heat: face {face}{' of ' + c.on if c.on else ''} has two conditions "
                              f"({seen[key]}, {c.name or c.type})")
            seen[key] = c.name or c.type
        keys = {"convection": ("h", "T_ext"), "radiation": ("emissivity", "T_ext")}.get(c.type, ("value",))
        for key in keys:
            try:
                evt(getattr(c, key))
            except Exception as exc:
                errors.append(f"{where}: {key}: {exc}")
        if c.type == "flux":
            check_pulse(where, c.pulse)
            check_profile(where, c.profile)
        if c.type == "radiation":
            try:
                if not 0 < ev(c.emissivity) <= 1:
                    errors.append(f"{where}: emissivity must be between 0 and 1")
            except Exception:
                pass                                        # reported above
    for q in h.sources:
        if q.material not in material_names:
            errors.append(f"heat source '{q.name}': material '{q.material}' not defined")
        try:
            evt(q.value)
        except Exception as exc:
            errors.append(f"heat source '{q.name}': {exc}")
        check_pulse(f"heat source '{q.name or q.material}'", q.pulse)
        check_profile(f"heat source '{q.name or q.material}'", q.profile)
    for c in h.interfaces:
        errors += _check_interface(c, ev, objects, project)
    st = h.problem
    if st.type not in HEAT_PROBLEM_TYPES:
        errors.append(f"heat.problem.type must be one of {', '.join(HEAT_PROBLEM_TYPES)}")
    if st.type == "frequency":
        try:
            if ev(st.frequency) <= 0:
                errors.append("heat.problem.frequency must be positive")
        except Exception as exc:
            errors.append(f"heat.problem.frequency: {exc}")
    if st.type == "transient":
        try:
            if ev(st.t_end) <= ev(st.t_start):
                errors.append("heat.problem: t_end must be later than t_start")
        except Exception as exc:
            errors.append(f"heat.problem.t_start / t_end: {exc}")
        if st.steps < 1 or st.save_every < 1:
            errors.append("heat.problem: steps and save_every must be >= 1")
        if not 0.5 <= st.theta <= 1:
            errors.append("heat.problem.theta must be between 0.5 and 1")
    if not 0 < st.tol < 1 or st.max_iter < 1:
        errors.append("heat.problem: tol must be between 0 and 1, max_iter >= 1")
    try:
        from ... import fields
        fields.check(h.initial, ev.values, fields=(), scalars=("time",))     # may use x, y, z
    except Exception as exc:
        errors.append(f"heat.initial: {exc}")
    return errors


def interface_title(c) -> str:
    """'Pad | Membrane', 'Pad | any', 'material interfaces'."""
    if not c.between:
        return "material interfaces"
    return " | ".join(c.between) if len(c.between) == 2 else f"{c.between[0]} | any"


def _check_interface(c, ev, objects, project) -> list[str]:
    from ...materials import REFERENCE_T, TEMPERATURE, depends_on_T
    errors = []
    where = f"heat interface '{c.name or interface_title(c)}'"
    if c.type not in INTERFACE_TYPES:
        errors.append(f"{where}: type must be one of {', '.join(INTERFACE_TYPES)}")
    if len(c.between) > 2:
        errors.append(f"{where}: between takes at most two objects")
    if len(c.between) == 2 and c.between[0] == c.between[1]:
        errors.append(f"{where}: the two objects must differ")
    for obj in c.between:
        if obj not in objects:
            errors.append(f"{where}: unknown object '{obj}' (an object or material name)")
    if c.type == "resistance":
        try:
            evr = ev.with_overrides({TEMPERATURE: REFERENCE_T}) if depends_on_T(c.R) else ev
            if evr(c.R) <= 0:
                errors.append(f"{where}: R must be positive")
        except Exception as exc:
            errors.append(f"{where}: R: {exc}")
    elif c.type == "layer":
        if c.material not in {m.name for m in project.materials}:
            errors.append(f"{where}: material '{c.material}' not defined")
        else:
            from .properties import Material
            try:
                Material(c.material, project.material(c.material).props, ev)
            except Exception as exc:
                errors.append(f"{where}: {exc}")
        try:
            if ev(c.thickness) <= 0:
                errors.append(f"{where}: thickness must be positive")
        except Exception as exc:
            errors.append(f"{where}: thickness: {exc}")
    return errors
