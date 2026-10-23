"""Heat transfer panels: physics overview, initial values, conditions, sources with pulses, problem,
solver and the temperature field."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QSpinBox,
)

from ...physics.heat.model import HeatCondition, HeatInterface, HeatSource, interface_title
from ...physics.heat.properties import nonlinear
from ...selection import ALL_FACES, FACES, selection_text
from .problem import solver_box
from .common import (
    ERROR_STYLE, ExprEdit, Panel, bypass_row, bypassed_label, equations, field_quantity_rows, muted_label, objects,
    outer_labels, selection_box,
)


# ---------------------------------------------------------------------------
# Heat transfer
HEAT_TITLES = {"temperature": "Temperature", "flux": "Heat flux", "convection": "Convection",
               "radiation": "Radiation"}


def heat_condition_label(c) -> str:
    where = selection_text(c)
    if c.type == "flux" and c.profile.shape:
        where += ", Gaussian" if c.profile.shape == "gauss" else ", profile"
    if c.type == "flux" and c.pulse.shape:
        where += ", oscillating" if c.pulse.shape == "sine" else ", pulsed"
    return bypassed_label(c, f"{c.name or HEAT_TITLES.get(c.type, c.type)} ({where})")


INTERFACE_TITLES = {"resistance": "Contact resistance", "layer": "Thin layer"}


def interface_label(c) -> str:
    return bypassed_label(c, f"{c.name or INTERFACE_TITLES.get(c.type, c.type)} ({interface_title(c)})")


def source_label(q) -> str:
    return bypassed_label(q, f"{q.name or 'Heat source'} (in {q.material})")


PULSE_TITLES = {"": "Constant", "rect": "Rectangular pulse", "gauss": "Gaussian pulse", "sine": "Sinusoidal"}


def pulse_box(panel, pulse, refresh, value=None) -> QGroupBox:
    """Time profile of a flux or source: constant, a (repeated) rectangular / Gaussian pulse, or a sinusoidal
    oscillation about the value; value() gives the current value (the default amplitude)."""
    box = QGroupBox("Time profile (transient problems)")
    f = QFormLayout(box)
    shape = QComboBox()
    for key, title in PULSE_TITLES.items():
        shape.addItem(title, key)
    shape.setCurrentIndex(max(0, shape.findData(pulse.shape)))

    def set_shape(i):
        was_sine, pulse.shape = pulse.shape == "sine", shape.itemData(i)
        if was_sine and pulse.shape != "sine":
            pulse.period = ""               # the oscillation's period is not a pulse repetition
        if pulse.shape == "sine":           # full modulation by default: from 0 to twice the mean
            pulse.amplitude = pulse.amplitude or (str(value()) if value else "")
            pulse.period = pulse.period or "1[us]"
        panel.win.mark_dirty()
        refresh()

    shape.currentIndexChanged.connect(set_shape)
    f.addRow("Shape", shape)

    def setter(attr, default=""):
        return lambda t: (setattr(pulse, attr, t or default), panel.win.mark_dirty())
    if pulse.shape == "sine":
        f.addRow("Amplitude", ExprEdit(panel.win, lambda: pulse.amplitude, setter("amplitude")))
        period = ExprEdit(panel.win, lambda: pulse.period, setter("period", "1[us]"))
        period.setToolTip("Period of the oscillation, 1/f; e.g. 1/(1[MHz])")
        f.addRow("Period", period)
        start = ExprEdit(panel.win, lambda: pulse.start, setter("start", "0[s]"))
        start.setToolTip("Phase: the sine rises through the mean at this time")
        f.addRow("Phase zero at", start)
        f.addRow(muted_label("value(t) = value + amplitude · sin(2π (t − t₀) / period): the value above is the "
                             "mean. Each time step uses the exact average over the step; take at least ~20 steps "
                             "per period to resolve the oscillation."))
    elif pulse.shape:
        gauss = pulse.shape == "gauss"
        f.addRow("Peak at" if gauss else "Start", ExprEdit(panel.win, lambda: pulse.start, setter("start", "0[s]")))
        f.addRow("FWHM" if gauss else "Duration",
                 ExprEdit(panel.win, lambda: pulse.duration, setter("duration", "10[ns]")))
        period = ExprEdit(panel.win, lambda: pulse.period, setter("period"), optional="once")
        period.setToolTip("Repeat the pulse with this period; empty = a single pulse")
        f.addRow("Repeat every", period)
        f.addRow(muted_label("The value above is the peak. Each time step uses the pulse averaged over the step, "
                             "so the deposited energy is exact even when steps are longer than the pulse edges; "
                             "use steps shorter than the pulse to resolve its shape."))
    else:
        f.addRow(muted_label("Values may also depend on time directly, e.g. 1e9[W/m^2]*(time < 50[ns])."))
    return box


PROFILE_TITLES = {"": "Uniform", "gauss": "Gaussian beam (x-y)", "expr": "Expression of x, y, z"}


def profile_box(panel, profile, refresh, surface=True) -> QGroupBox:
    """Spatial profile of a flux or source: uniform, a Gaussian beam along z, or an expression of x, y, z."""
    from ... import fields
    box = QGroupBox("Spatial profile")
    f = QFormLayout(box)
    shape = QComboBox()
    for key, title in PROFILE_TITLES.items():
        shape.addItem(title, key)
    shape.setCurrentIndex(max(0, shape.findData(profile.shape)))

    def set_shape(i):
        profile.shape = shape.itemData(i)
        panel.win.mark_dirty()
        refresh()

    shape.currentIndexChanged.connect(set_shape)
    f.addRow("Shape", shape)
    if profile.shape == "gauss":
        def set_center(k):
            def apply(t):
                profile.center[k] = t or "0"
                panel.win.mark_dirty()
            return apply
        for k, axis in enumerate("xy"):
            f.addRow(f"Centre {axis}", ExprEdit(panel.win, lambda k=k: profile.center[k], set_center(k)))
        f.addRow("FWHM", ExprEdit(panel.win, lambda: profile.fwhm,
                                  lambda t: (setattr(profile, "fwhm", t or "3[um]"), panel.win.mark_dirty())))
        f.addRow(muted_label("Factor exp(−4 ln2 r² / FWHM²), r the distance from the centre in the x-y plane: the "
                             "value above is the peak. A beam of total power P has the peak "
                             + ("flux 4 ln2 P / (π FWHM²)" if surface else "4 ln2 P / (π FWHM² d) when absorbed "
                                "uniformly over a depth d") + ", e.g. 4*log(2)*P/(pi*w^2)."))
    elif profile.shape == "expr":
        e = QLineEdit(str(profile.expr))
        note = muted_label("A dimensionless factor of the position x, y, z (m) and the variables, e.g. "
                           "exp((z - z_top) / delta) for absorption over a depth delta.")

        def commit():
            text = e.text().strip() or "1"
            try:
                fields.check(text, panel.p.evaluator().values, fields=(), scalars=())
            except Exception as exc:
                e.setStyleSheet(ERROR_STYLE)
                e.setToolTip(str(exc))
                return
            e.setStyleSheet("")
            e.setToolTip("")
            profile.expr = text
            panel.win.mark_dirty()

        e.editingFinished.connect(commit)
        f.addRow("Factor", e)
        f.addRow(note)
    return box


class HeatPanel(Panel):
    title = "Heat Transfer"
    help = ("Conduction in the meshed structure. Conditions act on faces of the model's bounding box or of one "
            "object (e.g. the top of a pad); heat sources fill all domains of a material, and fluxes and "
            "sources can be pulsed in transient problems. Interfaces between objects are perfect contacts "
            "unless given a contact resistance or a thin layer.")
    formalism = equations(
        ("ρ c<sub>p</sub> ∂T/∂t = ∇·(k ∇T) + Q", "Heat conduction (Fourier's law), with the heat sources Q "
         "(W/m³). A stationary problem drops the time term and solves ∇·(k ∇T) + Q = 0."),
        ("n·(k ∇T) = 0", "On every face without a condition: insulated — the same as a mirror-symmetry plane "
         "(Symmetry names such planes and keeps conditions on all faces off them)."),
        ("C Ṫ + K T = f", "The assembled system (quadratic tetrahedra). Stationary: K T = f, one sparse solve. "
         "Transient: θ-scheme time steps (backward Euler or Crank–Nicolson), the matrix factorised once if the "
         "properties are constant. "
         "Frequency domain: (K + iωC) T̂ = f̂ for the complex amplitude of the temperature."),
        "Radiation, or properties written with the temperature (k, c<sub>p</sub>, ρ using T), make the problem "
        "nonlinear: it is then solved by Picard iteration until the temperature stops changing, re-assembling "
        "only the matrix that varies.",
    )

    def build(self):
        h = self.p.heat
        f = self.form("Conditions and sources")
        for i, c in enumerate(h.conditions):
            b = QPushButton(heat_condition_label(c))
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.clicked.connect(lambda _=False, i=i: self.win.select_node("heat_cond", i))
            f.addRow(b)
        for i, q in enumerate(h.sources):
            b = QPushButton(source_label(q))
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.clicked.connect(lambda _=False, i=i: self.win.select_node("heat_src", i))
            f.addRow(b)
        for i, c in enumerate(h.interfaces):
            b = QPushButton(interface_label(c))
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.clicked.connect(lambda _=False, i=i: self.win.select_node("heat_iface", i))
            f.addRow(b)
        if not h.conditions and not h.sources:
            f.addRow(muted_label("No conditions yet: all faces are insulated."))
        row = self.button_row(("Heat source", lambda: add_source(self.win)))
        for typ in reversed(HEAT_TITLES):
            b = QPushButton(HEAT_TITLES[typ])
            b.clicked.connect(lambda _=False, t=typ: add_condition(self.win, t))
            row.insertWidget(0, b)
        row.insertWidget(0, QLabel("Add:"))
        row = self.button_row(*[(INTERFACE_TITLES[t], lambda _=False, t=t: add_interface(self.win, t))
                                for t in INTERFACE_TITLES])
        row.insertWidget(0, QLabel("Interfaces:"))


class HeatSymmetryPanel(Panel):
    title = "Symmetry"
    help = ("Faces of the model's bounding box that are mirror planes of a structure modelled only in half (or a "
            "quarter). They are insulated, and no condition reaches them: a selection of all faces, of the model "
            "or of an object, stops at the plane. Physically the same as leaving the face without a condition; "
            "naming it says why it is empty.")
    formalism = equations(
        ("n·(k ∇T) = 0", "Mirror symmetry of the temperature field: no heat crosses the plane. Implementation: "
         "nothing is added on those faces (the natural condition of the weak form), and their boundary "
         "triangles are removed from every condition's selection."),
    )

    def build(self):
        h = self.p.heat
        box = QGroupBox("Faces of the bounding box")
        form = QFormLayout(box)
        for face in FACES:
            cb = QCheckBox("Symmetry plane")
            cb.setChecked(face in h.symmetry)

            def changed(on, face=face):
                if face in h.symmetry:
                    h.symmetry.remove(face)
                if on:
                    h.symmetry.append(face)
                    h.symmetry.sort(key=FACES.index)
                self.win.physics_changed()

            cb.toggled.connect(changed)
            form.addRow(face, cb)
        self.outer.addWidget(box)


class HeatInitialPanel(Panel):
    title = "Initial values"
    help = ("The temperature a transient problem starts from. May depend on position, e.g. "
            "300[K] + 10[K]*exp(-(x^2+y^2)/(100[nm])^2).")
    formalism = equations(
        ("T(r, 0) = T₀(r)", "The initial condition of a transient problem, evaluated at the nodes of the mesh. "
         "In stationary and frequency-domain problems it is only the starting guess of the nonlinear "
         "iteration (a linear model ignores it)."),
    )

    def build(self):
        from ... import fields
        h = self.p.heat
        f = self.form()
        e = QLineEdit(str(h.initial))
        note = muted_label("")

        def commit():
            text = e.text().strip() or "293.15[K]"
            try:
                fields.check(text, {**self.p.evaluator().values, "x": 0, "y": 0, "z": 0})
                e.setStyleSheet("")
                note.setText("")
                h.initial = text
                self.win.mark_dirty()
            except Exception as exc:
                e.setStyleSheet(ERROR_STYLE)
                note.setText(str(exc))

        e.editingFinished.connect(commit)
        f.addRow("Temperature", e)
        f.addRow(note)


HEAT_CONDITION_HELP = {
    "temperature": "The selected faces are held at a given temperature.",
    "flux": "Heat entering the body through the selected faces, e.g. absorbed laser power: uniform or with a "
            "spatial profile (a Gaussian beam), constant or pulsed / oscillating in time.",
    "convection": "Cooling or heating by a fluid around the structure.",
    "radiation": "Thermal radiation to the surroundings — often the main heat loss in vacuum.",
}

HEAT_CONDITION_FORMALISM = {
    "temperature": equations(
        ("T = T₀", "A fixed temperature (Dirichlet condition) on the selected faces. Implementation: every "
         "node of those boundary faces is held at T₀ — its degree of freedom is removed from the system. "
         "Where selections overlap, the later condition's temperature wins (fluxes, in contrast, add up)."),
    ),
    "flux": equations(
        ("−n·(k ∇T) = q", "The heat flux into the body through the selected faces (W/m², positive = heating). "
         "Implementation: ∫ v q dS over the selected boundary triangles, added to the right-hand side; no "
         "degree of freedom is removed, so the faces are free to warm up. A spatial profile multiplies q "
         "over the face, a pulse multiplies it in time (step-averaged over each time step)."),
    ),
    "convection": equations(
        ("−n·(k ∇T) = h (T<sub>ext</sub> − T)", "Convective cooling or heating (a Robin condition) with the "
         "heat-transfer coefficient h and the temperature T<sub>ext</sub> of the surrounding fluid. "
         "Implementation: h ∫ v T dS joins the matrix and h T<sub>ext</sub> ∫ v dS the right-hand side, so the "
         "condition stays linear — no iteration needed."),
    ),
    "radiation": equations(
        ("−n·(k ∇T) = ε σ (T<sub>amb</sub>⁴ − T⁴)", "Surface-to-ambient thermal radiation, with the emissivity "
         "ε (0 to 1) and the Stefan-Boltzmann constant σ. Implementation: the fourth power is Newton-linearised "
         "about the current temperature (4 ε σ T³ in the matrix) and the solve is iterated until it settles. "
         "It adds to other conditions on the same faces, e.g. convection."),
    ),
}


class HeatConditionPanel(Panel):
    def __init__(self, win, index):
        self.index = index
        self.c = win.project.heat.conditions[index]
        self.title = f"{HEAT_TITLES.get(self.c.type, self.c.type)}: {self.c.name}" if self.c.name else \
            HEAT_TITLES.get(self.c.type, self.c.type)
        self.help = HEAT_CONDITION_HELP.get(self.c.type, "")
        self.formalism = HEAT_CONDITION_FORMALISM.get(self.c.type, "")
        super().__init__(win)

    def build(self):
        c = self.c
        bypass_row(self, c, ("heat_cond", self.index), "condition", lambda: delete_item(self.win, "heat_cond", self.index))
        f = self.form()
        name = QLineEdit(c.name)
        name.editingFinished.connect(lambda: (setattr(c, "name", name.text().strip()), self.win.mark_dirty(),
                                              self.win.rebuild_tree(("heat_cond", self.index), refresh_panel=False)))
        f.addRow("Name", name)
        if c.type == "convection":
            f.addRow("h", ExprEdit(self.win, lambda: c.h, lambda t: self._set("h", t)))
            f.addRow("T_ext", ExprEdit(self.win, lambda: c.T_ext, lambda t: self._set("T_ext", t)))
        elif c.type == "radiation":
            f.addRow("Emissivity ε", ExprEdit(self.win, lambda: c.emissivity, lambda t: self._set("emissivity", t)))
            f.addRow("Ambient T_amb", ExprEdit(self.win, lambda: c.T_ext, lambda t: self._set("T_ext", t)))
        else:
            label = "Temperature" if c.type == "temperature" else \
                "Mean heat flux (inward)" if c.pulse.shape == "sine" else \
                "Peak heat flux (inward)" if c.profile.shape else "Heat flux (inward)"
            f.addRow(label, ExprEdit(self.win, lambda: c.value, lambda t: self._set("value", t)))
        self.outer.addWidget(selection_box(self, c))
        if c.type == "flux":
            self.outer.addWidget(profile_box(self, c.profile, self._refresh))
            self.outer.addWidget(pulse_box(self, c.pulse, self._refresh, lambda: c.value))

    def _refresh(self):
        self.win.rebuild_tree(("heat_cond", self.index))

    def _set(self, attr, text):
        setattr(self.c, attr, text)
        self.win.mark_dirty()


class HeatSourcePanel(Panel):
    title = "Heat source"
    help = ("Heat generated inside the structure, e.g. Joule heating or absorption. In transient problems it "
            "may be pulsed or oscillate about its value.")
    formalism = equations(
        ("Q(r, t)  [W/m³]", "The source term of the heat equation, filling every domain made of the selected "
         "material. Implementation: ∫ v Q dV over the elements of those domains, added to the right-hand "
         "side; a spatial profile shapes Q over the volume and a pulse shapes it in time."),
    )

    def __init__(self, win, index):
        self.index = index
        self.q = win.project.heat.sources[index]
        super().__init__(win)

    def build(self):
        q = self.q
        bypass_row(self, q, ("heat_src", self.index), "heat source", lambda: delete_item(self.win, "heat_src", self.index))
        f = self.form()
        name = QLineEdit(q.name)
        name.editingFinished.connect(lambda: (setattr(q, "name", name.text().strip()), self.win.mark_dirty()))
        f.addRow("Name", name)
        m = QComboBox()
        m.addItems([x.name for x in self.p.materials])
        m.setCurrentText(q.material)
        m.currentTextChanged.connect(lambda v: (setattr(q, "material", v), self.win.mark_dirty(),
                                                self.win.rebuild_tree(("heat_src", self.index), refresh_panel=False)))
        f.addRow("Material", m)
        f.addRow("Mean Q" if q.pulse.shape == "sine" else "Peak Q" if q.profile.shape else "Q",
                 ExprEdit(self.win, lambda: q.value, lambda t: (setattr(q, "value", t), self.win.mark_dirty())))
        refresh = lambda: self.win.rebuild_tree(("heat_src", self.index))    # noqa: E731
        self.outer.addWidget(profile_box(self, q.profile, refresh, surface=False))
        self.outer.addWidget(pulse_box(self, q.pulse, refresh, lambda: q.value))


INTERFACE_SIDES = ("Implementation: the nodes on the interface are duplicated, one copy for each side, and "
                   "the two copies coupled by ∫ (T₁ − T₂)(v₁ − v₂) / R dS over the interface triangles. "
                   "Junctions where several objects meet stay connected. Without an interface condition the "
                   "sides share their nodes: a perfect contact.")

INTERFACE_HELP = {
    "resistance": "A thermal contact between two objects: the temperature jumps across it.",
    "layer": "A layer of material too thin to mesh, e.g. a 5 nm Cr adhesion layer under a gold pad.",
}

INTERFACE_FORMALISM = {
    "resistance": equations(
        ("q = (T₁ − T₂) / R", "The heat flux through the interface; the temperature jumps across it by "
         "ΔT = q R. A thermal boundary (Kapitza) resistance is typically 1e-9 to 1e-8 m²K/W between solids "
         "(conductances of 1 GW/(m²K) down to 100 MW/(m²K)), much more for a poor mechanical contact. R may "
         "depend on T, the mean of both sides. R = 0 is the perfect contact (e.g. the end of a sweep of R): the "
         "interface is then not cut, and T is continuous across it."),
        INTERFACE_SIDES,
    ),
    "layer": equations(
        ("R = d / k", "The resistance of a layer of thickness d, with k taken across the layer; heat does not "
         "flow along it. In transient problems the layer also carries its heat capacity ρ c<sub>p</sub> d, "
         "split between the two sides. Its properties may depend on T."),
        INTERFACE_SIDES,
    ),
}


class HeatInterfacePanel(Panel):
    def __init__(self, win, index):
        self.index = index
        self.c = win.project.heat.interfaces[index]
        title = INTERFACE_TITLES.get(self.c.type, self.c.type)
        self.title = f"{title}: {self.c.name}" if self.c.name else title
        self.help = INTERFACE_HELP.get(self.c.type, "")
        self.formalism = INTERFACE_FORMALISM.get(self.c.type, "")
        super().__init__(win)

    def build(self):
        c, p = self.c, self.p
        bypass_row(self, c, ("heat_iface", self.index), "interface",
                   lambda: delete_item(self.win, "heat_iface", self.index))
        f = self.form()
        name = QLineEdit(c.name)
        name.editingFinished.connect(lambda: (setattr(c, "name", name.text().strip()), self.win.mark_dirty(),
                                              self.win.rebuild_tree(("heat_iface", self.index), refresh_panel=False)))
        f.addRow("Name", name)
        typ = QComboBox()
        for key, title in INTERFACE_TITLES.items():
            typ.addItem(title, key)
        typ.setCurrentIndex(max(0, typ.findData(c.type)))
        typ.currentIndexChanged.connect(lambda i: self._change(type=typ.itemData(i)))
        f.addRow("Problem type", typ)
        if c.type == "layer":
            mat = QComboBox()
            names = [m.name for m in p.materials]
            mat.addItems(names if c.material in names else names + [c.material])
            mat.setCurrentText(c.material)
            mat.currentTextChanged.connect(lambda v: (setattr(c, "material", v), self.win.physics_changed()))
            f.addRow("Layer material", mat)
            f.addRow("Thickness d", ExprEdit(self.win, lambda: c.thickness, lambda t: self._set("thickness", t)))
        else:
            R = ExprEdit(self.win, lambda: c.R, lambda t: self._set("R", t))
            R.setToolTip("Thermal boundary resistance (m²K/W), e.g. 5e-9[m^2*K/W] or 1/200[MW/(m^2*K)]; "
                         "may depend on T")
            f.addRow("Resistance R", R)

        g = self.form("Interface")
        first = QComboBox()
        first.addItem("Every material interface", "")
        second = QComboBox()
        second.addItem("Any other object", "")
        for obj in objects(p):
            first.addItem(obj, obj)
            second.addItem(obj, obj)
        a = c.between[0] if c.between else ""
        b = c.between[1] if len(c.between) > 1 else ""
        for combo, value in ((first, a), (second, b)):
            if value and combo.findData(value) < 0:
                combo.addItem(f"{value} (missing)", value)
            combo.setCurrentIndex(max(0, combo.findData(value)))
        second.setEnabled(bool(a))

        def set_between(_=None):
            a, b = first.currentData(), second.currentData()
            second.setEnabled(bool(a))
            self._change(between=[x for x in (a, b if a else "") if x])

        first.currentIndexChanged.connect(set_between)
        second.currentIndexChanged.connect(set_between)
        first.setToolTip("An object (one of the Objects, or all domains of a material) whose interfaces get the "
                         "contact; or every interface between two different materials")
        g.addRow("Between", first)
        g.addRow("and", second)

    def _change(self, **kw):
        for key, value in kw.items():
            setattr(self.c, key, value)
        self.win.physics_changed()
        self.win.rebuild_tree(("heat_iface", self.index))

    def _set(self, attr, text):
        setattr(self.c, attr, text)
        self.win.mark_dirty()


HEAT_PROBLEM_FORMALISM = {
    "stationary": equations(
        ("(K + A<sub>bc</sub>) T = b", "K: conduction, A<sub>bc</sub>: convection and contact terms, b: heat "
         "sources and fluxes. Fixed temperatures are eliminated from the system; one sparse direct solve."),
        ("Σ<sub>i</sub> (C T)<sub>i</sub> = Σ<sub>i</sub> (C T₀)<sub>i</sub>", "Without any heat sink (no fixed "
         "temperature, convection or radiation) the temperature level is undetermined: it is fixed by keeping "
         "the heat content of the initial values. A steady state then exists only if the net heat input is "
         "zero."),
        "A nonlinear model (radiation, temperature-dependent properties or contacts) is solved by Picard "
        "iteration: the properties are taken at the previous iterate until the temperature stops changing.",
    ),
    "transient": equations(
        ("(C/Δt + θA) T<sup>n+1</sup> = (C/Δt − (1 − θ) A) T<sup>n</sup> + b<sup>n+θ</sup>", "θ-scheme "
         "steps of Δt = (t<sub>end</sub> − t<sub>start</sub>) / steps from T₀ (Initial values) at "
         "t<sub>start</sub>; A = K + A<sub>bc</sub>, C the heat capacity matrix. θ = 1: backward Euler "
         "(first order, never oscillates); θ = ½: Crank–Nicolson (second order, may ring after sudden "
         "changes)."),
        ("b<sup>n+θ</sup>", "Pulsed sources and fluxes are averaged over each step, so a pulse shorter than a "
         "step still deposits its full energy."),
        "The step matrix is factorised once when the properties are constant; a nonlinear model iterates "
        "(Picard) within every step.",
    ),
    "frequency": equations(
        ("(K + A<sub>bc</sub> + R + iωC) T̂ = b̂", "The small-signal response at ω = 2πf: b̂ holds the "
         "oscillation amplitudes of the sources and fluxes, fixed temperatures have T̂ = 0 (they do not "
         "oscillate). One complex sparse direct solve per frequency."),
        ("R = 4εσT₀³", "Linearisation about the initial temperature T₀: radiation acts as a sink of this "
         "strength, and temperature-dependent properties are taken at T₀."),
        ("T(t) = T₀ + Re(T̂ e<sup>iωt</sup>)", "The results: Tamp = |T̂|, Tphase = arg T̂ (degrees, negative "
         "= lagging the drive), Tre and Tim."),
    ),
}


HEAT_PROBLEM_FORMALISM["hydrodynamic"] = equations(
    ("q + k∇T = ℓ² (∇²q + 2∇(∇·q)),  ∇·q = Q", "Steady hydrodynamic (Guyer–Krumhansl) phonon heat transport: the heat "
     "flux q is an unknown next to T, with the viscous length ℓ² = ητ<sub>R</sub>. For ℓ → 0 it is Fourier's law."),
    ("[A G; Gᵀ 0] [q; T] = [f; −F]", "Like Stokes flow (q: velocity, T: pressure): quadratic q and linear T "
     "(Taylor–Hood, quadratic mesh needed), one sparse direct solve."),
    "Boundaries: temperature = heat bath (T prescribed, q free); flux = prescribed normal flux, no slip along the "
    "face; other faces = insulated walls, no slip (q = 0); symmetry planes: normal q = 0, tangential q free.",
)

HEAT_PROBLEM_TITLES = {"stationary": "Stationary", "transient": "Transient", "frequency": "Frequency domain",
                       "hydrodynamic": "Hydrodynamic (phonon)"}


class HeatProblemPanel(Panel):
    help = ("Stationary: the steady temperature field. Transient: the temperature evolution from the initial "
            "values at the start time, implicit time stepping. Frequency domain: the amplitude and phase of the "
            "temperature oscillation driven by sources at one frequency (3ω, FDTR). A variable sweep (Problem → "
            "Variable sweep) repeats any of them, e.g. over the frequency. Hydrodynamic: steady phonon flow with "
            "viscosity (Guyer–Krumhansl): Poiseuille flow and heat vortices in narrow structures.")
    action = ("Run ▶︎", "compute")

    def __init__(self, win):
        typ = win.project.heat.problem.type
        self.title = f"Problem: {HEAT_PROBLEM_TITLES.get(typ, typ)}"
        self.formalism = HEAT_PROBLEM_FORMALISM.get(typ, "")
        super().__init__(win)

    def build(self):
        st = self.p.heat.problem
        f = self.form()
        typ = QComboBox()
        for key, title in HEAT_PROBLEM_TITLES.items():
            typ.addItem(title, key)
        typ.setCurrentIndex(max(0, typ.findData(st.type)))
        typ.currentIndexChanged.connect(lambda i: set_problem_type(self.win, typ.itemData(i)))
        f.addRow("Problem type", typ)
        if st.type == "frequency":
            freq = ExprEdit(self.win, lambda: st.frequency, lambda t: setattr(st, "frequency", t or "f"))
            freq.setToolTip("Frequency of the oscillation; a variable (f) can be swept: Problem → Variable sweep, "
                            "e.g. logspace(3, 6, 16) * 1[Hz]")
            f.addRow("Frequency", freq)
            f.addRow(muted_label("Small-signal response: every heat flux and source oscillates at this frequency "
                                 "with its value as amplitude (or its sine amplitude); fixed temperatures do not "
                                 "oscillate, convection and radiation are sinks. Results are the complex amplitude "
                                 "of the temperature: Tamp, Tphase (degrees, negative = lagging), Tre, Tim. "
                                 "Temperature-dependent properties are taken at the initial temperature."))
        if st.type == "hydrodynamic":
            length = ExprEdit(self.win, lambda: st.viscous_length, lambda t: setattr(st, "viscous_length", t or "1[um]"))
            length.setToolTip("Viscous length ℓ = √(ητ_R) of the Guyer-Krumhansl equation: Fourier's law for ℓ ≪ the "
                              "structure, Poiseuille flow and heat vortices for ℓ comparable to its width "
                              "(χ = w²/ℓ² ≲ 10)")
            f.addRow("Viscous length", length)
            f.addRow(muted_label("Steady phonon hydrodynamics: the heat flux is solved for together with the "
                                 "temperature. Temperature conditions are heat baths, flux conditions prescribe the "
                                 "normal flux, and the other faces are insulated no-slip walls; symmetry planes are "
                                 "free-slip. Needs quadratic mesh elements, constant isotropic conductivity, and no "
                                 "convection, radiation or thermal contacts."))
        if st.type == "transient":
            start = ExprEdit(self.win, lambda: st.t_start, lambda t: setattr(st, "t_start", t or "0[s]"))
            start.setToolTip("Time of the initial values; may be negative, e.g. -50[ns] to start before a pulse "
                             "at t = 0")
            f.addRow("Start time", start)
            f.addRow("End time", ExprEdit(self.win, lambda: st.t_end, lambda t: setattr(st, "t_end", t or "1[us]")))
            steps = QSpinBox()
            steps.setRange(1, 1_000_000)
            steps.setValue(st.steps)
            steps.valueChanged.connect(lambda v: (setattr(st, "steps", v), self.win.mark_dirty()))
            f.addRow("Time steps", steps)
            every = QSpinBox()
            every.setRange(1, 1_000_000)
            every.setValue(st.save_every)
            every.setToolTip("Keep every n-th step for plots and fields")
            every.valueChanged.connect(lambda v: (setattr(st, "save_every", v), self.win.mark_dirty()))
            f.addRow("Save every", every)
            scheme = QComboBox()
            scheme.addItem("Backward Euler (robust)", 1.0)
            scheme.addItem("Crank–Nicolson (accurate)", 0.5)
            scheme.setCurrentIndex(0 if st.theta >= 0.75 else 1)
            scheme.currentIndexChanged.connect(lambda i: (setattr(st, "theta", scheme.itemData(i)),
                                                          self.win.mark_dirty()))
            f.addRow("Scheme", scheme)
        solver_box(self)
        if nonlinear(self.p):
            self._nonlinear_box(st)

    def _nonlinear_box(self, st):
        """Iterations of a nonlinear model (radiation, temperature-dependent properties)."""
        g = self.form("Nonlinear solver")
        g.addRow(muted_label("The model is nonlinear (radiation or temperature-dependent properties or contacts): "
                             + ("every time step is" if st.type == "transient" else "it is")
                             + " solved by iteration, with the properties at the previous iterate and the "
                             "radiation linearised about it."))
        tol = QLineEdit(f"{st.tol:g}")
        tol.setToolTip("Stop when the largest temperature change of an iteration is below this times the "
                       "largest temperature")

        def set_tol():
            try:
                v = float(tol.text())
                if not 0 < v < 1:
                    raise ValueError
            except ValueError:
                tol.setStyleSheet(ERROR_STYLE)
                return
            tol.setStyleSheet("")
            st.tol = v
            self.win.mark_dirty()

        tol.editingFinished.connect(set_tol)
        g.addRow("Relative tolerance", tol)
        it = QSpinBox()
        it.setRange(1, 10_000)
        it.setValue(st.max_iter)
        it.valueChanged.connect(lambda v: (setattr(st, "max_iter", v), self.win.mark_dirty()))
        g.addRow("Maximum iterations", it)


class TemperaturePanel(Panel):
    title = "Field"
    help = "A quantity of the solution on the structure. Pick the time step (transient) and the variable value (sweep)."

    def build(self):
        r, sel = self.win.results, self.win.selection
        f = self.form("Dataset")
        if r is None or not self.win.has_field_results:
            f.addRow(muted_label("No results yet. Configure the Problem and press Run; then pick the time step "
                                 "and variable value here (or click a point in a plot)."))
            return self._settings()
        f.addRow(muted_label(self.win.results_summary()))
        if r.outer_name:
            outer = QComboBox()
            outer.addItems(outer_labels(self.win))
            outer.setCurrentIndex(sel["i"])
            outer.currentIndexChanged.connect(lambda i: self._select(i=i))
            f.addRow("Variable sweep", outer)
        if len(r.times) > 1:
            from ...results.style import time_axis
            scale, unit = time_axis(self.p)
            row = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, len(r.times) - 1)
            slider.setValue(sel["j"])
            lbl = QLabel(f"{r.times[sel['j']] / scale:.4g} {unit}")
            lbl.setMinimumWidth(80)
            slider.valueChanged.connect(lambda j: (lbl.setText(f"{r.times[j] / scale:.4g} {unit}"),
                                                   self._select(j=j)))
            row.addWidget(slider, 1)
            row.addWidget(lbl)
            f.addRow("Time", row)
        self._settings()
        self.win.show_field()

    def _settings(self):
        q = self.form("Quantity")
        field_quantity_rows(self, q, "Any quantity of the solution on the surface of the structure, e.g. T, qz or "
                                     "sqrt(qx^2+qy^2+qz^2); gradients and fluxes are averaged at the nodes.")
        d = self.form("Style")
        st, viewer = self.win.field_style, self.win.viewer
        from ..viewer import COLORMAPS
        first = ["magma", "inferno", "plasma", "turbo", "coolwarm"]
        cmap = QComboBox()
        cmap.addItems(first + [c for c in COLORMAPS if c not in first])
        cmap.setCurrentText(st["cmap"])
        cmap.currentTextChanged.connect(lambda c: self.win.set_field_style(cmap=c))
        d.addRow("Colour scale", cmap)
        grid = QGridLayout()
        checks = [("Colour bar", st["colorbar"], lambda on: self.win.set_field_style(colorbar=on)),
                  ("Title", st["title"], lambda on: self.win.set_field_style(title=on)),
                  ("Rulers", viewer.scale_action.isChecked(), viewer.set_rulers),
                  ("Axes (xyz)", viewer.show_triad, viewer.set_triad)]
        for n, (label, on, fn) in enumerate(checks):
            cb = QCheckBox(label)
            cb.setChecked(on)
            cb.toggled.connect(fn)
            grid.addWidget(cb, n // 2, n % 2)
        d.addRow(grid)

    def _select(self, **kw):
        self.win.selection.update(kw)
        self.win.show_field()


# ---- actions (tree context menu, panel buttons) ------------------------------------------
def add_condition(win, typ: str):
    used = {f for c in win.project.heat.conditions for f in c.faces}
    free = [f for f in FACES if f not in used]
    c = HeatCondition(typ, faces=free[:1], name="")
    if typ == "flux":
        c.value = "1e6[W/m^2]"
    if typ == "radiation":         # usually from the whole outer surface; adds to other conditions
        c.faces = [ALL_FACES]
    win.project.heat.conditions.append(c)
    win.mark_dirty()
    win.rebuild_tree(("heat_cond", len(win.project.heat.conditions) - 1))


def add_source(win):
    material = win.project.materials[0].name if win.project.materials else ""
    win.project.heat.sources.append(HeatSource(material, "1e12[W/m^3]"))
    win.mark_dirty()
    win.rebuild_tree(("heat_src", len(win.project.heat.sources) - 1))


def add_interface(win, typ: str):
    h = win.project.heat
    c = HeatInterface(typ)
    if typ == "layer":
        names = [m.name for m in win.project.materials]
        c.material = "Cr" if "Cr" in names else (names[0] if names else "")
    h.interfaces.append(c)
    win.physics_changed()
    win.rebuild_tree(("heat_iface", len(h.interfaces) - 1))


def delete_item(win, kind: str, index: int):
    h = win.project.heat
    lst = {"heat_cond": h.conditions, "heat_src": h.sources, "heat_iface": h.interfaces}[kind]
    del lst[index]
    win.mark_dirty()
    win.rebuild_tree(("physics", None))


def set_problem_type(win, typ: str):
    """Stationary / transient / frequency domain. A transient problem gets a time plot of the average temperature
    unless it already has one; a frequency-domain problem needs a frequency (a variable f by default, so it can
    be swept) and shows the oscillation amplitude in field views."""
    from ...physics.heat.plots import time_plot
    p = win.project
    p.heat.problem.type = typ
    if typ == "frequency" and p.heat.problem.frequency == "f" and "f" not in p.variables:
        p.variables["f"] = "1[kHz]"
    for pl in p.plots:
        if pl.kind == "field" and pl.quantity in ("T", "Tamp"):
            pl.quantity = "Tamp" if typ == "frequency" else "T"
    if typ == "transient" and not any(pl.kind == "time" for pl in win.project.plots):
        names = {pl.name for pl in win.project.plots}
        plot = time_plot()
        if plot.name not in names:
            win.project.plots.append(plot)
    win.problem_changed()
    win.rebuild_tree(("problem", None))
