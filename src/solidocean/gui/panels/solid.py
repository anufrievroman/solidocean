"""Solid mechanics panels: physics overview, periodicity, symmetry, boundary conditions, problem
(eigenfrequency / stationary) with its solutions, wavevector sweep or mesh, solver and the displacement
field."""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ... import units
from .. import theme
from ...physics.solid.model import (LOAD_TYPES, SOLID_CONDITION_TITLES, PROBLEM_TITLES, Fourier, KMesh,
                                    KPath, SolidCondition)
from ...project import AXES, Feature
from ...selection import FACES, selection_text
from .common import (ERROR_STYLE, ExprEdit, Panel, VectorEdit, bypass_row, bypassed_label, equations,
                     field_quantity_rows, muted_label, outer_labels, pretty_point, references, selection_box,
                     theme_text)
from .problem import SweepPanel, solver_box, sweep_summary
from .lattice import _lattice_kind, hexagonal_vectors, lattice_mode, use_lattice_cell


def physics_summary(project) -> dict[str, str]:
    """Short text per physics node, used in the tree and the overview panel."""
    ph, lt = project.solid, project.lattice
    sym = list(ph.symmetry) + [f"{f} anti" for f in ph.antisymmetry]
    periodicity = ", ".join(lt.periodic) or "none"
    if lt.vectors:
        periodicity = {"hexagonal": "hexagonal", "1d": "1D lattice", "cubic": "3D lattice"}.get(
            _lattice_kind(project), f"{len(lt.vectors)}D lattice")
    return {"periodicity": periodicity, "symmetry": ", ".join(sym) or "none"}


def solid_condition_label(c) -> str:
    return bypassed_label(c, f"{c.name or SOLID_CONDITION_TITLES.get(c.type, c.type)} ({selection_text(c)})")


class SolidPanel(Panel):
    title = "Solid Mechanics"
    help = ("Linear elasticity; what is solved follows from Problem → Problem type. Fixed constraints, prescribed "
            "displacements and loads act on faces of the model or of one object (bounding-box faces or all "
            "faces, optionally only inside a box) and are highlighted in the graphics window.")
    formalism = equations(
        ("∇·(C : ∇u) + ρ ω² u = 0", "Eigenfrequency problem: the free vibrations of the structure. Assembled as "
         "(K − ω² M) u = 0 and solved for the lowest ω² by shift-invert Lanczos."),
        ("∇·(C : ∇u) + f = 0", "Stationary problem: the displacement under loads, assembled as K u = f."),
        ("(K + i K<sub>loss</sub> + iω C − ω² M) û = f̂", "Frequency-domain problem: the steady response to a "
         "drive at ω = 2πf, with the loss factors η of the problem and the materials (K<sub>loss</sub> = Σ "
         "η<sub>e</sub> K<sub>e</sub>) and Rayleigh damping C = αM + βK; û and f̂ are complex amplitudes, "
         "u(t) = Re(û e<sup>iωt</sup>)."),
        ("n·(C : ∇u) = 0", "On every face without a condition: traction-free, a free surface."),
        "C is the material's stiffness in the global frame (each material's crystal orientation is applied to "
        "it); u is interpolated with quadratic tetrahedra.",
    )

    def build(self):
        f = self.form("Boundary conditions")
        summary = physics_summary(self.p)
        for key, label in (("periodicity", "Periodicity"), ("symmetry", "Symmetry")):
            b = QPushButton(f"{label}: {summary[key]}")
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.clicked.connect(lambda _=False, k=key: self.win.select_node(k))
            f.addRow(b)
        for i, c in enumerate(self.p.solid.conditions):
            b = QPushButton(solid_condition_label(c))
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.clicked.connect(lambda _=False, i=i: self.win.select_node("solid_cond", i))
            f.addRow(b)
        row = self.button_row()
        row.insertWidget(0, QLabel("Add:"))
        for n, typ in enumerate(SOLID_CONDITION_TITLES):
            b = QPushButton(SOLID_CONDITION_TITLES[typ])
            b.clicked.connect(lambda _=False, t=typ: add_condition(self.win, t))
            row.insertWidget(n + 1, b)


class PeriodicityPanel(Panel):
    title = "Periodicity"
    help = ("Rectangular cells repeat along x, y, z; hexagonal and custom lattices use the lattice vectors "
            "a1, a2 (a3) and a unit-cell object. Periodic face pairs are tinted in the graphics "
            "window (a1/x red, a2/y green, a3/z blue).")
    formalism = equations(
        ("u(r + R) = e<sup>i k·R</sup> u(r)", "The Bloch-Floquet condition, for every lattice translation "
         "R = n₁a₁ + n₂a₂ + n₃a₃ and the wavevector k of the current sweep point."),
        "Implementation: each periodic face is paired with its partner by translating it along R, and their "
        "nodes are matched (KD-tree). The partner's degrees of freedom are then not solved for: they are "
        "e<sup>i k·R</sup> times those of its source, written as a reduction u = T(k) u<sub>r</sub>. The solver "
        "assembles the real K and M once and reduces them per wavevector, K<sub>r</sub> = Tᴴ K T, "
        "M<sub>r</sub> = Tᴴ M T — so a k-sweep never re-meshes or re-assembles.",
    )

    def build(self):
        ph = self.p.lattice
        mode, const = lattice_mode(ph)
        if mode == "hex" and getattr(self.win, "custom_lattice", False):
            mode = "custom"  # chosen explicitly, even though the vectors are still hexagonal
        f = self.form("Lattice")
        combo = QComboBox()
        combo.addItem("Rectangular (axis-aligned)", "rect")
        combo.addItem("Hexagonal", "hex")
        combo.addItem("Custom lattice vectors", "custom")
        combo.setCurrentIndex(combo.findData(mode))
        combo.currentIndexChanged.connect(lambda i: self._switch(combo.itemData(i)))
        f.addRow("Lattice", combo)
        if mode == "rect":
            self._rect_box()
        elif mode == "hex":
            f.addRow("Lattice constant", ExprEdit(self.win, lambda: const,
                                                   lambda t: self._set_lattice(hexagonal_vectors(t or "a"))))
            f.addRow(muted_label("a1 = a (1, 0, 0),  a2 = a (1/2, √3/2, 0). Standard path Γ–M–K–Γ."))
        else:
            self._custom_box()
        if mode != "rect":
            self._cell_hint()
        g = self.form("Bloch wave vector (rad/m, used when no k-path sweep is defined)")
        g.addRow(VectorEdit(self.win, self.p.solid.problem.k, ("kx", "ky", "kz")))

    # ---- modes -----------------------------------------------------------
    def _switch(self, mode):
        ph = self.p.lattice
        self.win.custom_lattice = mode == "custom"
        if mode == "rect":
            ph.vectors = []
            ph.periodic = {"x": "a", "y": "a"}
            for n, ft in enumerate(self.p.objects):  # a lattice cell becomes the matching block
                if ft.type == "cell":
                    self.p.objects[n] = Feature("block", ft.name, ft.op, ft.material, {
                        "center": list(ft.params.get("center", ["0", "0", "0"])),
                        "size": ["a", "a", ft.params.get("height", "1")]}, enabled=ft.enabled)
                    self.win.log(f"Replaced the lattice cell '{ft.name}' with a block")
            self.win.preview = None
        elif mode == "hex":
            ph.periodic = {}
            ph.vectors = hexagonal_vectors("a")
        else:
            vecs = ph.vectors or [[ph.periodic.get("x", "a"), "0", "0"], ["0", ph.periodic.get("y", "a"), "0"]]
            ph.periodic = {}
            ph.vectors = [list(v) for v in vecs]
        for ax in ph.periodic_axes():
            clear_face_conditions(self.win, ax)
        old = self.p.solid.problem.kpath
        if old is not None:        # the old path's named points belong to the old lattice; a list is the
            kp = default_kpath(self.p, enabled=old.enabled)      # user's own and is kept as it is
            kp.mode, kp.k = old.mode, [list(v) for v in old.k]
            self.p.solid.problem.kpath = kp
        self.win.physics_changed()
        self.win.rebuild_tree(("periodicity", None))

    def _set_lattice(self, vecs):
        self.p.lattice.vectors = vecs
        self.win.physics_changed()

    def _rect_box(self):
        ph = self.p.lattice
        box = QGroupBox("Lattice periods")
        grid = QGridLayout(box)
        for i, ax in enumerate(AXES):
            cb = QCheckBox(f"Periodic in {ax}")
            cb.setChecked(ax in ph.periodic)
            ed = ExprEdit(self.win, lambda ax=ax: ph.periodic.get(ax, "a"),
                          lambda t, ax=ax: self._set_period(ax, t))
            ed.setEnabled(cb.isChecked())

            def toggle(on, ax=ax, ed=ed):
                if on:
                    ph.periodic[ax] = ed.line.text() or "a"
                    clear_face_conditions(self.win, ax)
                else:
                    ph.periodic.pop(ax, None)
                ed.setEnabled(on)
                self.win.physics_changed()

            cb.toggled.connect(toggle)
            grid.addWidget(cb, i, 0)
            grid.addWidget(ed, i, 1)
        self.outer.addWidget(box)

    def _custom_box(self):
        ph = self.p.lattice
        f = self.form("Lattice vectors")
        for n, vec in enumerate(ph.vectors):
            f.addRow(f"a{n + 1}", VectorEdit(self.win, vec))
        three = QCheckBox("Periodic in a third direction (a3)")
        three.setChecked(len(ph.vectors) == 3)

        def toggle(on):
            if on and len(ph.vectors) == 2:
                ph.vectors.append(["0", "0", "a"])
            elif not on and len(ph.vectors) == 3:
                ph.vectors.pop()
            self.win.physics_changed()
            self.win.rebuild_tree(("periodicity", None))

        three.toggled.connect(toggle)
        f.addRow(three)

    def _cell_hint(self):
        has_cell = any(ft.type == "cell" and ft.enabled for ft in self.p.objects)
        if has_cell:
            self.outer.addWidget(muted_label("The geometry uses a 'cell' object: the unit cell spanned by a1, a2."))
            return
        self.outer.addWidget(muted_label(
            "The geometry must be one unit cell of this lattice. A 'cell' object (parallelogram spanned by "
            "a1, a2, extruded along z) builds it; holes and pillars go inside."))
        self.button_row(("Use lattice cell as base geometry", lambda: use_lattice_cell(self.win)))

    def _set_period(self, ax, text):
        if ax in self.p.lattice.periodic:
            self.p.lattice.periodic[ax] = text
            self.win.physics_changed()


class SymmetryPanel(Panel):
    title = "Symmetry"
    help = ("Planes of a model that is only half (or a quarter) of the real structure. Running it once with "
            "symmetry and once with antisymmetry on the same plane gives all the modes of the full structure.")
    formalism = equations(
        ("u · n = 0", "Symmetry plane: no motion through the plane, free motion in it — the mirror-symmetric "
         "half of the modes. Implementation: on every node of those faces the displacement along the face "
         "normal is held at zero (that degree of freedom is removed from the system). The planes are "
         "bounding-box faces, so the normal is always an axis."),
        ("u − (u·n) n = 0", "Antisymmetry plane: no motion in the plane, free motion through it — the "
         "mirror-antisymmetric half. Implementation: the two in-plane components of every node on those faces "
         "are held at zero."),
    )

    def build(self):
        ph = self.p.solid
        box = QGroupBox("Faces of the bounding box")
        form = QFormLayout(box)
        for face in FACES:
            combo = QComboBox()
            combo.addItems(["—", "Symmetry", "Antisymmetry"])
            combo.setCurrentIndex(1 if face in ph.symmetry else 2 if face in ph.antisymmetry else 0)
            reason = "periodic" if face[0] in self.p.lattice.periodic_axes() else ""
            if reason:
                combo.setEnabled(False)
                combo.setToolTip(f"{face} is {reason}")

            def changed(idx, face=face):
                for lst in (ph.symmetry, ph.antisymmetry):
                    if face in lst:
                        lst.remove(face)
                if idx == 1:
                    ph.symmetry.append(face)
                elif idx == 2:
                    ph.antisymmetry.append(face)
                self.win.physics_changed()

            combo.currentIndexChanged.connect(changed)
            form.addRow(face + (f"  ({reason})" if reason else ""), combo)
        self.outer.addWidget(box)


def default_kpath(project, enabled: bool = False) -> KPath:
    """The standard Brillouin-zone path of the current lattice."""
    from ... import lattice as lat
    lattice = project.lattice
    if lattice.vectors:
        kind = _lattice_kind(project)
        points, path = lat.KPATHS.get(kind if kind != "custom" else "oblique", lat.KPATHS["oblique"])
        return KPath(copy.deepcopy(points), list(path), 12, enabled=enabled)
    axes = [a for a in AXES if a in lattice.periodic]
    unit = {a: [1.0 if b == a else 0.0 for b in AXES] for a in AXES}
    half = lambda *ax: [0.5 * sum(unit[a][i] for a in ax) for i in range(3)]  # noqa: E731
    if len(axes) <= 1:
        ax = axes[0] if axes else "x"
        return KPath({"G": [0, 0, 0], "X": half(ax)}, ["G", "X"], 20, enabled=enabled)
    if len(axes) == 2:
        return KPath({"G": [0, 0, 0], "X": half(axes[0]), "M": half(*axes)}, ["G", "X", "M", "G"], 12,
                     enabled=enabled)
    return KPath({"G": [0, 0, 0], "X": half("x"), "M": half("x", "y"), "R": half("x", "y", "z")},
                 ["G", "X", "M", "G", "R", "X"], 10, enabled=enabled)


def default_kmesh(project, enabled: bool = True) -> KMesh:
    """A mesh of the irreducible wedge where the lattice has a standard one, else of the whole zone."""
    from ... import lattice as lat
    try:
        kind = lat.classify(project.lattice, project.evaluator())
    except Exception:
        kind = ""
    return KMesh(n=12, region="irreducible" if kind in lat.WEDGES else "full", enabled=enabled)


def default_fourier(project, enabled: bool = True) -> Fourier:
    """First-order components along every periodic direction of the lattice, none along the others."""
    from ... import lattice as lat
    try:
        slots = lat.slots(project.lattice, project.evaluator())
    except Exception:
        slots = [None, None, None]
    return Fourier(orders=[1 if v is not None else 0 for v in slots], enabled=enabled)


def fourier_labels(project) -> list[tuple[str, bool]]:
    """Label of each reciprocal vector ('b1 (x)', 'b2 (a2)') and whether the lattice is periodic along it."""
    from ... import lattice as lat
    try:
        slots = lat.slots(project.lattice, project.evaluator())
    except Exception:
        slots = [None, None, None]
    axes = ("a1", "a2", "a3") if project.lattice.vectors else AXES
    return [(f"b{i + 1} ({axes[i]})" + ("" if v is not None else ", not periodic"), v is not None)
            for i, v in enumerate(slots)]


def fourier_count(fo: Fourier) -> int:
    """Reciprocal vectors the grid of orders holds."""
    count = 1
    for o in fo.orders:
        count *= 2 * int(o) + 1
    return count


def kmesh_counts(project, km: KMesh) -> tuple[int, int] | None:
    """(points, triangles) the mesh would hold, or None if the lattice cannot carry one."""
    from ... import lattice as lat
    try:
        ev = project.evaluator()
        recip = [b for b in lat.reciprocal(lat.slots(project.lattice, ev)) if b is not None]
        if len(recip) != 2:
            return None
        points, tris = lat.zone_mesh(*recip, km.n, km.region, lat.classify(project.lattice, ev))
    except Exception:
        return None
    return len(points), len(tris)


def accent_fill():
    """A one-colour colormap of the accent, for filling a region of a sketch."""
    from matplotlib.colors import ListedColormap
    return ListedColormap([theme.ACCENT])


def paint_zone(ax, recip, draw):
    """The zone and what `draw(ax)` puts on it, onto a fresh axes."""
    from ... import lattice as lat
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    zone = lat.brillouin_zone_2d(*recip)
    poly = np.vstack([zone, zone[:1]])
    ax.fill(poly[:, 0], poly[:, 1], color=theme.SHORE[4], alpha=0.2 if not theme.is_dark() else 0.3)
    ax.plot(poly[:, 0], poly[:, 1], color=theme.shore_line(), lw=1.1)
    draw(ax)


def zone_sketch(outer, recip, draw, height: int = 190):
    """A small drawing of the 2D Brillouin zone in the app's own colours, with `draw(ax)` putting the
    wavevectors on top of it. The canvas is added to `outer` and returned, so that a panel can repaint it
    (`update_sketch`) while the user edits instead of waiting for the panel to be rebuilt."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(2.6, 2.4), layout="constrained")
    fig.patch.set_alpha(0)
    paint_zone(fig.add_subplot(), recip, draw)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setFixedHeight(height)
    canvas.setStyleSheet("background: transparent;")
    outer.addWidget(canvas)
    return canvas


def update_sketch(canvas, recip, draw):
    """Draw a zone sketch again, in place: the wavevectors move as they are edited."""
    if canvas is None:
        return
    paint_zone(canvas.figure.axes[0], recip, draw)
    canvas.draw_idle()


def problem_summary(project) -> dict[str, str]:
    """Short text per problem node (empty string = inactive)."""
    st = project.solid.problem
    kp, km, fo = st.active_kpath, st.active_kmesh, st.active_fourier
    counts = kmesh_counts(project, km) if km else None
    if not kp:
        kpath = ""
    elif kp.mode == "list":
        kpath = f"{len(kp.k)} wavevector{'s' if len(kp.k) != 1 else ''}"
    else:
        kpath = "–".join(pretty_point(p) for p in kp.path)
    return {**sweep_summary(project.problem), "problem_type": PROBLEM_TITLES.get(st.type, st.type),
            "kpath": kpath,
            "kmesh": (f"{counts[0]} points" if counts else f"{km.region}, n = {km.n}") if km else "",
            "fourier": f"{fourier_count(fo)} vectors" if fo else ""}


PROBLEM_HELP = {
    "eigenfrequency": "For every value of the variable sweep (outer loop), the wavevector sweep (inner loop) is "
                      "computed and at each point the requested eigenfrequencies are found. Geometry and mesh are "
                      "rebuilt only when a variable they depend on changes; a wavevector sweep reuses them. Loads "
                      "are ignored; prescribed displacements hold their components at zero.",
    "stationary": "The static displacement, strain and stress under the boundary loads and prescribed "
                  "displacements, for every value of the variable sweep. The model must be held against "
                  "rigid-body motion (fixed constraints, prescribed displacements or symmetry planes). Periodic "
                  "models give the periodic solution (wavevector 0).",
    "conductance": "Coherent (ballistic) heat conduction: the carriers of heat are the phonons of the "
                   "structure itself, so the band structure has to be known over the whole Brillouin zone, "
                   "not just along a path. The same eigenproblem as an eigenfrequency run is therefore "
                   "solved on a triangular mesh of the zone (Wavevector mesh below), giving a surface "
                   "ω(k) per band from which spectral quantities follow. Geometry, mesh and matrices are "
                   "built once and reused at every k point.",
    "frequency": "The harmonic response to loads and prescribed displacements oscillating at one frequency: "
                 "(K (1 + iη) + iω(αM + βK) − ω² M) u = f, the loads and displacements being the amplitudes of the drive. The "
                 "result is the complex displacement amplitude (use abs(uz) for an amplitude). A loss factor η "
                 "damps it: for every material (below) and per material (property eta, which may depend on x, y, "
                 "z: graded absorbing regions at the ends of a long structure keep waves from reflecting). A loss "
                 "factor damps every frequency alike (damping ratio η/2); Rayleigh damping C = αM + βK is viscous "
                 "instead, with a damping ratio α/2ω + βω/2 that falls with frequency through α and rises through "
                 "β, for fitting a measured Q over a band. Sweep the frequency with a variable (f) in Problem → "
                 "Variable sweep.",
}


PROBLEM_FORMALISM = {
    "eigenfrequency": equations(
        ("(K(k) − ω² M(k)) u = 0", "At every point of the wavevector sweep: the assembled stiffness K and mass M, "
         "reduced to the free degrees of freedom by the periodicity, symmetry and fixed constraints "
         "(K(k) = T(k)ᴴ K T(k), M(k) = T(k)ᴴ M T(k)). K and M are assembled once per mesh; only T(k) changes "
         "along a wavevector sweep."),
        ("(K − σM)⁻¹ M u = u / (ω² − σ)", "Shift-invert Lanczos (ARPACK): the n eigenvalues ω² closest to the "
         "shift σ = (2π f<sub>search</sub>)², or a tiny negative σ when searching from 0, which keeps "
         "K − σM invertible despite rigid-body modes. One sparse factorisation of K − σM per point; small "
         "systems are solved densely instead."),
        ("f = ω / 2π", "The eigenfrequencies reported, sorted; the eigenvectors are the mode shapes."),
    ),
    "conductance": equations(
        ("(K(k) − ω² M(k)) u = 0", "The eigenproblem of an eigenfrequency run, solved at every point of a "
         "mesh of the Brillouin zone instead of along a path: K and M are assembled once and only the "
         "Bloch reduction T(k) changes from point to point."),
        ("ω<sub>j</sub>(k), j = 1 … N", "The bands over the zone: at each k the eigenfrequencies in "
         "ascending order, so a band is the j-th lowest surface rather than a branch followed through its "
         "crossings (where surfaces cross, the group velocities read low)."),
        ("Mesh of the zone", "Either the irreducible wedge (Γ–X–M for a square lattice, Γ–M–K for a "
         "hexagonal one), which assumes the contents of the cell share the point symmetry of the lattice, "
         "or the whole zone. Each edge of the region is divided n times; a wedge counts S times (8 for a "
         "square lattice, 12 for a hexagonal one)."),
        ("D(ω) = (S / 4π²) Σ<sub>j,m</sub> L<sub>jm</sub> / |∇<sub>k</sub>ω<sub>jm</sub>|",
         "Density of states per unit area: over each triangle m a band is the plane through its corner "
         "frequencies, so the level ω<sub>j</sub>(k) = ω is a segment of length L<sub>jm</sub> and the "
         "gradient is constant. The group velocity is v<sub>j</sub> = ∇<sub>k</sub>ω<sub>j</sub>."),
        ("P(T) = (S/4π²) Σ<sub>j</sub> ∫ d²k ħω<sub>j</sub> n(ω<sub>j</sub>, T) |∇<sub>k</sub>ω<sub>j</sub>| / π",
         "Power a boundary at temperature T radiates into the structure per unit of its length, with the "
         "Bose-Einstein occupation n = 1/(e<sup>ħω/k<sub>B</sub>T</sup> − 1) and the speed |v|/π at which "
         "states leave a convex boundary, averaged over its directions, as in Eq. (1) of [1]. Integrated over "
         "the zone, each triangle divided into 64 sampled at their centroids."),
        ("G(T) = dP/dT", "The ballistic thermal conductance per unit length of the boundary, with ∂n/∂T in "
         "place of n. Both stop where the computed bands stop being complete (the lowest value of the "
         "topmost band over the zone), so they are lower bounds once k<sub>B</sub>T/ħ approaches it: a "
         "point is reliable up to about a ninth of that frequency."),
        references("N. Zen, T. A. Puurtinen, T. J. Isotalo, S. Chaudhuri and I. J. Maasilta, Nat. Commun. 5, 3435 (2014)"),
    ),
    "stationary": equations(
        ("K u = f", "The assembled stiffness K and load vector f (surface loads integrated over their faces)."),
        ("u = T u<sub>r</sub> + g", "Constraints: T keeps the free degrees of freedom (symmetry planes and "
         "periodic images map onto their source), g holds the prescribed displacements and zeros of fixed "
         "faces."),
        ("Tᵀ K T u<sub>r</sub> = Tᵀ (f − K g)", "The reduced system, one sparse direct solve. A residual "
         "check catches a model that is not held against rigid-body motion (singular K)."),
    ),
    "frequency": equations(
        ("(K + i K<sub>loss</sub> + iω C − ω² M) û = f̂", "At the drive frequency ω = 2πf: K and M as in the "
         "eigenfrequency problem, K<sub>loss</sub> = Σ η<sub>e</sub> K<sub>e</sub> with each element's loss factor "
         "(the problem's η plus the material's eta at the element, so it may vary in space). The loads and "
         "prescribed displacements are the amplitudes f̂ and ĝ of the drive."),
        ("C = α M + β K", "Rayleigh damping: viscous, proportional to mass (α, 1/s) and stiffness (β, s). Every "
         "mode keeps its shape and gets the damping ratio ζ = α/2ω + βω/2, so two measured ζ₁, ζ₂ at ω₁, ω₂ "
         "fix α = 2ω₁ω₂(ζ₁ω₂ − ζ₂ω₁)/(ω₂² − ω₁²) and β = 2(ζ₂ω₂ − ζ₁ω₁)/(ω₂² − ω₁²)."),
        ("Tᵀ A T û<sub>r</sub> = Tᵀ (f̂ − A ĝ)", "With A the matrix above and the same constraint "
         "reduction as a stationary problem: one complex sparse direct solve per frequency. Without loss or "
         "damping A is singular at an eigenfrequency."),
        ("u(t) = Re(û e<sup>iωt</sup>)", "The complex amplitude û is the result: abs(û) the amplitude, "
         "its argument the phase relative to the drive."),
    ),
}


class SolidProblemPanel(Panel):
    action = ("Run ▶\ufe0e", "compute")

    def __init__(self, win):
        st = win.project.solid.problem
        self.title = f"Problem: {PROBLEM_TITLES.get(st.type, st.type)}"
        self.help = PROBLEM_HELP.get(st.type, "")
        self.formalism = PROBLEM_FORMALISM.get(st.type, "")
        super().__init__(win)

    def build(self):
        st = self.p.solid.problem
        t = self.form()
        typ = QComboBox()
        for key, title in PROBLEM_TITLES.items():
            typ.addItem(title, key)
        typ.setCurrentIndex(max(0, typ.findData(st.type)))
        typ.currentIndexChanged.connect(lambda i: set_problem_type(self.win, typ.itemData(i)))
        t.addRow("Problem type", typ)
        if st.harmonic:
            freq = ExprEdit(self.win, lambda: st.frequency, lambda x: setattr(st, "frequency", x or "f"))
            freq.setToolTip("Drive frequency; a variable (f) can be swept: Problem → Variable sweep")
            t.addRow("Frequency", freq)
            loss = ExprEdit(self.win, lambda: st.loss, lambda x: setattr(st, "loss", x or "0"))
            loss.setToolTip("Loss factor η of every material: stiffness C (1 + iη); materials may add their own "
                            "(property eta)")
            t.addRow("Loss factor η", loss)
            alpha = ExprEdit(self.win, lambda: st.rayleigh_alpha, lambda x: setattr(st, "rayleigh_alpha", x or "0"))
            alpha.setToolTip("Rayleigh damping C = αM + βK, mass-proportional part α (1/s): a damping ratio α/2ω, "
                             "strongest at low frequencies")
            t.addRow("Rayleigh α (mass)", alpha)
            beta = ExprEdit(self.win, lambda: st.rayleigh_beta, lambda x: setattr(st, "rayleigh_beta", x or "0"))
            beta.setToolTip("Rayleigh damping C = αM + βK, stiffness-proportional part β (s): a damping ratio βω/2, "
                            "strongest at high frequencies")
            t.addRow("Rayleigh β (stiffness)", beta)
        elif not st.field_problem:
            n = QSpinBox()
            n.setRange(1, 1000)
            n.setValue(st.num_eigs)
            n.setToolTip("How many eigenfrequencies to compute at each sweep point")
            n.valueChanged.connect(lambda v: (setattr(st, "num_eigs", v), self.win.mark_dirty()))
            t.addRow("Number of solutions", n)
            around = ExprEdit(self.win, lambda: st.search_freq, lambda x: setattr(st, "search_freq", x or "0[GHz]"))
            around.setToolTip("The frequency the solutions are searched around: 0 finds the lowest modes, a higher "
                              "value the modes closest to it. Units are allowed, e.g. 5[GHz]")
            t.addRow("Around", around)
        f = self.form("Configuration")
        summary = problem_summary(self.p)
        if st.field_problem:
            nodes = (("sweep", "Variable sweep"),)
        elif st.coherent:
            nodes = (("sweep", "Variable sweep"), ("kmesh", "Wavevector mesh"))
        else:
            nodes = (("sweep", "Variable sweep"), ("kpath", "Wavevector sweep"),
                     ("fourier", "Fourier components"))
        for key, label in nodes:
            b = QPushButton(f"{label}: {summary[key] or 'off'}")
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.clicked.connect(lambda _=False, k=key: self.win.select_node(k))
            f.addRow(b)
        self._solver()

    def _solver(self):
        """Linear solver; for eigenfrequencies also the Lanczos tolerance and iterations."""
        st = self.p.solid.problem
        f = solver_box(self)
        if st.field_problem:
            return
        tol = QLineEdit(f"{st.tol:g}")
        tol.setToolTip("Relative accuracy of the eigenvalues; 0 = machine precision")

        def set_tol():
            try:
                st.tol = max(0.0, float(tol.text()))
                tol.setStyleSheet("")
                self.win.mark_dirty()
            except ValueError:
                tol.setStyleSheet(ERROR_STYLE)

        tol.editingFinished.connect(set_tol)
        f.addRow("Eigenvalue tolerance", tol)
        it = QSpinBox()
        it.setRange(0, 1_000_000)
        it.setSpecialValueText("default")
        it.setValue(st.maxiter)
        it.setToolTip("Maximum Lanczos iterations; default = ARPACK's")
        it.valueChanged.connect(lambda v: (setattr(st, "maxiter", v), self.win.mark_dirty()))
        f.addRow("Max. iterations", it)
        from ...fem import eigen
        f.addRow(muted_label(f"Eigenvalues are found with shift-invert Lanczos (ARPACK); problems below "
                             f"{eigen.DENSE_LIMIT} DOFs use a dense eigensolver."))


class KPathPanel(SweepPanel):
    title = "Wavevector sweep"
    help = ("Inner loop: the Bloch wavevector follows a path through the Brillouin zone (the phonon "
            "dispersion), or takes exactly the values of a list — for the directions of an experiment, which "
            "are not round fractions of the zone. Both are in reduced coordinates: fractions of the "
            "reciprocal lattice vectors b1, b2, b3 (b_i · a_j = 2π δ_ij; for a rectangular cell "
            "b = 2π/period along each axis).")
    attr = "kpath"

    def holder(self):
        return self.p.solid.problem

    def build(self):
        st = self.p.solid.problem
        self._draft = st.kpath or default_kpath(self.p)
        kp = self._draft
        self.enable_box("Enable wavevector sweep", kp)
        if not self.p.lattice.is_periodic:
            self.outer.addWidget(muted_label("No periodic axes yet: enable Periodicity under Solid Mechanics."))
        f = self.form()
        mode = QComboBox()
        mode.addItem("Path through the zone", "path")
        mode.addItem("List of wavevectors", "list")
        mode.setCurrentIndex(max(0, mode.findData(kp.mode)))
        mode.currentIndexChanged.connect(
            lambda i: (setattr(kp, "mode", mode.itemData(i)), self.changed(),
                       self.win.rebuild_tree(("kpath", None))))
        f.addRow("Wavevectors", mode)
        if kp.mode == "list":
            self._list_box(kp)
            self._bz_sketch()
            return
        box = QGroupBox("High-symmetry points")
        bl = QVBoxLayout(box)
        self.ktable = QTableWidget(0, 4)
        self.ktable.setHorizontalHeaderLabels(["Point", "b1 (x)", "b2 (y)", "b3 (z)"])
        self.ktable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ktable.verticalHeader().setVisible(False)
        self.ktable.setMaximumHeight(170)
        for name, vec in kp.points.items():
            self._add_kpoint(name, vec)
        self.ktable.itemChanged.connect(self._sync_kpoints)
        bl.addWidget(self.ktable)
        brow = QHBoxLayout()
        for text, fn in (("Add point", lambda: self._add_kpoint("P", [0, 0, 0]) or self._sync_kpoints()),
                         ("Remove point", self._remove_kpoint), ("Standard path", self._standard_path)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            brow.addWidget(b)
        brow.addStretch(1)
        bl.addLayout(brow)
        bl.addWidget(muted_label("Name the Γ point 'G' (shown as Γ in plots)."))
        self.outer.addWidget(box)

        # one line: Path [G, X, M, G] with [12] points per segment
        row = QHBoxLayout()
        row.addWidget(QLabel("Path"))
        path = QLineEdit(", ".join(kp.path))
        path.setToolTip("Point names from the table, separated by commas")
        path.editingFinished.connect(lambda: (setattr(kp, "path", [s.strip() for s in path.text().split(",") if s.strip()]),
                                              self.changed(), self._update_sketch()))
        row.addWidget(path, 1)
        row.addWidget(QLabel("with"))
        npts = QSpinBox()
        npts.setRange(1, 1000)
        npts.setValue(kp.n)
        npts.valueChanged.connect(lambda v: (setattr(kp, "n", v), self.changed()))
        row.addWidget(npts)
        row.addWidget(QLabel("points per segment"))
        self.outer.addLayout(row)
        self._bz_sketch()  # the Brillouin-zone sketch last, below all settings

    def _list_box(self, kp):
        """The explicit wavevectors: one row each, in reduced coordinates (expressions allowed)."""
        box = QGroupBox("Wavevectors")
        bl = QVBoxLayout(box)
        self.ktable = QTableWidget(0, 3)
        self.ktable.setHorizontalHeaderLabels(["b1", "b2", "b3"])
        self.ktable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ktable.setMaximumHeight(220)
        for point in kp.k:
            self._add_kvector(point)
        self.ktable.itemChanged.connect(self._sync_kvectors)
        bl.addWidget(self.ktable)
        brow = QHBoxLayout()
        for text, fn in (("Add wavevector", lambda: self._add_kvector(["0", "0", "0"]) or self._sync_kvectors()),
                         ("Remove wavevector", self._remove_kvector)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            brow.addWidget(b)
        brow.addStretch(1)
        bl.addLayout(brow)
        bl.addWidget(muted_label("One eigenvalue solve per wavevector, per value of the variable sweep. "
                                 "Expressions are allowed, so an angle can be swept: "
                                 "0.4*cos(th[deg]), 0.4*sin(th[deg]), 0."))
        self.outer.addWidget(box)

    def _add_kvector(self, point):
        self.ktable.blockSignals(True)
        r = self.ktable.rowCount()
        self.ktable.insertRow(r)
        for c in range(3):
            self.ktable.setItem(r, c, QTableWidgetItem(str(point[c]) if c < len(point) else "0"))
        self.ktable.blockSignals(False)

    def _remove_kvector(self):
        r = self.ktable.currentRow()
        if r >= 0:
            self.ktable.removeRow(r)
            self._sync_kvectors()

    def _sync_kvectors(self, *_):
        points = []
        for r in range(self.ktable.rowCount()):
            items = [self.ktable.item(r, c) for c in range(3)]
            points.append([(it.text().strip() or "0") if it is not None else "0" for it in items])
        self._draft.k = points
        self.changed()
        self._update_sketch()

    def _standard_path(self):
        std = default_kpath(self.p, enabled=self._draft.enabled)
        self._draft.points, self._draft.path = std.points, std.path
        self.changed()
        self.win.rebuild_tree(("kpath", None))

    def _bz_sketch(self):
        """Small drawing of the 2D Brillouin zone with the wavevectors of the sweep on it. The canvas is
        kept so that edits can repaint it in place (`_update_sketch`)."""
        from ... import lattice as lat
        self._sketch = self._recip = self._zone = None
        try:
            recip = lat.reciprocal(lat.slots(self.p.lattice, self.p.evaluator()))
        except Exception:
            return
        b = [v for v in recip if v is not None]
        if len(b) < 2 or abs(b[0][2]) > 0 or abs(b[1][2]) > 0:
            return
        self._recip, self._zone = recip, (b[0], b[1])
        self._sketch = zone_sketch(self.outer, self._zone, self._draw_wavevectors)

    def _update_sketch(self):
        """Draw the sketch again where it is, so a wavevector moves as it is typed or added."""
        if getattr(self, "_sketch", None) is not None:
            update_sketch(self._sketch, self._zone, self._draw_wavevectors)

    def _draw_wavevectors(self, ax):
        """The sweep on the zone: the named points with the path through them, or the listed wavevectors.
        Read afresh at every repaint; an entry that does not (yet) evaluate is left out rather than
        emptying the sketch while it is being typed."""
        kp = self._draft

        def pos(reduced):
            return sum((r * b for r, b in zip(reduced, self._recip) if b is not None), np.zeros(3))[:2]

        if kp.mode == "list":
            ev = self.p.evaluator()
            for point in kp.k:
                try:
                    x, y = pos(ev.vector(point))
                except Exception:
                    continue
                ax.plot(x, y, "o", color=theme.ACCENT, ms=5, zorder=4)
            return
        path = [pos(kp.points[n]) for n in kp.path if n in kp.points]
        if len(path) > 1:
            xy = np.array(path)
            ax.plot(xy[:, 0], xy[:, 1], color=theme.ACCENT, lw=3.0, solid_capstyle="round", zorder=2)
            ax.plot(xy[:, 0], xy[:, 1], color=theme.shore_line(), lw=1.0, solid_capstyle="round", zorder=3)
        for name in kp.points:
            x, y = pos(kp.points[name])
            ax.plot(x, y, "o", color=theme.shore_mark(), ms=3.4, zorder=4)
            ax.annotate(pretty_point(name), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=9,
                        color=theme_text())

    def _add_kpoint(self, name, vec):
        self.ktable.blockSignals(True)
        r = self.ktable.rowCount()
        self.ktable.insertRow(r)
        self.ktable.setItem(r, 0, QTableWidgetItem(name))
        for c in range(3):
            self.ktable.setItem(r, c + 1, QTableWidgetItem(f"{vec[c]:g}"))
        self.ktable.blockSignals(False)

    def _remove_kpoint(self):
        r = self.ktable.currentRow()
        if r >= 0:
            self.ktable.removeRow(r)
            self._sync_kpoints()

    def _sync_kpoints(self, *_):
        pts = {}
        for r in range(self.ktable.rowCount()):
            try:
                name = self.ktable.item(r, 0).text().strip()
                pts[name] = [float(self.ktable.item(r, c).text()) for c in (1, 2, 3)]
            except (ValueError, AttributeError):
                continue
        self._draft.points = pts
        self.changed()
        self._update_sketch()


class KMeshPanel(SweepPanel):
    title = "Wavevector mesh"
    help = ("Inner loop: instead of a path, the Bloch wavevector covers a triangular mesh of the Brillouin "
            "zone, so that every band becomes a surface ω(k) over the zone. Meshing the irreducible wedge "
            "(an eighth of a square zone, a twelfth of a hexagonal one) assumes the contents of the unit "
            "cell share the point symmetry of the lattice — true for an empty cell or a centred hole, not "
            "for a tilted or off-centre inclusion; mesh the whole zone if in doubt.")
    attr = "kmesh"

    def holder(self):
        return self.p.solid.problem

    def build(self):
        st = self.p.solid.problem
        self._draft = st.kmesh or default_kmesh(self.p)
        km = self._draft
        self.enable_box("Enable wavevector mesh", km)
        if not self.p.lattice.is_periodic:
            self.outer.addWidget(muted_label("No periodic axes yet: enable Periodicity under Solid Mechanics."))
        f = self.form("Mesh of the zone")
        region = QComboBox()
        region.addItem("Irreducible wedge", "irreducible")
        region.addItem("Whole zone", "full")
        region.setCurrentIndex(max(0, region.findData(km.region)))
        region.currentIndexChanged.connect(
            lambda i: (setattr(km, "region", region.itemData(i)), self.changed(),
                       self.win.rebuild_tree(("kmesh", None))))
        f.addRow("Region", region)
        n = QSpinBox()
        n.setRange(1, 400)
        n.setValue(km.n)
        n.setToolTip("Each edge of the region is divided this many times; a triangular wedge then holds "
                     "(n + 1)(n + 2) / 2 points")
        n.valueChanged.connect(lambda v: (setattr(km, "n", v), self.changed(),
                                          self.win.rebuild_tree(("kmesh", None))))
        f.addRow("Divisions per edge", n)
        counts = kmesh_counts(self.p, km)
        if counts is None:
            self.outer.addWidget(muted_label("The mesh needs a lattice periodic in exactly two directions; "
                                             "a lattice with no standard wedge must use the whole zone."))
        else:
            self.outer.addWidget(muted_label(f"{counts[0]} wavevectors, {counts[1]} triangles — one "
                                             f"eigenvalue solve per wavevector, per value of the variable "
                                             f"sweep."))
        self._mesh_sketch(km)

    def _mesh_sketch(self, km: KMesh):
        """The zone with the meshed region drawn in it."""
        from ... import lattice as lat
        try:
            ev = self.p.evaluator()
            recip = [v for v in lat.reciprocal(lat.slots(self.p.lattice, ev)) if v is not None]
            if len(recip) != 2 or abs(recip[0][2]) > 0 or abs(recip[1][2]) > 0:
                return
            kind = lat.classify(self.p.lattice, ev)
            shown = KMesh(n=min(km.n, 12), region=km.region)
            reduced, tris = lat.zone_mesh(*recip, shown.n, shown.region, kind)
        except Exception:
            return

        def draw(ax):
            xy = reduced[:, :1] * recip[0][:2] + reduced[:, 1:] * recip[1][:2]
            ax.tripcolor(xy[:, 0], xy[:, 1], tris, facecolors=np.zeros(len(tris)), cmap=accent_fill(),
                         vmin=0, vmax=1, zorder=2)                   # the meshed region in the accent
            ax.triplot(xy[:, 0], xy[:, 1], tris, color=theme.shore_line(), lw=0.6, alpha=0.8, zorder=3)
            for name, corner in zip(lat.WEDGE_LABELS.get(kind, ()) if km.region == "irreducible" else (),
                                    lat.WEDGES.get(kind, ())):
                x, y = corner[0] * recip[0][:2] + corner[1] * recip[1][:2]
                ax.plot(x, y, "o", color=theme.shore_mark(), ms=3.4, zorder=4)
                ax.annotate(pretty_point(name), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=9,
                            color=theme_text())
            if km.n > shown.n:
                ax.set_title(f"n = {shown.n} shown", fontsize=8, color=theme_text())

        zone_sketch(self.outer, (recip[0], recip[1]), draw)


class FourierPanel(SweepPanel):
    title = "Fourier components"
    help = ("For every mode, the complex amplitudes of its plane-wave components over the unit cell: "
            "ŝ(G) = (1/V) ∫ f(r) e^{−i(k+G)·r} dV for the three displacement components and the six strains, "
            "at the reciprocal vectors G = h₁b₁ + h₂b₂ + h₃b₃ of a grid of orders. They go to the run folder "
            "(fourier.csv, results.npz), not to a plot.")
    attr = "fourier"

    def holder(self):
        return self.p.solid.problem

    def build(self):
        st = self.p.solid.problem
        self._draft = st.fourier or default_fourier(self.p)
        fo = self._draft
        self.enable_box("Write the Fourier components of every mode", fo)
        if not self.p.lattice.is_periodic:
            self.outer.addWidget(muted_label("No periodic axes yet: the components are over the unit cell of "
                                             "a lattice (enable Periodicity under Solid Mechanics)."))
        f = self.form("Orders of the reciprocal vectors")
        for i, (label, periodic) in enumerate(fourier_labels(self.p)):
            n = QSpinBox()
            n.setRange(0, 20)
            n.setValue(int(fo.orders[i]))
            n.setEnabled(periodic)
            n.setToolTip("Highest order |h| along this reciprocal vector; 0 keeps only h = 0. A direction "
                         "that is not periodic has no reciprocal vector and stays at 0.")
            n.valueChanged.connect(lambda v, i=i: (self._set_order(i, v), self.changed(),
                                                   self.win.rebuild_tree(("fourier", None))))
            f.addRow(label, n)
        self.outer.addWidget(muted_label(
            f"{fourier_count(fo)} reciprocal vectors × 9 fields per mode. Modes are mass-normalised "
            f"(∫ρ|u|² dV = 1 over the cell, phase rotated so that Σ u·u > 0), which is what makes the "
            f"components of different modes comparable."))
        self.outer.addWidget(muted_label(
            "Where modes are degenerate their eigenvectors are an arbitrary basis of the subspace, so the "
            "components of one of them mean nothing while their sum over the group does: the run flags "
            "those modes in fourier_degeneracy.csv."))

    def _set_order(self, i: int, value: int):
        orders = [int(o) for o in self._draft.orders]
        orders += [0] * (3 - len(orders))
        orders[i] = int(value)
        self._draft.orders = orders


class DisplacementPanel(Panel):
    title = "Displacement field"
    help = ("3D mode shape: the structure deformed by the mode's displacement. |u| is the vibration amplitude; "
            "ux, uy, uz are the components at the displayed phase. Animate cycles the phase to show the motion.")

    def __init__(self, win):
        if win.project.solid.problem.field_problem:
            self.help = ("The structure deformed by the stationary displacement, coloured by the displacement or "
                         "the von Mises stress. Pick the variable value of a sweep here or click a point in a plot.")
        super().__init__(win)

    @property
    def static(self) -> bool:
        return self.p.solid.problem.field_problem

    def build(self):
        r, sel = self.win.results, self.win.selection
        f = self.form("Dataset")
        if self.static:
            if r is None or not self.win.has_field_results:
                f.addRow(muted_label("No results yet. Configure the Problem and press Run; then pick the variable "
                                     "value here (or click a point in a plot)."))
                return self._settings()
            f.addRow(muted_label(self.win.results_summary()))
            if r.outer_name:
                outer = QComboBox()
                outer.addItems(outer_labels(self.win))
                outer.setCurrentIndex(min(self.win.selection["i"], len(r.outer_values) - 1))
                outer.currentIndexChanged.connect(lambda i: (self.win.selection.update(i=i), self.win.show_field()))
                f.addRow("Variable sweep", outer)
            self._settings()
            self.win.show_field()
            return
        if r is None or self.win.has_field_results:
            f.addRow(muted_label("No results yet. Configure the Problem and press Run; then pick the mode here "
                                 "(or click a point in a plot)."))
            return self._settings()
        n_outer, n_inner, n_eigs = r.freqs.shape
        f.addRow(muted_label(self.win.results_summary()))
        if r.outer_name:
            outer = QComboBox()
            outer.addItems(outer_labels(self.win))
            outer.setCurrentIndex(sel["i"])
            outer.currentIndexChanged.connect(lambda i: self._select(i=i))
            f.addRow("Variable sweep", outer)
        if n_inner > 1:
            row = QHBoxLayout()
            self.k_slider = QSlider(Qt.Orientation.Horizontal)
            self.k_slider.setRange(0, n_inner - 1)
            self.k_slider.setValue(sel["j"])
            self.k_label = QLabel()
            self.k_label.setMinimumWidth(150)
            self.k_slider.valueChanged.connect(self._k_moved)
            self.k_slider.sliderReleased.connect(lambda: self._select(j=self.k_slider.value()))
            row.addWidget(self.k_slider, 1)
            row.addWidget(self.k_label)
            f.addRow("Wavevector", row)
            self._update_k_label(sel["j"])
        mode = QComboBox()
        self.mode_box = mode
        self._fill_modes()
        mode.setCurrentIndex(sel["b"])
        mode.currentIndexChanged.connect(lambda b: self._select(b=b) if b >= 0 else None)
        f.addRow("Mode", mode)
        self._settings()
        self.button_row(("Export animation…", self.win.export_animation))
        self.win.show_field()

    def _settings(self):
        """Quantity (what the colours show), Plot (deformation, overlays) and Style (colour scale)."""
        q = self.form("Quantity")
        if self.static:
            field_quantity_rows(self, q, "Any quantity of the solution on the deformed structure (displacements "
                                         "in the length unit, stresses in MPa), e.g. mises, uz or sxx; the "
                                         "deformation is exaggerated by the scale below.")
        else:
            comp = QComboBox()
            for c, label in (("|u|", "|u|  amplitude"), ("ux", "ux"), ("uy", "uy"), ("uz", "uz")):
                comp.addItem(label, c)
            comp.setCurrentIndex(max(0, comp.findData(self.win.field_component)))
            comp.currentIndexChanged.connect(lambda i: self.win.set_field_component(comp.itemData(i)))
            q.addRow("Colour by", comp)
            q.addRow(muted_label("Displacement of the mode, normalised to a largest amplitude of 1; ux, uy, uz at "
                                 "the displayed phase."))

        d = self.form("Style")
        st, viewer = self.win.field_style, self.win.viewer
        cmap = QComboBox()
        from ..viewer import COLORMAPS
        cmap.addItems(COLORMAPS + ([st["cmap"]] if st["cmap"] not in COLORMAPS else []))
        cmap.setCurrentText(st["cmap"])
        cmap.currentTextChanged.connect(lambda c: self.win.set_field_style(cmap=c))
        d.addRow("Colour scale", cmap)
        scale = QSlider(Qt.Orientation.Horizontal)
        scale.setRange(0, 40)
        scale.setValue(int(self.win.deform_scale * 100))
        scale.setToolTip("Largest displacement as a percentage of the model size")
        scale.valueChanged.connect(lambda v: self.win.set_deform_scale(v / 100))
        d.addRow("Deformation scale", scale)

        grid = QGridLayout()
        checks = [("Colour bar", st["colorbar"], lambda on: self.win.set_field_style(colorbar=on)),
                  ("Title", st["title"], lambda on: self.win.set_field_style(title=on)),
                  ("Undeformed outline", st["outline"], lambda on: self.win.set_field_style(outline=on)),
                  ("Rulers", viewer.scale_action.isChecked(), viewer.set_rulers),
                  ("Axes (xyz)", viewer.show_triad, viewer.set_triad)]
        if not self.static:
            checks.append(("Animate", self.win.mode_animate, self.win.set_mode_animate))
        for n, (label, on, fn) in enumerate(checks):
            cb = QCheckBox(label)
            cb.setChecked(on)
            cb.toggled.connect(fn)
            grid.addWidget(cb, n // 2, n % 2)
        d.addRow(grid)

    def _fill_modes(self):
        r, sel = self.win.results, self.win.selection
        unit = self.p.frequency_unit
        self.mode_box.blockSignals(True)
        self.mode_box.clear()
        for b, fr in enumerate(r.freqs[sel["i"], sel["j"]]):
            self.mode_box.addItem(f"{b + 1}:  {fr / units.UNITS[unit]:.5g} {unit}")
        self.mode_box.setCurrentIndex(sel["b"])
        self.mode_box.blockSignals(False)

    def _update_k_label(self, j):
        r = self.win.results
        k_scale, k_unit = units.axis_scale(units.INV_LENGTH, self.p.length_unit)
        names = {round(pos, 12): lbl for pos, lbl in r.ticks}
        name = names.get(round(float(r.x[j]), 12), "")
        kv = ", ".join(f"{v * k_scale:.3g}" for v in r.k[self.win.selection["i"], j])
        self.k_label.setText(f"{pretty_point(name) + ' ' if name else ''}({kv}) {k_unit}")

    def _k_moved(self, j):
        self._update_k_label(j)
        if not self.k_slider.isSliderDown():  # keyboard / click: select immediately
            self._select(j=j)

    def _select(self, **kw):
        self.win.selection.update(kw)
        if "b" not in kw:
            self._fill_modes()
        self.win.show_field()


# ---------------------------------------------------------------------------
# Solid mechanics conditions
LOAD_TITLES = {"area": "Force per area", "total": "Total force", "pressure": "Pressure"}


SOLID_HELP = {
    "fixed": "The selected faces are clamped.",
    "displacement": "The selected faces are moved by a given amount; an empty component is left free.",
    "load": "Force on the selected faces. Stationary and frequency-domain problems only — an eigenfrequency "
            "problem has no loads.",
}

SOLID_FORMALISM = {
    "fixed": equations(
        ("u = 0", "The selected faces cannot move. Implementation: all three displacement components of every "
         "node on those boundary faces are removed from the system, so they are never solved for. Where "
         "selections overlap, the later condition in the list wins."),
    ),
    "displacement": equations(
        ("u<sub>i</sub> = g<sub>i</sub>", "Prescribed displacement, component by component; an empty component "
         "is free (only u<sub>x</sub> given: the face is moved along x and slides freely in y and z). "
         "Implementation: the given components of every node on those faces are held at g. In stationary and "
         "frequency-domain problems the prescribed values move to the right-hand side; an eigenfrequency problem "
         "holds them at zero, since a mode shape has no amplitude of its own."),
    ),
    "load": equations(
        ("t = F / A", "Force per area (traction, Pa) on the selected faces."),
        ("t = F<sub>total</sub> / A<sub>selected</sub>", "A total force (N), spread uniformly over the area "
         "the selection covers."),
        ("t = −p n", "A pressure p on the surface, pushing against its outward normal n (negative = suction)."),
        "Implementation: ∫ v·t dS over the selected boundary triangles, added to the right-hand side. Loads "
        "from several conditions on the same face add up.",
    ),
}


class SolidConditionPanel(Panel):
    def __init__(self, win, index):
        self.index = index
        self.c = win.project.solid.conditions[index]
        kind = SOLID_CONDITION_TITLES.get(self.c.type, self.c.type)
        self.title = f"{kind}: {self.c.name}" if self.c.name else kind
        self.help = SOLID_HELP.get(self.c.type, "")
        self.formalism = SOLID_FORMALISM.get(self.c.type, "")
        super().__init__(win)

    def build(self):
        c = self.c
        bypass_row(self, c, ("solid_cond", self.index), "condition", lambda: delete_condition(self.win, self.index))
        f = self.form()
        name = QLineEdit(c.name)
        name.editingFinished.connect(lambda: (setattr(c, "name", name.text().strip()), self.win.mark_dirty(),
                                              self.win.rebuild_tree(("solid_cond", self.index), refresh_panel=False)))
        f.addRow("Name", name)
        if c.type == "displacement":
            for i, axis in enumerate(AXES):
                def setter(t, i=i):
                    c.displacement[i] = t
                    self.win.physics_changed()
                f.addRow(f"u{axis}", ExprEdit(self.win, lambda i=i: c.displacement[i], setter, optional="free"))
            f.addRow(muted_label("Empty = free component."))
        elif c.type == "load":
            typ = QComboBox()
            for key in LOAD_TYPES:
                typ.addItem(LOAD_TITLES[key], key)
            typ.setCurrentIndex(max(0, typ.findData(c.load_type)))
            typ.currentIndexChanged.connect(lambda i: (setattr(c, "load_type", typ.itemData(i)),
                                                       self.win.physics_changed(), self._refresh()))
            f.addRow("Load type", typ)
            if c.load_type == "pressure":
                f.addRow("Pressure", ExprEdit(self.win, lambda: c.pressure,
                                              lambda t: setattr(c, "pressure", t or "0[Pa]")))
            else:
                unit = "N" if c.load_type == "total" else "Pa"
                for i, axis in enumerate(AXES):
                    def setter(t, i=i):
                        c.load[i] = t or "0"
                    f.addRow(f"F{axis} ({unit})", ExprEdit(self.win, lambda i=i: c.load[i], setter))
            if not self.p.solid.problem.field_problem:
                f.addRow(muted_label("Loads act in stationary problems only (Problem → Problem type)."))
        self.outer.addWidget(selection_box(self, c))

    def _refresh(self):
        self.win.rebuild_tree(("solid_cond", self.index))


# ---- actions (tree context menu, panel buttons) ------------------------------------------
def add_condition(win, typ: str):
    c = SolidCondition(typ, faces=[])
    if typ == "load":
        c.load_type, c.pressure = "pressure", "1[MPa]"
    win.project.solid.conditions.append(c)
    win.mark_dirty()
    win.rebuild_tree(("solid_cond", len(win.project.solid.conditions) - 1))


def delete_condition(win, index: int):
    del win.project.solid.conditions[index]
    win.mark_dirty()
    win.rebuild_tree(("physics", None))


def set_problem_type(win, typ: str):
    """Eigenfrequency / stationary (solid mechanics). Results are cleared; a stationary problem gets a
    displacement field plot unless it has one."""
    from ...physics.solid.plots import field_plot
    st = win.project.solid.problem
    if typ == st.type:
        return
    st.type = typ
    p = win.project
    if typ == "frequency" and st.frequency == "f" and "f" not in p.variables:
        p.variables["f"] = "1[MHz]"
    if typ == "conductance":
        if st.kmesh is None:
            st.kmesh = default_kmesh(p)
        from ...physics.solid.plots import band_map, band_surfaces, conductance_plot, spectrum_plot
        for n, (kind, make) in enumerate((("kmap", band_map), ("surfaces", band_surfaces),
                                          ("spectrum", spectrum_plot), ("conductance", conductance_plot))):
            if not any(pl.kind == kind for pl in p.plots):      # the plots a zone run is for
                p.plots.insert(n, make())
    if not any(pl.kind == "field" for pl in win.project.plots):
        win.project.plots.append(field_plot())
    win.clear_results()
    win.problem_changed()
    win.rebuild_tree(("problem", None))


def clear_face_conditions(win, axis: str):
    """Periodic faces cannot carry other conditions: drop them when periodicity is enabled."""
    ph = win.project.solid
    for kind in ("symmetry", "antisymmetry"):
        lst = getattr(ph, kind)
        for face in [f for f in lst if f[0] == axis]:
            lst.remove(face)
            win.log(f"Removed {kind} on {face}: the face is now periodic")
    for c in ph.conditions:
        if c.on:
            continue
        for face in [f for f in c.faces if f in FACES and f[0] == axis]:
            c.faces.remove(face)
            win.log(f"Removed face {face} from '{c.name or c.type}': the face is now periodic")
