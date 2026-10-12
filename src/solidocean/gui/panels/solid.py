"""Solid mechanics panels: physics overview, periodicity, symmetry, boundary conditions, study
(eigenfrequency / stationary), solutions, wavevector sweep, solver and the displacement field."""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ... import units
from ...physics.solid.model import LOAD_TYPES, SOLID_CONDITION_TITLES, STUDY_TITLES, KPath, SolidCondition
from ...project import AXES, Feature
from ...selection import FACES, selection_text
from .common import (ERROR_STYLE, ExprEdit, Panel, VectorEdit, field_quantity_rows, muted_label, outer_labels,
                     pretty_point, selection_box, theme_text)
from .study import SweepPanel, solver_box, sweep_summary
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
    return f"{c.name or SOLID_CONDITION_TITLES.get(c.type, c.type)} ({selection_text(c)})"


class SolidPanel(Panel):
    title = "Solid Mechanics"
    help = ("Linear elasticity: eigenfrequencies of  ∇·(C:∇u) + ρω²u = 0, or the stationary displacement under "
            "loads,  ∇·(C:∇u) = 0  (Study → Study type). Boundaries are traction-free unless a condition applies. "
            "Fixed constraints, prescribed displacements and loads act on faces of the model or of one object "
            "(bounding-box faces or all faces, optionally only inside a box) and are highlighted in the graphics "
            "window.")

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
    help = ("Bloch-Floquet condition u(r + R) = exp(i k·R) u(r) for every lattice translation R. Rectangular cells "
            "repeat along x, y, z; hexagonal and custom lattices use lattice vectors a1, a2 (a3) and a unit-cell "
            "geometry feature. Periodic face pairs are tinted (a1/x red, a2/y green, a3/z blue).")

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
        g.addRow(VectorEdit(self.win, self.p.solid.study.k, ("kx", "ky", "kz")))

    # ---- modes -----------------------------------------------------------
    def _switch(self, mode):
        ph = self.p.lattice
        self.win.custom_lattice = mode == "custom"
        if mode == "rect":
            ph.vectors = []
            ph.periodic = {"x": "a", "y": "a"}
            for n, ft in enumerate(self.p.geometry):  # a lattice cell becomes the matching block
                if ft.type == "cell":
                    self.p.geometry[n] = Feature("block", ft.name, ft.op, ft.material, {
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
        if self.p.solid.study.kpath is not None:  # the old path's points belong to the old lattice
            self.p.solid.study.kpath = default_kpath(self.p, enabled=self.p.solid.study.kpath.enabled)
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
        has_cell = any(ft.type == "cell" and ft.enabled for ft in self.p.geometry)
        if has_cell:
            self.outer.addWidget(muted_label("The geometry uses a 'cell' feature: the unit cell spanned by a1, a2."))
            return
        self.outer.addWidget(muted_label(
            "The geometry must be one unit cell of this lattice. A 'cell' feature (parallelogram spanned by "
            "a1, a2, extruded along z) builds it; holes and pillars go inside."))
        self.button_row(("Use lattice cell as base geometry", lambda: use_lattice_cell(self.win)))

    def _set_period(self, ax, text):
        if ax in self.p.lattice.periodic:
            self.p.lattice.periodic[ax] = text
            self.win.physics_changed()


class SymmetryPanel(Panel):
    title = "Symmetry"
    help = ("Symmetry plane: zero normal displacement, free tangential motion (mirror-symmetric modes). "
            "Antisymmetry: zero tangential displacement, free normal motion (mirror-antisymmetric modes). "
            "Modelling half a structure with each in turn gives all modes of the full structure.")

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


def study_summary(project) -> dict[str, str]:
    """Short text per study node (empty string = inactive)."""
    st = project.solid.study
    kp = st.active_kpath
    return {**sweep_summary(project.study), "study_type": STUDY_TITLES.get(st.type, st.type),
            "solutions": str(st.num_eigs), "kpath": "–".join(pretty_point(p) for p in kp.path) if kp else ""}


STUDY_HELP = {
    "eigenfrequency": "For every value of the variable sweep (outer loop), the wavevector sweep (inner loop) is "
                      "computed and at each point the requested eigenfrequencies are found. Geometry and mesh are "
                      "rebuilt only when a variable they depend on changes; a wavevector sweep reuses them. Loads "
                      "are ignored; prescribed displacements hold their components at zero.",
    "stationary": "The static displacement, strain and stress under the boundary loads and prescribed "
                  "displacements, for every value of the variable sweep. The model must be held against "
                  "rigid-body motion (fixed constraints, prescribed displacements or symmetry planes). Periodic "
                  "models give the periodic solution (wavevector 0).",
    "frequency": "The harmonic response to loads and prescribed displacements oscillating at one frequency: "
                 "(K (1 + iη) − ω² M) u = f, the loads and displacements being the amplitudes of the drive. The "
                 "result is the complex displacement amplitude (use abs(uz) for an amplitude). A loss factor η "
                 "damps it: for every material (below) and per material (property eta, which may depend on x, y, "
                 "z: graded absorbing regions at the ends of a long structure keep waves from reflecting). Sweep "
                 "the frequency with a variable (f) in Study → Variable sweep.",
}


class SolidStudyPanel(Panel):
    action = ("Run ▶\ufe0e", "compute")

    def __init__(self, win):
        st = win.project.solid.study
        self.title = f"Study: {STUDY_TITLES.get(st.type, st.type)}"
        self.help = STUDY_HELP.get(st.type, "")
        super().__init__(win)

    def build(self):
        st = self.p.solid.study
        t = self.form()
        typ = QComboBox()
        for key, title in STUDY_TITLES.items():
            typ.addItem(title, key)
        typ.setCurrentIndex(max(0, typ.findData(st.type)))
        typ.currentIndexChanged.connect(lambda i: set_study_type(self.win, typ.itemData(i)))
        t.addRow("Study type", typ)
        if st.harmonic:
            freq = ExprEdit(self.win, lambda: st.frequency, lambda x: setattr(st, "frequency", x or "f"))
            freq.setToolTip("Drive frequency; a variable (f) can be swept: Study → Variable sweep")
            t.addRow("Frequency", freq)
            loss = ExprEdit(self.win, lambda: st.loss, lambda x: setattr(st, "loss", x or "0"))
            loss.setToolTip("Loss factor η of every material: stiffness C (1 + iη); materials may add their own "
                            "(property eta)")
            t.addRow("Loss factor η", loss)
        f = self.form("Configuration")
        summary = study_summary(self.p)
        nodes = (("sweep", "Variable sweep"), ("solver", "Solver")) if st.field_study else \
            (("solutions", "Solutions"), ("sweep", "Variable sweep"), ("kpath", "Wavevector sweep"),
             ("solver", "Solver"))
        for key, label in nodes:
            b = QPushButton(f"{label}: {summary[key] or 'off'}")
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.clicked.connect(lambda _=False, k=key: self.win.select_node(k))
            f.addRow(b)


class SolutionsPanel(Panel):
    title = "Solutions"
    help = ("How many eigenfrequencies to compute at each sweep point, and the frequency they are searched "
            "around: 0 finds the lowest modes; a higher value finds the modes closest to it. Units are "
            "allowed, e.g. 5[GHz].")

    def build(self):
        st = self.p.solid.study
        row = QHBoxLayout()
        row.addWidget(QLabel("Find"))
        n = QSpinBox()
        n.setRange(1, 1000)
        n.setValue(st.num_eigs)
        n.valueChanged.connect(lambda v: (setattr(st, "num_eigs", v), self.win.study_changed()))
        row.addWidget(n)
        row.addWidget(QLabel("solutions around"))
        freq = ExprEdit(self.win, lambda: st.search_freq, lambda t: setattr(st, "search_freq", t or "0"),
                        show_value=False)
        row.addWidget(freq, 1)
        row.addWidget(QLabel("Hz"))
        self.outer.addLayout(row)


class KPathPanel(SweepPanel):
    title = "Wavevector sweep"
    help = ("Inner loop: the Bloch wavevector follows a path through the Brillouin zone, giving the phonon "
            "dispersion. Points are in reduced coordinates: fractions of the reciprocal lattice vectors "
            "b1, b2, b3 (b_i · a_j = 2π δ_ij; for a rectangular cell b = 2π/period along each axis).")
    attr = "kpath"

    def holder(self):
        return self.p.solid.study

    def build(self):
        st = self.p.solid.study
        self._draft = st.kpath or default_kpath(self.p)
        kp = self._draft
        self.enable_box("Enable wavevector sweep", kp)
        if not self.p.lattice.is_periodic:
            self.outer.addWidget(muted_label("No periodic axes yet: enable Periodicity under Solid Mechanics."))
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
                                              self.changed()))
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

    def _standard_path(self):
        std = default_kpath(self.p, enabled=self._draft.enabled)
        self._draft.points, self._draft.path = std.points, std.path
        self.changed()
        self.win.rebuild_tree(("kpath", None))

    def _bz_sketch(self):
        """Small drawing of the 2D Brillouin zone with the high-symmetry points and the path."""
        from ... import lattice as lat
        try:
            recip = lat.reciprocal(lat.slots(self.p.lattice, self.p.evaluator()))
        except Exception:
            return
        b = [v for v in recip if v is not None]
        if len(b) < 2 or abs(b[0][2]) > 0 or abs(b[1][2]) > 0:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(2.6, 2.4), layout="constrained")
        fig.patch.set_alpha(0)
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        ax.axis("off")
        bz = lat.brillouin_zone_2d(b[0], b[1])
        poly = np.vstack([bz, bz[:1]])
        ax.fill(poly[:, 0], poly[:, 1], color="#cfd8e3", alpha=0.5)
        ax.plot(poly[:, 0], poly[:, 1], color="#8a99ab", lw=1)
        kp = self._draft

        def pos(name):
            r = kp.points[name]
            return sum((ri * bi for ri, bi in zip(r, recip) if bi is not None), np.zeros(3))[:2]

        path = [pos(n) for n in kp.path if n in kp.points]
        if len(path) > 1:
            xy = np.array(path)
            ax.plot(xy[:, 0], xy[:, 1], color="#d62728", lw=1.6)
        for name in kp.points:
            x, y = pos(name)
            ax.plot(x, y, "o", color="#d62728", ms=3)
            ax.annotate(pretty_point(name), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=9,
                        color=theme_text())
        canvas = FigureCanvasQTAgg(fig)
        canvas.setFixedHeight(190)
        canvas.setStyleSheet("background: transparent;")
        self.outer.addWidget(canvas)

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


class SolverPanel(Panel):
    title = "Solver"
    help = ("Eigenvalues are found with shift-invert Lanczos (ARPACK). Each iteration solves a sparse linear "
            "system; the direct solver used for it dominates the run time. A stationary study solves one sparse "
            "system per variable value with the same solver.")

    def build(self):
        solver_box(self)
        st = self.p.solid.study
        f = self.form("Eigenvalues")
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
        it.valueChanged.connect(lambda v: (setattr(st, "maxiter", v), self.win.mark_dirty()))
        f.addRow("Max. iterations", it)
        from ...fem import eigen
        self.outer.addWidget(muted_label(f"Problems below {eigen.DENSE_LIMIT} DOFs use a dense eigensolver."))


class DisplacementPanel(Panel):
    title = "Displacement field"
    help = ("3D mode shape: the structure deformed by the mode's displacement. |u| is the vibration amplitude; "
            "ux, uy, uz are the components at the displayed phase. Animate cycles the phase to show the motion.")

    def __init__(self, win):
        if win.project.solid.study.field_study:
            self.help = ("The structure deformed by the stationary displacement, coloured by the displacement or "
                         "the von Mises stress. Pick the variable value of a sweep here or click a point in a plot.")
        super().__init__(win)

    @property
    def static(self) -> bool:
        return self.p.solid.study.field_study

    def build(self):
        r, sel = self.win.results, self.win.selection
        f = self.form("Dataset")
        if self.static:
            if r is None or not self.win.has_field_results:
                f.addRow(muted_label("No results yet. Configure the Study and press Run; then pick the variable "
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
            f.addRow(muted_label("No results yet. Configure the Study and press Run; then pick the mode here "
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
    "fixed": "Zero displacement (clamped) on the selected faces.",
    "displacement": "Prescribed displacement on the selected faces, per component; an empty component is free "
                    "(e.g. only ux: the face is moved along x and slides freely in y and z). Eigenfrequency studies "
                    "hold the given components at zero.",
    "load": "Force on the selected faces (stationary studies; eigenfrequency studies ignore it): a force per area "
            "(traction, Pa), a total force (N) spread uniformly over the selected area, or a pressure pushing on "
            "the surface (against its outward normal; negative = suction).",
}


class SolidConditionPanel(Panel):
    def __init__(self, win, index):
        self.index = index
        self.c = win.project.solid.conditions[index]
        kind = SOLID_CONDITION_TITLES.get(self.c.type, self.c.type)
        self.title = f"{kind}: {self.c.name}" if self.c.name else kind
        self.help = SOLID_HELP.get(self.c.type, "")
        super().__init__(win)

    def build(self):
        c = self.c
        self.button_row(("Delete", lambda: delete_condition(self.win, self.index)))
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
            if not self.p.solid.study.field_study:
                f.addRow(muted_label("Loads act in stationary studies only (Study → Study type)."))
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


def set_study_type(win, typ: str):
    """Eigenfrequency / stationary (solid mechanics). Results are cleared; a stationary study gets a
    displacement field plot unless it has one."""
    from ...physics.solid.plots import field_plot
    st = win.project.solid.study
    if typ == st.type:
        return
    st.type = typ
    p = win.project
    if typ == "frequency" and st.frequency == "f" and "f" not in p.variables:
        p.variables["f"] = "1[MHz]"
    if not any(pl.kind == "field" for pl in win.project.plots):
        win.project.plots.append(field_plot())
    win.clear_results()
    win.study_changed()
    win.rebuild_tree(("study", None))


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
