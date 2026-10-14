"""Panels of the model shared by every physics: model, variables, materials, objects, mesh."""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QMenu, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ... import materials as mat
from ... import physics
from ... import units
from ...project import (ARRAY_KEY, AXES, FEATURE_TITLES, FEATURE_TYPES, OPS, ROTATION_KEY, SPACING_KEY, Feature,
                         Material)
from .. import theme
from .common import (ExprEdit, Panel, VectorEdit, display_evaluator, feature_swatch, fmt, model_box, muted_label,
                     set_value_item, show, swatch)
from .lattice import hexagonal_vectors, lattice_mode


# ---------------------------------------------------------------------------
class ModelPanel(Panel):
    title = "Model"

    # preferred units: (project attribute, label, choices, what they apply to)
    UNITS = (("length_unit", "Length", units.LENGTH_UNITS, "Lengths and wavevectors in the GUI and plots."),
             ("frequency_unit", "Frequency", units.FREQUENCY_UNITS,
              "Frequencies in the GUI, plots and dispersion.csv."),
             ("time_unit", "Time", units.TIME_UNITS, "Times in the GUI and plots."))

    def build(self):
        """Labels on top of full-width fields; the preferred units as one setting with a row per quantity."""
        name = QLineEdit(self.p.name)
        name.editingFinished.connect(lambda: (setattr(self.p, "name", name.text()), self.win.mark_dirty(),
                                              self.win.rebuild_tree()))
        desc = QPlainTextEdit(self.p.description)
        desc.setFixedHeight(6 * desc.fontMetrics().lineSpacing() + 12)   # about six lines
        desc.textChanged.connect(lambda: (setattr(self.p, "description", desc.toPlainText()), self.win.mark_dirty()))
        def heading(text, first=False):
            if not first:
                self.outer.addSpacing(6)
            self.outer.addWidget(QLabel(text))

        for n, (label, widget) in enumerate((("Name", name), ("Description", desc),
                                             ("Physics", physics_selector(self.win)))):
            heading(label, first=n == 0)
            self.outer.addWidget(widget)
        heading("Preferred units")
        row = QHBoxLayout()   # length, frequency, time side by side: the units themselves say which is which
        for key, label, choices, tip in self.UNITS:
            box = QComboBox()
            for u in choices:
                box.addItem(units.unit_label(u), u)
            box.setCurrentIndex(max(0, box.findData(getattr(self.p, key))))
            box.setToolTip(f"{label}: {tip} Inputs accept any unit, e.g. 500[nm]; values are computed and saved in "
                           "SI units.")
            box.currentIndexChanged.connect(lambda i, key=key, box=box: self.win.set_unit(key, box.itemData(i)))
            row.addWidget(box, 1)
        self.outer.addLayout(row)
        heading("File")
        self.outer.addWidget(muted_label(str(self.p.path) if self.p.path else "(not saved)"))


def physics_selector(win) -> QComboBox:
    """The physics of the model: one of the single physics (coupled setups, such as thermal expansion or Joule
    heating, will be further choices)."""
    phys = QComboBox()
    for kind, spec in physics.REGISTRY.items():
        phys.addItem(spec.title, kind)
        phys.setItemData(phys.count() - 1, spec.description, Qt.ItemDataRole.ToolTipRole)
    phys.setCurrentIndex(max(0, phys.findData(win.project.physics_kind)))
    phys.setToolTip("Physics of this model; it defines the boundary conditions and problems available. "
                    "Switching clears the results.")
    phys.currentIndexChanged.connect(lambda i: win.set_physics(phys.itemData(i)))
    return phys


class PhysicsListPanel(Panel):
    title = "Physics"
    help = ("The physics solved on the model, each with its own conditions and settings (its node below). One "
            "physics at a time for now; coupled options, such as thermal expansion (heat transfer driving solid "
            "mechanics) or Joule heating, will be further choices of the setup.")

    def build(self):
        self.outer.addWidget(physics_selector(self.win))
        g = self.form("Physics in this model")
        for spec in physics.active(self.p):
            b = QPushButton(spec.title)
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.setToolTip(spec.description)
            b.clicked.connect(lambda _=False: self.win.select_node("physics"))
            g.addRow(b)
            g.addRow(muted_label(spec.description))


class VariablesPanel(Panel):
    title = "Variables"
    help = "Global variables usable in any expression. Units in brackets: 500[nm], 165[GPa]."

    def build(self):
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Expression", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        rows = 15   # models usually have many variables: room for about 15 without scrolling
        self.table.setMinimumHeight(self.table.horizontalHeader().sizeHint().height()
                                    + rows * self.table.verticalHeader().defaultSectionSize()
                                    + 2 * self.table.frameWidth())
        self.outer.addWidget(self.table)
        for name, expr in self.p.variables.items():
            self._add_row(name, str(expr))
        self.table.itemChanged.connect(self._sync)
        self.button_row(("Add", lambda: (self._add_row("", ""), None)), ("Remove", self._remove))
        self._refresh_values()

    def _add_row(self, name, expr):
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(expr))
        v = QTableWidgetItem("")
        v.setFlags(v.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(r, 2, v)
        self.table.blockSignals(False)

    def _remove(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._sync()

    def _sync(self, *_):
        params = {}
        for r in range(self.table.rowCount()):
            n, e = self.table.item(r, 0), self.table.item(r, 1)
            if n and n.text().strip():
                params[n.text().strip()] = e.text().strip() if e else ""
        self.p.variables.clear()
        self.p.variables.update(params)
        self.win.mark_dirty()
        self._refresh_values()

    def _refresh_values(self):
        """Values of the variables; a broken one (or one that uses it) shows its own error, the others their
        values."""
        from ...expressions import Evaluator, resolve_each
        self.table.blockSignals(True)
        values, errors = resolve_each(self.p.variables)
        ev = Evaluator({k: self.p.variables[k] for k in values})     # the variables that resolve
        for r in range(self.table.rowCount()):
            n = self.table.item(r, 0)
            name = n.text().strip() if n else ""
            item = self.table.item(r, 2)
            if name in values:
                set_value_item(item, show(self.p, ev, self.p.variables[name], values[name]), True,
                               f"SI: {fmt(values[name])}")
            else:
                set_value_item(item, "error" if name else "", False, errors.get(name, ""))
        self.table.blockSignals(False)


# ---------------------------------------------------------------------------
class MaterialsPanel(Panel):
    title = "Materials"
    help = "Materials are assigned to objects. Elastic constants in Voigt notation, crystal axes along x, y, z."

    def build(self):
        self.list = QListWidget()
        self.list.addItems([m.name for m in self.p.materials])
        self.list.itemDoubleClicked.connect(lambda it: self.win.select_node("material", self.list.row(it)))
        self.outer.addWidget(self.list)
        row = self.button_row(("Remove", self._remove))
        add = QPushButton("Add material")
        menu = QMenu(add)
        for name in mat.LIBRARY:
            menu.addAction(name, lambda n=name: self._add(n, n))
        menu.addAction("Blank (isotropic)", lambda: self._add("Material", None))
        add.setMenu(menu)
        row.insertWidget(0, add)

    def _add(self, name, library):
        existing = {m.name for m in self.p.materials}
        base, i = name, 2
        while name in existing:
            name, i = f"{base}{i}", i + 1
        if library:
            model, props = mat.resolve_properties({"library": library})
        else:
            model, props = "isotropic", {"rho": "1000[kg/m^3]", "E": "100[GPa]", "nu": "0.3"}
        self.p.materials.append(Material(name, {"model": model, **props}))
        self.win.mark_dirty()
        self.win.rebuild_tree(("material", len(self.p.materials) - 1))

    def _remove(self):
        r = self.list.currentRow()
        if r >= 0:
            self.win.delete_material(r)


class MaterialPanel(Panel):
    def __init__(self, win, index):
        self.m: Material = win.project.materials[index]
        self.index = index
        self.title = f"Material: {self.m.name}"
        super().__init__(win)

    def build(self):
        name = QLineEdit(self.m.name)
        name.editingFinished.connect(lambda: self._rename(name.text().strip()))
        self.field("Name", name)
        model, props = mat.resolve_properties(self.m.props)
        self.model_box = QComboBox()
        self.model_box.addItems(list(mat.MODELS))
        self.model_box.setCurrentText(model)
        self.model_box.currentTextChanged.connect(self._change_model)
        self.field_row(("Model", self.model_box), ("Colour", self._color_box()))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Property", "Expression", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.outer.addWidget(self.table)
        self.aniso_k = QCheckBox("Anisotropic thermal conductivity (kxx, kyy, kzz in the crystal frame)")
        self.aniso_k.setChecked(any(k in props for k in ("kxx", "kyy", "kzz")))
        self._fill(model, props)
        self.table.itemChanged.connect(self._sync)
        self.aniso_k.toggled.connect(self._toggle_aniso)
        self.outer.addWidget(self.aniso_k)
        self.outer.addWidget(muted_label(
            "ρ: density.  Cij: elastic stiffness constants (Voigt).  E: Young's modulus, ν (nu): Poisson's ratio.  "
            "k: thermal conductivity, cp: specific heat capacity (only needed for heat transfer; leave empty "
            "otherwise). In heat transfer k, cp and ρ may depend on the temperature T (K), e.g. "
            "148[W/(m*K)]*(300[K]/T)^1.3; the value column shows them at 293.15 K."))

        # crystal orientation
        box = QGroupBox("Crystal orientation")
        bl = QVBoxLayout(box)
        rot = self.m.props.get(mat.ROTATION_KEY, ["0", "0", "0"])
        self.rotation = [str(a) for a in rot]
        row = QHBoxLayout()
        for i, axis in enumerate(AXES):
            row.addWidget(QLabel(f"about {axis}"))
            row.addWidget(ExprEdit(self.win, lambda i=i: self.rotation[i],
                                   lambda t, i=i: self._set_rotation(i, t), show_value=False), 1)
        row.addWidget(QLabel("deg"))
        bl.addLayout(row)
        bl.addWidget(muted_label(
            "Rotates the crystal relative to the global axes: first about x, then y, then z (degrees; numbers "
            "or expressions, e.g. a variable swept by the problem). The constants above stay in the crystal frame; the stiffness used by the solver is "
            "transformed. Example: 45 about z puts a Si <110> direction along x."))
        self.outer.addWidget(box)

        box = QGroupBox("Effective stiffness in the global frame (GPa)")
        bl = QVBoxLayout(box)
        self.ctable = QTableWidget(6, 6)
        labels = ["1 xx", "2 yy", "3 zz", "4 yz", "5 xz", "6 xy"]
        self.ctable.setHorizontalHeaderLabels(labels)
        self.ctable.setVerticalHeaderLabels(labels)
        self.ctable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ctable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ctable.setFixedHeight(self.ctable.horizontalHeader().height() + 6 * 26 + 4)
        for r in range(6):
            self.ctable.setRowHeight(r, 26)
        bl.addWidget(self.ctable)
        self.outer.addWidget(box)
        self._refresh_effective()

    def _color_box(self) -> QComboBox:
        """Display colour: automatic (by the order of the materials) or one of the palette's colours."""
        box = QComboBox()
        auto = mat.DEFAULT_COLORS[self.p.materials.index(self.m) % len(mat.DEFAULT_COLORS)]
        box.addItem(swatch(auto), "Automatic", "")
        for group, colors in mat.COLOR_CHOICES.items():
            box.insertSeparator(box.count())
            for n, c in enumerate(colors):
                box.addItem(swatch(c), f"{group} {n + 1}", c)
        if self.m.color and box.findData(self.m.color) < 0:          # set in the file, not in the palette
            box.addItem(swatch(self.m.color), self.m.color, self.m.color)
        box.setCurrentIndex(max(0, box.findData(self.m.color)))

        def choose(i):
            self.m.color = box.itemData(i) or ""
            self.win.mark_dirty()
            self.win.rebuild_tree(("material", self.index), refresh_panel=False)
            self.win.refresh_view()

        box.currentIndexChanged.connect(choose)
        return box

    def _set_rotation(self, i, text):
        self.rotation[i] = text or "0"
        if all(a.strip() in ("0", "0.0", "") for a in self.rotation):
            self.m.props.pop(mat.ROTATION_KEY, None)
        else:
            self.m.props[mat.ROTATION_KEY] = list(self.rotation)
        self._refresh_effective()

    def _refresh_effective(self):
        if not hasattr(self, "ctable"):
            return
        try:
            C, _ = mat.evaluate(self.m.props, self.win.project.evaluator())
        except Exception as exc:
            C, err = None, str(exc)
        for r in range(6):
            for c in range(6):
                item = self.ctable.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.ctable.setItem(r, c, item)
                if C is None:
                    set_value_item(item, "–", False, err)
                else:
                    v = C[r, c] / 1e9
                    set_value_item(item, "0" if v == 0 else f"{v:.4g}", True)

    def _fill(self, model, props):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        thermal = ["kxx", "kyy", "kzz", "cp"] if self.aniso_k.isChecked() else ["k", "cp"]
        for key in mat.MODELS[model] + thermal:
            r = self.table.rowCount()
            self.table.insertRow(r)
            k = QTableWidgetItem(key)
            k.setFlags(k.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, k)
            default = "" if key in mat.THERMAL_KEYS else "0"
            self.table.setItem(r, 1, QTableWidgetItem(str(props.get(key, default))))
            v = QTableWidgetItem("")
            v.setFlags(v.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 2, v)
        self.table.blockSignals(False)
        # show every row without an inner scrollbar
        self.table.setFixedHeight(self.table.horizontalHeader().sizeHint().height()
                                  + sum(self.table.rowHeight(r) for r in range(self.table.rowCount())) + 4)
        self._refresh_values()

    def _refresh_values(self):
        self.table.blockSignals(True)
        ev = display_evaluator(self.win.project)
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 2)
            try:
                expr = self.table.item(r, 1).text()
                if not expr.strip():
                    set_value_item(item, "–", True)
                    continue
                if mat.depends_on_T(expr):     # a temperature-dependent (thermal) property: at REFERENCE_T
                    v = ev.with_overrides({mat.TEMPERATURE: mat.REFERENCE_T})(expr)
                    unit = {"rho": " kg/m³", "cp": " J/(kg·K)"}.get(self.table.item(r, 0).text(), " W/(m·K)")
                    set_value_item(item, f"{fmt(v)}{unit} at {mat.REFERENCE_T:g} K", True)
                    continue
                set_value_item(item, show(self.win.project, ev, expr, ev(expr)), True)
            except Exception as exc:
                set_value_item(item, "error", False, str(exc))
        self.table.blockSignals(False)

    def _sync(self, *_):
        props = {"model": self.model_box.currentText()}
        for r in range(self.table.rowCount()):
            value = self.table.item(r, 1).text().strip()
            if value:  # empty thermal properties are simply not set
                props[self.table.item(r, 0).text()] = value
        if mat.ROTATION_KEY in self.m.props:
            props[mat.ROTATION_KEY] = self.m.props[mat.ROTATION_KEY]
        self.m.props = props
        self.win.mark_dirty()
        self._refresh_values()
        self._refresh_effective()

    def _toggle_aniso(self, on):
        _, props = mat.resolve_properties(self.m.props)
        if on and "k" in props:
            props.update({k: props["k"] for k in ("kxx", "kyy", "kzz")})
        elif not on and "kxx" in props:
            props["k"] = props["kxx"]
        self._fill(self.model_box.currentText(), props)
        self._sync()

    def _change_model(self, model):
        _, props = mat.resolve_properties(self.m.props)
        self._fill(model, props)
        self._sync()

    def _rename(self, new):
        old = self.m.name
        if not new or new == old:
            return
        self.m.name = new
        for f in self.p.objects:
            if f.material == old:
                f.material = new
        self.win.mark_dirty()
        self.win.rebuild_tree(("material", self.index))


def default_feature(project, ftype: str) -> Feature:
    """A new feature sized and placed relative to the existing geometry."""
    names = {f.name for f in project.objects}
    stem = FEATURE_TITLES.get(ftype, ftype.capitalize())   # e.g. "Cylinder 1", "Lattice cell 1"
    i = 1
    while f"{stem} {i}" in names:
        i += 1
    material = project.materials[0].name if project.materials else ""
    lo, hi = model_box(project)
    c, span = (lo + hi) / 2, hi - lo
    r = 0.25 * min(span[0], span[1])
    L = lambda v: units.length_expr(v, project.length_unit)  # noqa: E731
    params = copy.deepcopy(FEATURE_TYPES[ftype])
    base = [L(c[0]), L(c[1]), L(lo[2])]
    if ftype == "block":
        params.update(center=[L(x) for x in c], size=[L(x / 2) for x in span])
    elif ftype == "cylinder":
        params.update(center=base, radius=L(r), height=L(span[2]))
    elif ftype == "cone":
        params.update(center=base, radius1=L(r), radius2=L(r / 2), height=L(span[2]))
    elif ftype == "sphere":
        params.update(center=[L(x) for x in c], radius=L(0.25 * span.min()))
    elif ftype == "cell":
        params.update(center=[L(x) for x in c], height=L(span[2]))
    elif ftype == "prism":
        params.update(points=[[L(c[0] - r), L(c[1] - r)], [L(c[0] + r), L(c[1] - r)], [L(c[0]), L(c[1] + r)]],
                      z0=L(lo[2]), height=L(span[2]))
    elif ftype == "triangle":
        params.update(center=[L(x) for x in c], base=L(span[0] / 2), height=L(span[1] / 2), thickness=L(span[2]))
    elif ftype == "loft":   # a square frustum: base 2r, top r
        params.update(sections=[{"z": L(z), "points": [[L(c[0] + sx * h), L(c[1] + sy * h)]
                                                       for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]}
                                for z, h in ((lo[2], r), (hi[2], r / 2))])
    return Feature(ftype, f"{stem} {i}", "add", material, params)


BUILD_GEOMETRY = ("Build ◱", "build_geometry")


class ObjectsPanel(Panel):
    title = "Objects"
    action = BUILD_GEOMETRY
    help = ("Objects are applied top to bottom. 'add' creates solid with a material (later objects win where "
            "they overlap); 'subtract' removes material from everything above it.")

    def build(self):
        self.list = QListWidget()
        for f in self.p.objects:
            text = (f"{f.name}  ({f.type}, {f.op}{', ' + f.material if f.op == 'add' else ''}"
                    f"{', array ' + f.array_label if f.arrayed else ''})")
            self.list.addItem(text + ("  — bypassed" if not f.enabled else ""))
            self.list.item(self.list.count() - 1).setIcon(feature_swatch(self.p, f))
            if not f.enabled:
                item = self.list.item(self.list.count() - 1)
                item.setForeground(QColor(theme.disabled_text()))
                font = QFont(item.font())
                font.setItalic(True)
                item.setFont(font)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.itemDoubleClicked.connect(lambda it: self.win.select_node("feature", self.list.row(it)))
        self.outer.addWidget(self.list)
        row = self.button_row(("Bypass / enable", lambda: self._on_selected(self.win.toggle_feature)),
                              ("Delete", lambda: self._on_selected(self.win.delete_feature)))
        self.button_row(("Move up", lambda: self._on_selected(self.win.move_feature, -1)),
                        ("Move down", lambda: self._on_selected(self.win.move_feature, 1)))
        add = QPushButton("Add object")
        menu = QMenu(add)
        for t in FEATURE_TYPES:
            menu.addAction(FEATURE_TITLES[t], lambda t=t: self._add(t))
        add.setMenu(menu)
        row.insertWidget(0, add)

    def _add(self, ftype):
        self.win.add_feature(ftype)

    def _on_selected(self, fn, *args):
        r = self.list.currentRow()
        if r >= 0:
            fn(r, *args)


class FeaturePanel(Panel):
    LABELS = {"corner": "Corner", "size": "Size", "center": "Center", "axis": "Axis direction",
              "radius": "Radius", "radius1": "Base radius", "radius2": "Top radius", "height": "Height",
              "points": "Polygon (x, y per line)", "z0": "Base z", "base": "Base (along x)",
              "thickness": "Thickness (along z)"}
    CELL_HELP = ("Unit cell of the periodic lattice: the parallelogram (rectangle, rhombus, …) spanned by the "
                 "lattice vectors a1, a2, centred on Center and extruded along z by Height. It always matches "
                 "the periodicity. Add holes or pillars inside it.")

    def __init__(self, win, index):
        self.f: Feature = win.project.objects[index]
        self.index = index
        self.title = f"{FEATURE_TITLES.get(self.f.type, self.f.type)}: {self.f.name}" + ("  (bypassed)" if not self.f.enabled else "")
        super().__init__(win)

    def build(self):
        i = self.index
        up = self.button_row(("Build up to here ◱", lambda: self.win.build_geometry(upto=i)))
        up.itemAt(0).widget().setToolTip("Build the geometry from the first object down to this one; later "
                                         "objects are left out")
        self.button_row(("Enable" if not self.f.enabled else "Bypass", lambda: self.win.toggle_feature(i)),
                        ("Duplicate", lambda: self.win.duplicate_feature(i)),
                        ("Delete", lambda: self.win.delete_feature(i)),
                        ("Move up", lambda: self.win.move_feature(i, -1)),
                        ("Move down", lambda: self.win.move_feature(i, 1)))
        if not self.f.enabled:
            self.outer.addWidget(muted_label("This object is bypassed: it is kept in the model but ignored "
                                             "when building the geometry."))
        if self.f.type == "cell":
            self.outer.addWidget(muted_label(self.CELL_HELP))
            self._lattice_box()
        name = QLineEdit(self.f.name)
        name.editingFinished.connect(self._rename(name))
        self.field("Name", name)
        op = QComboBox()
        op.addItems(OPS)
        op.setCurrentText(self.f.op)
        material = QComboBox()
        colors = self.p.material_colors()
        for m in self.p.materials:
            material.addItem(swatch(colors[m.name]), m.name)
        material.setCurrentText(self.f.material)
        material.setEnabled(self.f.op == "add")
        self.field_row(("Operation", op), ("Material", material))

        def set_op(v):
            self.f.op = v
            material.setEnabled(v == "add")
            self.win.mark_dirty()
            self.win.rebuild_tree(("feature", self.index), refresh_panel=False)

        op.currentTextChanged.connect(set_op)
        material.currentTextChanged.connect(lambda v: (setattr(self.f, "material", v), self.win.mark_dirty(),
                                                       self.win.rebuild_tree(refresh_panel=False)))

        g = self.form("Variables")
        params = self.f.params
        for key in FEATURE_TYPES[self.f.type]:
            label = self.LABELS.get(key, key)
            if key == "center" and self.f.type in ("cylinder", "cone"):
                label = "Base center"
            if key == "sections":
                self._sections_box(g)
            elif key == "points":
                edit = QPlainTextEdit("\n".join(", ".join(str(c) for c in pt) for pt in params[key]))
                edit.setMaximumHeight(120)

                def set_points(edit=edit):
                    pts = [[c.strip() for c in line.split(",")] for line in edit.toPlainText().splitlines() if line.strip()]
                    params["points"] = [p for p in pts if len(p) == 2]
                    self.win.mark_dirty()

                edit.textChanged.connect(set_points)
                g.addRow(label, edit)
            elif isinstance(params[key], list):
                g.addRow(label, VectorEdit(self.win, params[key]))
            else:
                g.addRow(label, ExprEdit(self.win, lambda k=key: params[k], lambda t, k=key: params.__setitem__(k, t)))
        if self.f.type == "triangle":
            g.addRow(muted_label("Isosceles triangle: base along x, apex towards +y, extruded along z; centred on "
                                 "its bounding box. Rotate it for other orientations."))
        if self.f.type == "loft":
            g.addRow(muted_label("Polygons in x-y planes at heights z, joined by straight side faces: corner i of "
                                 "one section joins corner i of the next, so all sections need the same number "
                                 "of corners."))
        if ROTATION_KEY in params:
            rot = VectorEdit(self.win, params[ROTATION_KEY])
            rot.setToolTip("Degrees about the global x, then y, then z axis, around the "
                           + ("base center" if self.f.type in ("cylinder", "cone") else "center"))
            g.addRow("Rotation (deg)", rot)
        if ARRAY_KEY in params:
            self._array_box()

    def _sections_box(self, g):
        """Loft sections: z and a polygon each, plus adding/removing sections."""
        sections = self.f.params["sections"]
        for n, sec in enumerate(sections, 1):
            g.addRow(f"Section {n}: z", ExprEdit(self.win, lambda sec=sec: sec["z"],
                                                 lambda t, sec=sec: sec.__setitem__("z", t)))
            edit = QPlainTextEdit("\n".join(", ".join(str(c) for c in pt) for pt in sec["points"]))
            edit.setMaximumHeight(120)

            def set_points(edit=edit, sec=sec):
                pts = [[c.strip() for c in line.split(",")] for line in edit.toPlainText().splitlines() if line.strip()]
                sec["points"] = [p for p in pts if len(p) == 2]
                self.win.mark_dirty()

            edit.textChanged.connect(set_points)
            g.addRow("  Polygon (x, y per line)", edit)
        row = QHBoxLayout()
        add = QPushButton("Add section")
        add.setToolTip("Append a copy of the last section; then change its z and corners")
        add.clicked.connect(lambda: self._edit_sections(lambda: sections.append(copy.deepcopy(sections[-1]))))
        remove = QPushButton("Remove last section")
        remove.setEnabled(len(sections) > 2)
        remove.clicked.connect(lambda: self._edit_sections(sections.pop))
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        g.addRow(row)

    def _edit_sections(self, change):
        change()
        self.win.mark_dirty()
        self.win.rebuild_tree(("feature", self.index))

    def _array_box(self):
        """Copies of the feature along x, y, z, one object together."""
        params = self.f.params
        a = self.form("Array")
        a.addRow(muted_label("Copies along x, y and z (1 = no array), spacing apart. All copies form one object: "
                             "one name, one material, one selection; they merge where they touch."))
        a.addRow("Copies", VectorEdit(self.win, params[ARRAY_KEY], on_change=lambda: self.win.rebuild_tree(
            ("feature", self.index), refresh_panel=False)))
        spacing = VectorEdit(self.win, params[SPACING_KEY])
        spacing.setToolTip("Shift between neighbouring copies along each axis (negative: towards -x, -y, -z); "
                           "the object itself is the first copy")
        a.addRow("Spacing", spacing)

    def _lattice_box(self):
        """The lattice that shapes the cell, editable here (same setting as Periodicity)."""
        ph = self.p.lattice
        box = QGroupBox("Lattice (shared with Solid Mechanics → Periodicity)")
        f = QFormLayout(box)
        self.outer.addWidget(box)
        mode, const = lattice_mode(ph)
        if mode == "hex" and getattr(self.win, "custom_lattice", False):
            mode = "custom"
        if mode == "rect":
            f.addRow("Type", QLabel("Rectangular"))
            for ax in ("x", "y"):
                if ax in ph.periodic:
                    f.addRow(f"Period in {ax}", ExprEdit(self.win, lambda ax=ax: ph.periodic[ax],
                                                         lambda t, ax=ax: self._set_period(ax, t)))
        elif mode == "hex":
            f.addRow("Type", QLabel("Hexagonal"))
            f.addRow("Lattice constant", ExprEdit(self.win, lambda: const, self._set_hex_constant))
        else:
            f.addRow("Type", QLabel("Custom vectors"))
            for n, vec in enumerate(ph.vectors[:2]):
                f.addRow(f"a{n + 1}", VectorEdit(self.win, vec))
        self.cell_info = muted_label(self._cell_text())
        f.addRow(self.cell_info)
        row = QHBoxLayout()
        b = QPushButton("Change lattice type…")
        b.clicked.connect(lambda: self.win.select_node("periodicity"))
        row.addWidget(b)
        row.addStretch(1)
        f.addRow(row)

    def _set_period(self, ax, text):
        self.p.lattice.periodic[ax] = text or "a"
        self.win.physics_changed()
        self.cell_info.setText(self._cell_text())

    def _set_hex_constant(self, text):
        self.p.lattice.vectors = hexagonal_vectors(text or "a")
        self.win.physics_changed()
        self.cell_info.setText(self._cell_text())

    def _cell_text(self) -> str:
        """Side lengths, angle and area of the cell in the display length unit."""
        from ... import lattice as lat
        try:
            vecs = [v for v in lat.slots(self.p.lattice, self.p.evaluator()) if v is not None][:2]
            a1, a2 = vecs
        except Exception:
            return ""
        l1, l2 = np.linalg.norm(a1), np.linalg.norm(a2)
        angle = np.degrees(np.arccos(np.clip(a1 @ a2 / (l1 * l2), -1, 1)))
        area = abs(np.cross(a1, a2)[2])
        lu = self.p.length_unit
        scale = units.LENGTH_UNITS.get(lu, 1e-6)
        label = units.unit_label(lu)
        return (f"Cell: |a1| = {l1 / scale:.4g} {label}, |a2| = {l2 / scale:.4g} {label}, angle {angle:.4g}°, "
                f"area {area / scale ** 2:.4g} {label}²")

    def _rename(self, edit):
        def fn():
            if edit.text().strip() and edit.text().strip() != self.f.name:
                self.f.name = edit.text().strip()
                self.win.mark_dirty()
                self.win.rebuild_tree(("feature", self.index))
        return fn


# ---------------------------------------------------------------------------
def mesh_nodes(project) -> list[tuple[str, tuple]]:
    """Tree nodes under Mesh: the default element size (the whole model, or everything outside the regions),
    then the regions."""
    ms = project.mesh
    default = "Everything else" if ms.regions else "Whole model"
    return ([(f"{default} (max {ms.max_size})", ("mesh_default", None))]
            + [(f"{r.name or r.on or '?'} (max {r.max_size})", ("mesh_region", i)) for i, r in enumerate(ms.regions)])


class MeshDefaultPanel(Panel):
    help = ("The maximum element size of the model, outside the mesh regions (all of it without regions). Curved "
            "and small features get smaller elements automatically.")

    def __init__(self, win):
        self.title = "Everything else" if win.project.mesh.regions else "Whole model"
        super().__init__(win)

    def build(self):
        ms = self.p.mesh
        f = self.form()
        f.addRow("Maximum element size", ExprEdit(self.win, lambda: ms.max_size,
                                                  lambda t: (setattr(ms, "max_size", t or "1"),
                                                             self.win.rebuild_tree(refresh_panel=False))))
        self.button_row(("Add mesh region", self.win.add_mesh_region))


class MeshRegionPanel(Panel):
    help = ("A maximum element size for one object: one of the Objects (what is left of it after later objects) or "
            "all domains of a material. Regions apply in order; a later one wins where they overlap. The mesh stays "
            "continuous across the object's interfaces, where the smaller size applies, and grades away from them.")

    def __init__(self, win, index):
        self.index = index
        self.r = win.project.mesh.regions[index]
        self.title = f"Mesh region: {self.r.name or self.r.on}"
        super().__init__(win)

    def build(self):
        from .common import objects
        r = self.r
        self.button_row(("Delete", lambda: self.win.delete_mesh_region(self.index)))
        f = self.form()
        name = QLineEdit(r.name)
        name.editingFinished.connect(lambda: (setattr(r, "name", name.text().strip()), self.win.mark_dirty(),
                                              self.win.rebuild_tree(refresh_panel=False)))
        f.addRow("Name", name)
        on = QComboBox()
        names = objects(self.p)
        on.addItems(names if r.on in names else names + [r.on])
        on.setCurrentText(r.on)
        on.currentTextChanged.connect(lambda v: (setattr(r, "on", v), self.win.mark_dirty(),
                                                 self.win.rebuild_tree(refresh_panel=False)))
        f.addRow("Object", on)
        f.addRow("Maximum element size", ExprEdit(self.win, lambda: r.max_size,
                                                  lambda t: (setattr(r, "max_size", t or "1"),
                                                             self.win.rebuild_tree(refresh_panel=False))))


class MeshPanel(Panel):
    title = "Mesh"
    action = ("Mesh ◪", "build_mesh")
    help = "Free tetrahedral mesh. Quadratic elements are strongly recommended for elasticity."

    # name: maximum element size as a fraction of the model size (middle bounding-box dimension); the minimum
    # size is a fifth of the maximum
    PRESETS = {"Coarse": 1 / 4, "Normal": 1 / 8, "Fine": 1 / 14, "Very fine": 1 / 24}

    def _model_size(self) -> float:
        lo, hi = model_box(self.p)
        return float(np.median(hi - lo))

    def _preset_sizes(self, name) -> tuple[float, float]:
        h = self._model_size() * self.PRESETS[name]
        return h, h / 5

    def _current_preset(self) -> str:
        try:
            ev = self.p.evaluator()
            hmax, hmin = ev(self.p.mesh.max_size), ev(self.p.mesh.min_size)
        except Exception:
            return "Custom"
        for name in self.PRESETS:
            a, b = self._preset_sizes(name)
            if abs(hmax - a) <= 0.01 * a and abs(hmin - b) <= 0.01 * b:
                return name
        return "Custom"

    def _apply_preset(self, name):
        if name not in self.PRESETS:
            return
        ms = self.p.mesh
        hmax, hmin = self._preset_sizes(name)

        def length(v):  # nm, um, mm or m so that the number stays below 1000, e.g. 62.5[nm]
            for unit, f in (("nm", 1e-9), ("um", 1e-6), ("mm", 1e-3), ("m", 1.0)):
                if v < 1000 * f:
                    return f"{v / f:.3g}[{unit}]"
            return f"{v:.3g}[m]"

        ms.max_size, ms.min_size = length(hmax), length(hmin)
        self.win.mark_dirty()
        self.win.log(f"Mesh preset '{name}': maximum {ms.max_size}, minimum {ms.min_size}")
        self.win.rebuild_tree(("mesh", None))

    def build(self):
        ms = self.p.mesh
        f = self.form()
        preset = QComboBox()
        preset.addItems(list(self.PRESETS) + ["Custom"])
        preset.setCurrentText(self._current_preset())
        preset.setToolTip("Sets the element sizes below relative to the model size; edit them for a custom mesh")
        preset.currentTextChanged.connect(self._apply_preset)
        f.addRow("Element size", preset)

        def set_size(attr, text):  # a manual edit: the selector follows (Custom unless it matches a preset)
            setattr(ms, attr, text)
            preset.blockSignals(True)
            preset.setCurrentText(self._current_preset())
            preset.blockSignals(False)

        f.addRow("Minimum element size", ExprEdit(self.win, lambda: ms.min_size, lambda t: set_size("min_size", t)))
        order = QComboBox()
        order.addItems(["Linear (P1)", "Quadratic (P2)"])
        order.setCurrentIndex(ms.order - 1)
        order.currentIndexChanged.connect(lambda i: (setattr(ms, "order", i + 1), self.win.mark_dirty()))
        f.addRow("Element order", order)
        opt = QCheckBox()
        opt.setChecked(ms.optimize)
        opt.toggled.connect(lambda v: (setattr(ms, "optimize", v), self.win.mark_dirty()))
        f.addRow("Optimize", opt)
        g = self.form("Element sizes")
        for label, node in mesh_nodes(self.p):
            b = QPushButton(label)
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.clicked.connect(lambda _=False, node=node: self.win.select_node(*node))
            g.addRow(b)
        g.addRow(muted_label("The maximum element size of the model, and optionally different ones in some objects "
                             "(mesh regions, e.g. finer in a thin film or near a heater). The mesh stays "
                             "continuous across their interfaces and grades away from them."))
        self.button_row(("Add mesh region", self.win.add_mesh_region))
        self.outer.addWidget(muted_label(self.win.mesh_summary()))
        self.button_row(("Export mesh (.msh)…", self.win.export_mesh))
