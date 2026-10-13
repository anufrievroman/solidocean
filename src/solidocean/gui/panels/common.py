"""Shared building blocks of the settings panels: expression editors, the Panel base class,
the model's bounding box and the selection editor of boundary conditions (any physics)."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from ... import materials as mat
from ... import units
from ...geometry import evaluate_features
from ...project import AXES
from ...selection import ALL_FACES, FACES
from .. import theme


ERROR_STYLE = "QLineEdit, QPlainTextEdit { border: 1px solid #d04040; }"


def display_evaluator(project):
    """An evaluator of the variables that resolve, for showing values in the GUI: one broken variable
    marks only the fields that use it, not every field of the model."""
    from ...expressions import Evaluator, resolve_each
    values, _ = resolve_each(project.variables)
    return Evaluator({k: project.variables[k] for k in values})


def fmt(v: float) -> str:
    return f"{v:.6g}"


def show(project, ev, expr, value: float) -> str:
    """Value formatted with units in the project's preferred units (falls back to plain SI)."""
    try:
        return units.format_value(value, units.dimension(expr, ev), project.display_units)
    except Exception:
        return fmt(value)


def set_value_item(item: QTableWidgetItem, text: str, ok: bool, tooltip: str = ""):
    """Read-only value cell: default text colour when OK (works in dark mode), red on error."""
    item.setText(text)
    item.setToolTip(tooltip)
    if ok:
        item.setData(Qt.ItemDataRole.ForegroundRole, None)
    else:
        item.setForeground(QColor(theme.error()))


def theme_text() -> str:
    """Normal text colour for matplotlib drawings embedded in panels."""
    return "#e6e6e6" if theme.is_dark() else "#222222"


def labelled(label: str, widget: QWidget) -> QWidget:
    """A widget under its own label, to be used full width (the label never steals room from the field)."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    lay.addWidget(QLabel(label))
    lay.addWidget(widget)
    return box


def spoiler(win, title: str, html: str, state: str) -> tuple[QToolButton, QLabel]:
    """A collapsible block of help text: a title that folds the text away. Its open/closed state is
    remembered on the window for the session, under the attribute `state`. Returns the two widgets, for
    the caller to put in a form or a layout."""
    toggle = QToolButton()
    toggle.setCheckable(True)
    toggle.setChecked(getattr(win, state, False))
    toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    toggle.setStyleSheet("QToolButton { border: none; background: transparent; font-weight: 600; }")
    info = QLabel(html)
    info.setWordWrap(True)
    info.setTextFormat(Qt.TextFormat.RichText)
    info.setStyleSheet(f"color: {theme.muted()}; padding-left: 18px;")

    def show(on):
        toggle.setText(("▾" if on else "▸") + f"  {title}")   # small text triangles
        info.setVisible(on)
        setattr(win, state, on)

    toggle.toggled.connect(show)
    show(toggle.isChecked())
    return toggle, info


def equations(*blocks: tuple[str, str] | str) -> str:
    """The text of a panel's Formalism block: an (equation, meaning) pair puts the equation on a line of
    its own, a plain string is a paragraph. Rich text, so <sub>/<sup> and entities may be used."""
    out = []
    for block in blocks:
        if isinstance(block, str):
            out.append(f"<p style='margin:4px 0'>{block}</p>")
        else:
            eq, meaning = block
            out.append(f"<p style='margin:6px 0 2px 10px'><b>{eq}</b></p>"
                       f"<p style='margin:0 0 4px 0'>{meaning}</p>")
    return "".join(out)


def muted_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color: {theme.muted()};")
    return lbl


class ExprEdit(QWidget):
    """Expression line edit with its evaluated SI value alongside."""

    def __init__(self, win, getter: Callable, setter: Callable, show_value=True, width=None, optional=""):
        super().__init__()
        self.win, self.setter = win, setter
        self.optional = optional  # text shown when the field may stay empty (e.g. "once")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.line = QLineEdit(str(getter()))
        if width:
            self.line.setFixedWidth(width)
        lay.addWidget(self.line, 1)
        self.value = QLabel() if show_value else None
        if self.value is not None:
            self.value.setMinimumWidth(90)
            self.value.setStyleSheet(f"color: {theme.muted()};")
            lay.addWidget(self.value)
        self.line.editingFinished.connect(self._commit)
        self.refresh()

    def _commit(self):
        self.setter(self.line.text().strip())
        self.win.mark_dirty()
        self.refresh()

    def refresh(self):
        if self.optional and not self.line.text().strip():
            self.line.setStyleSheet("")
            self.line.setPlaceholderText(self.optional)
            if self.value is not None:
                self.value.setText("")
            return
        try:
            ev = display_evaluator(self.win.project)
            from ... import materials as mat
            timed = re.search(r"\btime\b", self.line.text()) is not None
            if timed:  # time-dependent value: shown at t = 0
                ev = ev.with_overrides({"time": 0.0})
            hot = mat.depends_on_T(self.line.text())
            if hot:    # temperature-dependent value: shown at the reference temperature
                ev = ev.with_overrides({mat.TEMPERATURE: mat.REFERENCE_T})
            v = ev(self.line.text())
            text = show(self.win.project, ev, self.line.text(), v) + (" at t = 0" if timed else "") + \
                (f" at {mat.REFERENCE_T:g} K" if hot else "")
            self.line.setStyleSheet("")
            self.line.setToolTip(f"= {text}   (SI: {fmt(v)})")
            if self.value is not None:
                self.value.setText(f"= {text}")
        except Exception as exc:
            self.line.setStyleSheet(ERROR_STYLE)
            self.line.setToolTip(str(exc))
            if self.value is not None:
                self.value.setText("error")


class ExprBox(QPlainTextEdit):
    """Multi-line expression editor (monospace); Ctrl+Enter calls `ctrl_enter` if set."""

    def __init__(self, text: str, lines: int = 3):
        super().__init__(text)
        self.setTabChangesFocus(True)
        line = self.fontMetrics().lineSpacing()
        self.setFixedHeight(lines * line + 14)
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

    ctrl_enter = None  # optional callback for Ctrl+Enter

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.ctrl_enter:
                self.ctrl_enter()
            return
        super().keyPressEvent(event)

    def text(self) -> str:
        return self.toPlainText()


class VectorEdit(QWidget):
    def __init__(self, win, vec: list, labels=("x", "y", "z"), on_change: Callable | None = None):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        def set_component(t, i):
            changed = vec[i] != t
            vec[i] = t
            if changed and on_change:
                on_change()

        for i, lbl in enumerate(labels):
            lay.addWidget(QLabel(lbl))
            lay.addWidget(ExprEdit(win, lambda i=i: vec[i], lambda t, i=i: set_component(t, i), show_value=False), 1)


class Panel(QWidget):
    title = ""
    help = ""
    # equations of this part of the physics and how they are imposed, collapsed under "Formalism" (under the
    # description)
    formalism = ""
    action: tuple[str, str] | None = None  # (button label, main-window method) shown under the title

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.p = win.project
        self.outer = QVBoxLayout(self)
        head = QLabel(self.title)
        head.setStyleSheet("font-size: 15px; font-weight: 600; padding-bottom: 4px;")
        self.outer.addWidget(head)
        if self.help:
            self.outer.addWidget(muted_label(self.help))
        if self.formalism:    # reference material, with the description, before the panel's controls
            self.spoiler("Formalism", self.formalism, "show_formalism")
        if self.action:
            label, method = self.action
            self.button_row((label, getattr(self.win, method)))
        self.build()
        self.outer.addStretch(1)

    def build(self):
        pass

    def form(self, title: str | None = None) -> QFormLayout:
        box = QGroupBox(title) if title else QWidget()
        f = QFormLayout(box)
        self.outer.addWidget(box)
        return f

    def field(self, label: str, widget: QWidget):
        """A full-width field with its label on top."""
        self.outer.addWidget(labelled(label, widget))

    def field_row(self, *fields: tuple[str, QWidget], stretch: bool = True):
        """Small fields side by side on one line, each with its label beside it. `stretch`: share the
        width between them (off: each stays its natural size, the row fills up on the right)."""
        row = QHBoxLayout()
        for n, (label, widget) in enumerate(fields):
            if n:
                row.addSpacing(12)
            row.addWidget(QLabel(label))
            row.addWidget(widget, 1 if stretch else 0)
        if not stretch:
            row.addStretch(1)
        self.outer.addLayout(row)

    def spoiler(self, title: str, html: str, state: str):
        """A collapsible block of help text, added where the panel is at."""
        for widget in spoiler(self.win, title, html, state):
            self.outer.addWidget(widget)

    def button_row(self, *buttons: tuple[str, Callable]) -> QHBoxLayout:
        row = QHBoxLayout()
        for text, fn in buttons:
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        self.outer.addLayout(row)
        return row


def outer_labels(win) -> list[str]:
    """Display labels for each value of the variable sweep."""
    r = win.results
    dim = r.info.get("outer_dim")
    return [f"{r.outer_name} = {units.format_value(v, tuple(dim) if dim else None, win.project.display_units)}"
            for v in r.outer_values]


def no_results(panel) -> bool:
    if panel.win.results is None:
        panel.outer.addWidget(muted_label("No results yet. Configure the Study and press Run."))
        return True
    return False


# ---------------------------------------------------------------------------
def model_box(project, only: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Approximate bounding box of the enabled solid features (1 µm cube if unknown); `only`: the features
    of that name or material."""
    try:
        feats = evaluate_features(project, project.evaluator())
    except Exception:
        feats = []
    lo, hi = [], []
    for ef in feats:
        v, t = ef.values, ef.feature.type
        if ef.feature.op != "add" or (only and only not in (ef.feature.name, ef.feature.material)):
            continue
        n_before = len(lo)
        if t == "block":
            half = np.abs(v["size"]) / 2
            lo.append(v["center"] - half)
            hi.append(v["center"] + half)
        elif t == "sphere":
            lo.append(v["center"] - v["radius"])
            hi.append(v["center"] + v["radius"])
        elif t in ("cylinder", "cone"):
            n = np.linalg.norm(v["axis"]) or 1.0
            axis = v["axis"] / n
            end = v["center"] + axis * v["height"]
            r = max(v.get("radius", 0), v.get("radius1", 0), v.get("radius2", 0))
            r = r * np.sqrt(np.clip(1 - axis ** 2, 0, 1))   # end discs extend only perpendicular to the axis
            lo.append(np.minimum(v["center"], end) - r)
            hi.append(np.maximum(v["center"], end) + r)
        elif t == "cell":
            from ... import lattice as lat
            try:
                vecs = [x for x in lat.slots(project.lattice, project.evaluator()) if x is not None][:2]
                corners = lat.cell_corners(vecs, v["center"])
                lo.append(np.r_[corners[:, :2].min(axis=0), v["center"][2] - v["height"] / 2])
                hi.append(np.r_[corners[:, :2].max(axis=0), v["center"][2] + v["height"] / 2])
            except Exception:
                pass
        elif t == "prism" and len(v["points"]):
            z = sorted([v["z0"], v["z0"] + v["height"]])
            lo.append(np.r_[v["points"].min(axis=0), z[0]])
            hi.append(np.r_[v["points"].max(axis=0), z[1]])
        elif t == "triangle":
            half = np.abs([v["base"], v["height"], v["thickness"]]) / 2
            lo.append(v["center"] - half)
            hi.append(v["center"] + half)
        angles = np.asarray(v.get("rotation", np.zeros(3)), dtype=float)
        if len(lo) > n_before and angles.shape == (3,) and np.any(angles != 0):  # box of the rotated box corners
            from ...geometry import rotation_pivot
            flo, fhi = lo.pop(), hi.pop()
            corners = np.array([[(flo, fhi)[i][0], (flo, fhi)[j][1], (flo, fhi)[k][2]]
                                for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            p0 = rotation_pivot(ef)
            rotated = (corners - p0) @ mat.rotation_matrix(angles).T + p0
            lo.append(rotated.min(axis=0))
            hi.append(rotated.max(axis=0))
        if len(lo) > n_before and "array" in v:   # an array spans from its first to its last copy
            from ...geometry import array_shifts
            try:
                shifts = array_shifts(ef)
                lo[-1], hi[-1] = lo[-1] + shifts.min(axis=0), hi[-1] + shifts.max(axis=0)
            except ValueError:
                pass
    if not lo:
        return np.full(3, -0.5e-6), np.full(3, 0.5e-6)
    lo, hi = np.min(lo, axis=0), np.max(hi, axis=0)
    span = hi - lo
    fallback = span.max() if span.max() > 0 else 1e-6
    span = np.where(span > 0, span, fallback)
    c = (lo + hi) / 2
    return c - span / 2, c + span / 2


def objects(project) -> list[str]:
    """Objects a condition or a reduction can refer to: geometry features that add material, then
    materials (all their domains)."""
    feats = [f.name for f in project.geometry if f.op == "add" and f.enabled]
    return feats + [m.name for m in project.materials if m.name not in feats]


def selection_box(panel, c) -> QGroupBox:
    """Where a condition applies: the outer surface of the whole model or of one object, its bounding-box faces
    (or all of it), optionally only the faces inside a box. Shared by heat and solid conditions."""
    win, p = panel.win, panel.p
    box = QGroupBox("Selection")
    f = QFormLayout(box)
    on = QComboBox()
    on.addItem("Whole model", "")
    for obj in objects(p):
        on.addItem(obj, obj)
    if c.on and on.findData(c.on) < 0:
        on.addItem(f"{c.on} (missing)", c.on)
    on.setCurrentIndex(max(0, on.findData(c.on)))
    on.setToolTip("The outer surface of the whole model, or of one object (a geometry feature or all domains of a "
                  "material). An object's interfaces with other objects are not part of its outer surface, e.g. "
                  "the bottom of a pad bonded to a membrane.")
    on.currentIndexChanged.connect(lambda i: (setattr(c, "on", on.itemData(i)), win.physics_changed(),
                                              panel._refresh()))
    f.addRow("Applies to", on)

    every = QCheckBox("All faces (the whole outer surface)")
    every.setChecked(ALL_FACES in c.faces)
    grid = QGridLayout()
    checks = []
    for i, face in enumerate(FACES):
        cb = QCheckBox(face)
        cb.setChecked(face in c.faces)
        cb.setEnabled(ALL_FACES not in c.faces)

        def toggle(state, face=face):
            if state and face not in c.faces:
                c.faces.append(face)
            elif not state and face in c.faces:
                c.faces.remove(face)
            win.physics_changed()

        cb.toggled.connect(toggle)
        grid.addWidget(cb, i // 2, i % 2)
        checks.append(cb)

    def set_all(state):
        c.faces[:] = [ALL_FACES] if state else [cb.text() for cb in checks if cb.isChecked()]
        for cb in checks:
            cb.setEnabled(not state)
        win.physics_changed()

    every.toggled.connect(set_all)
    f.addRow(every)
    f.addRow(f"Bounding-box faces of '{c.on}'" if c.on else "Bounding-box faces", grid)

    inside = QCheckBox("Only faces inside a box")
    inside.setChecked(bool(c.box))
    inside.setToolTip("Keeps the selected faces whose corners all lie inside the box (edges included), e.g. a "
                      "patch of the top face. The box may use variables, so it follows a variable sweep.")

    def set_inside(state):
        if state:
            lo, hi = model_box(p, only=c.on or None)
            lu = p.length_unit
            c.box = {"min": [units.length_expr(float(v), lu) for v in lo],
                     "max": [units.length_expr(float(v), lu) for v in hi]}
        else:
            c.box = {}
        win.physics_changed()
        panel._refresh()

    inside.toggled.connect(set_inside)
    f.addRow(inside)
    if c.box:
        grid = QGridLayout()
        grid.addWidget(QLabel("from"), 0, 1)
        grid.addWidget(QLabel("to"), 0, 2)
        for i, axis in enumerate(AXES):
            grid.addWidget(QLabel(axis), i + 1, 0)
            for col, key in ((1, "min"), (2, "max")):
                def setter(t, key=key, i=i):
                    c.box[key][i] = t or "0"
                    win.physics_changed()
                grid.addWidget(ExprEdit(win, lambda key=key, i=i: c.box[key][i], setter, show_value=False), i + 1, col)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        f.addRow("Box", grid)
    return box


def pretty_point(name: str) -> str:
    return {"G": "Γ", "Gamma": "Γ"}.get(name, name)


def swatch(color: str | None, size: int = 14):
    """A small square icon of a colour (material colours in lists, menus and the model tree); None: an empty
    outlined square (no material, e.g. a subtracting feature), the same size so names stay aligned."""
    from PySide6.QtGui import QIcon, QPainter, QPen, QPixmap
    pm = QPixmap(size, size)
    if color is not None:
        pm.fill(QColor(color))
        return QIcon(pm)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setPen(QPen(QColor(theme.muted()), 1))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(pm)


def feature_swatch(project, feature, size: int = 14):
    """The swatch of a geometry feature: its material's colour, or empty if it adds no material."""
    colors = project.material_colors()
    return swatch(colors.get(feature.material) if feature.op == "add" else None, size)


def quantity_menu(project, pick) -> QComboBox:
    """A "Quantities…" menu of the physics' named quantities (spec.quantities); choosing one calls
    pick(expression) and the menu returns to its title."""
    from ... import physics
    box = QComboBox()
    box.addItem("Quantities…", None)
    for label, expr in physics.of(project).quantities(project):
        box.addItem(f"{label}:  {expr}" if label != expr else expr, expr)
    box.setToolTip("Insert a physical quantity of this study")

    def chosen(i):
        expr = box.itemData(i)
        if expr is not None:
            box.blockSignals(True)
            box.setCurrentIndex(0)
            box.blockSignals(False)
            pick(expr)

    box.currentIndexChanged.connect(chosen)
    return box


REDUCTION_LABELS = {"": "Value", "mean": "Mean", "integral": "Integral", "max": "Maximum", "min": "Minimum"}


def _uses_field(expr: str, field_names) -> bool:
    return any(name in field_names for name in re.findall(r"[A-Za-z_]\w*", expr))


def builder_label(project, expr: str) -> str:
    """Axis label of an expression the quantity builder writes ("Maximum temperature in Pad (K)"); "" for any
    other expression."""
    from ... import physics
    labels = {e: lbl for lbl, e in physics.of(project).quantities(project)}
    m = re.fullmatch(r'(?:(mean|integral|max|min)\((.+?)(?:,\s*"([^"]*)")?\))|(.+)', expr.strip())
    if not m:
        return ""
    op, inner, place = m.group(1) or "", m.group(2) or m.group(4), m.group(3) or ""
    if inner not in labels:
        return ""
    label = labels[inner]
    unit = re.search(r"\s*\(([^()]*)\)$", label)
    name = label[:unit.start()] if unit else label
    if op == "integral":
        unit = None     # the unit changes (times m³)
    if op:
        name = (f"Integral of {name[0].lower() + name[1:]}" if op == "integral"
                else f"{REDUCTION_LABELS[op]} {name[0].lower() + name[1:]}")
    if place:
        name += f" in {place}"
    return name + (f" ({unit.group(1)})" if unit else "")


def quantity_builder(panel, editor, run, reduce: bool = True) -> tuple[QHBoxLayout, list[QPushButton]]:
    """One line that writes a plot's expression: <operation> of <quantity> in <place>; returns it and its
    buttons: Use replaces the expression (and computes it), Insert puts it at the cursor. `reduce` = False (values along a line or
    on a plane): the quantity alone. Fields must be reduced (mean, integral, max, min); scalars (time, freq,
    variables) are taken as they are. A y label written by the builder follows the expression."""
    from ... import physics
    p, pl = panel.p, getattr(panel, "pl", None)
    spec = physics.of(p)
    row = QHBoxLayout()
    op, qty, place = QComboBox(), QComboBox(), QComboBox()
    for box in (op, qty, place):
        box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        box.setMinimumContentsLength(5)
    for fn, label in REDUCTION_LABELS.items():
        op.addItem(label, fn)
    op.setCurrentIndex(op.findData("mean"))
    for label, expr in spec.quantities(p):
        qty.addItem(label, expr)
        qty.setItemData(qty.count() - 1, expr, Qt.ItemDataRole.ToolTipRole)
    place.addItem("whole model", "")
    for o in objects(p):
        place.addItem(o, o)
    op.setToolTip("How a field is reduced to one number (scalars such as time are taken as they are)")
    qty.setToolTip("The quantity; its expression is shown when hovering an item")
    place.setToolTip("The whole model, or one geometry feature or material")

    def sync():
        field = _uses_field(qty.currentData() or "", spec.field_names)
        if not field:
            op.setCurrentIndex(0)
        elif op.currentIndex() == 0:
            op.setCurrentIndex(op.findData("mean"))
        op.model().item(0).setEnabled(not field)
        op.setEnabled(field)
        place.setEnabled(field)

    def built() -> str:
        expr, fn, o = qty.currentData() or "", op.currentData(), place.currentData()
        if not reduce or not fn:
            return expr
        return f'{fn}({expr}, "{o}")' if o else f"{fn}({expr})"

    def use():
        old = editor.text().strip()
        expr = built()
        editor.setPlainText(expr)
        if pl is not None and hasattr(pl, "y_label") and (not pl.y_label or pl.y_label == builder_label(p, old)):
            pl.y_label = builder_label(p, expr)
            panel.win.mark_dirty()
        run()

    def insert():
        editor.insertPlainText(built())
        editor.setFocus()

    use_b, insert_b = QPushButton("Use"), QPushButton("Insert")
    use_b.setToolTip("Replace the expression with this one and compute it")
    insert_b.setToolTip("Insert this at the cursor, e.g. to build a ratio")
    use_b.clicked.connect(use)
    insert_b.clicked.connect(insert)
    if not reduce:
        row.addWidget(QLabel("Quantity"))
    if reduce:
        qty.currentIndexChanged.connect(sync)
        sync()
        row.addWidget(op, 2)
        row.addWidget(QLabel("of"))
    row.addWidget(qty, 4)
    if reduce:
        row.addWidget(QLabel("in"))
        row.addWidget(place, 3)
    return row, [use_b, insert_b]


def field_quantity_rows(panel, form, note: str):
    """The quantity a 3D field view shows, edited like every plot's expression: a large editor (empty = the
    physics' default) with Compute and the Quantities menu under it; evaluated at the nodes of the
    structure."""
    from ... import fields, physics
    win = panel.win
    spec = physics.of(panel.p)
    edit = ExprBox(win.field_quantity)
    edit.setPlaceholderText(spec.field_quantity(panel.p))
    edit.setToolTip("Expression; may span several lines. Press Compute (or Ctrl+Enter) to show it.")
    status = muted_label("")
    status.setVisible(False)

    def commit():
        text = edit.text().strip()
        try:
            if text:
                fields.check(text, display_evaluator(panel.p).values, spec.field_names, spec.scalar_names)
        except Exception as exc:
            edit.setStyleSheet(ERROR_STYLE)
            status.setText(str(exc))
            status.setVisible(True)
            return
        edit.setStyleSheet("")
        status.setVisible(False)
        win.set_field_quantity(" ".join(text.split()))

    def pick(expr):
        edit.setPlainText(expr)
        commit()

    edit.ctrl_enter = commit
    compute = QPushButton("Compute")
    compute.clicked.connect(commit)
    row = QHBoxLayout()
    row.addWidget(compute)
    row.addStretch(1)
    row.addWidget(quantity_menu(panel.p, pick))
    form.addRow(edit)
    form.addRow(row)
    form.addRow(status)
    form.addRow(muted_label(note))
