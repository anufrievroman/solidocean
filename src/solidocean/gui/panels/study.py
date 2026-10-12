"""Study panels shared by every physics: variable sweep."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit

from ... import units
from ...project import VariableSweep
from .common import ERROR_STYLE, Panel, muted_label


class SweepPanel(Panel):
    """Shared enable checkbox for the optional sweeps."""
    attr = ""

    def holder(self):
        """The settings object holding the sweep (shared study; a physics' own study for its sweeps)."""
        return self.p.study

    def sweep(self):
        st = self.holder()
        if getattr(st, self.attr) is None:
            setattr(st, self.attr, self._draft)
        return getattr(st, self.attr)

    def changed(self):
        self.sweep()  # store the draft on first edit
        self.win.study_changed()

    def enable_box(self, label: str, obj):
        cb = QCheckBox(label)
        cb.setChecked(getattr(self.holder(), self.attr) is not None and obj.enabled)
        cb.toggled.connect(lambda on: (setattr(obj, "enabled", on), self.changed()))
        self.outer.addWidget(cb)


class VariableSweepPanel(SweepPanel):
    title = "Variable sweep"
    help = ("Outer loop: the whole study is repeated for each value of one variable, for example a hole "
            "radius, a thickness, or a wavevector component used in Periodicity. The geometry and mesh are "
            "rebuilt when they depend on the variable.\n\n"
            "Example:   r  in  range(0.1*a, 0.05*a, 0.4*a)   → r = 0.1a, 0.15a, … 0.4a\n\n"
            "Values: range(start, step, stop), linspace(start, stop, n), or a list [v1, v2, …]. "
            "Units are allowed, e.g. [100[nm], 150[nm], 200[nm]].")
    attr = "sweep"

    def build(self):
        st = self.p.study
        self._draft = st.sweep or VariableSweep(next(iter(self.p.variables), ""), "", enabled=False)
        par = self._draft
        self.enable_box("Enable variable sweep", par)

        row = QHBoxLayout()
        pname = QComboBox()
        pname.setEditable(True)
        pname.addItems(list(self.p.variables))
        pname.setCurrentText(par.variable)
        pname.setMinimumWidth(90)
        row.addWidget(pname)
        row.addWidget(QLabel("in"))
        values = QLineEdit(par.values)
        values.setPlaceholderText("range(start, step, stop)")
        row.addWidget(values, 1)
        self.outer.addLayout(row)
        count = muted_label("")
        self.outer.addWidget(count)

        def preview():
            try:
                ev = self.p.evaluator()
                arr = ev.array(par.values)
                dim = units.dimension(self.p.variables.get(par.variable, "0"), ev)
                lo, hi = (units.format_value(v, dim, self.p.display_units) for v in (arr[0], arr[-1]))
                count.setText(f"{len(arr)} value{'s' if len(arr) != 1 else ''}: {lo} … {hi}" if len(arr) > 1
                              else f"1 value: {lo}")
                values.setStyleSheet("")
            except Exception as exc:
                count.setText(str(exc) if par.values else "")
                values.setStyleSheet(ERROR_STYLE if par.values else "")

        def set_values():
            par.values = values.text().strip()
            self.changed()
            preview()

        def set_param(v):
            par.variable = v.strip()
            self.changed()
            preview()

        values.editingFinished.connect(set_values)
        pname.currentTextChanged.connect(set_param)
        preview()


def sweep_summary(st) -> dict[str, str]:
    """Short text of the study nodes every physics has (empty string = inactive)."""
    from ...fem import eigen
    sw = st.active_sweep
    try:
        solver = eigen.backend(st.solver)
    except Exception:
        solver = st.solver
    return {"sweep": sw.variable if sw else "",
            "solver": ("auto: " if st.solver == "auto" else "") + solver}


def solver_box(panel):
    """The sparse direct solver every study uses (study.solver)."""
    import os
    from ...fem import eigen
    st = panel.p.study
    avail = eigen.available_backends()
    f = panel.form()
    combo = QComboBox()
    for key, label in [("auto", f"Auto (fastest available: {avail[0]})"), ("mumps", "MUMPS"),
                       ("superlu", "SuperLU (scipy)")]:
        combo.addItem(label + ("" if key == "auto" or key in avail else "  — not installed"), key)
        if key != "auto" and key not in avail:
            combo.model().item(combo.count() - 1).setEnabled(False)
    combo.setCurrentIndex(max(0, combo.findData(st.solver)))
    combo.currentIndexChanged.connect(lambda i: (setattr(st, "solver", combo.itemData(i)), panel.win.study_changed()))
    f.addRow("Linear solver", combo)
    panel.outer.addWidget(muted_label(
        f"Installed linear solvers: {', '.join(avail)}.  CPU cores: {os.cpu_count()}.  MUMPS (conda-forge "
        "python-mumps) is typically 10-20x faster than SuperLU for 3D models."))
    return f
