"""Results panels shared by every physics: the results overview and the 1D plot settings."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox,
)

from ... import physics, units
from .. import theme
from ...plots import MAP_KINDS, VECTOR_KINDS, plot_label
from ...project import AXES
from .common import (ERROR_STYLE, ExprBox, ExprEdit, Panel, muted_label, outer_labels, pretty_point,
                     quantity_builder, spoiler)


# ---------------------------------------------------------------------------
class ResultsPanel(Panel):
    title = "Results"

    def build(self):
        self.outer.addWidget(muted_label(self.win.results_summary()))
        spec = physics.of(self.p)
        f = self.form("Plots")
        for n, pl in enumerate(self.p.plots):
            quantity = " ".join(pl.quantity.split())
            what = f"  ({quantity})" if quantity and pl.kind not in ("dispersion", "colored") else ""
            b = QPushButton(plot_label(pl.kind, pl.name) + what)
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            why = self.win.plot_unavailable(pl)
            if why:
                b.setToolTip(why)
                b.setStyleSheet(b.styleSheet() + f" color: {theme.disabled_text()};")
            b.clicked.connect(lambda _=False, n=n: self.win.select_node("plot", n))
            f.addRow(b)
        add = QGroupBox("Add a plot")
        grid = QGridLayout(add)
        for n, k in enumerate(spec.plot_kinds):
            b = QPushButton(plot_label(k, spec.plot_titles[k]))
            b.clicked.connect(lambda _=False, k=k: self.win.add_plot_kind(k))
            grid.addWidget(b, n // 2, n % 2)
        self.outer.addWidget(add)
        if self.win.results is not None:
            self.button_row(("Save this run", lambda: self.win.save_run()), ("Load results…", self.win.load_results))
            self.outer.addWidget(muted_label(
                "Save this run writes quantities.csv, the fields as .npz and the model into the run folder of "
                "the Output section — the same files the command line writes. Single plots: the Save plot "
                "button of a plot." if self.win.field_problem else
                "Save this run writes dispersion.csv (wavevectors in rad/m, frequencies in the selected "
                "unit), results.npz (SI) and the model into the run folder of the Output section — the same "
                "files the command line writes. Single plots: the Save plot button of a plot."))
        else:
            self.button_row(("Load results…", self.win.load_results))


# qualitative palettes: neighbouring bands get clearly different colours
DISPERSION_PALETTES = ["SolidOcean", "Paired", "Pastel1", "Pastel2", "Set1", "Set2", "Set3", "tab10",
                       "tab20", "Dark2", "Accent"]


CONTINUOUS_MAPS = ["Blues", "Reds", "Greens", "Purples", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu_r", "turbo", "jet", "gray"]


PLOT_HELP = {
    "dispersion": "Eigenfrequencies against the sweep. Click a point to see that mode's displacement field. "
                  "Right-click Results to add plots.",
    "quantity": "Any quantity, one value per mode, against the sweep. Click a point to see that mode's "
                "displacement field.",
    "colored": "Eigenfrequencies against the sweep, each point coloured by a quantity, e.g. the out-of-plane "
               "fraction mean(abs(uz)^2): 0 = in-plane mode, 1 = flexural. Click a point to see the mode.",
}


class PlotPanel(Panel):

    def __init__(self, win, index):
        self.index = index
        self.pl = win.project.plots[index]
        self.title = self.pl.name
        from ..physics import of as ui_of
        self.ui = ui_of(win.project)
        self.help = self.ui.plot_help(win.project, self.pl.kind) or PLOT_HELP.get(self.pl.kind, "")
        self.formalism = self.ui.plot_formalism(win.project, self.pl)
        self.q_status = None
        super().__init__(win)

    def build(self):
        # Dataset (what was computed / which part of it) -> Quantity (what is plotted) -> Plot (axes,
        # ranges, labels) -> Style (cosmetics)
        self._dataset_box()
        self._quantity_box()
        self._plot_box()
        self._style_box()
        if self.win.results is not None:
            self.button_row(("Save plot…", self.win.export_plot), ("Save data…", self.win.export_csv))

    def _refresh(self):
        self.win.mark_dirty()
        self.win.show_plot(self.index)

    def _restyle(self, attr):
        def fn(v):
            setattr(self.pl.style, attr, v)
            self._refresh()
        return fn

    @property
    def fieldres(self) -> bool:
        """Field results (one field per variable value and time step) rather than modes."""
        return self.win.field_problem

    def _dataset_box(self):
        """The computed solutions this plot uses: which sweep is on the x axis and which part of the other
        sweep (variable value, wavevector or time step) is shown."""
        r, pl = self.win.results, self.pl
        d = self.form("Dataset")
        if r is None:
            d.addRow(muted_label("No results yet. Configure the Problem and press Run; the computed sweeps and "
                                 "time steps can then be chosen here."))
            return
        d.addRow(muted_label(self.win.results_summary()))
        if self.fieldres and (pl.kind == "line" or pl.kind in MAP_KINDS):
            self._line_dataset_rows(d, overlay=pl.kind == "line")
        elif self.fieldres:
            if (pl.kind == "time" and r.outer_name) or (pl.kind == "quantity" and len(r.times) > 1):
                self._heat_sweep_rows(d)
        elif pl.kind in ("spectrum", "conductance") and r.is_kmesh:
            if r.outer_name:
                at = QComboBox()
                for n, label in enumerate(outer_labels(self.win)):
                    at.addItem(label, n)
                at.setCurrentIndex(max(0, at.findData(max(pl.at, 0))))
                at.currentIndexChanged.connect(lambda i: (setattr(pl, "at", at.itemData(i)), self._refresh()))
                d.addRow("Variable sweep", at)
        elif pl.kind in ("kmap", "surfaces") and r.is_kmesh:
            self._map_rows(d)
        elif r.is_kpath and r.outer_name:
            self._sweep_rows(d)

    def _quantity_box(self):
        from ... import fields
        pl, kind = self.pl, self.pl.kind
        d = self.form("Colour by quantity" if kind == "colored" else "Vector and background" if kind in VECTOR_KINDS
                      else "Quantity")
        if kind in VECTOR_KINDS:
            self._vector_rows(d)
        if kind == "dispersion":
            d.addRow(muted_label("Eigenfrequencies (freq), one curve per band. For other quantities add a "
                                 "'Quantity vs sweep' or a coloured dispersion plot."))
            return
        if kind == "spectrum":
            return self._spectrum_rows(d)
        if kind == "conductance":
            return self._conductance_rows(d)
        attr = "color_by" if kind == "colored" else "quantity"
        if kind == "colored":
            d.addRow(muted_label("The points are the eigenfrequencies; this expression sets their colour:"))
        q = ExprBox(getattr(pl, attr))
        q.setPlaceholderText(self.ui.quantity_placeholder(self.p))
        q.setToolTip("Expression; may span several lines. Press Compute (or Ctrl+Enter) to evaluate and plot.")
        d.addRow(q)
        compute = QPushButton("Compute")
        row = QHBoxLayout()
        row.addWidget(compute)
        row.addStretch(1)
        self.q_status = muted_label("")
        d.addRow(self.q_status)

        def run_compute():
            text = q.text().strip()
            try:
                if not text and kind not in VECTOR_KINDS:      # the colour map behind a vector plot is optional
                    raise ValueError("enter an expression")
                if text:
                    fields.check(text, self.p.evaluator().values)
            except Exception as exc:
                q.setStyleSheet(ERROR_STYLE)
                self.q_status.setText(str(exc))
                self.q_status.setVisible(True)
                return
            q.setStyleSheet("")
            if text != getattr(pl, attr):
                setattr(pl, attr, text)
                self.win.mark_dirty()
            if self.win.results is None:
                self.q_status.setText("No results yet: the quantity is computed with the next Run.")
                self.q_status.setVisible(True)
                return
            self.win.compute_plot(self.index)

        compute.clicked.connect(run_compute)
        q.ctrl_enter = run_compute
        builder, buttons = quantity_builder(self, q, run_compute, reduce=kind not in ("line", "section") + VECTOR_KINDS)
        for b in buttons:
            row.addWidget(b)
        d.insertRow(d.rowCount() - 1, builder)     # the builder, then Compute / Use / Insert
        d.insertRow(d.rowCount() - 1, row)
        if kind == "line":
            d.addRow(muted_label(self.ui.line_help(self.p)))
        elif kind == "section":
            d.addRow(muted_label(self.ui.line_help(self.p).replace("along the line", "on the plane")
                                 + " Empty: the default quantity of the physics."))
        elif kind in VECTOR_KINDS:
            d.addRow(muted_label("Optional colour map behind the " + ("arrows" if kind == "quiver" else "streamlines")
                                 + " (a scalar quantity). Empty: they are coloured by their in-plane magnitude."))
        self._quantity_help(d)
        self.update_status()

    def _vector_rows(self, d):
        """The three components of the vector of an arrow or streamline plot; its part in the plane is drawn."""
        from ... import fields
        pl = self.pl
        spec = physics.of(self.p)
        d.addRow(muted_label("Components of the vector (expressions); its part lying in the plane is drawn."))
        row = QHBoxLayout()
        for n, axis in enumerate(AXES):
            row.addWidget(QLabel(axis))
            e = QLineEdit(pl.vector[n])
            e.setPlaceholderText(spec.vector[n])
            e.setToolTip(f"The {axis} component, e.g. {spec.vector[n]}")

            def commit(e=e, n=n):
                text = e.text().strip()
                try:
                    if not text:
                        raise ValueError("enter an expression")
                    fields.check(text, self.p.evaluator().values, spec.field_names, spec.scalar_names)
                except Exception as exc:
                    e.setStyleSheet(ERROR_STYLE)
                    e.setToolTip(str(exc))
                    return
                e.setStyleSheet("")
                pl.vector[n] = text
                self._refresh()

            e.editingFinished.connect(commit)
            row.addWidget(e, 1)
        d.addRow(row)

    def _vector_style_rows(self, d):
        """Arrows: how many, how long, how thick; streamlines: how dense, how thick, the size of the heads; the
        colour used over a colour map."""
        pl, st = self.pl, self.pl.style

        def spin(attr, lo, hi, step, decimals, tip, target=pl):
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            box.setValue(getattr(target, attr))
            box.setToolTip(tip)
            box.valueChanged.connect(lambda v: (setattr(target, attr, v), self._refresh()))
            return box
        if pl.kind == "quiver":
            n = QSpinBox()
            n.setRange(2, 200)
            n.setValue(pl.arrows)
            n.setToolTip("Arrows across the longer side of the plane")
            n.valueChanged.connect(lambda v: (setattr(pl, "arrows", v), self._refresh()))
            d.addRow("Arrows", n)
            d.addRow("Arrow length", spin("arrow_scale", 0.1, 10, 0.1, 2, "Relative to the spacing between arrows: "
                                          "1 makes the longest arrow as long as one spacing"))
            d.addRow("Arrow width", spin("arrow_width", 0.1, 10, 0.1, 2, "Relative to the default width"))
        else:
            d.addRow("Density", spin("stream_density", 0.1, 5, 0.1, 1, "Closeness of the streamlines"))
            d.addRow("Line width", spin("line_width", 0.2, 6, 0.2, 1, "Width of the streamlines", st))
            d.addRow("Arrow size", spin("arrow_scale", 0.1, 10, 0.1, 2, "Size of the arrow heads along the lines"))
        swatch = QPushButton()
        swatch.setFixedWidth(46)
        swatch.setToolTip("Colour of the arrows or lines drawn over a colour map; without one they are coloured by "
                          "their magnitude")

        def paint():
            swatch.setStyleSheet(f"background: {st.color}; border: 1px solid #777; border-radius: 4px;")

        def pick():
            from PySide6.QtWidgets import QColorDialog
            c = QColorDialog.getColor(QColor(st.color), self, "Colour")
            if c.isValid():
                self._restyle("color")(c.name())
                paint()

        swatch.clicked.connect(pick)
        paint()
        row = QHBoxLayout()
        row.addWidget(swatch)
        row.addStretch(1)
        d.addRow("Colour over a map", row)

    def _line_dataset_rows(self, d, overlay: bool = True):
        """Line plot: which variable value(s) and which time step; cross-sections: one variable value."""
        r, pl = self.win.results, self.pl
        if r.outer_name:
            at = QComboBox()
            if overlay:
                at.addItem("All (overlaid)", -1)
            for n, label in enumerate(outer_labels(self.win)):
                at.addItem(label, n)
            at.setCurrentIndex(max(0, at.findData(pl.at if overlay else max(pl.at, 0))))
            at.currentIndexChanged.connect(lambda i: (setattr(pl, "at", at.itemData(i)), self._refresh()))
            d.addRow("Variable sweep", at)
        if len(r.times) > 1:
            from ...results.style import time_axis
            scale, unit = time_axis(self.p)
            tm = QComboBox()
            tm.addItem("Final time", -1)
            for j, t in enumerate(r.times):
                tm.addItem(f"{t / scale:.4g} {unit}", j)
            tm.setCurrentIndex(max(0, tm.findData(pl.at_time)))
            tm.currentIndexChanged.connect(lambda i: (setattr(pl, "at_time", tm.itemData(i)), self._refresh()))
            d.addRow("Time", tm)

    def _heat_sweep_rows(self, d):
        """Time plot: which variable value (or all overlaid); sweep plot: at which time step."""
        r, pl = self.win.results, self.pl
        at = QComboBox()
        if pl.kind == "time":
            at.addItem("All (overlaid)", -1)
            for n, label in enumerate(outer_labels(self.win)):
                at.addItem(label, n)
            label = "Variable sweep"
        else:
            from ...results.style import time_axis
            scale, unit = time_axis(self.p)
            at.addItem("Final time", -1)
            for j, t in enumerate(r.times):
                at.addItem(f"{t / scale:.4g} {unit}", j)
            label = "Time"
        at.setCurrentIndex(max(0, at.findData(pl.at)))
        at.currentIndexChanged.connect(lambda i: (setattr(pl, "at", at.itemData(i)), self._refresh()))
        d.addRow(label, at)

    def _map_rows(self, d):
        """Band map and band surfaces: which band(s) of the wavevector mesh are drawn, and for which value
        of the variable sweep."""
        r, pl = self.win.results, self.pl
        n_bands = r.freqs.shape[2]
        surfaces = pl.kind == "surfaces"
        band = QSpinBox()
        band.setRange(1, n_bands)
        band.setValue(min(pl.band, n_bands - 1) + 1)
        band.setToolTip("Bands are numbered by ascending frequency at each wavevector, so a band is the "
                        "n-th lowest surface over the zone (branches are not followed through crossings)")
        band.valueChanged.connect(lambda v: (setattr(pl, "band", v - 1), self._refresh()))
        d.addRow("First band" if surfaces else "Band", band)
        if surfaces:
            count = QSpinBox()
            count.setRange(0, n_bands)
            count.setSpecialValueText("all")
            count.setValue(min(pl.bands, n_bands))
            count.setToolTip("How many bands are drawn from the first one up; 0 draws all of them")
            count.valueChanged.connect(lambda v: (setattr(pl, "bands", v), self._refresh()))
            d.addRow("Bands drawn", count)
            d.addRow(muted_label("Drag inside the plot to turn the view."))
        if r.outer_name:
            at = QComboBox()
            for n, label in enumerate(outer_labels(self.win)):
                at.addItem(label, n)
            at.setCurrentIndex(max(0, at.findData(max(pl.at, 0))))
            at.currentIndexChanged.connect(lambda i: (setattr(pl, "at", at.itemData(i)), self._refresh()))
            d.addRow("Variable sweep", at)

    def _sweep_rows(self, d):
        from ...results.modal import sweep_of
        r, pl = self.win.results, self.pl
        sw = QComboBox()
        sw.addItem("Wavevector path", "kpath")
        sw.addItem(f"Variable {r.outer_name}", "variable")
        sw.setCurrentIndex(max(0, sw.findData(sweep_of(r, pl))))
        d.addRow("Sweep axis", sw)
        at = QComboBox()
        at_label = QLabel()
        d.addRow(at_label, at)

        def fill_at():
            # the other sweep: variable values when x is the k-path, k points when x is the variable
            at_label.setText("Variable sweep" if sweep_of(r, pl) == "kpath" else "Wavevector")
            at.blockSignals(True)
            at.clear()
            at.addItem("All (overlaid)", -1)
            if sweep_of(r, pl) == "kpath":
                labels = outer_labels(self.win)
            else:
                names = {round(pos, 12): lbl for pos, lbl in r.ticks}
                labels = [f"k point {j + 1}" + (f" ({pretty_point(names[round(float(x), 12)])})"
                                                if round(float(x), 12) in names else "")
                          for j, x in enumerate(r.x)]
            for n, label in enumerate(labels):
                at.addItem(label, n)
            at.setCurrentIndex(max(0, at.findData(pl.at)))
            at.blockSignals(False)

        fill_at()
        sw.currentIndexChanged.connect(
            lambda i: (setattr(pl, "sweep", sw.itemData(i)), setattr(pl, "at", -1), fill_at(), self._refresh()))
        at.currentIndexChanged.connect(lambda i: (setattr(pl, "at", at.itemData(i)), self._refresh()))

    def _quantity_help(self, form):
        """Collapsible reference; its open/closed state is remembered for the session."""
        help_items = physics.of(self.p).quantity_help(self.p)
        html = "<br>".join(f"<b>{k}:</b> {v}" for k, v in help_items.items())
        for widget in spoiler(self.win, "Available expressions", html, "show_quantity_help"):
            form.addRow(widget)

    def update_status(self):
        """Note below the expression: only shown when Compute will have to re-solve the sweep."""
        from ... import fields
        if self.q_status is None:
            return
        r = self.win.results
        pending = [e for e in self.pl.expressions() if r is not None and r.values(e) is None]
        text = ""
        try:
            if pending and any(fields.uses_fields(e) for e in pending) and not r.has_all_modes:
                text = "Mode shapes were not kept in memory: Compute will re-solve all points."
        except Exception:
            pass
        self.q_status.setText(text)
        self.q_status.setVisible(bool(text))

    def _range_row(self, attrs, unit_label, log: str = ""):
        """[min] – [max] unit [Log]: plain numbers are in the displayed unit, or give units e.g. 10[GHz];
        `log`: the PlotStyle flag of a logarithmic axis, ticked at the end of the row."""
        st = self.pl.style
        row = QHBoxLayout()
        edits = []
        for n, attr in enumerate(attrs):
            e = QLineEdit(str(getattr(st, attr)))
            e.setPlaceholderText("auto")
            e.setMaximumWidth(80)

            def commit(e=e, attr=attr):
                text = e.text().strip()
                try:
                    if text:
                        self.p.evaluator()(text)
                    e.setStyleSheet("")
                    self._restyle(attr)(text)
                except Exception as exc:
                    e.setStyleSheet(ERROR_STYLE)
                    e.setToolTip(str(exc))

            e.editingFinished.connect(commit)
            row.addWidget(e)
            if n == 0:
                row.addWidget(QLabel("–"))
            edits.append(e)
        unit = QLabel(unit_label)
        row.addWidget(unit)
        if log:
            box = QCheckBox("Log")
            box.setChecked(getattr(st, log))
            box.setToolTip("Logarithmic axis (only positive values are shown)")
            box.toggled.connect(self._restyle(log))
            row.addWidget(box)
            edits.append(box)
        row.addStretch(1)
        return row, edits, unit

    def _line_rows(self, d):
        """Start and end of the line (expressions), number of points and the x axis."""
        pl = self.pl
        for attr, label in (("start", "Line start"), ("end", "Line end")):
            row = QHBoxLayout()
            vec = getattr(pl, attr)
            for n, axis in enumerate(AXES):
                row.addWidget(QLabel(axis))

                def setter(t, vec=vec, n=n):
                    vec[n] = t or "0"
                    self._refresh()
                    if self.win.results is None:
                        self.win.show_line_scene()

                row.addWidget(ExprEdit(self.win, lambda vec=vec, n=n: vec[n], setter, show_value=False), 1)
            d.addRow(label, row)
        npts = QSpinBox()
        npts.setRange(2, 100000)
        npts.setValue(pl.points)
        npts.setToolTip("Number of evaluation points along the line")
        npts.valueChanged.connect(lambda v: (setattr(pl, "points", v), self._refresh()))
        d.addRow("Points", npts)
        xa = QComboBox()
        for key, label in (("s", "Distance from the start"), ("x", "x coordinate"), ("y", "y coordinate"),
                           ("z", "z coordinate")):
            xa.addItem(label, key)
        xa.setCurrentIndex(max(0, xa.findData(pl.line_axis)))
        xa.currentIndexChanged.connect(lambda i: (setattr(pl, "line_axis", xa.itemData(i)), self._refresh()))
        d.addRow("X axis", xa)

    def _section_rows(self, d):
        """The plane of a cross-section: a coordinate plane, rotated about x, y, z, through an origin; its
        sampling; colour range and label."""
        from ...plots import SECTION_PLANES
        pl, st = self.pl, self.pl.style
        plane = QComboBox()
        for key in SECTION_PLANES:
            plane.addItem(f"{key} plane", key)
        plane.setCurrentIndex(max(0, plane.findData(pl.plane)))
        plane.currentIndexChanged.connect(lambda i: self._section_changed("plane", plane.itemData(i)))
        d.addRow("Plane", plane)
        for attr, label, tip in (("origin", "Through point", "A point of the plane (expressions)"),
                                 ("rotation", "Rotation (deg)", "Rotation of the plane about x, then y, then z, "
                                                                "around the point above")):
            row = QHBoxLayout()
            vec = getattr(pl, attr)
            for n, axis in enumerate(AXES):
                row.addWidget(QLabel(axis))

                def setter(t, vec=vec, n=n):
                    vec[n] = t or "0"
                    self._section_changed()
                e = ExprEdit(self.win, lambda vec=vec, n=n: vec[n], setter, show_value=False)
                e.setToolTip(tip)
                row.addWidget(e, 1)
            d.addRow(label, row)
        res = QSpinBox()
        res.setRange(10, 1000)
        res.setValue(pl.resolution)
        res.setToolTip("Sample points along the longer side of the section")
        res.valueChanged.connect(lambda v: self._section_changed("resolution", v))
        d.addRow("Resolution", res)
        d.addRow("Colour label", self._text_row("color_label", "automatic (the quantity)"))
        row, _, _ = self._range_row(("color_min", "color_max"), "")
        d.addRow("Colour range", row)

    def _section_changed(self, attr=None, value=None):
        if attr:
            setattr(self.pl, attr, value)
        self._refresh()
        if self.win.results is None:
            self.win.show_line_scene()      # where the plane cuts the structure

    def _text_row(self, attr, placeholder="automatic"):
        e = QLineEdit(getattr(self.pl, attr))
        e.setPlaceholderText(placeholder)
        e.editingFinished.connect(lambda: (setattr(self.pl, attr, e.text().strip()), self._refresh()))
        return e

    def _plot_box(self):
        """Axes: titles, units, axis type and ranges; colour range and label; title and colour bar."""
        from ...results.modal import sweep_of
        r, st, pl = self.win.results, self.pl.style, self.pl
        d = self.form("Plot")
        if pl.kind == "line":
            self._line_rows(d)
        if pl.kind in MAP_KINDS:
            return self._section_rows(d)
        if pl.kind in ("kmap", "surfaces"):
            return self._zone_rows(d)
        if pl.kind == "spectrum":
            return self._spectrum_plot_rows(d)
        if pl.kind == "conductance":
            return self._conductance_plot_rows(d)
        is_freq = pl.quantity.strip() == "freq" and not self.fieldres
        if is_freq:
            fu = QComboBox()
            fu.addItems(list(units.FREQUENCY_UNITS))
            fu.setCurrentText(self.p.frequency_unit)
            d.addRow("Frequency unit", fu)
        row, _, yunit = self._range_row(("y_min", "y_max"), self.p.frequency_unit if is_freq else "", "y_log")
        d.addRow("Frequency range" if is_freq else "Y range", row)
        if is_freq:
            def set_unit(u):
                yunit.setText(u)
                self._set_unit(u)
            fu.currentTextChanged.connect(set_unit)
        if pl.kind in ("quantity", "time", "line"):
            d.addRow("Y axis title", self._text_row("y_label"))

        if self.fieldres:
            row, _, _ = self._range_row(("x_min", "x_max"), "", "x_log")
            d.addRow("X range", row)
        elif r is not None and sweep_of(r, pl) == "kpath":
            from ...plots import K_UNITS
            from ...results.modal import k_axis_label
            xa = QComboBox()
            xa.addItem("Symmetry points", "points")
            xa.addItem("Wavevector", "k")
            xa.setCurrentIndex(max(0, xa.findData(st.x_axis)))
            ku = QComboBox()
            for u in K_UNITS:
                ku.addItem(k_axis_label(u), u)
            ku.setCurrentIndex(max(0, ku.findData(st.k_unit)))
            row = QHBoxLayout()
            row.addWidget(xa)
            row.addWidget(ku)
            row.addStretch(1)
            d.addRow("X axis", row)
            xrow, xedits, xunit = self._range_row(("x_min", "x_max"), k_axis_label(st.k_unit))
            d.addRow("X range", xrow)

            def sync():
                k = st.x_axis == "k"
                ku.setEnabled(k)
                for w in xedits:
                    w.setEnabled(k)
                xunit.setText(k_axis_label(st.k_unit) if k else "")

            xa.currentIndexChanged.connect(lambda i: (self._restyle("x_axis")(xa.itemData(i)), sync()))
            ku.currentIndexChanged.connect(lambda i: (self._restyle("k_unit")(ku.itemData(i)), sync()))
            sync()
        elif r is not None:
            from ...results.style import outer_axis
            _, label = outer_axis(r, self.p.display_units)
            unit = label.split("(")[-1].rstrip(")") if "(" in label else ""
            xrow, _, _ = self._range_row(("x_min", "x_max"), unit, "x_log")
            d.addRow("X range", xrow)

        d.addRow("X axis title", self._text_row("x_label"))
        if pl.kind == "colored":
            d.addRow("Colour label", self._text_row("color_label", "automatic (the expression)"))
            row, _, _ = self._range_row(("color_min", "color_max"), "")
            d.addRow("Colour range", row)

    def _conductance_plot_rows(self, d):
        """The temperature axis: the range the curve is computed over, and how many points it takes."""
        pl = self.pl
        row = QHBoxLayout()
        for attr in ("t_min", "t_max"):
            edit = QLineEdit(str(getattr(pl, attr)))
            edit.setMaximumWidth(80)

            def commit(edit=edit, attr=attr):
                text = edit.text().strip()
                try:
                    self.p.evaluator()(text)
                    edit.setStyleSheet("")
                    setattr(pl, attr, text)
                    self._refresh()
                except Exception as exc:
                    edit.setStyleSheet(ERROR_STYLE)
                    edit.setToolTip(str(exc))

            edit.editingFinished.connect(commit)
            row.addWidget(edit)
            if attr == "t_min":
                row.addWidget(QLabel("–"))
        row.addWidget(QLabel("K"))
        points = QSpinBox()
        points.setRange(2, 2000)
        points.setValue(pl.t_points)
        points.setPrefix("points ")
        points.valueChanged.connect(lambda v: (setattr(pl, "t_points", v), self._refresh()))
        row.addWidget(points)
        row.addStretch(1)
        d.addRow("Temperature range", row)
        d.addRow("", self._checks((("Log temperature", "x_log"),)))
        row, _, _ = self._range_row(("y_min", "y_max"), "", "y_log")
        d.addRow("Y range", row)
        d.addRow("Y axis title", self._text_row("y_label"))
        d.addRow("X axis title", self._text_row("x_label"))

    def _spectrum_plot_rows(self, d):
        """A spectrum has frequency on the x axis and the quantity on the y axis, both with a range."""
        fu = QComboBox()
        fu.addItems(list(units.FREQUENCY_UNITS))
        fu.setCurrentText(self.p.frequency_unit)
        d.addRow("Frequency unit", fu)
        row, _, xunit = self._range_row(("x_min", "x_max"), self.p.frequency_unit, "x_log")
        d.addRow("Frequency range", row)

        def set_unit(u):
            xunit.setText(u)
            self._set_unit(u)

        fu.currentTextChanged.connect(set_unit)
        row, _, _ = self._range_row(("y_min", "y_max"), "", "y_log")
        d.addRow("Y range", row)
        d.addRow("Y axis title", self._text_row("y_label"))
        d.addRow("X axis title", self._text_row("x_label"))

    def _zone_rows(self, d):
        """Plots over the Brillouin zone: the unit and range of the quantity (the colour scale of a map, the
        vertical axis of the surfaces) and the ranges of the two wavevector axes. Neither has a logarithmic
        axis: a 3D axes cannot, and the colour scale of a map is linear."""
        from ...plots import K_UNITS
        from ...results.modal import k_axis_label
        pl = self.pl
        is_freq = pl.quantity.strip() == "freq"
        if is_freq:
            fu = QComboBox()
            fu.addItems(list(units.FREQUENCY_UNITS))
            fu.setCurrentText(self.p.frequency_unit)
            d.addRow("Frequency unit", fu)
        row, _, yunit = self._range_row(("y_min", "y_max"), self.p.frequency_unit if is_freq else "")
        label = "Frequency range" if is_freq else "Value range"
        d.addRow(f"{label} (colour)" if pl.kind == "kmap" else label, row)
        if is_freq:
            def set_unit(u):
                yunit.setText(u)
                self._set_unit(u)
            fu.currentTextChanged.connect(set_unit)
        ku = QComboBox()
        ku.addItems(list(K_UNITS))
        ku.setCurrentText(pl.style.k_unit)
        ku.currentTextChanged.connect(self._restyle("k_unit"))
        d.addRow("Wavevector unit", ku)
        for attrs, name in ((("x_min", "x_max"), "kx range"), (("ky_min", "ky_max"), "ky range")):
            row, _, unit = self._range_row(attrs, "")
            unit.setText(k_axis_label(pl.style.k_unit))
            ku.currentTextChanged.connect(lambda u, lbl=unit: lbl.setText(k_axis_label(u)))
            d.addRow(name, row)

    def _complete_note(self) -> str:
        """Where the computed bands stop being complete, and where they reach: the two are not the same,
        since the topmost band dips somewhere in the zone (often at Gamma, where folded bands crowd)."""
        from ...physics.solid import spectra
        r, pl = self.win.results, self.pl
        if r is None or not getattr(r, "is_kmesh", False):
            return "no results yet."
        whole = spectra.complete_to(r, max(pl.at, 0), pl.band, pl.bands)
        reached = spectra.highest(r, max(pl.at, 0), pl.band, pl.bands)
        return (f"every band is present below {whole / 1e9:.3g} GHz, though some reach "
                f"{reached / 1e9:.3g} GHz.")

    def _spectrum_rows(self, d):
        """A spectrum is summed over every band of the mesh, so it is chosen from a list rather than
        written as an expression; the heat flux also needs a temperature."""
        from ...physics.solid.spectra import SPECTRA, SPECTRUM_HELP, SPECTRUM_TITLES
        pl = self.pl
        which = QComboBox()
        for name in SPECTRA:
            which.addItem(SPECTRUM_TITLES[name], name)
        which.setCurrentIndex(max(0, which.findData(pl.spectrum)))
        help_text = muted_label(SPECTRUM_HELP.get(pl.spectrum, ""))

        def changed(i):
            pl.spectrum = which.itemData(i)
            if pl.name in SPECTRUM_TITLES.values():      # a plot still carrying its default name
                pl.name = SPECTRUM_TITLES[pl.spectrum]
            self._refresh()
            self.win.rebuild_tree(("plot", self.index))

        which.currentIndexChanged.connect(changed)
        d.addRow("Spectrum", which)
        d.addRow(help_text)
        if pl.spectrum == "flux":
            temp = ExprEdit(self.win, lambda: pl.temperature,
                            lambda t: (setattr(pl, "temperature", t or "1[K]"), self._refresh()))
            temp.setToolTip("Temperature of the radiating boundary, e.g. 0.3[K]")
            d.addRow("Temperature", temp)
        samples = QSpinBox()
        samples.setRange(2, 20000)
        samples.setValue(pl.bins)
        samples.setToolTip("How many frequencies the spectrum is sampled at")
        samples.valueChanged.connect(lambda v: (setattr(pl, "bins", v), self._refresh()))
        d.addRow("Frequency samples", samples)
        d.addRow("", self._checks((("Carry past the complete range", "show_incomplete"),), on=pl))
        d.addRow(muted_label("Every band of the mesh takes part, and the spectrum stops where the computed "
                             "bands stop being complete — " + self._complete_note() + " Beyond it states "
                             "are missing, so the curve only sags; carried on, it is a lower bound and the "
                             "line is marked."))

    def _conductance_rows(self, d):
        """The heat flux integrated over the bands: which integral, and a warning when the computed bands do not reach the occupied frequencies."""
        import numpy as np

        from ...physics.solid import spectra
        from ...physics.solid.spectra import (TRANSPORT, TRANSPORT_HELP, TRANSPORT_TITLES,
                                              thermal_frequency)
        pl = self.pl
        which = QComboBox()
        for name in TRANSPORT:
            which.addItem(TRANSPORT_TITLES[name], name)
        which.setCurrentIndex(max(0, which.findData(pl.transport)))

        def changed(i):
            pl.transport = which.itemData(i)
            if pl.name in TRANSPORT_TITLES.values():
                pl.name = TRANSPORT_TITLES[pl.transport]
            self._refresh()
            self.win.rebuild_tree(("plot", self.index))

        which.currentIndexChanged.connect(changed)
        d.addRow("Quantity", which)
        d.addRow(muted_label(TRANSPORT_HELP.get(pl.transport, "")))
        d.addRow("", self._checks((("Carry past the complete range", "show_incomplete"),), on=pl))
        r = self.win.results
        if r is not None and getattr(r, "is_kmesh", False):
            top = (spectra.highest if pl.show_incomplete else spectra.complete_to)(r, max(pl.at, 0),
                                                                                   pl.band, pl.bands)
            try:
                hottest = self.p.evaluator()(pl.t_max)
            except Exception:
                hottest = 0.0
            occupied = thermal_frequency(hottest)
            note = (self._complete_note() + f" At {hottest:.3g} K the occupied frequencies reach about "
                    f"{occupied / 1e9:.3g} GHz.")
            if top < 9 * occupied:
                note += (f" The integral stops there, so the curve is a lower bound and flattens above "
                         f"roughly {top / (9 * thermal_frequency(1.0)):.3g} K. Compute more "
                         f"eigenfrequencies (Problem → Number of solutions) to carry it higher.")
            d.addRow(muted_label(note))

    def _zone_style_box(self):
        """Style of the plots over the zone: a map can show its wavevectors and a colour bar; the surfaces
        take one colour per band from the palette. Lines, single colours and legends play no part in either."""
        st, pl = self.pl.style, self.pl
        d = self.form("Style")
        if pl.kind == "kmap":
            row = QHBoxLayout()
            pts = QCheckBox("Show")
            pts.setChecked(st.markers)
            pts.toggled.connect(self._restyle("markers"))
            size = QDoubleSpinBox()
            size.setRange(0.5, 15)
            size.setSingleStep(0.5)
            size.setValue(st.marker_size)
            size.setPrefix("size ")
            size.valueChanged.connect(self._restyle("marker_size"))
            row.addWidget(pts)
            row.addWidget(size)
            row.addStretch(1)
            d.addRow("Wavevectors", row)
            levels = QSpinBox()
            levels.setRange(2, 200)
            levels.setValue(st.levels)
            levels.valueChanged.connect(self._restyle("levels"))
            d.addRow("Contour levels", levels)
        maps = CONTINUOUS_MAPS if pl.kind == "kmap" else DISPERSION_PALETTES
        cmap = QComboBox()
        cmap.addItems(list(maps) + ([st.colormap] if st.colormap not in maps else []))
        cmap.setCurrentText(st.colormap)
        cmap.setToolTip("Colour scale of the map" if pl.kind == "kmap" else "Distinct colours, one per band")
        cmap.currentTextChanged.connect(self._restyle("colormap"))
        d.addRow("Colour scale", cmap)
        checks = (("Grid", "grid"), ("Title", "title"), ("Colour bar", "colorbar")) if pl.kind == "kmap" \
            else (("Grid", "grid"), ("Title", "title"))
        d.addRow("", self._checks(checks))

    def _curve_style_box(self):
        """Style of a single curve (a spectrum): its points, its line and its colour."""
        st, setter = self.pl.style, self._restyle
        d = self.form("Style")
        row = QHBoxLayout()
        pts = QCheckBox("Show")
        pts.setChecked(st.markers)
        pts.toggled.connect(setter("markers"))
        size = QDoubleSpinBox()
        size.setRange(0.5, 15)
        size.setSingleStep(0.5)
        size.setValue(st.marker_size)
        size.setPrefix("size ")
        size.valueChanged.connect(setter("marker_size"))
        row.addWidget(pts)
        row.addWidget(size)
        row.addStretch(1)
        d.addRow("Points", row)
        row = QHBoxLayout()
        line = QComboBox()
        line.addItems(["solid", "dashed", "dotted", "none"])
        line.setCurrentText(st.line_style)
        line.currentTextChanged.connect(setter("line_style"))
        width = QDoubleSpinBox()
        width.setRange(0.2, 6)
        width.setSingleStep(0.2)
        width.setValue(st.line_width)
        width.setPrefix("width ")
        width.valueChanged.connect(setter("line_width"))
        row.addWidget(line)
        row.addWidget(width)
        row.addStretch(1)
        d.addRow("Lines", row)
        swatch = QPushButton()
        swatch.setFixedWidth(46)

        def paint():
            swatch.setStyleSheet(f"background: {st.color}; border: 1px solid #777; border-radius: 4px;")

        def pick():
            from PySide6.QtWidgets import QColorDialog
            c = QColorDialog.getColor(QColor(st.color), self, "Curve colour")
            if c.isValid():
                setter("color")(c.name())
                paint()

        swatch.clicked.connect(pick)
        paint()
        row = QHBoxLayout()
        row.addWidget(swatch)
        row.addStretch(1)
        d.addRow("Colour", row)
        d.addRow("", self._checks((("Grid", "grid"), ("Title", "title"))))

    def _style_box(self):
        if self.pl.kind in ("spectrum", "conductance"):
            return self._curve_style_box()
        if self.pl.kind in ("kmap", "surfaces"):
            return self._zone_style_box()
        if self.pl.kind == "colored":
            return self._colored_style_box()
        if self.pl.kind in MAP_KINDS:
            d = self.form("Style")
            cmap = QComboBox()
            st = self.pl.style
            cmap.addItems(CONTINUOUS_MAPS + ([st.colormap] if st.colormap not in CONTINUOUS_MAPS else []))
            cmap.setCurrentText(st.colormap)
            cmap.currentTextChanged.connect(self._restyle("colormap"))
            d.addRow("Colour scale", cmap)
            d.addRow("", self._checks((("Title", "title"), ("Colour bar", "colorbar"))))
            if self.pl.kind in VECTOR_KINDS:
                self._vector_style_rows(d)
            return
        st = self.pl.style
        d = self.form("Style")
        setter = self._restyle

        row = QHBoxLayout()
        pts = QCheckBox("Show")
        pts.setChecked(st.markers)
        pts.toggled.connect(setter("markers"))
        size = QDoubleSpinBox()
        size.setRange(0.5, 15)
        size.setSingleStep(0.5)
        size.setValue(st.marker_size)
        size.setPrefix("size ")
        size.valueChanged.connect(setter("marker_size"))
        row.addWidget(pts)
        row.addWidget(size)
        row.addStretch(1)
        d.addRow("Points", row)

        row = QHBoxLayout()
        line = QComboBox()
        line.addItems(["solid", "dashed", "dotted", "none"])
        line.setCurrentText(st.line_style)
        line.currentTextChanged.connect(setter("line_style"))
        width = QDoubleSpinBox()
        width.setRange(0.2, 6)
        width.setSingleStep(0.2)
        width.setValue(st.line_width)
        width.setPrefix("width ")
        width.valueChanged.connect(setter("line_width"))
        row.addWidget(line)
        row.addWidget(width)
        row.addStretch(1)
        d.addRow("Lines", row)

        row = QHBoxLayout()
        mode = QComboBox()
        mode.addItem("One colour per curve" if self.fieldres else "One colour per band", "bands")
        mode.addItem("Single colour", "single")
        mode.setCurrentIndex(max(0, mode.findData(st.colors)))
        swatch = QPushButton()
        swatch.setFixedWidth(46)
        swatch.setToolTip("Line colour")

        def paint_swatch():
            swatch.setStyleSheet(f"background: {st.color}; border: 1px solid #777; border-radius: 4px;")
            swatch.setEnabled(st.colors == "single")

        def pick_colour():
            from PySide6.QtWidgets import QColorDialog
            c = QColorDialog.getColor(QColor(st.color), self, "Line colour")
            if c.isValid():
                setter("color")(c.name())
                paint_swatch()

        swatch.clicked.connect(pick_colour)
        mode.currentIndexChanged.connect(lambda i: (setter("colors")(mode.itemData(i)), paint_swatch()))
        paint_swatch()
        row.addWidget(mode)
        row.addWidget(swatch)
        row.addStretch(1)
        d.addRow("Colours", row)

        cmap = QComboBox()
        cmap.addItems(DISPERSION_PALETTES + ([st.colormap] if st.colormap not in DISPERSION_PALETTES else []))
        cmap.setCurrentText(st.colormap)
        cmap.setToolTip("Distinct colours for 'one colour per band' and for overlaid variable-sweep values")
        cmap.currentTextChanged.connect(setter("colormap"))
        d.addRow("Colour scale", cmap)
        d.addRow("", self._checks((("Grid", "grid"), ("Title", "title"), ("Legend", "legend"))))

    def _colored_style_box(self):
        st, setter = self.pl.style, self._restyle
        d = self.form("Style")
        size = QDoubleSpinBox()
        size.setRange(0.5, 15)
        size.setSingleStep(0.5)
        size.setValue(st.marker_size)
        size.valueChanged.connect(setter("marker_size"))
        d.addRow("Point size", size)

        row = QHBoxLayout()
        line = QComboBox()
        line.addItems(["none", "solid", "dashed", "dotted"])
        line.setCurrentText(st.line_style)
        line.setToolTip("Thin gray lines joining the points of each band")
        line.currentTextChanged.connect(setter("line_style"))
        width = QDoubleSpinBox()
        width.setRange(0.2, 6)
        width.setSingleStep(0.2)
        width.setValue(st.line_width)
        width.setPrefix("width ")
        width.valueChanged.connect(setter("line_width"))
        row.addWidget(line)
        row.addWidget(width)
        row.addStretch(1)
        d.addRow("Guide lines", row)

        cmap = QComboBox()
        cmap.addItems(CONTINUOUS_MAPS + ([st.colormap] if st.colormap not in CONTINUOUS_MAPS else []))
        cmap.setCurrentText(st.colormap)
        cmap.currentTextChanged.connect(setter("colormap"))
        d.addRow("Colour scale", cmap)
        d.addRow("", self._checks((("Grid", "grid"), ("Title", "title"), ("Colour bar", "colorbar"))))

    def _checks(self, items, on=None) -> QHBoxLayout:
        """A row of check boxes: (label, attribute) of the plot's style, or of `on` (the plot itself)."""
        row = QHBoxLayout()
        holder = self.pl.style if on is None else on

        def setter(attr):
            def fn(v):
                setattr(holder, attr, v)
                self._refresh()
                if on is not None:
                    self.win.select_node("plot", self.index)    # the notes below depend on it
            return fn

        for label, attr in items:
            cb = QCheckBox(label)
            cb.setChecked(getattr(holder, attr))
            cb.toggled.connect(setter(attr))
            row.addWidget(cb)
        row.addStretch(1)
        return row

    def _set_unit(self, u):
        self.win.set_unit("frequency_unit", u)   # the model's preferred frequency unit
        self._refresh()
