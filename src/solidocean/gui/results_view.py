"""Results side of the main window: running the problem, results plots, the selected solution and its 3D field,
exports. Physics-neutral: there are two families of results, field results (results/field.FieldResults: one
field per variable value and time step, shown by the physics' GUI adapter) and modal results (modes per
sweep point, shown as deformed, animated mode shapes)."""

from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog

from .. import meshing, units
from ..expressions import UNITS
from ..results.field import FieldResults
from .panels.common import model_box, pretty_point


class ResultsMixin:
    """Mixed into MainWindow; uses its project, spec, ui, viewer, plot, tree and task runner."""

    def init_results_state(self):
        self.mode_animate = True   # mode shapes animate by default
        self._pending_mode = False
        self._selected_in = None   # results in which the user picked / viewed a solution (for the plot circle)
        self.selection = {"i": 0, "j": 0, "b": 0}   # variable value, sweep point / time step, mode
        self.field_cache = {}      # derived nodal fields of the displayed solution (e.g. von Mises stress)
        self.results = self.results_project = None
        self.solver = None
        self._mode = None          # (mesh, u, title) currently displayed
        self._mode_cache = {}

    def clear_results(self):
        self.results = self.results_project = None
        self.solver = None
        self.current_plot = 0
        self.selection = {"i": 0, "j": 0, "b": 0}
        self._mode, self._mode_cache, self.field_cache = None, {}, {}
        self.plot.show_message("No results yet")
        self.viewer.stop_animation()

    @property
    def field_problem(self) -> bool:
        """The model's problem gives field results (rather than modes)."""
        return self.spec.field_results(self.project)

    @property
    def has_field_results(self) -> bool:
        return isinstance(self.results, FieldResults)

    def results_solver(self):
        """The solver the current results came from. Results loaded from a file have none, so one is built
        from the model they were loaded with: recomputing a mode shape or a quantity needs the model the
        results belong to, not whatever is being edited now."""
        if self.solver is None and self.results_project is not None:
            from .. import physics
            self.solver = physics.of(self.results_project).solver(self.results_project, log=self.log)
        return self.solver

    # ---- running ----------------------------------------------------------
    def compute(self):
        if not self._check():
            return
        snap = copy.deepcopy(self.project)
        solver = self.spec.solver(snap)
        quantities = [q for pl in snap.plots if not self.plot_unavailable(pl) for q in pl.expressions()]
        label = self.ui.run_label(snap)

        def work(sig):
            solver.log, solver.progress, solver.cancel = sig.log.emit, sig.progress.emit, self.cancel
            solver.mesh_cancel = self.cancel        # ✕ stops the meshing step too
            if self.field_problem:
                return solver.run(quantities)
            return solver.run(quantities, store_modes=True)

        def done(results):
            self.results, self.results_project, self.solver = results, snap, solver
            self.selection = {"i": 0, "j": len(getattr(results, "times", [0])) - 1, "b": 0}
            self._mode_cache, self._mode, self.field_cache = {}, None, {}
            if solver._model is not None:
                self.mesh = solver._model.mesh
            if snap.output.autosave:
                self.save_run(quiet=True)
            first = next((n for n, pl in enumerate(self.project.plots)
                          if pl.kind != "field" and not self.plot_unavailable(pl)), None)
            if first is None:
                first = next((n for n, pl in enumerate(self.project.plots) if pl.kind == "field"), None)
            self.rebuild_tree(("plot", first) if first is not None else ("results", None))

        self.run_task(label, work, done, cancellable=True)

    def results_summary(self) -> str:
        r = self.results
        if r is None:
            return "No results yet. Configure the Problem and press Run."
        if isinstance(r, FieldResults):
            return self.ui.results_summary(self, r)
        n_outer, n_inner, n_eigs = r.freqs.shape
        parts = [f"{n_outer * n_inner} solve(s), {n_eigs} eigenfrequencies each"]
        if r.outer_name:
            parts.append(f"variable sweep over {r.outer_name} ({n_outer} values)")
        if r.is_kpath:
            parts.append(f"k-path {'–'.join(pretty_point(lbl) for _, lbl in r.ticks)} ({n_inner} points)"
                         if r.ticks else f"{n_inner} wavevectors")
        if r.is_kmesh:
            parts.append(f"wavevector mesh ({n_inner} points, {len(r.ktris)} triangles)")
        if r.fourier is not None:
            from ..physics.solid.fourier import degeneracies
            n_degenerate = int((degeneracies(r.freqs) > 1).sum())
            parts.append(f"Fourier components at {r.fourier.shape[3]} reciprocal vectors"
                         + (f" ({n_degenerate} degenerate modes, basis-dependent)" if n_degenerate else ""))
        if r.info.get("elapsed_s") is not None:
            parts.append(f"{r.info['elapsed_s']} s")
        return "; ".join(parts) + "."

    # ---- plots --------------------------------------------------------------
    def show_plot(self, index: int | None = None, switch_tab: bool = True):
        """Draw results plot `index` (default: the current one), evaluating its quantity if needed."""
        if self.results is None or not self.project.plots:
            return
        index = self.current_plot if index is None else index
        index = min(max(index, 0), len(self.project.plots) - 1)
        self.current_plot = index
        pdef = self.project.plots[index]
        if pdef.kind == "field":
            return self.show_field()
        if switch_tab:
            self.views.setCurrentWidget(self.plot)
        why = self.plot_unavailable(pdef)
        if why:
            self.plot.show_message(why)
            return
        if isinstance(self.results, FieldResults):
            return self._show_field_plot(pdef)
        from ..physics.solid.modal import NeedsModes, compute_quantity
        evaluated = []
        for expr in pdef.expressions():
            vals = self.results.values(expr)
            if vals is None:
                try:
                    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                    vals = compute_quantity(self.results, self.results_project, expr)
                except NeedsModes:
                    self.plot.show_message(f"'{' '.join(expr.split())}' needs mode shapes.\n"
                                           "Press Compute in the Data box (re-solves all points).")
                    self._refresh_plot_panel()
                    return
                except Exception as exc:
                    self.plot.show_message(f"Cannot evaluate '{expr}':\n{exc}")
                    return
                finally:
                    QApplication.restoreOverrideCursor()
            evaluated.append(vals)
        values = evaluated[0] if evaluated else None     # a spectrum evaluates nothing per mode
        colors = evaluated[1] if len(evaluated) > 1 else None
        self.plot.show_results(self.results, self.project, pdef, values, colors)
        self.mark_selection()
        self._refresh_plot_panel()

    def _show_field_plot(self, pdef):
        """Plots of field results: against time or the swept variable, or along a line."""
        from ..results.field import draw
        if pdef.kind in ("line", "section"):  # evaluated point by point by the drawer itself
            self.show_line_scene()
            try:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self.plot.show_results(self.results, self.project, pdef, None, drawer=draw)
            except Exception as exc:
                where = "along the line" if pdef.kind == "line" else "on the cross-section"
                self.plot.show_message(f"Cannot plot '{pdef.quantity}' {where}:\n{exc}")
            finally:
                QApplication.restoreOverrideCursor()
            self._refresh_plot_panel()
            return
        vals = self.results.values(pdef.quantity)
        if vals is None:
            try:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                vals = self.results.compute(self.results_project, pdef.quantity)
            except Exception as exc:
                self.plot.show_message(f"Cannot evaluate '{pdef.quantity}':\n{exc}")
                return
            finally:
                QApplication.restoreOverrideCursor()
        self.plot.show_results(self.results, self.project, pdef, vals, drawer=draw)
        self.mark_selection()
        self._refresh_plot_panel()

    def _refresh_plot_panel(self):
        from .panels.results import PlotPanel
        panel = self.settings.widget()
        if isinstance(panel, PlotPanel):
            panel.update_status()

    def compute_plot(self, index: int):
        """Evaluate a plot's expressions and draw it, re-solving only if mode shapes are missing."""
        from .. import fields
        if self.results is None:
            return
        if isinstance(self.results, FieldResults):  # fields are always stored
            return self.show_plot(index)
        missing = [e for e in self.project.plots[index].expressions() if self.results.values(e) is None]
        if any(fields.uses_fields(e) for e in missing) and not self.results.has_all_modes:
            self.evaluate_plot(index)
        else:
            self.show_plot(index)

    def evaluate_plot(self, index: int):
        """Re-solve the sweep to evaluate a field quantity whose mode shapes were not kept."""
        if self.results is None:
            return
        solver = self.results_solver()
        if solver is None:
            return
        snap = copy.deepcopy(self.results_project)
        exprs = [e for e in self.project.plots[index].expressions() if self.results.values(e) is None]
        q = ", ".join(exprs)

        def work(sig):
            solver.project = snap
            solver.log = sig.log.emit
            solver.progress = sig.progress.emit
            solver.cancel = self.cancel
            solver.mesh_cancel = self.cancel
            return solver.run(quantities=exprs, store_modes=False)

        def done(res):
            self.results.quantities.update(res.quantities)
            self.show_plot(index)

        self.run_task(f"Evaluating {q}…", work, done, cancellable=True)
    def plot_unavailable(self, pl) -> str:
        """Why a plot cannot be shown with the current problem ("" if it can)."""
        return self.spec.plot_unavailable(self.project, pl)

    def add_plot_kind(self, kind: str):
        """Add a plot of one of the kinds available for the current physics (lines and cross-sections are placed
        in the middle of the model)."""
        self.add_plot(self.spec.new_plot(self.project, kind, model_box(self.project)))

    def add_plot(self, pdef=None):
        names = {pl.name for pl in self.project.plots}
        if pdef is None:
            return self.add_plot_kind(next(k for k in self.spec.plot_kinds if k in ("time", "quantity")))
        base, n = pdef.name, 2
        while pdef.name in names:
            pdef.name, n = f"{base} {n}", n + 1
        self.project.plots.append(pdef)
        self.mark_dirty()
        self.rebuild_tree(("plot", len(self.project.plots) - 1))

    def rename_plot(self, index: int):
        from PySide6.QtWidgets import QInputDialog
        pl = self.project.plots[index]
        name, ok = QInputDialog.getText(self, "Rename plot", "Name:", text=pl.name)
        name = name.strip()
        if ok and name and name not in {p.name for p in self.project.plots if p is not pl}:
            pl.name = name
            self.mark_dirty()
            self.rebuild_tree(("plot", index))
    def delete_plot(self, index: int):
        del self.project.plots[index]
        self.current_plot = 0
        self.mark_dirty()
        self.rebuild_tree(("results", None))
    # ---- the selected solution and its field --------------------------------------
    # ---- displacement field ---------------------------------------------
    def mode_title(self, i, j, b) -> str:
        r = self.results
        unit, lu = self.project.frequency_unit, self.project.length_unit
        k_scale, k_unit = units.axis_scale(units.INV_LENGTH, lu)
        title = (f"Mode {b + 1}: f = {r.freqs[i, j, b] / UNITS[unit]:.5g} {unit}, "
                 f"k = ({', '.join(f'{v * k_scale:.3g}' for v in r.k[i, j])}) {k_unit}")
        if r.outer_name:
            dim = r.info.get("outer_dim")
            title += f", {r.outer_name} = {units.format_value(r.outer_values[i], tuple(dim) if dim else None, self.project.display_units)}"
        return title
    def pick_mode(self, i, j, b):
        """Clicked in a results plot: open the field of that solution (mode, variable value, time step)."""
        self.selection = {"i": i, "j": j, "b": b}
        self.rebuild_tree(("plot", self.field_plot_index(create=True)))  # its panel shows the field

    def deselect(self):
        """The selected point was clicked again: no selection (the plot stays; the field view keeps showing
        the last solution until another point is picked)."""
        self._selected_in = None
    def mark_selection(self):
        """Circle the selected solution in the current 1D plot; kept across redraws and moved when the
        selection changes in the field view."""
        if self.results is None or self._selected_in is not self.results:  # nothing chosen in these results
            return
        current = self.project.plots[self.current_plot] if 0 <= self.current_plot < len(self.project.plots) else None
        if current is not None and current.kind == "line":
            self.plot.mark(-1, -1, -1)  # a solution is a whole curve here: nothing to circle
        else:
            sel = self.selection
            self.plot.mark(sel["i"], sel["j"], sel["b"] if not isinstance(self.results, FieldResults) else 0)

    def show_field(self):
        """Show the field of the selected solution: a mode shape, or the field drawn by the physics."""
        r = self.results
        if r is None:
            return
        if not isinstance(r, FieldResults):
            return self.show_mode()
        self._selected_in = r
        self.mark_selection()
        self._mode = None
        self._view = "field"
        self.views.setCurrentWidget(self.viewer)
        self.ui.show_field(self)

    def show_mode(self, i=None, j=None, b=None):
        """Compute (or reuse) the modes at sweep point (i, j) and display mode b."""
        r, snap = self.results, self.results_project
        if r is None or snap is None:
            return
        sel = self.selection
        i, j, b = (sel["i"] if i is None else i), (sel["j"] if j is None else j), (sel["b"] if b is None else b)
        self.selection = {"i": i, "j": j, "b": b}
        self._selected_in = r
        self.mark_selection()
        if (i, j) in r.modes and i in r.meshes:  # kept from the run
            self._mode_cache[(i, j)] = (r.meshes[i], r.modes[(i, j)].astype(complex))
        if (i, j) in self._mode_cache:
            mesh, u = self._mode_cache[(i, j)]
            self._display_mode(mesh, u[b], self.mode_title(i, j, b))
            return
        if self.busy:
            self._pending_mode = True  # shown when the current task finishes
            return
        ev = snap.evaluator({r.outer_name: r.outer_values[i]} if r.outer_name else None)
        k = r.k[i, j]
        solver = self.results_solver()
        if solver is None:
            return

        def work(sig):
            solver.log = sig.log.emit
            _, u = solver.solve_point(ev, k, n=r.freqs.shape[2], modes=True)
            return solver._model.mesh, u

        def done(res):
            if len(self._mode_cache) >= 8:  # keep memory bounded
                self._mode_cache.pop(next(iter(self._mode_cache)))
            self._mode_cache[(i, j)] = res
            self.show_mode()

        self.run_task("Computing displacement field…", work, done)

    def _display_mode(self, mesh, u, title):
        self._mode = (mesh, u, title)
        self._view = "mode"
        self.views.setCurrentWidget(self.viewer)
        comp = self.field_component if self.field_component in self.viewer.COMPONENTS else "|u|"
        self.viewer.show_mode(mesh, u, title, self.deform_scale, comp, style=self.field_style)
        if self.mode_animate:
            self.viewer.start_animation()

    # ---- field plots: their settings live in the plot (saved with the model) ----
    def field_plot_index(self, create: bool = False) -> int | None:
        """The field plot to show: the selected one, else the first; created if asked and missing."""
        plots = self.project.plots
        if 0 <= self.current_plot < len(plots) and plots[self.current_plot].kind == "field":
            return self.current_plot
        n = next((n for n, pl in enumerate(plots) if pl.kind == "field"), None)
        if n is None and create:
            plots.append(self.spec.field_plot())
            self.mark_dirty()
            n = len(plots) - 1
        return n

    def _field_plot(self):
        n = self.field_plot_index()
        return self.project.plots[n] if n is not None else self.spec.field_plot()

    @property
    def field_style(self) -> dict:
        st = self._field_plot().style
        return {"cmap": st.colormap, "colorbar": st.colorbar, "outline": st.outline, "title": st.title}

    @property
    def deform_scale(self) -> float:
        return self._field_plot().style.deform_scale

    @property
    def field_component(self) -> str:
        return self._field_plot().component

    @property
    def field_quantity(self) -> str:
        """The expression the 3D field view shows ("" = the physics' default)."""
        return self._field_plot().quantity

    def set_field_quantity(self, text: str):
        self._field_plot().quantity = text
        self.mark_dirty()
        self._redraw_field()

    def set_deform_scale(self, s):
        self._field_plot().style.deform_scale = s
        self.mark_dirty()
        self.viewer.set_mode_scale(s)

    def set_field_component(self, component: str):
        self._field_plot().component = component
        self.mark_dirty()
        self._redraw_field()

    def set_field_style(self, **kw):
        st = self._field_plot().style
        for key, value in kw.items():
            setattr(st, "colormap" if key == "cmap" else key, value)
        self.mark_dirty()
        self._redraw_field()

    def _redraw_field(self):
        if self._view == "field":
            self.show_field()
        elif self._mode is not None and self._view == "mode":
            self._display_mode(*self._mode)

    def set_mode_animate(self, on: bool):
        self.mode_animate = on
        if on and self._view == "mode":
            self.viewer.start_animation()
        else:
            self.viewer.stop_animation()
    def load_results(self):
        from ..physics.solid.modal import ModalResults
        path, _ = QFileDialog.getOpenFileName(self, "Load results", self.results_start(),
                                              "Results (results.npz)")
        if path:
            self.results = ModalResults.load(self.used_dir(path))
            self.results_project = copy.deepcopy(self.project)
            self._mode_cache = {}
            self._mode = None
            self.selection = {"i": 0, "j": 0, "b": 0}
            self.rebuild_tree(("plot", 0) if self.project.plots else ("results", None))
            self.log(f"Loaded {path}")

    # ---- saving results -------------------------------------------------
    def results_start(self, name: str = "") -> str:
        """Where a dialog for a result file opens: the results root once it exists, else next to the
        model file."""
        root = self.project.results_root() if self.project.path else None
        if root is not None and root.is_dir():
            return str(root / name) if name else str(root)
        return self.start_dir(name)

    def _run_project(self):
        """The project as it was run, with the [output] settings and the plots as they are now (plots added or
        changed after the run are post-processing of the same results)."""
        snap = copy.deepcopy(self.results_project)
        snap.output = copy.deepcopy(self.project.output)
        snap.plots = copy.deepcopy(self.project.plots)
        snap.path = self.project.path
        return snap

    def save_run(self, directory=None, quiet: bool = False) -> Path | None:
        """Write the files of the run in memory ([output] contents) into `directory`, by default the folder
        the [output] template resolves to. The same files the command line writes."""
        if self.results is None:
            self.log("Nothing to save: no results.")
            return None
        snap = self._run_project()
        if directory is None:
            if snap.path is None:
                self.log("Not saved: the model has no file yet — save the model, or use “Save to folder…”.")
                return None
            directory = snap.run_dir()
        plots = [pl for pl in snap.plots if pl.kind != "field" and not self.plot_unavailable(pl)]
        if isinstance(self.results, FieldResults):  # fields are stored: evaluate quantities of plots not shown yet
            for pl in plots:
                if pl.kind in ("quantity", "time") and self.results.values(pl.quantity) is None:
                    try:
                        self.results.compute(self.results_project, pl.quantity)
                    except Exception as exc:
                        self.log(f"Not saved: cannot evaluate '{pl.quantity}' ({pl.name}): {exc}")
        try:
            written = self.results.write_outputs(snap, Path(directory), plots)
        except Exception as exc:
            self.log(f"Error: could not write the results: {exc}")
            return None
        if quiet:
            self.log(f"Autosaved {len(written)} file(s) to {directory}")
        else:
            for f in written:
                self.log(f"Wrote {f}")
        return Path(directory)

    def export_results(self):
        """Write the files of the run in memory into a folder chosen by the user."""
        if self.results is None:
            self.log("Nothing to save: no results.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Save results to folder", self.results_start())
        if folder:
            self._last_dir = Path(folder)
            self.save_run(folder)

    def export_csv(self):
        """Save the current plot's data (CSV) under a name of the user's choice."""
        if self.results is None or not self.project.plots:
            self.log("Nothing to save: no results.")
            return
        pdef = self.project.plots[self.current_plot]
        if pdef.kind == "field":
            self.log("A 3D view has no table: use “Save this run” for the fields, or add a line or section plot.")
            return
        field = isinstance(self.results, FieldResults)
        is_freq = (pdef.quantity.strip() == "freq" and not field
                   and pdef.kind not in ("spectrum", "conductance"))
        default = "dispersion.csv" if is_freq else f"{pdef.name.replace(' ', '_')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save data", self.results_start(default),
                                              "CSV table (*.csv)")
        if not path:
            return
        self.used_dir(path)
        if not path.lower().endswith(".csv"):
            path += ".csv"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        from ..results.field import save_line_table, save_section_table
        try:
            if pdef.kind in ("spectrum", "conductance"):
                from ..physics.solid.spectra import save_table
                save_table(self.results, self.results_project, pdef, Path(path))
            elif is_freq:
                self.results.save_csv(Path(path), self.project.frequency_unit)
            elif field and pdef.kind == "line":
                save_line_table(self.results, self.results_project, pdef, Path(path))
            elif field and pdef.kind == "section":
                save_section_table(self.results, self.results_project, pdef, Path(path))
            else:
                if self.results.values(pdef.quantity) is None and field:
                    self.results.compute(self.results_project, pdef.quantity)
                if self.results.values(pdef.quantity) is None:
                    self.log(f"'{pdef.quantity}' has not been evaluated yet (press Compute).")
                    return
                self.results.save_table(Path(path), pdef.quantity)
        except Exception as exc:
            self.log(f"Error: could not write {path}: {exc}")
            return
        self.log(f"Wrote {path}")
    def export_mesh(self):
        if not self._check(geometry_only=True):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export mesh", self.results_start("mesh.msh"),
                                              "Gmsh mesh (*.msh)")
        if not path:
            return
        self.used_dir(path)
        snap = copy.deepcopy(self.project)
        self.run_task("Exporting mesh…",
                      lambda sig: meshing.generate(snap, snap.evaluator(), save_msh=path, cancel=self.cancel),
                      lambda mesh: self.log(f"Wrote {path}"), cancellable=True)
    def export_animation(self):
        if self._view != "mode" or self._mode is None:
            self.log("Show a displacement field first.")
            return
        sel = self.selection
        name = f"mode{sel['b'] + 1}_k{sel['j']}" + (f"_p{sel['i']}" if self.results.outer_name else "")
        path, _ = QFileDialog.getSaveFileName(self, "Export animation", self.results_start(f"{name}.gif"),
                                              "Animated GIF (*.gif)")
        if not path:
            return
        self.used_dir(path)
        if not path.lower().endswith(".gif"):
            path += ".gif"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            n = self.viewer.export_animation(path)
            self.log(f"Wrote {path} ({n} frames)")
        except Exception as exc:
            self.log(f"Error: could not export animation: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
    def export_plot(self):
        pl = self.project.plots[self.current_plot] if self.project.plots else None
        name = (pl.name.replace(" ", "_") if pl else "plot") + ".png"
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", self.results_start(name),
                                              "Images (*.png *.pdf *.svg)")
        if path:
            self.used_dir(path)
            self.plot.figure.savefig(path, dpi=200)
            self.log(f"Wrote {path}")
