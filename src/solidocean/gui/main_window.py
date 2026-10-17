"""Main window: model tree | settings | graphics, toolbar on top, messages below.

The window is physics-neutral: everything specific to a physics comes from its GUI adapter (gui/physics/,
`self.ui`) and its core PhysicsSpec (`self.spec`). Running problems, results plots and field views are in
results_view.ResultsMixin.
"""

from __future__ import annotations

import copy
import threading
import time
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QPlainTextEdit,
    QProgressBar, QScrollArea, QSplitter, QTreeWidget, QTreeWidgetItem, QWidget,
)

from .. import meshing, physics, units
from ..plots import plot_label
from ..project import FEATURE_TITLES, FEATURE_TYPES, Project
from . import physics as gui_physics
from . import theme
from .panels import model as mp
from .panels import output as op
from .panels import results as rp
from .panels.common import model_box, swatch
from .panels.problem import VariableSweepPanel
from .plot_canvas import PlotCanvas
from .view_area import ViewArea
from .results_view import ResultsMixin
from .viewer import DEFAULT_OPACITY, Viewer
from .worker import Task


def new_project() -> Project:
    from ..templates import create
    return create("empty")


class MainWindow(ResultsMixin, QMainWindow):
    SECTIONS = [("variables", "Variables"), ("materials", "Materials"), ("objects", "Objects"),
                ("mesh", "Mesh"), ("physics_list", "Physics"), ("problem", "Problem"), ("results", "Results"),
                ("output", "Output")]
    GEOMETRY_KINDS = ("objects", "feature", "materials", "material")
    PANELS = {"model": mp.ModelPanel, "variables": mp.VariablesPanel, "materials": mp.MaterialsPanel,
              "physics_list": mp.PhysicsListPanel, "mesh_default": mp.MeshDefaultPanel,
              "objects": mp.ObjectsPanel, "mesh": mp.MeshPanel, "sweep": VariableSweepPanel,
              "results": rp.ResultsPanel, "output": op.OutputPanel}

    def __init__(self, path: str | None = None):
        super().__init__()
        self.pool = QThreadPool.globalInstance()
        self.cancel = threading.Event()
        self.busy = False
        self._stashed_plots = {}   # plots of the other physics, kept while switching back and forth
        self._last_dir = Path.home()   # where file dialogs open while the model has no file of its own
        self.init_results_state()
        self.current_plot = 0
        self._view = None
        self._build_ui()
        self.resize(1500, 900)
        if path:
            self.open_file(path)
        else:
            self.set_project(new_project())

    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        tb = self.addToolBar("Main")
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        def action(text, fn, shortcut=None, tip=""):
            a = QAction(text, self)
            a.triggered.connect(fn)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setToolTip(tip or text)
            tb.addAction(a)
            return a

        # unicode symbols as lightweight icons; U+FE0E forces text (not emoji) rendering
        T = "\ufe0e"
        action("New", self.new, QKeySequence.StandardKey.New, "New model")
        action("Open…", self.open_dialog, QKeySequence.StandardKey.Open, "Open a model file")
        action("Save", self.save, QKeySequence.StandardKey.Save, "Save the model")
        action("Save As…", self.save_as, QKeySequence.StandardKey.SaveAs, "Save the model to a new file")
        tb.addSeparator()
        self.run_actions = [
            action("Build ◱", self.build_geometry, "F6", "Build the geometry (F6)"),
            action("Mesh ◪", self.build_mesh, "F7", "Build the mesh (F7)"),
            action(f"Run ▶{T}", self.compute, "F8", "Run the problem (F8)"),
        ]
        # right end of the toolbar: what a task is doing, its progress and the cancel button (shown while busy)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        self.status = QLabel()
        self.status.setMaximumWidth(450)
        self.status.setStyleSheet(f"color: {theme.muted()}; padding-right: 6px;")
        tb.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(250)
        self.progress_action = tb.addWidget(self.progress)   # a toolbar widget is shown/hidden via its action
        self.progress_action.setVisible(False)
        self.cancel_button = QToolButton()
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setText("✕")
        self.cancel_button.setToolTip("Cancel the run (stops after the current step)")
        self.cancel_button.setAutoRaise(True)
        self.cancel_button.clicked.connect(self.stop)
        self.cancel_action = tb.addWidget(self.cancel_button)
        self.cancel_action.setVisible(False)

        # the model's sections as a sidebar: headings (Model, Variables, ..., Results) with their items below,
        # always expanded, no tree lines or arrows
        self.tree = QTreeWidget()
        self.tree.setObjectName("sidebar")
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setItemsExpandable(False)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.setIndentation(14)
        self.tree.currentItemChanged.connect(self._on_tree_select)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        for key in (QKeySequence.StandardKey.Delete, QKeySequence(Qt.Key.Key_Backspace)):
            sc = QShortcut(key, self.tree)
            sc.setContext(Qt.ShortcutContext.WidgetShortcut)
            sc.activated.connect(self._delete_current)

        self.settings = QScrollArea()
        self.settings.setWidgetResizable(True)
        # fit panels to the width (wrap text) instead of scrolling sideways
        self.settings.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.viewer = Viewer()
        self.plot = PlotCanvas()
        self.plot.picked.connect(self.pick_mode)
        self.plot.deselected.connect(self.deselect)
        self.views = ViewArea(self.viewer, self.plot)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.tree)
        split.addWidget(self.settings)
        split.addWidget(self.views)
        split.setSizes([230, 470, 800])
        self.setCentralWidget(split)

        self.messages = QPlainTextEdit()
        self.messages.setObjectName("messages")
        self.messages.setReadOnly(True)
        self.messages.setMaximumBlockCount(5000)
        self.log_dock = dock = QDockWidget("Messages", self)
        dock.setWidget(self.messages)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setTitleBarWidget(QWidget())  # no "Messages" title bar: the log speaks for itself
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.setVisible(False)   # collapsed by default: ↑ Log in the status bar opens it
        self.log_button = QToolButton()
        self.log_button.setObjectName("logButton")
        self.log_button.setAutoRaise(True)
        self.log_button.clicked.connect(lambda: self.show_log(not self.log_dock.isVisible()))
        self.statusBar().addWidget(self.log_button)
        self.show_log(False)

    def show_log(self, on: bool):
        """Open (about four lines; the user can drag it taller) or collapse the messages panel."""
        self.log_dock.setVisible(on)
        self.log_button.setText("↓ Log" if on else "↑ Log")
        self.log_button.setToolTip("Hide the messages" if on else "Show the messages")
        if on:
            line = self.messages.fontMetrics().lineSpacing()
            self.resizeDocks([self.log_dock], [4 * line + 12], Qt.Orientation.Vertical)

    def show_status(self, text: str, ms: int = 0):
        """Task status at the right of the toolbar; with `ms` it clears itself after that time."""
        self.status.setText(text)
        self.status.setToolTip(text)
        if ms:
            QTimer.singleShot(ms, lambda: self.status.text() == text and self.status.clear())

    # ---- project / tree -------------------------------------------------
    @property
    def spec(self):
        """The core description of the model's physics (physics.PhysicsSpec)."""
        return physics.of(self.project)

    @property
    def ui(self):
        """The GUI adapter of the model's physics (gui/physics)."""
        return gui_physics.of(self.project)

    def set_project(self, project: Project):
        self.project = project
        self._stashed_plots = {}
        self.preview = None
        self.mesh = None
        self.custom_lattice = False
        self._view = None
        self.dirty = False
        self.viewer.clear()
        self.viewer.set_length_unit(project.length_unit)
        self.clear_results()
        self.rebuild_tree(("model", None))
        self._update_title()
        QTimer.singleShot(0, self._build_on_load)

    def _build_on_load(self):
        """A model that was just opened shows its geometry right away (the surface preview is quick); a model
        whose geometry has errors is left alone (they show when the user builds it)."""
        if self.preview is None and not self.busy and not self.project.validate_geometry():
            self.build_geometry()

    def rebuild_tree(self, select=None, refresh_panel=True):
        cur = select or self._current_node()
        self.tree.blockSignals(True)
        self.tree.clear()
        ui = self.ui
        heading = QFont(self.tree.font())
        heading.setBold(True)
        items = {}
        for key, label in [("model", "Model")] + self.SECTIONS:
            if key == "problem":
                label = ui.problem_label(self.project)
            it = QTreeWidgetItem([label])
            it.setFont(0, heading)
            it.setSizeHint(0, QSize(0, 30))           # air above each heading
            it.setData(0, Qt.ItemDataRole.UserRole, (key, None))
            self.tree.addTopLevelItem(it)
            items[(key, None)] = it
        root = items[("model", None)]
        root.setToolTip(0, self.project.name or "")
        semibold = QFont(self.tree.font())
        semibold.setWeight(QFont.Weight.DemiBold)
        for spec in physics.active(self.project):     # the physics of the model (one for now), its conditions below
            it = QTreeWidgetItem(items[("physics_list", None)], [spec.title])
            it.setFont(0, semibold)
            it.setData(0, Qt.ItemDataRole.UserRole, ("physics", None))
            items[("physics", None)] = it
        colors = self.project.material_colors()
        for i, m in enumerate(self.project.materials):
            it = QTreeWidgetItem(items[("materials", None)], [m.name])
            it.setIcon(0, swatch(colors[m.name], 10))
            it.setData(0, Qt.ItemDataRole.UserRole, ("material", i))
            items[("material", i)] = it
        for i, f in enumerate(self.project.objects):
            label = f"{f.name} ({f.type}{', ' + f.array_label if f.arrayed else ''})" + (("  −" + (f" from {f.on}" if f.on else "")) if f.op == "subtract" else "")
            if not f.enabled:
                label += "  (bypassed)"
            it = QTreeWidgetItem(items[("objects", None)], [label])
            it.setIcon(0, swatch(colors.get(f.material) if f.op == "add" else None, 10))
            if not f.enabled:
                it.setForeground(0, QColor(theme.disabled_text()))
                font = QFont(it.font(0))
                font.setItalic(True)
                it.setFont(0, font)
            it.setData(0, Qt.ItemDataRole.UserRole, ("feature", i))
            items[("feature", i)] = it
        for label, node in mp.mesh_nodes(self.project):     # the element sizes: the default, then the regions
            it = QTreeWidgetItem(items[("mesh", None)], [label])
            it.setData(0, Qt.ItemDataRole.UserRole, node)
            items[node] = it
        for parent, nodes in (("physics", ui.physics_nodes(self.project)), ("problem", ui.problem_nodes(self.project))):
            for label, node, dimmed in nodes:
                it = QTreeWidgetItem(items[(parent, None)], [label])
                if dimmed:
                    it.setForeground(0, QColor(theme.disabled_text()))
                it.setData(0, Qt.ItemDataRole.UserRole, node)
                items[node] = it
        for n, pl in enumerate(self.project.plots):
            it = QTreeWidgetItem(items[("results", None)], [plot_label(pl.kind, pl.name)])
            why = self.plot_unavailable(pl)
            if self.results is None or why:
                it.setForeground(0, QColor(theme.disabled_text()))
            it.setToolTip(0, why or self.spec.plot_titles.get(pl.kind, pl.kind))
            it.setData(0, Qt.ItemDataRole.UserRole, ("plot", n))
            items[("plot", n)] = it
        self.tree.expandAll()
        self.tree.blockSignals(False)
        target = items.get(tuple(cur) if cur else ("model", None), root)
        if refresh_panel:
            self.tree.setCurrentItem(target)
        else:
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(target)
            self.tree.blockSignals(False)

    def select_node(self, kind, index=None):
        self.rebuild_tree((kind, index))
    def _current_node(self):
        it = self.tree.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it else None
    def _panel(self, kind, idx):
        """The settings panel of a tree node."""
        ui = self.ui
        if kind == "plot":
            self.current_plot = idx
            return ui.field_panel(self) if self.project.plots[idx].kind == "field" else rp.PlotPanel(self, idx)
        if kind == "material":
            return mp.MaterialPanel(self, idx)
        if kind == "feature":
            return mp.FeaturePanel(self, idx)
        if kind == "mesh_region":
            return mp.MeshRegionPanel(self, idx)
        if kind in ui.indexed_panels:
            return ui.indexed_panels[kind](self, idx)
        if kind == "physics":
            return ui.physics_panel(self)
        if kind == "problem":
            return ui.problem_panel(self)
        return {**self.PANELS, **ui.node_panels}[kind](self)

    def _on_tree_select(self, item, _prev):
        if item is None:
            return
        # A widget of the old panel may have the keyboard focus (e.g. its Run button, or a field being
        # edited). Detaching the panel would then make Qt pass the focus on inside a widget that is no
        # longer in the scroll area, which crashes; so give the focus to the tree first. That can commit an
        # edit, which may rebuild the tree (deleting `item`): look the current node up again afterwards.
        old = self.settings.widget()
        focused = QApplication.focusWidget()
        if old is not None and focused is not None and old.isAncestorOf(focused):
            self.tree.setFocus(Qt.FocusReason.OtherFocusReason)
            item = self.tree.currentItem()
            if item is None or self.settings.widget() is not old:   # a rebuild already showed its panel
                return
        kind, idx = item.data(0, Qt.ItemDataRole.UserRole)
        panel = self._panel(kind, idx)
        # fit the panel to the available width (rows shrink) rather than clipping its right edge
        panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        # Panels often trigger this from one of their own widgets' signals (e.g. a combo box that
        # rebuilds the tree). Deleting the old panel right away would destroy that widget while its
        # signal is still running (crash), so detach it now and let Qt delete it afterwards.
        old = self.settings.takeWidget()
        self.settings.setWidget(panel)
        if old is not None:
            old.hide()
            old.deleteLater()
        # show the matching graphics
        if kind in self.GEOMETRY_KINDS:
            self.refresh_view("geometry")
        elif kind in ("physics", "physics_list") or kind in self.ui.highlight_nodes:
            if not (self.refresh_view("geometry") or self.refresh_view("mesh")) \
                    and not self.busy and not self.project.validate_geometry():
                self.build_geometry()  # boundary highlights need something to draw on
        elif kind in ("mesh", "mesh_default", "mesh_region"):
            self.refresh_view("mesh")
        elif kind == "plot" and self.results is not None:
            self.show_plot(idx)
        elif kind == "plot" and self._line_plot() is not None:
            self.show_line_scene()  # before a run: where the line will be

    def _line_plot(self):
        """The selected plot if it probes the structure along a line or a plane (line, cross-section)."""
        node = self._current_node()
        if node and node[0] == "plot" and 0 <= node[1] < len(self.project.plots):
            pl = self.project.plots[node[1]]
            return pl if pl.kind in ("line", "section") else None
        return None
    def show_line_scene(self):
        """Line or cross-section plot selected: the structure, semi-transparent, with the line or the plane drawn
        through it (in the Structure tab; the plot stays in front)."""
        if self.preview is None and self.mesh is None:
            if not self.busy and not self.project.validate_geometry():
                self.build_geometry(show=False)
            return
        self.refresh_view("geometry" if self.preview is not None else "mesh", switch_tab=False)
    def _overlay_line(self):
        pl = self._line_plot()
        if pl is None:
            return
        try:
            if pl.kind == "section":
                from ..results.field import section_corners
                bounds = self.mesh.bounds() if self.mesh is not None else model_box(self.project)
                self.viewer.set_plane(section_corners(self.project, pl, bounds))
            else:
                ev = self.project.evaluator()
                self.viewer.set_line(ev.vector(pl.start), ev.vector(pl.end))
        except Exception:
            pass  # an invalid coordinate: the panel marks it
    def refresh_view(self, view: str | None = None, reset_camera=False, switch_tab=True) -> bool:
        """Redraw geometry or mesh in the style of the selected tree node.

        Boundary-condition tints appear only on physics nodes (semi-transparent so hidden faces show);
        geometry nodes show plain, slightly transparent material colours.
        """
        view = view or self._view
        node = self._current_node()
        kind = node[0] if node else ""
        conditions, opacity, periodic = {}, DEFAULT_OPACITY, ()   # slightly transparent: inner faces show
        if kind in ("physics", "physics_list") or kind in self.ui.highlight_nodes:
            conditions = self.ui.highlights(self.project)
            if self.project.lattice.vectors:
                from .. import lattice as lat
                try:   # the lattice vectors are expressions; an invalid one is marked in its panel
                    periodic = lat.translations(self.project.lattice, self.project.evaluator())
                except Exception:
                    periodic = ()
            opacity = 0.5
        elif self._line_plot() is not None:
            opacity = 0.3  # the line runs through the structure
        if view == "geometry" and self.preview is not None:
            self.viewer.show_geometry(self.preview, conditions, reset_camera, opacity, periodic,
                                      self.project.material_colors())
        elif view == "mesh" and self.mesh is not None:
            self.viewer.show_mesh(self.mesh, conditions, reset_camera, opacity, periodic, self.project.material_colors())
        else:
            return False
        self._view = view
        self._overlay_line()
        if switch_tab:
            self.views.setCurrentWidget(self.viewer)
        return True

    UNIT_KINDS = {"length_unit": "Lengths", "frequency_unit": "Frequencies", "time_unit": "Times"}

    def set_unit(self, key: str, unit: str):
        """Set a preferred display unit (project.length_unit, frequency_unit or time_unit) and redraw."""
        if getattr(self.project, key) == unit:
            return
        setattr(self.project, key, unit)
        if key == "length_unit":
            self.viewer.set_length_unit(unit)
        self.mark_dirty()
        if self.results is not None:  # redraw in place, without switching tabs
            self.show_plot(switch_tab=False)
        self.log(f"{self.UNIT_KINDS[key]} are now displayed in {units.unit_label(unit)}")
    def set_physics(self, kind: str):
        """Switch the model to another physics (results are cleared). Each physics has its own plots: the
        current ones are kept aside and restored when switching back; a physics used for the first time starts
        with its default plots."""
        p = self.project
        if kind == p.physics_kind:
            return
        self._stashed_plots[p.physics_kind] = p.plots
        p.physics_kind = kind
        p.plots = self._stashed_plots.pop(kind, None) or self.spec.default_plots(p)
        self.clear_results()
        if not self.refresh_view("geometry" if self.preview is not None else "mesh"):
            self.viewer.plotter.clear()    # no geometry yet: drop the old results view
            self.viewer.plotter.render()
            self._view = None
        self.mark_dirty()
        self.log(f"Physics: {self.spec.title}")
        self.rebuild_tree(("model", None))

    def problem_changed(self):
        self.mark_dirty()
        self.rebuild_tree(refresh_panel=False)
    def physics_changed(self):
        self.mark_dirty()
        self.rebuild_tree(refresh_panel=False)
        if self._view in ("geometry", "mesh"):
            self.refresh_view()
    # ---- tree editing ---------------------------------------------------
    def _tree_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        kind, idx = item.data(0, Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        if kind == "feature":
            f = self.project.objects[idx]
            menu.addAction("Enable" if not f.enabled else "Bypass", lambda: self.toggle_feature(idx))
            menu.addAction("Delete", lambda: self.delete_feature(idx))
            menu.addSeparator()
            menu.addAction("Duplicate", lambda: self.duplicate_feature(idx))
            menu.addAction("Move up", lambda: self.move_feature(idx, -1))
            menu.addAction("Move down", lambda: self.move_feature(idx, 1))
            menu.addSeparator()
            menu.addAction("Build up to this object", lambda: self.build_geometry(upto=idx))
        elif kind == "material":
            menu.addAction("Delete", lambda: self.delete_material(idx))
        elif kind == "objects":
            for t in FEATURE_TYPES:
                menu.addAction(f"Add {FEATURE_TITLES[t]}", lambda t=t: self.add_feature(t))
            menu.addSeparator()
            menu.addAction("Build", self.build_geometry)
        elif kind in ("mesh", "mesh_default"):
            menu.addAction("Add mesh region", self.add_mesh_region)
            menu.addAction("Mesh", self.build_mesh)
        elif kind == "mesh_region":
            menu.addAction("Delete", lambda: self.delete_mesh_region(idx))
            menu.addAction("Mesh", self.build_mesh)
        elif kind == "problem":
            menu.addAction("Run", self.compute)
        elif kind == "results":
            from ..plots import ResultPlot
            spec = self.spec
            for k in spec.plot_kinds:  # one entry per plot kind, no submenu
                label = "Add " + plot_label(k, spec.plot_titles[k])
                act = menu.addAction(label, lambda k=k: self.add_plot_kind(k))
                why = self.plot_unavailable(ResultPlot(kind=k, quantity="x"))
                if why:
                    act.setToolTip(why)
        elif kind == "plot":
            menu.addAction("Rename…", lambda: self.rename_plot(idx))
            menu.addAction("Duplicate", lambda: self.add_plot(copy.deepcopy(self.project.plots[idx])))
            menu.addSeparator()
            menu.addAction("Delete", lambda: self.delete_plot(idx))
        elif kind == "sweep":
            sweep = self.project.problem.sweep
            on = sweep is not None and sweep.enabled
            menu.addAction("Disable" if on else "Enable", self.toggle_sweep)
        else:
            self.ui.context_menu(self, kind, idx, menu)
        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def toggle_sweep(self):
        from ..project import VariableSweep
        st = self.project.problem
        if st.sweep is None:
            st.sweep = VariableSweep(next(iter(self.project.variables), ""), "", enabled=False)
        st.sweep.enabled = not st.sweep.enabled
        self.mark_dirty()
        self.rebuild_tree(("sweep", None))

    def _delete_current(self):
        node = self._current_node()
        if node and node[0] == "feature":
            self.delete_feature(node[1])
        elif node and node[0] == "material":
            self.delete_material(node[1])
        elif node and node[0] == "mesh_region":
            self.delete_mesh_region(node[1])
        elif node:
            self.ui.delete_node(self, *node)

    def add_mesh_region(self):
        """A mesh region: by default the last object, at half the model's element size."""
        from ..project import MeshRegion
        ms = self.project.mesh
        on = next((f.name for f in reversed(self.project.objects) if f.op == "add" and f.enabled), "")
        ms.regions.append(MeshRegion(on=on, max_size=f"({ms.max_size})/2"))
        self.mark_dirty()
        self.rebuild_tree(("mesh_region", len(ms.regions) - 1))

    def delete_mesh_region(self, idx: int):
        del self.project.mesh.regions[idx]
        self.mark_dirty()
        self.rebuild_tree(("mesh", None))

    def add_feature(self, ftype: str):
        f = mp.default_feature(self.project, ftype)
        self.project.objects.append(f)
        self.log(f"Added object '{f.name}'")
        self.mark_dirty()
        self.rebuild_tree(("feature", len(self.project.objects) - 1))
    def toggle_feature(self, i):
        f = self.project.objects[i]
        f.enabled = not f.enabled
        self.log(f"{'Enabled' if f.enabled else 'Bypassed'} object '{f.name}'")
        self.mark_dirty()
        self.rebuild_tree(("feature", i))
    def duplicate_feature(self, i):
        """Copy of feature i, inserted right after it (same place in the build order), with a unique name."""
        import re
        g = self.project.objects
        f = copy.deepcopy(g[i])
        names = {x.name for x in g}
        m = re.fullmatch(r"(.*?)(\d+)", f.name)
        stem, n = (m.group(1), int(m.group(2)) + 1) if m else (f.name + "_", 2)
        while f"{stem}{n}" in names:
            n += 1
        f.name = f"{stem}{n}"
        g.insert(i + 1, f)
        self.log(f"Duplicated object '{g[i].name}' as '{f.name}'")
        self.mark_dirty()
        self.rebuild_tree(("feature", i + 1))
    def delete_feature(self, i):
        f = self.project.objects.pop(i)
        self.log(f"Deleted object '{f.name}'")
        self.mark_dirty()
        self.rebuild_tree(("objects", None))
    def move_feature(self, i, d):
        g = self.project.objects
        if 0 <= i + d < len(g):
            g[i], g[i + d] = g[i + d], g[i]
            self.mark_dirty()
            self.rebuild_tree(("feature", i + d))
    def delete_material(self, i):
        m = self.project.materials[i]
        users = [f.name for f in self.project.objects if f.op == "add" and f.material == m.name]
        if users and self.ask("Delete material", f"Material '{m.name}' is used by: {', '.join(users)}. "
                              "Delete it anyway?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) \
                != QMessageBox.StandardButton.Yes:
            return
        del self.project.materials[i]
        self.log(f"Deleted material '{m.name}'")
        self.mark_dirty()
        self.rebuild_tree(("materials", None))
    def mark_dirty(self):
        if not self.dirty:
            self.dirty = True
            self._update_title()
    def _update_title(self):
        name = self.project.path.name if self.project.path else "Untitled"
        self.setWindowTitle(f"SolidOcean — {name}{' *' if self.dirty else ''}")
    def log(self, msg: str):
        self.messages.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")
    # ---- file ops -------------------------------------------------------
    def ask(self, title: str, text: str, buttons) -> QMessageBox.StandardButton:
        """A plain question dialog (QMessageBox.question adds a question-mark icon)."""
        box = QMessageBox(QMessageBox.Icon.NoIcon, title, text, buttons, self)
        return QMessageBox.StandardButton(box.exec())
    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = self.ask("Unsaved changes", "Save changes to the current model?",
                     QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                     | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return r == QMessageBox.StandardButton.Discard
    def new(self):
        """New model: choose the physics, then a template of that physics."""
        if not self._confirm_discard():
            return
        from PySide6.QtWidgets import QButtonGroup
        from .. import templates
        dlg = QDialog(self)
        dlg.setWindowTitle("New model")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Physics"))
        group = QButtonGroup(dlg)
        for kind, spec in physics.REGISTRY.items():
            b = self._physics_button(spec)
            b.setProperty("physics", kind)
            group.addButton(b)
            lay.addWidget(b)
        lay.addSpacing(6)
        lay.addWidget(QLabel("Template (edit its variables afterwards)"))
        lst = QListWidget()
        lst.setMinimumSize(440, 260)
        lst.itemDoubleClicked.connect(lambda _: dlg.accept())
        lay.addWidget(lst)

        def fill(button):
            lst.clear()
            for key, t in templates.by_physics(button.property("physics")).items():
                item = QListWidgetItem(t.title)
                item.setData(Qt.ItemDataRole.UserRole, key)
                lst.addItem(item)
            lst.setCurrentRow(0)

        group.buttonClicked.connect(fill)
        current = next(b for b in group.buttons() if b.property("physics") == self.project.physics_kind)
        current.setChecked(True)
        fill(current)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted or lst.currentItem() is None:
            return
        key = lst.currentItem().data(Qt.ItemDataRole.UserRole)
        t = templates.all_templates()[key]
        self.set_project(templates.create(key))
        self.log(f"New model ({physics.get(t.physics).title}) from template: {t.title}")
    @staticmethod
    def _physics_button(spec) -> QPushButton:
        """A full-width choice of the New dialog: the physics' name in bold, its description below. (A
        QPushButton has no rich text, so the two lines are labels inside it; they let clicks through and take
        the accent's text colour when it is chosen.)"""
        b = QPushButton()
        b.setCheckable(True)
        b.setStyleSheet(  # explicit borders: the native style barely marks a checked button
            f"QPushButton {{ text-align: left; border: 1px solid {theme.muted()}; border-radius: 6px; "
            "background: transparent; }"
            f"QPushButton:checked {{ background: {theme.ACCENT}; border-color: {theme.ACCENT}; }}")
        inner = QVBoxLayout(b)
        inner.setContentsMargins(12, 8, 12, 8)
        inner.setSpacing(2)
        title, about = QLabel(spec.title), QLabel(spec.description)
        about.setWordWrap(True)
        for lbl in (title, about):
            lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            inner.addWidget(lbl)

        def recolor(on: bool):   # unchosen: the normal text colour, the description muted
            title.setStyleSheet("font-weight: 600; background: transparent;" + (f" color: {theme.ACCENT_TEXT};" if on else ""))
            about.setStyleSheet(f"color: {theme.ACCENT_TEXT if on else theme.muted()}; background: transparent;")

        b.toggled.connect(recolor)
        recolor(False)
        b.setMinimumHeight(inner.sizeHint().height())
        return b

    def start_dir(self, name: str = "") -> str:
        """Where a file dialog opens: the folder of the model file, or the last folder used. `name` is a
        file name offered in it."""
        base = self.project.path.parent if self.project.path else self._last_dir
        if not base.is_dir():
            base = Path.home()
        return str(base / name) if name else str(base)

    def used_dir(self, path) -> Path:
        """Remember the folder the user picked a file in (the next dialog opens there)."""
        path = Path(path)
        self._last_dir = path.parent
        return path

    def open_dialog(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open model", self.start_dir(), "SolidOcean model (*.toml)")
        if path:
            self.open_file(self.used_dir(path))
    def open_file(self, path):
        try:
            self.set_project(Project.load(path))
            self.log(f"Opened {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open model", str(exc))
            self.set_project(new_project())
    def save(self) -> bool:
        if self.project.path is None:
            return self.save_as()
        self.project.save()
        self.dirty = False
        self._update_title()
        self.log(f"Saved {self.project.path}")
        return True
    def save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save model", self.start_dir(f"{self.project.name}.toml"),
                                              "SolidOcean model (*.toml)")
        if not path:
            return False
        self.project.path = self.used_dir(path)
        return self.save()
    def closeEvent(self, event):
        if self._confirm_discard():
            self.cancel.set()
            self.viewer.plotter.close()
            event.accept()
        else:
            event.ignore()
    # ---- background tasks ----------------------------------------------
    def _check(self, geometry_only: bool = False) -> bool:
        """Validate before an action; geometry and mesh need only the geometry part of the model."""
        errors = self.project.validate_geometry() if geometry_only else self.project.validate()
        if errors:
            self.log("Model has errors:\n  " + "\n  ".join(errors))
            QMessageBox.warning(self, "Model has errors", "\n".join(errors))
            return False
        return True
    def run_task(self, label: str, fn, on_done, cancellable: bool = False):
        if self.busy:
            self.show_status("Busy — wait for the current task, or cancel the run with ✕", 5000)
            return
        self.busy = True
        self.cancel.clear()
        for a in self.run_actions:
            a.setEnabled(False)
        self.cancel_action.setVisible(cancellable)
        self.progress_action.setVisible(True)
        self.progress.setRange(0, 0)
        self.show_status(label)
        self.log(label)
        task = Task(fn)
        task.signals.log.connect(self.log)
        task.signals.log.connect(lambda msg: self._task_stage(label, msg))
        task.signals.progress.connect(self._on_progress)
        task.signals.done.connect(lambda r: (self._task_finished(), on_done(r)))
        task.signals.failed.connect(self._task_failed)
        self._task = task  # keep signals alive
        self.pool.start(task)
    def _task_stage(self, label: str, msg: str):
        """Toolbar status while a task runs: its label and the step it is at (meshing, assembly, …)."""
        if self.busy:
            stage = " ".join(msg.split())
            self.show_status(f"{label} — {stage[:90]}" if stage else label)
    def _task_failed(self, msg: str):
        self._task_finished()
        if msg == "Cancelled":
            self.log("Run cancelled.")
            self.show_status("Run cancelled", 5000)
        else:
            self.log(f"Error: {msg}")
            self.show_status(msg.splitlines()[0][:120] if msg else "Error", 8000)
            self.show_log(True)   # the details are in the log
    def _on_progress(self, i, n):
        self.progress.setRange(0, n)
        self.progress.setValue(i)
    def _task_finished(self):
        self.busy = False
        if self._pending_mode:
            self._pending_mode = False
            QTimer.singleShot(0, self.show_mode)
        for a in self.run_actions:
            a.setEnabled(True)
        self.cancel_action.setVisible(False)
        self.progress_action.setVisible(False)
        self.status.clear()
    def stop(self):
        if self.busy:
            self.cancel.set()
            self.log("Stopping after the current step…")
    # ---- actions --------------------------------------------------------
    def build_geometry(self, *_, upto: int | None = None, show: bool = True):
        """Build the geometry; with `upto` only the features up to and including that one (later ones are
        left out), to inspect the construction step by step."""
        snap = copy.deepcopy(self.project)
        if upto is not None:
            snap.objects = snap.objects[:upto + 1]
            if not any(f.enabled and f.op == "add" for f in snap.objects):
                self.log(f"Nothing to build up to '{self.project.objects[upto].name}': no solid ('add') "
                         "object before it.")
                return
            # the left-out objects take their mesh regions with them: nothing is meshed here, and a
            # region on an object that is not in the snapshot would fail validation
            kept = {f.name for f in snap.objects} | {m.name for m in snap.materials}
            snap.mesh.regions = [r for r in snap.mesh.regions if r.on in kept]
        errors = snap.validate_geometry()
        if errors:
            self.log("Model has errors:\n  " + "\n  ".join(errors))
            QMessageBox.warning(self, "Model has errors", "\n".join(errors))
            return
        upto_name = self.project.objects[upto].name if upto is not None else None

        def done(preview):
            self.preview = preview
            self.refresh_view("geometry", switch_tab=show)    # keeps the user's view (the first one is set up)
            self.log(f"Geometry built{f' up to {upto_name!r}' if upto_name else ''}: "
                     f"{len(preview.material_names)} material domain(s)")

        label = f"Building geometry up to '{upto_name}'…" if upto_name else "Building geometry…"
        self.run_task(label, lambda sig: meshing.preview_geometry(snap, snap.evaluator()), done)
    def build_mesh(self):
        if not self._check(geometry_only=True):
            return
        snap = copy.deepcopy(self.project)

        def work(sig):
            t0 = time.perf_counter()
            mesh = meshing.generate(snap, snap.evaluator())
            sig.log.emit(f"Mesh built in {time.perf_counter() - t0:.1f} s")
            return mesh

        def done(mesh):
            self.mesh = mesh
            self.refresh_view("mesh")
            self.log(self.mesh_summary())
            if self._current_node() and self._current_node()[0] == "mesh":
                self.rebuild_tree(("mesh", None))

        self.run_task("Meshing…", work, done)
    def mesh_summary(self) -> str:
        m = self.mesh
        if m is None:
            return "No mesh built yet."
        q = m.quality()
        return (f"{m.info['n_elements']} tetrahedra (order {m.order}), {m.n_nodes} nodes, "
                f"{m.n_dofs} DOFs. Element quality: min {q.min():.3f}, mean {q.mean():.3f}")
