"""Main window behaviour that once crashed (runs an off-screen window)."""
import os
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_panel_swap_with_focus_inside_the_old_panel():
    """Selecting another node while a widget of the current panel has the keyboard focus (e.g. after
    pressing Run in the Problem panel, a run ends by opening a plot) used to segfault in Qt."""
    from PySide6.QtWidgets import QApplication, QPushButton
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    w.set_project(templates.create("square"))
    w.select_node("problem")
    run = next(b for b in w.settings.widget().findChildren(QPushButton) if b.text().startswith("Run"))
    run.setFocus()
    app.processEvents()
    assert w.settings.widget().isAncestorOf(QApplication.focusWidget())
    w.select_node("plot", 0)
    app.processEvents()
    assert w._current_node() == ("plot", 0)
    w.cancel.set()
    w.dirty = False
    w.close()



def test_eigenfrequency_solutions_in_the_problem_panel():
    """The number of solutions and the frequency they are searched around are rows of the Problem panel (like
    the drive frequency of a frequency-domain problem), not a node of their own."""
    from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QSpinBox
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    from solidocean.gui.panels.common import ExprEdit
    w = MainWindow()
    w.show()
    w.set_project(templates.create("square"))
    assert "solutions" not in [k for _, (k, _i), _off in w.ui.problem_nodes(w.project)]
    w.select_node("problem")
    panel = w.settings.widget()
    labels = [lab.text() for lab in panel.findChildren(QLabel)]
    assert "Number of solutions" in labels and "Around" in labels
    n = next(b for b in panel.findChildren(QSpinBox) if b.value() == w.project.solid.problem.num_eigs)
    n.setValue(7)
    assert w.project.solid.problem.num_eigs == 7
    around = panel.findChildren(ExprEdit)[0]
    around.line.setText("5[GHz]")
    around._commit()
    assert w.project.solid.problem.search_freq == "5[GHz]"
    app.processEvents()
    w.cancel.set()
    w.dirty = False
    w.close()


def test_solver_box_in_the_problem_panel():
    """The linear solver (and, for eigenfrequencies, the Lanczos settings) is a Solver box of the Problem panel
    of every physics, not a node of its own."""
    from PySide6.QtWidgets import QApplication, QComboBox, QGroupBox, QLabel
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    for name, eigen in (("square", True), ("heat_bar", False)):
        w.set_project(templates.create(name))
        assert "solver" not in [k for _, (k, _i), _off in w.ui.problem_nodes(w.project)]
        w.select_node("problem")
        box = next(b for b in w.settings.widget().findChildren(QGroupBox) if b.title() == "Solver")
        labels = [lab.text() for lab in box.findChildren(QLabel)]
        assert "Linear solver" in labels and ("Eigenvalue tolerance" in labels) == eigen
        combo = box.findChildren(QComboBox)[0]
        combo.setCurrentIndex(combo.findData("superlu"))
        assert w.project.problem.solver == "superlu"
    app.processEvents()
    w.cancel.set()
    w.dirty = False
    w.close()


def test_heat_problem_type_is_shown_on_the_problem_node():
    """Changing the heat problem type relabels the Problem node (like solid mechanics) and keeps it selected;
    there is no separate Problem type node repeating the panel."""
    from PySide6.QtWidgets import QApplication, QComboBox
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    w.set_project(templates.create("heat_bar"))
    w.select_node("problem")
    typ = w.settings.widget().findChildren(QComboBox)[0]
    typ.setCurrentIndex(typ.findData("transient"))
    app.processEvents()
    assert w.project.heat.problem.type == "transient"
    assert w._current_node() == ("problem", None)
    assert [k for _, (k, _i), _off in w.ui.problem_nodes(w.project)] == ["sweep"]
    assert w.ui.problem_label(w.project) == "Problem: Transient"
    w.cancel.set()
    w.dirty = False
    w.close()


def test_log_collapsed_and_progress_in_the_toolbar():
    """The messages panel starts collapsed behind ↑ Log; a task shows its progress at the right of the main
    toolbar, and a failed task opens the log."""
    import threading
    import time
    from PySide6.QtWidgets import QApplication, QToolBar
    app = QApplication.instance() or QApplication([])
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    app.processEvents()
    assert not w.log_dock.isVisible() and w.log_button.text() == "↑ Log"
    w.log_button.click()
    assert w.log_dock.isVisible() and w.log_button.text() == "↓ Log"
    w.log_button.click()
    assert not w.log_dock.isVisible()

    release = threading.Event()

    def work(signals):
        signals.log.emit("meshing")
        release.wait(5)
        raise ValueError("boom")

    deadline = time.time() + 30
    while w.busy and time.time() < deadline:   # the new model's geometry is built on start
        app.processEvents()
    w.run_task("Working", work, lambda r: None, cancellable=True)
    while "meshing" not in w.status.text() and time.time() < deadline:
        app.processEvents()
    app.processEvents()   # the toolbar lays out the shown widgets
    assert w.progress.isVisible() and w.cancel_button.isVisible()
    assert isinstance(w.progress.parent(), QToolBar) and w.status.text() == "Working — meshing"
    release.set()
    while w.busy and time.time() < deadline:
        app.processEvents()
    assert not w.progress.isVisible() and not w.cancel_button.isVisible()
    assert w.log_dock.isVisible() and "boom" in w.status.text()
    w.dirty = False
    w.close()


def test_output_panel_edits_the_model_and_saves_a_run(tmp_path):
    """The Output node edits [output], previews the folder of the next run, and 'Save this run' writes the
    results in memory into it."""
    import numpy as np
    from PySide6.QtWidgets import QApplication, QCheckBox, QLineEdit, QPushButton
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    from solidocean.physics.solid.modal import ModalResults
    w = MainWindow()
    p = templates.create("square")
    p.name = "Slab"
    p.save(tmp_path / "m.toml")
    w.set_project(p)
    w.select_node("output")
    panel = w.settings.widget()
    folder = next(e for e in panel.findChildren(QLineEdit) if e.placeholderText().startswith("all runs"))
    folder.setText("{name} a={a}")
    folder.textEdited.emit("{name} a={a}")
    assert p.output.folder == "{name} a={a}" and w.dirty
    assert str(tmp_path / "m" / "Slab a=400nm") in panel.shown.text()
    data = next(b for b in panel.findChildren(QCheckBox) if b.text().startswith("Full data"))
    data.setChecked(False)
    assert p.output.contents == ["tables", "model"]

    w.results = ModalResults(np.array([[[1e9, 2e9]]]), np.zeros((1, 1, 3)), np.zeros(1), "k")
    w.results_project = p
    w.select_node("output")
    panel = w.settings.widget()
    next(b for b in panel.findChildren(QPushButton) if b.text() == "Save this run").click()
    app.processEvents()
    run = tmp_path / "m" / "Slab a=400nm"
    assert {f.name for f in run.iterdir()} == {"dispersion.csv", "model.toml", "run.json"}
    next(b for b in panel.findChildren(QPushButton) if b.text() == "Save this run").click()
    assert (tmp_path / "m" / "Slab a=400nm (2)").is_dir()   # the first run is kept
    w.dirty = False
    w.close()


def test_file_dialogs_start_next_to_the_model(tmp_path, monkeypatch):
    """File dialogs open in the model's own folder (results dialogs in its results folder), never in the
    directory the program happened to be started from."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    assert w.start_dir() == str(pathlib.Path.home())          # no model file yet
    p = templates.create("square")
    p.save(tmp_path / "slab.toml")
    w.set_project(p)
    assert w.start_dir() == str(tmp_path)
    assert w.start_dir("slab.toml") == str(tmp_path / "slab.toml")
    assert w.results_start("dispersion.csv") == str(tmp_path / "dispersion.csv")   # no results folder yet
    (tmp_path / "slab").mkdir()
    assert w.results_start("dispersion.csv") == str(tmp_path / "slab" / "dispersion.csv")
    asked = []                                                # the 3D view's "Save Image..." too
    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getSaveFileName",
                        lambda *a, **k: (asked.append(a[2]), ("", ""))[1])
    w.viewer.save_image()
    w.save_as()                                               # both offer the model file's own name
    assert asked == [str(tmp_path / "slab" / "slab.png"), str(tmp_path / "slab.toml")]
    w.used_dir(tmp_path / "elsewhere" / "other.toml")          # remembered while a model has no file
    w.set_project(templates.create("square"))
    assert w.start_dir() == str(pathlib.Path.home())           # that folder does not exist: home
    asked.clear()                                              # no file yet: the model's title, "structure"
    w.viewer.save_image()
    w.save_as()
    assert asked == [str(pathlib.Path.home() / "structure.png"),
                     str(pathlib.Path.home() / f"{w.project.name}.toml")]
    w.dirty = False
    w.close()


def test_plots_added_after_the_run_are_saved(tmp_path, monkeypatch):
    """A line or section plot added after a frequency-domain run is written by 'Save this run' (with the
    quantity of a sweep plot not shown yet), and 'Save data…' on its page writes its table."""
    import csv
    from PySide6.QtWidgets import QApplication, QFileDialog
    app = QApplication.instance() or QApplication([])
    from solidocean.gui.main_window import MainWindow
    from solidocean.physics.solid.model import SolidCondition, SolidPhysics, SolidProblem
    from solidocean.physics.solid.solver import Solver
    from solidocean.plots import ResultPlot
    from solidocean.project import Feature, Material, MeshSettings, Project, Problem, VariableSweep
    props = {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.25", "eta": "0.05"}
    p = Project(name="bar", variables={"L": "2[um]", "w": "0.2[um]", "f": "1[GHz]"},
                materials=[Material("M", props)],
                objects=[Feature("block", "Bar", material="M",
                                  params={"center": ["L/2", "0", "0"], "size": ["L", "w", "w"]})],
                mesh=MeshSettings(max_size="w/2"),
                solid=SolidPhysics(symmetry=["ymin", "ymax", "zmin", "zmax"], conditions=[
                    SolidCondition("displacement", ["xmin"], displacement=["1[nm]", "", ""])],
                    problem=SolidProblem(type="frequency", frequency="f")),
                problem=Problem(sweep=VariableSweep("f", "[1, 2] * 1[GHz]")))
    p.save(tmp_path / "m.toml")
    w = MainWindow()
    w.set_project(p)
    w.results = Solver(p, log=lambda m: None).run()
    w.results_project = Project.load(tmp_path / "m.toml")   # the run's copy: without the plots added below
    w.results_project.path = p.path
    p.plots = [w.spec.new_plot(p, "line", ((0, 0, 0), (2e-6, 0.2e-6, 0.2e-6))),
               ResultPlot("Amplitude", "max(abs(ux))", kind="quantity")]
    p.plots[0].name, p.plots[0].quantity = "Profile", "abs(ux)"

    run = w.save_run(tmp_path / "run")
    assert (run / "Profile.csv").is_file()
    with open(run / "quantities.csv", encoding="utf-8") as fh:
        assert "Amplitude" in next(csv.reader(fh))            # columns are named after the plots

    out = tmp_path / "profile.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    w.current_plot = 0
    w.export_csv()
    rows = list(csv.reader(open(out, encoding="utf-8")))
    assert rows[0][1:] == ["f=1e+09", "f=2e+09"] and len(rows) > 10
    w.dirty = False
    w.close()


def test_loft_sections_are_added_and_removed_in_the_object_panel():
    """The loft panel edits one z and polygon per section; Add / Remove change the number of sections."""
    from PySide6.QtWidgets import QApplication, QPlainTextEdit, QPushButton
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    w.set_project(templates.create("square"))
    w.add_feature("loft")
    i = len(w.project.objects) - 1
    w.select_node("feature", i)
    app.processEvents()
    sections = w.project.objects[i].params["sections"]
    assert len(w.settings.widget().findChildren(QPlainTextEdit)) == 2

    def press(text):
        next(b for b in w.settings.widget().findChildren(QPushButton) if b.text() == text).click()
        app.processEvents()

    press("Add section")
    assert len(sections) == 3 and len(w.settings.widget().findChildren(QPlainTextEdit)) == 3
    w.settings.widget().findChildren(QPlainTextEdit)[2].setPlainText("0, 0\n1[um], 0\n0, 1[um]\n0, 0.5[um]")
    assert sections[2]["points"][1] == ["1[um]", "0"]
    press("Remove last section")
    assert len(sections) == 2
    w.cancel.set()
    w.dirty = False
    w.close()


def test_structure_and_plot_can_share_the_window():
    """The two result views sit behind tabs, side by side or stacked; the widgets are never reparented, so
    the 3D view keeps its render context, and a plot drawn while its view is hidden (a run ends with the
    Structure in front) must not fall over a zero-size figure."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    w.set_project(templates.create("square"))
    app.processEvents()
    views, parents = w.views, (w.viewer.parent(), w.plot.parent())

    assert views.mode == "tabs" and views.bar.isVisible()
    assert w.viewer.isVisible() and not w.plot.isVisible()

    views.set_mode("side")
    app.processEvents()
    assert w.viewer.isVisible() and w.plot.isVisible() and not views.bar.isVisible()
    assert views.splitter.orientation() == Qt.Orientation.Horizontal
    left, right = views.splitter.sizes()
    assert abs(left - right) <= 1                      # the handle starts in the middle

    views.set_mode("stacked")
    app.processEvents()
    assert views.splitter.orientation() == Qt.Orientation.Vertical
    top, bottom = views.splitter.sizes()
    assert abs(top - bottom) <= 1

    views.set_mode("tabs")
    views.setCurrentWidget(w.plot)
    app.processEvents()
    assert w.plot.isVisible() and not w.viewer.isVisible()
    assert views.bar.currentIndex() == 1               # the tab follows the code's choice
    assert (w.viewer.parent(), w.plot.parent()) == parents    # nothing was reparented on the way

    views.setCurrentWidget(w.viewer)
    app.processEvents()
    w.plot.show_message("drawn while hidden")          # used to raise: singular transform
    assert min(w.plot.figure.get_size_inches()) >= w.plot.MIN_INCHES
    w.dirty = False
    w.close()


def test_a_mode_can_be_shown_from_results_loaded_off_disk(tmp_path):
    """Results read back from a run have no solver behind them, so picking a point used to raise inside the
    worker ('NoneType' has no attribute 'log'). One is built from the model the results were loaded with."""
    import copy
    import time
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from solidocean import physics, templates
    from solidocean.gui.main_window import MainWindow
    from solidocean.physics.solid.modal import ModalResults

    project = templates.create("membrane")
    project.solid.problem.num_eigs, project.mesh.max_size = 6, "a"
    ModalResults.save(physics.of(project).solver(project, log=lambda m: None).run(), tmp_path)

    def wait_for(done, limit=60.0):
        """Spin the event loop until the work lands (the task starts on a thread, so waiting on `busy`
        alone can miss it) or the time is up."""
        end = time.time() + limit
        while time.time() < end and not done():
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    w = MainWindow()
    w.set_project(project)
    wait_for(lambda: not w.busy)
    w.results = ModalResults.load(tmp_path / "results.npz")
    w.results_project = copy.deepcopy(w.project)
    assert w.solver is None                      # nothing ran in this window
    assert not w.results.modes                   # mode shapes are not saved with the results

    w.pick_mode(0, 3, 1)
    wait_for(lambda: w._mode is not None)
    assert w.solver is not None                  # built on demand, from the loaded model
    assert w._mode is not None and "Mode 2" in w._mode[2]
    w.dirty = False
    w.close()


def test_size_grading_in_the_mesh_panel():
    """mesh.grading is a row of the Mesh panel: 1 reads as off, and an edit reaches the model."""
    import time
    from PySide6.QtWidgets import QApplication, QDoubleSpinBox
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    w.set_project(templates.create("heat_pad"))
    deadline = time.time() + 30
    while w.busy and time.time() < deadline:
        app.processEvents()
    w.select_node("mesh")
    app.processEvents()
    spin, = w.settings.widget().findChildren(QDoubleSpinBox)
    assert spin.value() == 1.0 and (spin.minimum(), spin.maximum()) == (1.0, 4.0)
    assert spin.specialValueText() == "1 (off)"          # 1 is not a grading, it is no grading
    spin.setValue(1.3)
    app.processEvents()
    assert w.project.mesh.grading == 1.3 and w.dirty
    w.dirty = False
    w.close()


def test_a_mesh_region_can_set_its_own_size_grading():
    """Per-region grading has three states in the panel: inherit the mesh's, grade at its own rate, or 1 to
    opt out — which is how elements get spent on the region that needs them and nowhere else."""
    import time
    from PySide6.QtWidgets import QApplication, QDoubleSpinBox
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    from solidocean.project import MeshRegion
    p = templates.create("heat_pad")
    p.mesh.grading = 1.3
    p.mesh.regions = [MeshRegion(on=p.objects[0].name, max_size="0.5[um]")]
    w = MainWindow()
    w.show()
    w.set_project(p)
    deadline = time.time() + 30
    while w.busy and time.time() < deadline:
        app.processEvents()
    w.select_node("mesh_region", 0)
    app.processEvents()
    spin, = w.settings.widget().findChildren(QDoubleSpinBox)
    region = p.mesh.regions[0]
    assert spin.specialValueText() == "Same as the mesh"
    assert spin.value() == spin.minimum() and region.grading is None    # starts out inheriting
    assert p.mesh.grading_of(region) == 1.3
    spin.setValue(1.0)                                                  # opt out, though the mesh grades
    app.processEvents()
    assert region.grading == 1.0 and p.mesh.grading_of(region) == 1.0 and w.dirty
    spin.setValue(1.15)                                                 # tighter than the mesh
    app.processEvents()
    assert region.grading == 1.15 and p.mesh.grading_of(region) == 1.15
    spin.setValue(spin.minimum())                                       # back to inheriting
    app.processEvents()
    assert region.grading is None and p.mesh.grading_of(region) == 1.3
    w.dirty = False
    w.close()


def test_meshing_can_be_stopped_from_the_toolbar():
    """A mesh that is taking too long has to be stoppable: the ✕ shows while meshing (it used to appear
    only for a run), and pressing it kills the mesher instead of waiting for gmsh to come back."""
    import time
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    p = templates.create("square")
    p.mesh.max_size = p.mesh.min_size = "a/80"     # about ten seconds of meshing, far longer than we wait
    w.set_project(p)
    deadline = time.time() + 120
    while w.busy and time.time() < deadline:       # the geometry a new model builds on arrival
        app.processEvents()

    w.build_mesh()
    while not w.cancel_button.isVisible() and time.time() < deadline:
        app.processEvents()
    assert w.cancel_button.isVisible(), "meshing offered no way to stop it"

    w.cancel_button.click()
    t0 = time.time()
    while w.busy and time.time() < deadline:
        app.processEvents()
    assert time.time() - t0 < 10, "the mesh ran on after the ✕"
    assert w.mesh is None, "a cancelled mesh was kept anyway"
    assert "Cancelled" in w.messages.toPlainText()
    app.processEvents()                            # the toolbar lays out the hidden widgets
    assert not w.cancel_button.isVisible()
    w.dirty = False
    w.close()
