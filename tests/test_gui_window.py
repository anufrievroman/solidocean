"""Main window behaviour that once crashed (runs an off-screen window)."""
import os
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_panel_swap_with_focus_inside_the_old_panel():
    """Selecting another node while a widget of the current panel has the keyboard focus (e.g. after
    pressing Run in the Study panel, a run ends by opening a plot) used to segfault in Qt."""
    from PySide6.QtWidgets import QApplication, QPushButton
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    w.set_project(templates.create("square"))
    w.select_node("study")
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


def test_file_dialogs_start_next_to_the_model(tmp_path):
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
    w.used_dir(tmp_path / "elsewhere" / "other.toml")          # remembered while a model has no file
    w.set_project(templates.create("square"))
    assert w.start_dir() == str(pathlib.Path.home())           # that folder does not exist: home
    w.dirty = False
    w.close()


def test_plots_added_after_the_run_are_saved(tmp_path, monkeypatch):
    """A line or section plot added after a frequency-domain run is written by 'Save this run' (with the
    quantity of a sweep plot not shown yet), and 'Save data…' on its page writes its table."""
    import csv
    from PySide6.QtWidgets import QApplication, QFileDialog
    app = QApplication.instance() or QApplication([])
    from solidocean.gui.main_window import MainWindow
    from solidocean.physics.solid.model import SolidCondition, SolidPhysics, SolidStudy
    from solidocean.physics.solid.solver import Solver
    from solidocean.plots import ResultPlot
    from solidocean.project import Feature, Material, MeshSettings, Project, Study, VariableSweep
    props = {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.25", "eta": "0.05"}
    p = Project(name="bar", variables={"L": "2[um]", "w": "0.2[um]", "f": "1[GHz]"},
                materials=[Material("M", props)],
                geometry=[Feature("block", "Bar", material="M",
                                  params={"center": ["L/2", "0", "0"], "size": ["L", "w", "w"]})],
                mesh=MeshSettings(max_size="w/2"),
                solid=SolidPhysics(symmetry=["ymin", "ymax", "zmin", "zmax"], conditions=[
                    SolidCondition("displacement", ["xmin"], displacement=["1[nm]", "", ""])],
                    study=SolidStudy(type="frequency", frequency="f")),
                study=Study(sweep=VariableSweep("f", "[1, 2] * 1[GHz]")))
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
        assert "max(abs(ux))" in next(csv.reader(fh))

    out = tmp_path / "profile.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    w.current_plot = 0
    w.export_csv()
    rows = list(csv.reader(open(out, encoding="utf-8")))
    assert rows[0][1:] == ["f=1e+09", "f=2e+09"] and len(rows) > 10
    w.dirty = False
    w.close()
