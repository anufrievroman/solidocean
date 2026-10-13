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
