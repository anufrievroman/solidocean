"""Main window behaviour that once crashed (runs an off-screen window)."""
import os

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
