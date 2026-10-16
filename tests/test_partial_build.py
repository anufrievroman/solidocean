"""'Build up to this object': the geometry of the first objects only, to check a model step by step."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from solidocean import templates
from solidocean.project import MeshRegion


@pytest.fixture
def window(monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    from solidocean.gui.main_window import MainWindow
    # a model with errors is reported in a modal dialog, which would block these tests for ever: collect
    # the warnings instead, so a build that is refused fails the test rather than hanging it
    warned: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text, *a, **k: warned.append(text))
    w = MainWindow()
    w.show()

    def open_model(project):
        w.set_project(project)
        settle(w, app)
        return w

    yield open_model, app, warned
    w.dirty = False
    w.close()


def settle(w, app):
    """Let the build, which runs in a worker thread, finish."""
    app.processEvents()
    deadline = time.time() + 60
    while w.busy and time.time() < deadline:
        app.processEvents()
    assert not w.busy, "the build did not finish"


def test_the_window_builds_up_to_every_object(window):
    """The regression: `build_geometry(upto=…)` cut the later objects out of its snapshot but left their
    mesh regions in, and a region on an object that is not there fails validation — so with a region on
    every object, every step but the last was refused, although a geometry preview meshes nothing."""
    open_model, app, warned = window
    p = templates.create("heat_pad")
    p.mesh.regions = [MeshRegion(on=f.name, max_size="0.5[um]") for f in p.objects if f.op == "add"]
    assert len(p.mesh.regions) > 1
    w = open_model(p)
    for i, f in enumerate(p.objects):
        w.preview = None
        w.build_geometry(upto=i, show=False)
        settle(w, app)
        assert not warned, f"building up to '{f.name}' was refused: {warned}"
        assert w.preview is not None, f"building up to '{f.name}' produced nothing"
    assert w.project.mesh.regions == p.mesh.regions      # the model itself is left alone


def test_a_partial_build_shows_only_the_objects_up_to_it(window):
    """Not just that it builds: the materials in the view are those of the objects it was given."""
    open_model, app, _ = window
    p = templates.create("heat_pad")
    w = open_model(p)
    seen = []
    for i in range(len(p.objects)):
        w.preview = None
        w.build_geometry(upto=i, show=False)
        settle(w, app)
        seen.append(set(w.preview.material_names))
    assert seen[0] < seen[-1]                            # the first object alone is not the whole model
    assert all(a <= b for a, b in zip(seen, seen[1:]))   # and each step only ever adds to the one before
    w.build_geometry(show=False)
    settle(w, app)
    assert set(w.preview.material_names) == seen[-1]     # the last step is the whole model


def test_the_full_model_is_still_validated():
    """Only a partial build drops the regions; a region naming nothing at all stays an error."""
    p = templates.create("heat_pad")
    p.mesh.regions = [MeshRegion(on="Nonexistent", max_size="0.3[um]")]
    assert any("Nonexistent" in e for e in p.validate_geometry())
