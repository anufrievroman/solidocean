"""Cancelling a mesh that is taking too long.

`gmsh.model.mesh.generate` is one call into the library: it does not come back to Python until it is
done, so no event a worker thread watches can stop it. Given a cancel event, meshing therefore runs in
a process we can kill, and the mesh comes back over a pipe. Without one (the CLI) it runs in the
calling process as before, which is what the last two tests here pin down.
"""
import multiprocessing
import threading
import time

import numpy as np
import pytest

from solidocean import meshing, templates
from solidocean.errors import Cancelled
from solidocean.physics.heat.solver import HeatSolver
from solidocean.physics.solid.solver import Solver as SolidSolver

SLOW = "a/80"        # a few hundred thousand elements: seconds of meshing, far longer than a cancel


def square(max_size=None):
    p = templates.create("square")
    if max_size is not None:
        p.mesh.max_size, p.mesh.min_size = max_size, max_size
    return p


def test_the_same_mesh_comes_back_through_the_process():
    """The cancellable path must be the ordinary one, not a degraded version of it."""
    p = square()
    here = meshing.generate(p, p.evaluator())
    apart = meshing.generate(p, p.evaluator(), cancel=threading.Event())
    assert np.array_equal(here.points, apart.points)
    assert np.array_equal(here.cells, apart.cells)
    assert np.array_equal(here.cell_domain, apart.cell_domain)
    assert np.array_equal(here.cell_feature, apart.cell_feature)
    assert here.info == apart.info
    assert (apart.order, apart.material_names, apart.feature_names) == \
           (here.order, here.material_names, here.feature_names)
    assert apart.length_scale == here.length_scale


def test_an_event_already_set_yields_no_mesh():
    """A user who has given up gets Cancelled, never a mesh that happened to land in time."""
    p = square()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        meshing.generate(p, p.evaluator(), cancel=cancel)


def test_cancelling_stops_a_long_mesh_long_before_it_would_finish():
    """The point of the whole exercise: the wait ends when the user asks, not when gmsh is done."""
    p = square(SLOW)
    cancel = threading.Event()
    threading.Timer(0.6, cancel.set).start()
    t0 = time.perf_counter()
    with pytest.raises(Cancelled):
        meshing.generate(p, p.evaluator(), cancel=cancel)
    assert time.perf_counter() - t0 < 5.0, "the mesh ran on well past the cancel"


def test_no_mesher_is_left_running_after_a_cancel():
    p = square(SLOW)
    cancel = threading.Event()
    threading.Timer(0.4, cancel.set).start()
    with pytest.raises(Cancelled):
        meshing.generate(p, p.evaluator(), cancel=cancel)
    for _ in range(50):          # the kill is a signal; give it a moment to be reaped
        if not multiprocessing.active_children():
            break
        time.sleep(0.1)
    assert multiprocessing.active_children() == []


def test_meshing_still_works_after_a_cancel():
    """The killed process held the gmsh session, so cancelling must leave nothing behind here."""
    slow = square(SLOW)
    cancel = threading.Event()
    threading.Timer(0.4, cancel.set).start()
    with pytest.raises(Cancelled):
        meshing.generate(slow, slow.evaluator(), cancel=cancel)
    p = square()
    assert meshing.generate(p, p.evaluator(), cancel=threading.Event()).info["n_elements"] > 0


def test_a_model_that_cannot_be_meshed_reports_the_same_error_either_way():
    """Failures have to read the same wherever the meshing happened, so the GUI log does not change."""
    p = square()
    p.objects = []
    with pytest.raises(ValueError) as here:
        meshing.generate(p, p.evaluator())
    with pytest.raises(ValueError) as apart:
        meshing.generate(p, p.evaluator(), cancel=threading.Event())
    assert str(here.value) == str(apart.value)


def test_without_an_event_meshing_stays_in_this_process(monkeypatch):
    """The CLI must not pay the start-up of a process it can never cancel."""
    def fail(*args, **kwargs):
        raise AssertionError("meshing started a process without a cancel event")

    monkeypatch.setattr(meshing, "_generate_apart", fail)
    p = square()
    assert meshing.generate(p, p.evaluator()).info["n_elements"] > 0


def test_a_solver_nobody_can_cancel_meshes_in_its_own_process():
    """`cancel` defaults to a private event the caller cannot reach, so it is not what meshing is
    given: a headless run would otherwise fork for nothing."""
    p = templates.create("square")
    assert SolidSolver(p).mesh_cancel is None
    assert HeatSolver(templates.create("heat_bar")).mesh_cancel is None
    given = threading.Event()
    assert SolidSolver(p, cancel=given).mesh_cancel is given
