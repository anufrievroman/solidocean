"""Fourier components of mode fields over the unit cell, and the explicit list of wavevectors they are
usually asked at. Checked against the two cases whose answer is known exactly: a uniform strain, which has
only the G = 0 component, and a plane wave in an unpatterned cell, which has only the component of its own
wavevector."""
import csv

import numpy as np
import pytest

from solidocean import fields as F
from solidocean import physics
from solidocean.lattice import Lattice
from solidocean.physics.solid import fourier
from solidocean.physics.solid.model import Fourier, KPath, SolidPhysics, SolidProblem
from solidocean.physics.solid.modal import ModalResults
from solidocean.physics.solid.solver import Solver
from solidocean.project import Feature, Material, MeshSettings, Project


def cube(orders=(1, 1, 1), num_eigs=4, k=("kx", "0", "0"), kpath=None, size="a/3") -> Project:
    """A Si cube of side a, periodic in all three directions: its cell is filled, so every mode is a plane
    wave and the mesh volume is the volume of the cell."""
    return Project(
        name="Cube", variables={"a": "1[um]", "kx": "0.3*pi/a"},
        materials=[Material("Si", {"library": "Si"})],
        objects=[Feature("block", "cell", material="Si",
                         params={"center": ["0", "0", "0"], "size": ["a", "a", "a"]})],
        mesh=MeshSettings(max_size=size, order=2),
        lattice=Lattice(periodic={"x": "a", "y": "a", "z": "a"}),
        solid=SolidPhysics(problem=SolidProblem(num_eigs=num_eigs, k=list(k), kpath=kpath,
                                                fourier=Fourier(orders=list(orders)))))


def zero_order(h: np.ndarray) -> int:
    """Row of the G = 0 vector in a grid of orders."""
    return int(np.argmin(np.linalg.norm(h, axis=1)))


def test_uniform_strain_has_only_the_zero_component():
    """u = (c x, 0, 0) is a uniform strain exx = c: its only non-zero component is ŝ(0) = c, and every other
    strain vanishes. What is left at G ≠ 0 is the error of the element quadrature on e^{-iG·r} (the fields
    themselves are exact), and it falls as the mesh is refined.

    Only the strains are checked: the displacement of a uniform strain grows with r instead of repeating,
    so it is not a periodic field and has components everywhere.
    """
    p = cube()
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    model = s.model(ev)
    quad = F.Quadrature.of(model.mesh)
    assert quad.volume == pytest.approx(ev("a") ** 3, rel=1e-12)
    h, G = fourier.vectors(p, ev, [1, 1, 1])
    assert len(h) == 27 and np.all(G[zero_order(h)] == 0)

    c = 3e-4
    u = np.zeros((model.mesh.n_nodes, 3), dtype=complex)
    u[:, 0] = c * model.mesh.points[:, 0]
    coefficients = fourier.components(quad, [u], model.cel, np.zeros(3), G)[0]
    for n, name in enumerate(fourier.FIELDS):
        if not name.startswith("e"):
            continue
        at_zero = coefficients[zero_order(h), n]
        assert at_zero == pytest.approx(c if name == "exx" else 0.0, abs=1e-9 * c)
        assert np.abs(np.delete(coefficients[:, n], zero_order(h))).max() < 1e-5 * c


def test_a_plane_wave_has_one_component():
    """Every mode of an unpatterned cell is a plane wave. The acoustic ones travel at the wavevector of the
    solve, so all they have is ŝ(0); the first folded mode is the same plane wave at k + G with |h| = 1, and
    it shows up there and nowhere else — which is what the phase e^{-i(k+G)·r} is for."""
    p = cube(num_eigs=4, k=["kx", "0.17*kx", "0"])
    results = physics.of(p).solver(p, log=lambda m: None).run()
    h = results.fourier_orders
    g0 = zero_order(h)
    for band in range(3):                                    # the three acoustic branches
        c = results.fourier[0, 0, band]
        assert np.abs(np.delete(c, g0, axis=0)).max() < 1e-4 * np.abs(c[g0]).max()
    folded = results.fourier[0, 0, 3]                        # the lowest folded branch, near v_T (2pi/a - k)
    assert results.freqs[0, 0, 3] == pytest.approx(4.97e9, rel=0.02)
    assert np.abs(folded[g0]).max() < 1e-4 * np.abs(folded).max()
    peak = np.unravel_index(int(np.argmax(np.abs(folded))), folded.shape)[0]
    assert np.abs(h[peak]).sum() == 1


def test_modes_are_mass_normalised():
    """The components of different modes are comparable because every mode carries the same modal mass:
    u^H M u = ∫ρ|u|² dV = 1 in SI units, with the phase rotated so that Σ u·u is real and positive."""
    p = cube(num_eigs=3)
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    _, u = s.solve_point(ev, s.k_of(ev), modes=True)
    M = s._model.M
    for raw in u:
        mode = fourier.mass_normalise(raw, M)
        flat = mode.reshape(-1)
        assert float(np.real(np.vdot(flat, M @ flat))) == pytest.approx(1.0, rel=1e-10)
        total = np.sum(mode * mode)
        assert total.real > 0 and abs(total.imag) < 1e-9 * abs(total)
        big = np.abs(raw) > 0.1 * np.abs(raw).max()      # the same mode: one scale and one phase, no more
        ratio = (mode[big] / raw[big])
        assert np.allclose(ratio, ratio[0], rtol=1e-9)


def test_degeneracies_group_equal_frequencies():
    """Modes closer than 1e-4 of the highest frequency of their solve count as one group; the group size is
    what is reported, 1 for a mode of its own."""
    freqs = np.array([[[1.0e9, 1.0e9 + 10.0, 2.0e9, np.nan]]])
    np.testing.assert_array_equal(fourier.degeneracies(freqs)[0, 0], [2, 2, 1, 1])
    freqs = np.array([[[1.0e9, 1.1e9, 2.0e9, 2.0e9 + 1e3]]])     # 1 kHz apart, well inside 1e-4 * 2 GHz
    np.testing.assert_array_equal(fourier.degeneracies(freqs)[0, 0], [1, 1, 2, 2])
    assert fourier.degeneracies(np.full((1, 1, 3), np.nan))[0, 0].tolist() == [1, 1, 1]


def test_degenerate_transverse_modes_are_flagged():
    """Along x of a cubic crystal the two transverse branches are degenerate by symmetry: the solver returns
    an arbitrary pair of them, which is exactly the case the flag is there for."""
    p = cube(num_eigs=4, k=["kx", "0", "0"])
    results = physics.of(p).solver(p, log=lambda m: None).run()
    f = results.freqs[0, 0]
    assert f[1] - f[0] < 1e-6 * f[1]                     # the transverse pair
    np.testing.assert_array_equal(fourier.degeneracies(results.freqs)[0, 0], [2, 2, 1, 1])


def test_explicit_wavevectors_are_solved_as_listed():
    """A list of wavevectors is solved point by point, in reduced coordinates and in the order given — no
    interpolation, and no named ticks on the plot axis."""
    listed = [["0.15", "0.05", "0"], ["0.15", "-0.05", "0"], ["0.3", "0.1", "0"]]
    p = cube(kpath=KPath(mode="list", k=listed), num_eigs=3)
    assert p.validate() == []
    results = physics.of(p).solver(p, log=lambda m: None).run()
    ev = p.evaluator()
    b = 2 * np.pi / ev("a")
    np.testing.assert_allclose(results.k[0], [[float(c) * b for c in point] for point in listed], rtol=1e-12)
    assert results.is_kpath and results.ticks == []
    np.testing.assert_allclose(results.x, [0, 2 * 0.05 * b, np.hypot(0.15, 0.15) * b + 2 * 0.05 * b], rtol=1e-9)
    direct = Solver(p, log=lambda m: None).solve_point(ev, results.k[0, 2], n=3)
    np.testing.assert_allclose(results.freqs[0, 2], direct, rtol=1e-9)


def test_a_wavevector_of_the_list_may_be_an_expression():
    """The entries are expressions, so a variable sweep can turn the direction of the wavevector."""
    p = cube(kpath=KPath(mode="list", k=[["0.4*cos(th[deg])", "0.4*sin(th[deg])", "0"]]), num_eigs=2)
    p.variables["th"] = "30"
    assert p.validate() == []
    results = physics.of(p).solver(p, log=lambda m: None).run()
    b = 2 * np.pi / p.evaluator()("a")
    np.testing.assert_allclose(results.k[0, 0], [0.4 * np.cos(np.pi / 6) * b, 0.4 * 0.5 * b, 0], rtol=1e-12)


def test_settings_round_trip(tmp_path):
    """Both settings survive the model file, disabled as well as enabled."""
    p = cube(orders=(2, 1, 0), kpath=KPath(mode="list", k=[["0.5", "0.1", "0"], ["0.5", "0.2", "0"]]))
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.solid.problem.kpath == p.solid.problem.kpath
    assert q.solid.problem.fourier == Fourier(orders=[2, 1, 0])
    assert q.solid.problem.active_fourier is q.solid.problem.fourier
    q.solid.problem.fourier.enabled = False
    q.solid.problem.kpath.mode = "path"
    r = Project.load(q.save(tmp_path / "off.toml"))
    assert r.solid.problem.fourier.enabled is False and r.solid.problem.active_fourier is None
    assert r.solid.problem.kpath.mode == "path" and r.solid.problem.kpath.path == ["G", "X"]


def test_components_belong_to_an_eigenfrequency_problem():
    """Modes are needed, so the components are only computed for an eigenfrequency problem."""
    p = cube()
    p.solid.problem.type = "stationary"
    assert p.solid.problem.active_fourier is None


@pytest.mark.parametrize("change, message", [
    (lambda st: setattr(st.fourier, "orders", [1, 1, 1.5]), "whole numbers"),
    (lambda st: setattr(st.fourier, "orders", [20, 20, 1]), "more than"),
    (lambda st: setattr(st.kpath, "k", []), "empty"),
    (lambda st: setattr(st.kpath, "k", [["0.5", "0.5"]]), "3 reduced coordinates"),
    (lambda st: setattr(st.kpath, "mode", "spiral"), "mode must be one of"),
])
def test_bad_settings_are_rejected(change, message):
    p = cube(kpath=KPath(mode="list", k=[["0.5", "0", "0"]]))
    change(p.solid.problem)
    assert any(message in e for e in p.validate()), p.validate()


def test_an_order_needs_a_periodic_direction():
    """There is no b3 to count orders along when the lattice is a membrane."""
    p = cube(orders=(1, 1, 1))
    p.lattice = Lattice(periodic={"x": "a", "y": "a"})
    assert any("not periodic along z" in e for e in p.validate())
    p.solid.problem.fourier.orders = [1, 1, 0]
    assert p.validate() == []


def test_a_run_writes_the_components_and_the_degeneracies(tmp_path):
    """The run folder gets fourier.csv (one complex column per mode) and the degeneracy flags beside it, and
    the arrays come back from results.npz unchanged."""
    p = cube(orders=(1, 0, 0), num_eigs=4, k=["kx", "0", "0"])
    results = physics.of(p).solver(p, log=lambda m: None).run()
    written = results.write_outputs(p, tmp_path / "run", [])
    assert {f.name for f in written} == {"dispersion.csv", "results.npz", "fourier.csv",
                                         "fourier_degeneracy.csv", "model.toml", "run.json"}
    rows = list(csv.DictReader(open(tmp_path / "run" / "fourier.csv", encoding="utf-8")))
    assert len(rows) == 3 * len(fourier.FIELDS)                  # three reciprocal vectors, nine fields
    h = results.fourier_orders
    for row in rows:
        orders = [int(row[f"h{i + 1}"]) for i in range(3)]
        g = int(np.nonzero((h == orders).all(axis=1))[0][0])
        c = results.fourier[0, 0, 2, g, fourier.FIELDS.index(row["field"])]
        assert complex(row["mode 3"]) == pytest.approx(c, rel=1e-9)
    flags = list(csv.DictReader(open(tmp_path / "run" / "fourier_degeneracy.csv", encoding="utf-8")))
    assert [flags[0][f"degenerate modes at mode {b + 1}"] for b in range(4)] == ["2", "2", "1", "1"]
    reloaded = ModalResults.load(tmp_path / "run" / "results.npz")
    np.testing.assert_allclose(reloaded.fourier, results.fourier)
    np.testing.assert_array_equal(reloaded.fourier_orders, h)
    np.testing.assert_allclose(reloaded.fourier_g, results.fourier_g)
    assert results.info["fourier"]["normalisation"].startswith("mass")


def test_the_panels_edit_the_model(tmp_path):
    """The GUI nodes write into the same model file: the wavevector list in the Wavevector sweep panel, the
    orders in the Fourier components panel."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6.QtWidgets import QApplication, QComboBox, QPushButton, QSpinBox, QTableWidget
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    w = MainWindow()
    w.set_project(templates.create("square"))

    w.select_node("kpath")
    panel = w.settings.widget()
    mode = next(c for c in panel.findChildren(QComboBox) if c.findData("list") >= 0)
    mode.setCurrentIndex(mode.findData("list"))
    app.processEvents()
    panel = w.settings.widget()
    next(b for b in panel.findChildren(QPushButton) if b.text() == "Add wavevector").click()
    table = panel.findChild(QTableWidget)
    table.item(0, 1).setText("0.123")
    kpath = w.project.solid.problem.kpath
    assert kpath.mode == "list" and kpath.k == [["0", "0.123", "0"]]

    markers = lambda: [tuple(line.get_xydata()[0]) for line in                       # noqa: E731
                       w.settings.widget().findChild(FigureCanvasQTAgg).figure.axes[0].lines
                       if line.get_marker() == "o"]
    b = 2 * np.pi / w.project.evaluator()("a")
    np.testing.assert_allclose(markers(), [[0, 0.123 * b]])    # the sketch follows the table, unrebuilt
    table.item(0, 0).setText("0.4")
    np.testing.assert_allclose(markers(), [[0.4 * b, 0.123 * b]])

    w.select_node("kpath")                     # the same in path mode: a moved point moves its dot
    app.processEvents()
    panel = w.settings.widget()
    mode = next(c for c in panel.findChildren(QComboBox) if c.findData("path") >= 0)
    mode.setCurrentIndex(mode.findData("path"))
    app.processEvents()
    panel = w.settings.widget()
    points = panel.findChild(QTableWidget)
    points.item(1, 1).setText("0.25")          # X = [0.25, 0, 0]
    assert min(abs(x - 0.25 * b) + abs(y) for x, y in markers()) < 1e-6 * b

    w.select_node("fourier")
    panel = w.settings.widget()
    boxes = panel.findChildren(QSpinBox)
    assert [b.isEnabled() for b in boxes] == [True, True, False]      # no b3: the lattice is a membrane
    boxes[0].setValue(3)
    fo = w.project.solid.problem.fourier
    assert fo.orders == [3, 1, 0] and fo.enabled
    q = Project.load(w.project.save(tmp_path / "m.toml"))      # the list survives the switch back to a path
    assert q.solid.problem.fourier == fo
    assert q.solid.problem.kpath.mode == "path" and q.solid.problem.kpath.k == [["0.4", "0.123", "0"]]
    w.dirty = False
    w.close()
