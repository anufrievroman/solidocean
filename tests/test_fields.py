"""Derived quantities evaluated over mode shapes."""
import numpy as np
import pytest

from solidocean import fields
from solidocean.expressions import ExpressionError
from solidocean.project import Feature, Material, MeshSettings, Project
from solidocean.lattice import Lattice
from solidocean.physics.solid.model import SolidPhysics, SolidProblem
from solidocean.physics.solid.plots import colored_plot
from solidocean.plots import ResultPlot
from solidocean.physics.solid.modal import compute_quantity, NeedsModes
from solidocean.physics.solid.solver import Solver


def bulk_cube():
    # bulk Si, periodic in x, y, z, k along x: modes 1,2 transverse (uy, uz), mode 3 longitudinal (ux)
    return Project(
        variables={"a": "1[um]", "kx": "0.1*pi/a"},
        materials=[Material("Si", {"library": "Si"})],
        objects=[Feature("block", "cell", material="Si",
                          params={"center": ["0", "0", "0"], "size": ["a", "a", "a"]})],
        mesh=MeshSettings(max_size="a/3"),
        lattice=Lattice(periodic={"x": "a", "y": "a", "z": "a"}),
        solid=SolidPhysics(problem=SolidProblem(k=["kx", "0", "0"])),
    )


def test_quadrature_volume_and_coordinates():
    p = bulk_cube()
    s = Solver(p, log=lambda m: None)
    mesh = s.model(p.evaluator()).mesh
    q = fields.Quadrature.of(mesh)
    assert q.volume == pytest.approx(1e-18, rel=1e-10)
    sc = {"freq": 1.0}
    assert fields.evaluate("mean(x)", sc, q) == pytest.approx(0, abs=1e-15)
    assert fields.evaluate("mean(x^2)", sc, q) == pytest.approx(1e-12 / 12, rel=1e-8)  # exact for P2
    assert fields.evaluate("integral(1)", sc, q) == pytest.approx(1e-18, rel=1e-10)
    assert fields.evaluate("max(z)", sc, q) < 0.5e-6


def test_polarization_of_bulk_modes():
    p = bulk_cube()
    p.solid.problem.num_eigs = 3
    p.plots = [ResultPlot(), ResultPlot("norm", "mean(abs(ux)^2+abs(uy)^2+abs(uz)^2)"),
               ResultPlot("long", "mean(abs(ux)^2)")]
    res = Solver(p, log=lambda m: None).run(quantities=[pl.quantity for pl in p.plots], store_modes=True)
    np.testing.assert_allclose(res.quantities["mean(abs(ux)^2+abs(uy)^2+abs(uz)^2)"][0, 0], 1, rtol=1e-6)
    np.testing.assert_allclose(res.quantities["mean(abs(ux)^2)"][0, 0], [0, 0, 1], atol=1e-6)
    # post-hoc evaluation from stored modes gives the same numbers
    again = compute_quantity(res, p, "mean(abs(uy)^2+abs(uz)^2)")
    np.testing.assert_allclose(again[0, 0], [1, 1, 0], atol=1e-6)
    np.testing.assert_allclose(compute_quantity(res, p, "freq/1e9")[0, 0], res.freqs[0, 0] / 1e9)


def test_needs_modes_and_errors():
    p = bulk_cube()
    p.solid.problem.num_eigs = 2
    res = Solver(p, log=lambda m: None).run()
    with pytest.raises(NeedsModes):
        compute_quantity(res, p, "mean(abs(uz)^2)")
    with pytest.raises(ExpressionError, match="reduce"):
        fields.evaluate("x", {"freq": 1.0}, fields.Quadrature.of(Solver(p).model(p.evaluator()).mesh))
    with pytest.raises(ExpressionError):
        fields.check("mean(abs(uq))", {})
    assert any("quantity" in e for e in Project(plots=[ResultPlot("bad", "nonsense(1)")]).validate())


def test_colored_dispersion(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solidocean.results.modal import draw
    p = bulk_cube()
    p.solid.problem.num_eigs = 3
    pl = colored_plot(color_by="mean(abs(ux)^2)")
    assert pl.quantity == "freq" and pl.expressions() == ["freq", "mean(abs(ux)^2)"]
    p.plots = [ResultPlot(), pl]
    res = Solver(p, log=lambda m: None).run(quantities=[q for x in p.plots for q in x.expressions()])
    fig, ax = plt.subplots()
    table = draw(res, ax, p, pl)
    sc = ax.collections[-1]
    np.testing.assert_allclose(sc.get_array(), [0, 0, 1], atol=1e-6)   # longitudinal mode is ux
    np.testing.assert_allclose(table[:, 1], res.freqs[0, 0] / 1e9)       # y is still frequency (GHz)
    assert len(fig.axes) == 2                                            # plot + colour bar
    plt.close(fig)
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.plots[1].kind == "colored" and q.plots[1].color_by == "mean(abs(ux)^2)"
