"""Every structure template is a valid, solvable model."""
import numpy as np
import pytest

from solidocean import templates
from solidocean.physics.solid.solver import Solver


@pytest.mark.parametrize("name", list(templates.all_templates()))
def test_template_is_valid_and_solves(name, tmp_path):
    p = templates.create(name)
    assert p.validate() == []
    p = type(p).load(p.save(tmp_path / f"{name}.toml"))         # survives the file round trip
    assert p.validate() == []
    if p.physics_kind == "heat":
        from solidocean.physics.heat.solver import HeatSolver
        p.study.sweep = None
        if p.heat.study.type == "transient":
            p.heat.study.steps = 4
        res = HeatSolver(p, log=lambda m: None).run([q for pl in p.plots for q in pl.expressions()])
        assert all(np.all(np.isfinite(v)) for v in res.quantities.values())
        return
    if p.solid.study.field_study:
        p.study.sweep = None
        res = Solver(p, log=lambda m: None).run([q for pl in p.plots for q in pl.expressions()])
        assert np.all(np.isfinite(res.U[0])) and np.abs(res.U[0]).max() > 0
        assert all(np.all(np.isfinite(v)) for v in res.quantities.values())
        return
    p.mesh.max_size = {"bulk": "a/2", "membrane": "a", "beam": "a"}.get(name, "a/4")
    s = Solver(p, log=lambda m: None)
    f = s.solve_point(p.evaluator(), np.zeros(3), n=4)
    assert np.all(np.isfinite(f))


def test_lattice_cell_matches_rectangular_periodicity():
    p = templates.create("square")
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    lo, hi = s.model(ev).mesh.bounds()
    np.testing.assert_allclose(hi[:2] - lo[:2], [ev("a")] * 2, rtol=1e-9)    # the cell is the a x a box
