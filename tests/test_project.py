import pytest

from solidocean import templates
from solidocean.cli import main
from solidocean.project import Project


@pytest.fixture
def EXAMPLE(tmp_path):
    """A model file: the square-lattice phononic crystal template."""
    return templates.create("square").save(tmp_path / "square.toml")


def test_example_is_valid_and_roundtrips(tmp_path, EXAMPLE):
    p = Project.load(EXAMPLE)
    assert p.validate() == []
    out = p.save(tmp_path / "copy.toml")
    q = Project.load(out)
    assert q.to_dict() == p.to_dict()


def test_cli_check(EXAMPLE):
    assert main([str(EXAMPLE), "--check", "-q"]) == 0


def test_cli_run(tmp_path, EXAMPLE):
    p = Project.load(EXAMPLE)
    p.mesh.max_size = "a/3"
    p.solid.study.kpath.n = 2
    p.solid.study.num_eigs = 4
    path = p.save(tmp_path / "m.toml")
    assert main([str(path), "-q"]) == 0
    out = tmp_path / "m"
    assert {f.name for f in out.iterdir()} == {"dispersion.csv", "results.npz", "model.toml",
                                               "run.json", "dispersion_colored.csv"}


def test_disabled_sweeps_roundtrip_and_are_skipped(tmp_path, EXAMPLE):
    p = Project.load(EXAMPLE)
    p.solid.study.kpath.enabled = False
    p.study.solver = "superlu"
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.solid.study.kpath is not None and q.solid.study.kpath.enabled is False
    assert q.solid.study.active_kpath is None
    assert q.study.solver == "superlu"


def test_solver_backends_agree(EXAMPLE):
    import numpy as np
    from solidocean.fem import eigen
    from solidocean.physics.solid.solver import Solver
    p = Project.load(EXAMPLE)
    p.mesh.max_size = "a/4"
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    k = np.array([2e6, 1e6, 0])
    results = {}
    for b in eigen.available_backends():
        p.study.solver = b
        results[b] = s.solve_point(ev, k, n=8)
    ref = results["superlu"]
    for f in results.values():
        np.testing.assert_allclose(f, ref, rtol=1e-8)
    p.study.solver = "nonsense"
    assert any("study.solver" in e for e in p.validate())


def test_save_csv_custom_name(tmp_path):
    import csv
    import numpy as np
    from solidocean.physics.solid.modal import ModalResults
    r = ModalResults(np.array([[[1e9, 2e9]]]), np.zeros((1, 1, 3)), np.zeros(1), "k")
    out = r.save_csv(tmp_path / "my_bands.csv", "GHz")
    rows = list(csv.reader(open(out)))
    assert rows[0][-2:] == ["f1 [GHz]", "f2 [GHz]"] and rows[1][-2:] == ["1", "2"]


def test_file_layout(tmp_path):
    """Shared sections, [lattice], and one table for the physics solved, without default values."""
    import tomllib
    for name, section, other in (("hexagonal", "solid", "heat"), ("heat_pad", "heat", "solid")):
        p = templates.create(name)
        d = tomllib.loads(p.save(tmp_path / f"{name}.toml").read_text())
        assert section in d and other not in d and "physics" not in d
        assert Project.load(tmp_path / f"{name}.toml").to_dict() == p.to_dict()
    d = tomllib.loads((tmp_path / "hexagonal.toml").read_text())
    assert d["lattice"] == {"vectors": [["a", "0", "0"], ["a/2", "sqrt(3)*a/2", "0"]]}
    assert d["solid"]["study"]["kpath"]["path"] == ["G", "M", "K", "G"]
    assert "tol" not in d["solid"]["study"] and "output" not in d          # defaults are left out
    assert '\ncenter = ["0", "0", "0"]\n' in (tmp_path / "hexagonal.toml").read_text()   # short arrays: one line
    import pytest
    with pytest.raises(ValueError, match="unknown section"):
        Project.from_dict({**d, "physics": {}})


def test_log_axes_round_trip_and_draw(tmp_path):
    """Log axes are saved only when set and turn the matplotlib axes logarithmic."""
    import matplotlib
    matplotlib.use("Agg")
    from solidocean import templates
    from solidocean.physics.heat.solver import HeatSolver
    p = templates.create("heat_convection")
    pl = p.plots[1]
    assert "y_log" not in pl.to_dict().get("style", {})
    pl.style.y_log = pl.style.x_log = True
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.plots[1].style.y_log and q.plots[1].style.x_log
    res = HeatSolver(q, log=lambda m: None).run([q.plots[1].quantity])
    import matplotlib.pyplot as plt
    from solidocean.results.field import draw
    fig, ax = plt.subplots()
    draw(res, ax, q, q.plots[1])
    assert ax.get_yscale() == "log" and ax.get_xscale() == "log"
    plt.close(fig)
