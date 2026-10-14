import numpy as np
import pytest

from solidocean.expressions import Evaluator
from solidocean.units import DIMENSIONLESS, FREQUENCY, INV_LENGTH, LENGTH, PRESSURE, axis_scale, dimension, format_value, length_expr


@pytest.fixture
def ev():
    return Evaluator({"a": "500[nm]", "r": "0.3*a", "kx": "pi/a", "C": "165.7[GPa]",
                      "rho": "2329[kg/m^3]", "f": "sqrt(C/rho)/a", "n": "4", "bad": "a + 1"})


def test_dimension_inference(ev):
    d = {k: dimension(ev.expressions[k], ev) for k in ev.expressions}
    assert d["a"] == LENGTH and d["r"] == LENGTH and d["kx"] == INV_LENGTH
    assert d["C"] == PRESSURE and d["f"] == FREQUENCY and d["n"] == DIMENSIONLESS
    assert d["bad"] is None  # inconsistent: length + number


def test_display_formatting(ev):
    assert format_value(ev.values["r"], LENGTH, "um") == "0.15 µm"
    assert format_value(ev.values["r"], LENGTH, "nm") == "150 nm"
    assert format_value(ev.values["C"], PRESSURE) == "165.7 GPa"
    assert format_value(ev.values["kx"], INV_LENGTH, "um") == f"{np.pi / 0.5:.6g} rad/µm"
    assert axis_scale(LENGTH, "nm") == (pytest.approx(1e9), "nm")
    assert length_expr(2.5e-7, "um") == "0.25[um]"


def test_sweep_plot_axis_uses_display_unit():
    from solidocean.project import VariableSweep
    from solidocean.physics.solid.solver import Solver
    from solidocean.results.style import outer_axis
    p = _example()
    p.solid.problem.kpath.enabled = False
    p.problem.sweep = VariableSweep("r", "[100[nm], 150[nm]]")
    p.mesh.max_size = "a/3"
    p.solid.problem.num_eigs = 4
    res = Solver(p, log=lambda m: None).run()
    np.testing.assert_allclose(res.outer_values, [1e-7, 1.5e-7])   # stored in SI
    x, label = outer_axis(res, "nm")
    np.testing.assert_allclose(x, [100, 150])
    assert label == "r (nm)"


def _example():
    """The square-lattice phononic crystal template, loaded from a model file."""
    import tempfile
    from pathlib import Path
    from solidocean import templates
    from solidocean.project import Project
    return Project.load(templates.create("square").save(Path(tempfile.mkdtemp()) / "square.toml"))


def _axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots()


def test_plot_single_variable_value():
    from solidocean.plots import ResultPlot
    from solidocean.physics.solid.modal import ModalResults
    from solidocean.results.modal import draw
    freqs = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2) * 1e9
    r = ModalResults(freqs, np.zeros((2, 3, 3)), np.array([0.0, 1.0, 2.0]), "k", [(0.0, "G"), (2.0, "X")],
                "r", np.array([1e-7, 2e-7]), {"outer_dim": [1, 0, 0]})
    p = _example()
    fig, ax = _axes()
    table = draw(r, ax, p, ResultPlot(at=1))
    assert len(ax.lines) == 2 + 2  # 2 bands + 2 tick lines
    np.testing.assert_allclose(ax.lines[0].get_ydata(), freqs[1, :, 0] / 1e9)
    assert set(table[:, 2]) == {1}  # only variable index 1 is pickable
    ax.clear()
    draw(r, ax, p, ResultPlot(at=-1))
    assert len(ax.lines) == 4 + 2  # both variable values overlaid
    ax.clear()
    draw(r, ax, p, ResultPlot(sweep="variable", at=2))  # frequency vs r at the 3rd k point
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.1, 0.2])   # r in µm
    np.testing.assert_allclose(ax.lines[0].get_ydata(), freqs[:, 2, 0] / 1e9)


def test_plot_style_applied_and_roundtrips(tmp_path):
    from solidocean.plots import PlotStyle, ResultPlot
    from solidocean.project import Project
    from solidocean.physics.solid.modal import ModalResults
    from solidocean.results.modal import draw
    freqs = np.arange(1 * 3 * 3, dtype=float).reshape(1, 3, 3) * 1e9
    r = ModalResults(freqs, np.zeros((1, 3, 3)), np.array([0.0, 1.0, 2.0]), "k", [(0.0, "G"), (2.0, "X")])
    st = PlotStyle(markers=False, line_style="dashed", colors="bands", colormap="plasma", grid=False, y_max="5")
    p = _example()
    fig, ax = _axes()
    draw(r, ax, p, ResultPlot(style=st))
    bands = ax.lines[:3]
    assert all(l.get_linestyle() == "--" and l.get_marker() in (None, "None", "") for l in bands)
    assert len({tuple(np.atleast_1d(l.get_color())) for l in bands}) == 3
    assert ax.get_ylim()[1] == 5
    p.plots = [ResultPlot(style=st), ResultPlot("Polarization", "mean(abs(uz)^2)")]
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.plots == p.plots


def test_wavevector_axis_and_ranges():
    from solidocean.plots import ResultPlot
    from solidocean.physics.solid.modal import ModalResults
    from solidocean.results.modal import inner_axis
    from solidocean.results.style import plot_ranges
    r = ModalResults(np.ones((1, 3, 1)) * 1e9, np.zeros((1, 3, 3)), np.array([0.0, 1e6, 2e6]), "k",
                [(0.0, "G"), (2e6, "X")])
    p = _example()
    pl = ResultPlot()
    pl.style.x_axis, pl.style.k_unit = "k", "1/cm"
    x, label = inner_axis(r, pl.style)
    np.testing.assert_allclose(x, [0, 1e4, 2e4])           # rad/m -> 1/cm
    assert "cm" in label
    pl.style.y_min, pl.style.y_max, pl.style.x_max = "2", "10[GHz]", "1.5e4"   # numbers in display units
    (ylo, yhi), (xlo, xhi) = plot_ranges(p, pl, y_scale=1e9)
    assert ylo == 2 and yhi == 10 and xlo is None and xhi == 1.5e4


def test_x_label_roundtrip_and_drawn(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solidocean import templates
    from solidocean.project import Project
    p = templates.create("heat_pad")
    p.plots[2].x_label = "t after switch-on (ns)"                     # "Peak temperature", max(T) vs time
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.plots[2].x_label == "t after switch-on (ns)" and q.plots[1].x_label == ""
    from solidocean.physics.heat.solver import HeatResults
    from solidocean.results.field import draw as draw_fields
    r = HeatResults(kind="transient", times=np.array([0.0, 1e-9]), T=[np.zeros((2, 3))],
                    quantities={"max(T)": np.ones((1, 2))})
    fig, ax = plt.subplots()
    draw_fields(r, ax, q, q.plots[2])
    assert ax.get_xlabel() == "t after switch-on (ns)"
    plt.close(fig)


def test_preferred_frequency_and_time_units():
    """Frequencies and times are shown in the project's preferred units, saved with the model."""
    from solidocean.project import Project
    from solidocean.units import TIME
    p = Project(frequency_unit="MHz", time_unit="ms")
    du = p.display_units
    assert format_value(2.5e6, FREQUENCY, du) == "2.5 MHz"
    assert format_value(3e-3, TIME, du) == "3 ms"
    assert axis_scale(FREQUENCY, du) == (pytest.approx(1e-6), "MHz")
    assert axis_scale(TIME, Project(time_unit="us").display_units) == (pytest.approx(1e6), "µs")
    q = Project.from_dict(p.to_dict())
    assert (q.frequency_unit, q.time_unit) == ("MHz", "ms")
    assert not any("unit" in e for e in q.validate())
    assert any("time_unit" in e for e in Project(time_unit="h").validate())
