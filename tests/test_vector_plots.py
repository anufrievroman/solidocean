"""Arrow (quiver) and streamline plots of a vector quantity on a cross-section."""
import os

import matplotlib
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg")

from solidocean import physics, templates
from solidocean.plots import PlotStyle, ResultPlot
from solidocean.project import Project
from solidocean.results.field import draw, save_vector_table, vector_values


@pytest.fixture(scope="module")
def bar():
    """The heat_bar template without its sweep: a uniform flux qx along the bar."""
    p = templates.create("heat_bar")
    p.problem.sweep = None
    return p, physics.of(p).solver(p, log=lambda m: None).run([])


def vector_plot(kind="quiver", **kw):
    return ResultPlot("Flux", kind=kind, vector=["qx", "qy", "qz"], **kw)


def test_in_plane_part_of_the_vector_follows_the_plane(bar):
    p, r = bar
    _, Vu, Vv = vector_values(r, p, vector_plot(plane="xy"))
    q = np.nanmean(Vu)
    assert abs(q) > 0 and np.nanmax(np.abs(Vu - q)) < 1e-3 * abs(q)           # the flux runs along x
    assert np.nanmax(np.abs(Vv)) < 1e-3 * abs(q)
    _, Vu, Vv = vector_values(r, p, vector_plot(plane="yz"))                  # it crosses this plane: nothing in it
    assert max(np.nanmax(np.abs(Vu)), np.nanmax(np.abs(Vv))) < 1e-3 * abs(q)
    _, Vu, Vv = vector_values(r, p, vector_plot(plane="xy", rotation=["0", "0", "90"]))   # e1 = y, e2 = -x
    assert np.nanmax(np.abs(Vu)) < 1e-3 * abs(q)
    np.testing.assert_allclose(np.nanmean(Vv), -q, rtol=1e-3)


def test_arrow_grid_is_square_on_the_screen_and_inside_the_plane(bar):
    p, r = bar
    g, Vu, _ = vector_values(r, p, vector_plot(arrows=20), 20, centered=True)
    assert max(g.nu, g.nv) == 20 and Vu.shape == (g.nv, g.nu)
    assert g.box[0] < g.ud[0] and g.ud[-1] < g.box[1] and g.vd[0] > g.box[2]   # cell centres, not edges


@pytest.mark.parametrize("kind", ["quiver", "stream"])
@pytest.mark.parametrize("background", ["", "T"])
def test_drawing(bar, kind, background):
    import matplotlib.pyplot as plt
    p, r = bar
    pl = vector_plot(kind, quantity=background, style=PlotStyle(colormap="viridis"))
    fig, ax = plt.subplots()
    assert draw(r, ax, p, pl).shape == (0, 5)
    arrows = ax.collections if kind == "quiver" else ax.patches + ax.collections
    assert len(arrows) >= 1
    assert len(fig.axes) == 2                                  # the colour bar: of the map, or of the lengths
    plt.close(fig)


def test_vector_table(bar, tmp_path):
    import csv
    p, r = bar
    pl = vector_plot(resolution=20, quantity="T")
    rows = list(csv.reader(open(save_vector_table(r, p, pl, tmp_path / "v.csv"), encoding="utf-8")))
    assert rows[0] == ["x [m]", "y [m]", "z [m]", "qx", "qy", "qz", "T"]
    assert float(rows[1][3]) == pytest.approx(float(rows[-1][3]), rel=1e-3)


def test_run_writes_the_table(bar, tmp_path):
    p, r = bar
    plots = [vector_plot(resolution=15), vector_plot("stream", resolution=15)]
    plots[1].name = "Lines"
    files = {f.name for f in r.write_outputs(p, tmp_path, plots)}
    assert {"flux.csv", "lines.csv"} <= files


def test_settings_roundtrip_and_validation(tmp_path):
    p = templates.create("heat_bar")
    pl = vector_plot(arrows=12, arrow_scale=1.5, arrow_width=2.0, quantity="T", plane="xz",
                     style=PlotStyle(colormap="magma", color="#ffffff"))
    st = vector_plot("stream", stream_density=2.5, arrow_scale=0.5)
    st.name = "Lines"
    p.plots += [pl, st]
    assert p.validate() == []
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.plots[-2:] == [pl, st]
    pl.vector, pl.arrows, pl.arrow_width = ["qx", "", "nonsense(T)"], 1, 0
    st.vector, st.stream_density = ["qx", "qy", "nonsense(T)"], 9
    errors = "\n".join(p.validate())
    for text in ("vector needs 3", "arrows must", "arrow_scale and arrow_width", "vector:", "stream_density"):
        assert text in errors


def test_physics_start_the_plots_with_their_vector():
    for name, vector in (("heat_bar", ["qx", "qy", "qz"]), ("tensile", ["ux", "uy", "uz"])):
        p = templates.create(name)
        spec = physics.of(p)
        box = ((0, 0, 0), (1e-6, 1e-6, 1e-6))
        for kind in ("quiver", "stream"):
            pl = spec.new_plot(p, kind, box)
            assert pl.kind == kind and pl.vector == vector and pl.quantity == ""
            p.plots.append(pl)
            pl.name += f" {kind}"
        assert p.validate() == []


def test_vector_plot_panel_in_the_window(tmp_path):
    """A model's arrow plot has its panel (vector, arrows, style) and draws in the window after a run."""
    from PySide6.QtWidgets import QApplication, QLineEdit
    app = QApplication.instance() or QApplication([])
    from solidocean.gui.main_window import MainWindow
    p = templates.create("heat_bar")
    p.problem.sweep = None
    w = MainWindow()
    w.set_project(p)
    w.results = physics.of(p).solver(p, log=lambda m: None).run([])
    w.results_project = p
    w.add_plot_kind("quiver")
    w.add_plot_kind("stream")
    for index in (len(p.plots) - 2, len(p.plots) - 1):
        w.select_node("plot", index)
        app.processEvents()
        edits = [e.text() for e in w.settings.widget().findChildren(QLineEdit)]
        assert "qx" in edits and "qy" in edits and "qz" in edits
        w.show_plot(index)
        assert w.plot.ax.collections or w.plot.ax.patches
    w.dirty = False
    w.close()
