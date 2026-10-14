"""Cross-sections, quantities of field views at the nodes, material colours and plot type labels."""
import csv

import numpy as np
import pytest

from solidocean import fields, physics, templates
from solidocean.plots import PlotStyle, ResultPlot, plot_label
from solidocean.project import Material, Project
from solidocean.results.field import save_section_table, section_values


def heated_bar():
    """The heat_bar template without its sweep: T is linear in x, from T_hot at -L/2 to T_cold at L/2."""
    p = templates.create("heat_bar")
    p.problem.sweep = None
    return p, physics.of(p).solver(p, log=lambda m: None).run([])


def test_section_of_a_linear_temperature_profile():
    p, r = heated_bar()
    ev = p.evaluator()
    L, Th, Tc = ev("L"), ev("T_hot"), ev("T_cold")
    pl = ResultPlot("Section", "T", kind="section", plane="xy", origin=["0", "0", "0"], resolution=60)
    u, v, vals, i, j, names = section_values(r, p, pl)
    assert names == ("x", "y") and vals.shape == (len(v), len(u))
    X = np.broadcast_to(u, vals.shape)
    ok = np.isfinite(vals)
    assert ok.mean() > 0.9                                         # the bar fills the plane's bounding box
    np.testing.assert_allclose(vals[ok], Th + (Tc - Th) * (X[ok] + L / 2) / L, atol=1e-6)


def test_section_axes_and_table_follow_the_origin(tmp_path):
    p, r = heated_bar()
    pl = ResultPlot("Section", "qx", kind="section", plane="xz", origin=["L/4", "0", "0"], resolution=40)
    u, v, vals, i, j, names = section_values(r, p, pl)
    lo, hi = r.meshes[0].bounds()
    assert names == ("x", "z") and u[0] == pytest.approx(lo[0]) and u[-1] == pytest.approx(hi[0])
    rows = list(csv.reader(open(save_section_table(r, p, pl, tmp_path / "s.csv"), encoding="utf-8")))
    assert rows[0] == ["x [m]", "y [m]", "z [m]", "qx"]            # displayed axes = model coordinates
    xs = [float(x) for x, _, _, _ in rows[1:]]
    assert min(xs) == pytest.approx(u[0]) and max(xs) == pytest.approx(u[-1])
    assert all(float(y) == pytest.approx(0.0) for _, y, _, _ in rows[1:])   # the plane y = 0
    pl.rotation = ["0", "0", "30"]
    assert section_values(r, p, pl)[5] == ("u", "v")
    rows = list(csv.reader(open(save_section_table(r, p, pl, tmp_path / "r.csv"), encoding="utf-8")))
    assert rows[0] == ["u [m]", "v [m]", "x [m]", "y [m]", "z [m]", "qx"]


def test_field_quantities_at_the_nodes():
    p, r = heated_bar()
    from solidocean.physics.heat.fields import at_nodes
    ev = p.evaluator()
    mesh = r.meshes[0]
    nf = at_nodes(mesh, r.T[0][0], r.kels[0])
    qx = fields.evaluate_nodal("qx", mesh.points, ev.values, nf)
    k = 148.0
    np.testing.assert_allclose(qx, k * (ev("T_hot") - ev("T_cold")) / ev("L"), rtol=1e-6)    # uniform flux
    q = fields.evaluate_nodal("sqrt(qx^2 + qy^2 + qz^2)", mesh.points, ev.values, nf)
    np.testing.assert_allclose(q, np.abs(qx), rtol=1e-6, atol=1e-6 * np.abs(qx).max())
    c = templates.create("tensile")
    c.problem.sweep = None
    rs = physics.of(c).solver(c, log=lambda m: None).run([])
    from solidocean.physics.solid.fields import at_nodes as solid_nodes
    m = fields.evaluate_nodal("mises", rs.meshes[0].points, c.evaluator().values, solid_nodes(rs.meshes[0], rs.U[0], rs.cels[0]))
    np.testing.assert_allclose(m, 130.13e9 * 1e-3, rtol=2e-4)       # uniaxial: E(100) x strain everywhere


def test_section_plot_settings_roundtrip_and_validation(tmp_path):
    p = templates.create("heat_bar")
    pl = ResultPlot("Section", "", kind="section", plane="yz", origin=["L/3", "0", "0"], rotation=["0", "15", "0"],
                    resolution=80, style=PlotStyle(colormap="magma"))
    p.plots.append(pl)
    assert p.validate() == []
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.plots[-1] == pl
    pl.plane, pl.resolution, pl.quantity = "zz", 5, "nonsense(T)"
    errors = "\n".join(p.validate())
    for text in ("plane must be", "resolution", "quantity"):
        assert text in errors


def test_material_colours(tmp_path):
    p = templates.create("pillars")
    assert list(p.material_colors().values()) == ["#9adcd2", "#e9d8a6"]     # automatic, in order
    p.materials[1].color = "#1b9aaa"
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.materials[1].color == "#1b9aaa" and q.material_colors()["Al"] == "#1b9aaa"
    assert "color" not in q.materials[0].to_dict()                          # automatic: not written
    p.materials[0] = Material("Si", dict(p.materials[0].props), color="blue")
    assert any("color must be" in e for e in p.validate())


def test_plot_type_labels():
    assert plot_label("field", "Temperature field") == "3D: Temperature field"
    assert plot_label("section", "Cut") == "2D: Cut"
    assert [plot_label(k, "p").split(":")[0] for k in ("dispersion", "quantity", "time", "line", "colored")] == \
        ["XY", "XY", "XY", "XY", "XYz"]
