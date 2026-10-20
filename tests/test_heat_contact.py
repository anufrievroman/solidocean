"""Thermal contact at interfaces: boundary resistances and thin resistive layers against the 1D series
solution, temperature-dependent R, layer heat capacity, connectivity at junctions, input checks."""
import numpy as np
import pytest
from scipy.optimize import brentq

from solidocean.physics.heat.model import HeatCondition, HeatInterface, HeatProblem
from solidocean.physics.heat.solver import HeatSolver
from solidocean.project import Feature, Material, Project

from test_heat import CP, K1, K2, RHO, bar, hot_cold

L = 1e-6


def run(p, quantities=()):
    return HeatSolver(p, log=lambda s: None).run(list(quantities))


def series(R):
    """Hot end 400 K at x = -L/2 (A, K1), cold end 300 K at L/2 (B, K2), interface at x = 0."""
    q = 100 / (L / 2 / K1 + R + L / 2 / K2)
    return q, 400 - q * L / 2 / K1, 300 + q * L / 2 / K2


def side_values(res, j=0):
    """Nodal temperatures (time step j) and x of the nodes of A and of B."""
    mesh, T = res.meshes[0], res.T[0][j]
    out = []
    for d in (0, 1):
        nodes = np.unique(mesh.cells[mesh.cell_domain == d])
        out.append((mesh.points[nodes, 0], T[nodes]))
    return out


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("between", [["A", "B"], ["B"], []])
def test_boundary_resistance_in_series(order, between):
    R = 5e-9
    p = bar(hot_cold(), two_materials=True, order=order)
    p.heat.interfaces = [HeatInterface("resistance", between, f"{R}[m^2*K/W]")]
    res = run(p, ["mean(qx)"])
    q, Ta, Tb = series(R)
    assert Ta - Tb == pytest.approx(q * R) and q * R > 10              # a jump of ~ 28 K
    assert res.quantities["mean(qx)"][0, 0] == pytest.approx(q, rel=1e-10)
    (xa, T_a), (xb, T_b) = side_values(res)
    np.testing.assert_allclose(T_a, 400 - q * (xa + L / 2) / K1, atol=1e-8)
    np.testing.assert_allclose(T_b, 300 + q * (L / 2 - xb) / K2, atol=1e-8)


def layer_material(**k):
    props = {"model": "isotropic", "rho": "3000[kg/m^3]", "E": "100[GPa]", "nu": "0.3", "cp": "400[J/(kg*K)]"}
    props.update({key: f"{v}[W/(m*K)]" for key, v in k.items()})
    return Material("Film", props)


@pytest.mark.parametrize("k", [{"k": 2.0}, {"kxx": 2.0, "kyy": 500.0, "kzz": 500.0}])
def test_thin_layer_resists_with_its_normal_conductivity(k):
    """A 10 nm layer with k = 2 W/(m K) across it (x): R = d / k."""
    p = bar(hot_cold(), two_materials=True)
    p.materials.append(layer_material(**k))
    p.heat.interfaces = [HeatInterface("layer", ["A", "B"], material="Film", thickness="10[nm]")]
    assert not p.validate()
    res = run(p, ["mean(qx)"])
    assert res.quantities["mean(qx)"][0, 0] == pytest.approx(series(10e-9 / 2.0)[0], rel=1e-10)


def test_layer_heat_capacity():
    """The capacity matrix holds the domains' rho cp V plus the layer's rho cp d A."""
    p = bar(hot_cold(), two_materials=True)
    p.materials.append(layer_material(k=2.0))
    p.heat.interfaces = [HeatInterface("layer", [], material="Film", thickness="10[nm]")]
    m = HeatSolver(p, log=lambda s: None).model(p.evaluator())
    w = 0.2e-6
    assert m.C.sum() == pytest.approx(RHO * CP * L * w * w + 3000 * 400 * 10e-9 * w * w, rel=1e-10)


@pytest.mark.parametrize("k", ["2[W/(m*K)]", "2[W/(m*K)] * T / 300[K]"])
def test_transient_energy_with_a_layer(k):
    """Heating through x = -L/2 of an insulated bar with a layer: the stored energy, including the layer's,
    equals the heat put in (linear and nonlinear time stepping)."""
    q0, t_end, w = 1e9, 20e-9, 0.2e-6
    p = bar([HeatCondition("flux", ["xmin"], f"{q0}[W/m^2]")], two_materials=True,
            problem=HeatProblem(type="transient", t_end=f"{t_end}[s]", steps=20, tol=1e-10))
    p.materials.append(layer_material(k=2.0))
    p.materials[-1].props["k"] = k
    p.heat.interfaces = [HeatInterface("layer", ["A", "B"], material="Film", thickness="10[nm]")]
    s = HeatSolver(p, log=lambda s: None)
    res = s.run()
    C = s.model(p.evaluator()).C
    stored = C @ (res.T[0][-1] - 300) @ np.ones(len(res.T[0][-1]))
    assert stored == pytest.approx(q0 * w * w * t_end, rel=1e-8)
    (_, T_a), (_, T_b) = side_values(res, -1)
    assert T_a.min() - T_b.max() > 1                                 # the layer holds the heat back


def test_resistance_depending_on_temperature():
    """R = R0 T / 300 K at the mean temperature of both sides: q R(Tm) = Ta - Tb, solved for q."""
    R0 = 5e-9
    p = bar(hot_cold(), two_materials=True)
    p.heat.problem = HeatProblem(tol=1e-10)
    p.heat.interfaces = [HeatInterface("resistance", ["A", "B"], f"{R0}[m^2*K/W] * T / 300[K]")]
    res = run(p, ["mean(qx)"])

    def gap(q):
        Ta, Tb = 400 - q * L / 2 / K1, 300 + q * L / 2 / K2
        return q * R0 * (Ta + Tb) / 2 / 300 - (Ta - Tb)
    q = brentq(gap, 1e8, 1e11)
    assert abs(q - series(R0)[0]) > 1e-3 * q                       # differs from the constant R0
    assert res.quantities["mean(qx)"][0, 0] == pytest.approx(q, rel=1e-8)


def junction(R=None):
    """Blocks A and B side by side on a plate C; the A | B interface meets C along a line."""
    m = [Material(n, {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                      "k": f"{k}[W/(m*K)]", "cp": "500[J/(kg*K)]"}) for n, k in (("A", 100), ("B", 25), ("C", 50))]
    geo = [Feature("block", "C", material="C", params={"center": ["0", "0", "-0.1[um]"], "size": ["1[um]", "0.4[um]", "0.2[um]"]}),
           Feature("block", "A", material="A", params={"center": ["-0.25[um]", "0", "0.1[um]"], "size": ["0.5[um]", "0.4[um]", "0.2[um]"]}),
           Feature("block", "B", material="B", params={"center": ["0.25[um]", "0", "0.1[um]"], "size": ["0.5[um]", "0.4[um]", "0.2[um]"]})]
    from solidocean.physics.heat.model import HeatPhysics
    from solidocean.project import MeshSettings
    p = Project(name="j", physics_kind="heat", materials=m, objects=geo, mesh=MeshSettings(max_size="0.1[um]"),
                heat=HeatPhysics(initial="300[K]", conditions=[
                    HeatCondition("temperature", ["xmin"], "400[K]", on="A"),
                    HeatCondition("temperature", ["xmax"], "300[K]")]))
    if R is not None:
        p.heat.interfaces = [HeatInterface("resistance", ["A", "B"], R)]
    return p


def test_sides_stay_connected_at_a_junction():
    """A vanishing resistance gives the solution without interface: the plate stays joined to both blocks
    along the line where the interface meets it; a large one cuts the direct path A -> B."""
    ref = run(junction())
    tiny = run(junction("1e-18[m^2*K/W]"))
    n = ref.meshes[0].n_nodes
    assert tiny.meshes[0].n_nodes > n
    np.testing.assert_allclose(tiny.T[0][0][:n], ref.T[0][0], atol=1e-6)
    big = run(junction("1e-5[m^2*K/W]"))
    assert big.T[0][0].max() == pytest.approx(400) and big.T[0][0][:n].mean() < ref.T[0][0].mean()


def test_interface_input_round_trip_and_checks(tmp_path):
    p = bar(hot_cold(), two_materials=True)
    p.materials.append(layer_material(k=2.0))
    p.heat.interfaces = [HeatInterface("resistance", ["A", "B"], "2e-9[m^2*K/W]", name="Kapitza"),
                         HeatInterface("layer", ["A"], material="Film", thickness="3[nm]")]
    q = Project.load(p.save(tmp_path / "m.toml"))
    a, b = q.heat.interfaces
    assert (a.type, a.between, a.R, a.name) == ("resistance", ["A", "B"], "2e-9[m^2*K/W]", "Kapitza")
    assert (b.type, b.between, b.material, b.thickness) == ("layer", ["A"], "Film", "3[nm]")
    assert not q.validate()
    with pytest.raises(ValueError, match="overlaps another interface"):
        run(q)
    a.between, a.R, b.material, b.thickness = ["A", "Nope"], "-1[m^2*K/W]", "Nope", "0[nm]"
    errors = " ".join(q.validate())
    for text in ("unknown object 'Nope'", "R must not be negative", "material 'Nope' not defined",
                 "thickness must be positive"):
        assert text in errors
    q.heat.interfaces = [HeatInterface("resistance", ["A", "A"])]
    assert any("must differ" in e for e in q.validate())
    p.objects[1].params["center"] = ["L", "0", "0"]                 # B no longer touches A
    p.heat.interfaces = p.heat.interfaces[:1]
    with pytest.raises(ValueError, match="share no faces"):
        run(p)


def test_bypassed_items_are_kept_but_not_applied(tmp_path):
    """enabled = false on a condition, source or interface: the run is the model without it, the file keeps it,
    and an unfinished bypassed item is not validated."""
    from solidocean.physics.heat.model import HeatSource
    base = bar(hot_cold(), two_materials=True)
    full = bar(hot_cold() + [HeatCondition("flux", ["ymax"], "1e9[W/m^2]", name="extra")], two_materials=True,
               sources=[HeatSource("A", "1e18[W/m^3]", name="Q")])
    full.heat.interfaces = [HeatInterface("resistance", ["A", "B"], "1e-8[m^2*K/W]")]
    for item in (full.heat.conditions[-1], full.heat.sources[0], full.heat.interfaces[0]):
        item.enabled = False
    full.heat.conditions[-1].on = "nowhere"        # unfinished, but bypassed: not an error
    assert full.validate() == []
    q = Project.load(full.save(tmp_path / "m.toml"))
    assert q.heat == full.heat
    assert "enabled = false" in (tmp_path / "m.toml").read_text()
    np.testing.assert_allclose(run(q, ["mean(T)"]).T[0][0], run(base, ["mean(T)"]).T[0][0], rtol=1e-12)
    q.heat.interfaces[0].enabled = True               # enabled again: it applies
    assert run(q, ["mean(T)"]).meshes[0].n_nodes > run(base, ["mean(T)"]).meshes[0].n_nodes


def test_zero_resistance_is_a_perfect_contact():
    """R = 0 (the end of a sweep down to perfect contact): the interface is not cut, so the solution is the
    bonded bar's exactly; the next value of the sweep cuts it again on the same mesh."""
    from solidocean.project import Problem, VariableSweep
    p = bar(hot_cold(), two_materials=True)
    p.variables["R"] = "0[m^2*K/W]"
    p.heat.interfaces = [HeatInterface("resistance", ["A", "B"], "R")]
    p.problem = Problem(sweep=VariableSweep("R", "[0, 5e-9, 0] * 1[m^2*K/W]"))
    assert p.validate() == []
    res = run(p, ["mean(qx)"])
    bonded = run(bar(hot_cold(), two_materials=True))
    for i in (0, 2):
        assert res.meshes[i].n_nodes == bonded.meshes[0].n_nodes
        np.testing.assert_allclose(res.T[i][0], bonded.T[0][0], rtol=1e-12)
    assert res.meshes[1].n_nodes > bonded.meshes[0].n_nodes
    assert res.quantities["mean(qx)"][:, 0] == pytest.approx([series(0)[0], series(5e-9)[0], series(0)[0]],
                                                             rel=1e-10)
    p.problem = Problem()
    p.heat.interfaces.append(HeatInterface("resistance", [], "1e-9[m^2*K/W]"))
    with pytest.raises(ValueError, match="overlaps another interface"):   # perfect, but still checked
        run(p)
    p.heat.interfaces[0].R = "0[m^2*K/W] * T/1[K]"         # zero only as a constant
    assert any("cannot be 0" in e for e in p.validate())
