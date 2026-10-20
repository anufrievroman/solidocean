"""Stationary solid mechanics and boundary selections: analytic checks."""
import numpy as np
import pytest

from solidocean import selection
from solidocean.physics.solid import fields as solid_fields
from solidocean.project import Feature, Material, MeshSettings, VariableSweep, Project
from solidocean.physics.heat.model import HeatCondition, HeatPhysics
from solidocean.lattice import Lattice
from solidocean.physics.solid.model import SolidCondition, SolidPhysics, SolidProblem
from solidocean.physics.solid.solver import Solver

E, NU = 100e9, 0.25
ISO = {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.25"}
C11_SI, C12_SI = 165.7e9, 63.9e9


def bar(conditions, symmetry=("xmin", "ymin", "zmin"), material=None, size=("L", "w", "w"), max_size="w/2",
        periodic=None):
    """Block [0, L] x [0, w] x [0, w] (L = 2 um, w = 0.5 um)."""
    return Project(
        variables={"L": "2[um]", "w": "0.5[um]"},
        materials=[Material("M", dict(material or ISO))],
        objects=[Feature("block", "Bar", material="M",
                          params={"center": [f"{s}/2" for s in size], "size": list(size)})],
        mesh=MeshSettings(max_size=max_size), lattice=Lattice(periodic=dict(periodic or {})),
        solid=SolidPhysics(symmetry=list(symmetry), conditions=conditions, problem=SolidProblem(type="stationary")))


def solve(p, quantities=()):
    return Solver(p, log=lambda m: None).run(list(quantities))


def test_uniaxial_prescribed_displacement_gives_youngs_modulus_and_poisson_ratio():
    p = bar([SolidCondition("displacement", ["xmax"], displacement=["2[nm]", "", ""])])
    r = solve(p, ["mean(sxx) / mean(exx)", "-mean(eyy) / mean(exx)", "max(mises)", "mean(syy)", "mean(exx)"])
    q = {k: v[0, 0] for k, v in r.quantities.items()}
    assert q["mean(exx)"] == pytest.approx(1e-3, rel=1e-9)
    assert q["mean(sxx) / mean(exx)"] == pytest.approx(E, rel=1e-9)
    assert q["-mean(eyy) / mean(exx)"] == pytest.approx(NU, rel=1e-9)
    assert q["max(mises)"] == pytest.approx(E * 1e-3, rel=1e-6)          # uniaxial: von Mises = |sxx|
    assert abs(q["mean(syy)"]) < 1e-6 * E * 1e-3
    vm = solid_fields.nodal_mises(r.meshes[0], r.U[0], r.cels[0])
    np.testing.assert_allclose(vm, E * 1e-3, rtol=1e-6)


@pytest.mark.parametrize("cond", [
    SolidCondition("load", ["xmax"], load_type="total", load=["0.5[uN]", "0", "0"]),
    SolidCondition("load", ["xmax"], load_type="area", load=["2[MPa]", "0", "0"]),
    SolidCondition("load", ["xmax"], load_type="pressure", pressure="-2[MPa]"),   # suction pulls outwards
])
def test_load_types_give_the_same_uniform_stress(cond):
    r = solve(bar([cond]), ["mean(sxx)", "max(ux)"])
    assert r.quantities["mean(sxx)"][0, 0] == pytest.approx(2e6, rel=1e-9)     # 0.5 uN / (0.5 um)^2
    assert r.quantities["max(ux)"][0, 0] == pytest.approx(2e6 / E * 2e-6, rel=1e-9)


@pytest.mark.parametrize("faces", [["xmax", "ymax", "zmax"], ["all"]])
def test_pressure_pushes_on_every_face_of_a_cube(faces):
    """Hydrostatic pressure on all free faces of a quarter... eighth cube held by three symmetry planes:
    uniform stress -p in every direction. "all" leaves the symmetry planes out."""
    p = bar([SolidCondition("load", faces, load_type="pressure", pressure="1[MPa]")],
            size=("w", "w", "w"))
    r = solve(p, ["mean(sxx)", "mean(syy)", "mean(szz)", "max(abs(sxy))"])
    for s in ("sxx", "syy", "szz"):
        assert r.quantities[f"mean({s})"][0, 0] == pytest.approx(-1e6, rel=1e-9)
    assert r.quantities["max(abs(sxy))"][0, 0] < 1e-6
    p.solid.conditions.append(SolidCondition("fixed", ["xmin"]))
    assert any("xmin is a symmetry plane" in e for e in p.validate())


def test_periodic_slab_under_pressure_is_in_uniaxial_strain():
    """Slab periodic in x and y, clamped at the bottom, pressure p on top: ezz = -p / C11, uz(top) = -p t / C11."""
    p = bar([SolidCondition("fixed", ["zmin"]),
             SolidCondition("load", ["zmax"], load_type="pressure", pressure="10[MPa]")],
            symmetry=(), material={"library": "Si"}, size=("w", "w", "w"), periodic={"x": "w", "y": "w"})
    r = solve(p, ["min(uz)", "mean(szz)", "mean(sxx)"])
    assert r.quantities["min(uz)"][0, 0] == pytest.approx(-10e6 * 0.5e-6 / C11_SI, rel=1e-9)
    assert r.quantities["mean(szz)"][0, 0] == pytest.approx(-10e6, rel=1e-9)
    assert r.quantities["mean(sxx)"][0, 0] == pytest.approx(-10e6 * C12_SI / C11_SI, rel=1e-9)


def test_cantilever_matches_euler_bernoulli():
    L, w, t, F = 8e-6, 0.4e-6, 0.4e-6, 1e-9
    p = Project(variables={"L": "8[um]", "t": "0.4[um]", "F": "1[nN]"}, materials=[Material("M", ISO)],
                objects=[Feature("block", "Beam", material="M",
                                  params={"center": ["L/2", "0", "0"], "size": ["L", "t", "t"]})],
                mesh=MeshSettings(max_size="t/2"),
                solid=SolidPhysics(problem=SolidProblem(type="stationary"), conditions=[
                    SolidCondition("fixed", ["xmin"]),
                    SolidCondition("load", ["xmax"], load_type="total", load=["0", "0", "-F"])]))
    tip = -solve(p, ["min(uz)"]).quantities["min(uz)"][0, 0]
    assert tip == pytest.approx(F * L ** 3 / (3 * E * w * t ** 3 / 12), rel=0.02)


def test_box_selection_spreads_a_total_force_over_the_box_only():
    """Load on a patch of the top face (a box selection): the force is F, wherever the box cuts the mesh."""
    p = bar([SolidCondition("fixed", ["xmin"]),
             SolidCondition("load", ["zmax"], load_type="total", load=["0", "0", "-1[nN]"],
                            box={"min": ["L - 0.5[um]", "0", "0"], "max": ["L", "w", "w"]})], symmetry=())
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    m = s.model(ev)
    f = s.load_vector(m, ev)
    assert f[:, 2].sum() == pytest.approx(-1e-9, rel=1e-12)
    loaded = np.abs(f[:, 2]) > 0
    assert np.all(m.mesh.points[loaded, 0] >= 1.5e-6 - 1e-12)                # only inside the box
    assert np.all(np.isclose(m.mesh.points[loaded, 2], 0.5e-6))              # only on the top face


def test_selection_of_an_object_and_all_faces():
    """'all' faces of a pad on a slab: the pad's outer surface without its bonded bottom."""
    p = Project(variables={"w": "2[um]", "t": "0.2[um]", "a": "0.5[um]"},
                materials=[Material("A", ISO), Material("B", ISO)],
                objects=[Feature("block", "Slab", material="A", params={"center": ["0", "0", "0"],
                                                                         "size": ["w", "w", "t"]}),
                          Feature("block", "Pad", material="B", params={"center": ["0", "0", "t/2 + a/2"],
                                                                        "size": ["a", "a", "a"]})],
                mesh=MeshSettings(max_size="0.25[um]"))
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    m = s.model(ev)
    from solidocean.fem import surface

    def area(c):
        idx = selection.select(m.mesh, m.boundary, c, ev)
        _, w = surface.surface_terms(m.mesh.points, m.boundary.tris[idx], m.mesh.order, load=1.0)
        return w.sum()

    a = 0.5e-6
    assert area(SolidCondition(faces=["all"], on="Pad")) == pytest.approx(5 * a * a, rel=1e-9)
    assert area(SolidCondition(faces=["zmin"], on="Pad")) == pytest.approx(0, abs=1e-30)    # bonded: not outer
    assert area(SolidCondition(faces=["zmax"], on="Pad")) == pytest.approx(a * a, rel=1e-9)
    slab_top = SolidCondition(faces=["zmax"], on="Slab")
    assert area(slab_top) == pytest.approx(4e-12 - a * a, rel=1e-9)          # the pad covers its centre
    box = SolidCondition(faces=["all"], box={"min": ["-w", "-w", "t/2"], "max": ["w", "w", "t/2"]})
    assert area(box) == pytest.approx(4e-12 - a * a, rel=1e-9)               # a flat box: the slab's top


def test_free_body_is_rejected():
    p = bar([SolidCondition("load", ["xmax"], load_type="total", load=["1[uN]", "0", "0"])], symmetry=())
    with pytest.raises(ValueError, match="not held"):
        solve(p)


def test_constraint_on_an_object_in_an_eigenfrequency_problem():
    """Clamping the bottom face of the whole model or of the only object is the same constraint."""
    def freqs(cond):
        p = bar([cond], symmetry=())
        p.solid.problem = SolidProblem(num_eigs=3)
        return Solver(p, log=lambda m: None).solve_point(p.evaluator(), np.zeros(3))
    a = freqs(SolidCondition("fixed", ["xmin"]))
    b = freqs(SolidCondition("fixed", ["xmin"], on="Bar"))
    c = freqs(SolidCondition("displacement", ["xmin"], displacement=["1[nm]", "0", "0"]))   # held at zero
    np.testing.assert_allclose(a, b, rtol=1e-12)
    np.testing.assert_allclose(a, c, rtol=1e-12)
    assert a[0] > 1e8


def test_mode_stresses_are_available():
    p = bar([SolidCondition("fixed", ["xmin"])], symmetry=())
    p.solid.problem = SolidProblem(num_eigs=2)
    r = Solver(p, log=lambda m: None).run(["max(mises)", "mean(abs(sxx))"])
    assert np.all(np.isfinite(r.quantities["max(mises)"])) and np.all(r.quantities["max(mises)"] > 0)


def test_condition_validation():
    p = bar([SolidCondition("displacement", ["xmax"], displacement=["", "", ""]),
             SolidCondition("load", [], load_type="area"),
             SolidCondition("fixed", ["xmin"], on="Nothing"),
             SolidCondition("fixed", ["all"], box={"min": ["0", "0"], "max": ["1", "1", "1"]})])
    errors = "\n".join(p.validate())
    for text in ("no displacement component", "no faces selected", "unknown object 'Nothing'", "3 coordinates"):
        assert text in errors
    p = bar([SolidCondition("fixed", ["xmin"])], symmetry=(), periodic={"x": "L"})
    assert any("periodic" in e for e in p.validate())


def test_stationary_sweep_reuses_the_mesh_for_load_values():
    p = bar([SolidCondition("load", ["xmax"], load_type="total", load=["F", "0", "0"])])
    p.variables["F"] = "1[uN]"
    p.problem.sweep = VariableSweep("F", "[1, 2, 3] * 1[uN]")
    s = Solver(p, log=lambda m: None)
    r = s.run(["mean(sxx)"])
    assert len({id(m) for m in r.meshes.values()}) == 1
    np.testing.assert_allclose(r.quantities["mean(sxx)"][:, 0], [4e6, 8e6, 12e6], rtol=1e-9)


def test_heat_flux_into_a_box_selection():
    """Flux q into a patch (box) of the top of a bar whose bottom is held at T0: the heat flow through the bar
    is q times the patch area, so integral(qz) = -q A_patch t."""
    from solidocean.physics.heat.solver import HeatSolver
    p = Project(physics_kind="heat", variables={"w": "1[um]", "t": "0.2[um]"},
                materials=[Material("M", {**ISO, "k": "100[W/(m*K)]", "cp": "700[J/(kg*K)]"})],
                objects=[Feature("block", "Slab", material="M", params={"center": ["0", "0", "0"],
                                                                         "size": ["w", "w", "t"]})],
                mesh=MeshSettings(max_size="0.1[um]"),
                heat=HeatPhysics(conditions=[
                    HeatCondition("temperature", ["zmin"], "300[K]"),
                    HeatCondition("flux", ["zmax"], "1e6[W/m^2]",
                                  box={"min": ["-w/2", "-w/2", "-t"], "max": ["0", "0", "t"]})]))
    assert p.validate() == []
    r = HeatSolver(p, log=lambda m: None).run(["integral(qz)"])
    assert r.quantities["integral(qz)"][0, 0] == pytest.approx(-1e6 * 0.25e-12 * 0.2e-6, rel=1e-6)


def test_bypassed_conditions_are_not_applied():
    """A bypassed fixed constraint and load leave the uniform stress of the load test unchanged."""
    p = bar([SolidCondition("load", ["xmax"], load_type="area", load=["2[MPa]", "0", "0"]),
             SolidCondition("fixed", ["xmax"], enabled=False),
             SolidCondition("load", ["xmax"], load_type="area", load=["5[MPa]", "0", "0"], enabled=False)])
    assert p.validate() == []
    r = solve(p, ["mean(sxx)"])
    assert r.quantities["mean(sxx)"][0, 0] == pytest.approx(2e6, rel=1e-9)
    assert Project.from_dict(p.to_dict()).solid == p.solid
