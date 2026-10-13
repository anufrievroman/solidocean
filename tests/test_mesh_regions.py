"""Mesh regions: different element sizes per object, with a conforming mesh across their interfaces."""
import numpy as np
import pytest

from solidocean import meshing
from solidocean.fem import surface
from solidocean.project import Feature, Material, MeshRegion, MeshSettings, Project


def two_blocks(regions=(), max_size="0.5[um]"):
    """Blocks A (x < 0) and B (x > 0), 2 x 1 x 1 um each, different materials."""
    m = {n: Material(n, {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                         "k": f"{k}[W/(m*K)]", "cp": "500[J/(kg*K)]"}) for n, k in (("MA", 100), ("MB", 20))}
    return Project(materials=list(m.values()), variables={"h": "0.5[um]"},
                   geometry=[Feature("block", "A", material="MA", params={"center": ["-1[um]", "0", "0"],
                                                                          "size": ["2[um]", "1[um]", "1[um]"]}),
                             Feature("block", "B", material="MB", params={"center": ["1[um]", "0", "0"],
                                                                          "size": ["2[um]", "1[um]", "1[um]"]})],
                   mesh=MeshSettings(max_size=max_size, regions=list(regions)))


def edges(mesh, name, where=None):
    """Mean edge of the elements of an object (optionally those whose centre x passes `where`)."""
    c = mesh.points[mesh.cells[mesh.region(name)][:, :4]]
    if where is not None:
        c = c[where(c[:, :, 0].mean(axis=1))]
    return np.mean([np.linalg.norm(c[:, i] - c[:, j], axis=1).mean() for i in range(4) for j in range(i)])


def conforming(mesh) -> bool:
    """No outer boundary face lies on the interface x = 0 (inside the model)."""
    tris = surface.boundary_faces(mesh.cells, mesh.order)
    x = mesh.points[tris[:, :3], 0]
    return not np.any(np.all(np.abs(x) < 1e-12, axis=1))


def test_region_refines_one_object_and_stays_conforming():
    base = meshing.generate(two_blocks(), two_blocks().evaluator())
    p = two_blocks([MeshRegion(on="A", max_size="h/4")])
    assert p.validate_geometry() == []
    fine = meshing.generate(p, p.evaluator())
    assert edges(fine, "A") < 0.5 * edges(base, "A")                 # A refined
    far = edges(fine, "B", lambda xc: xc > 1e-6)                    # B's far half: as without the region
    assert far == pytest.approx(edges(base, "B"), rel=0.2)
    assert edges(fine, "B", lambda xc: xc < 0.5e-6) < 0.6 * far        # graded down towards A
    assert conforming(base) and conforming(fine)


def test_later_region_wins_and_materials_select():
    p = two_blocks([MeshRegion(on="MA", max_size="h/4"), MeshRegion(on="A", max_size="h")])
    m = meshing.generate(p, p.evaluator())
    ref = meshing.generate(two_blocks(), two_blocks().evaluator())
    assert edges(m, "A") == pytest.approx(edges(ref, "A"), rel=0.1)  # the second region (= default) wins


def test_heat_across_a_refined_interface_is_exact():
    """Series conduction through A and B with A refined: the linear profiles are still exact (P2)."""
    from solidocean.physics.heat.model import HeatCondition, HeatPhysics
    from solidocean.physics.heat.solver import HeatSolver
    p = two_blocks([MeshRegion(on="A", max_size="h/4")])
    p.physics_kind, p.plots = "heat", []
    p.heat = HeatPhysics(initial="300[K]", conditions=[HeatCondition("temperature", ["xmin"], "400[K]"),
                                                       HeatCondition("temperature", ["xmax"], "300[K]")])
    res = HeatSolver(p, log=lambda s: None).run(["mean(qx)"])
    q = 100 / (2e-6 / 100 + 2e-6 / 20)
    assert res.quantities["mean(qx)"][0, 0] == pytest.approx(q, rel=1e-9)


def test_mesh_regions_round_trip_and_checks(tmp_path):
    p = two_blocks([MeshRegion(on="A", max_size="h/4", name="Fine")])
    q = Project.load(p.save(tmp_path / "m.toml"))
    r, = q.mesh.regions
    assert (r.on, r.max_size, r.name) == ("A", "h/4", "Fine")
    assert 'on = "A"' in (tmp_path / "m.toml").read_text(encoding="utf-8")
    q.mesh.regions[0].on = "Nope"
    q.mesh.regions[0].max_size = "-1[um]"
    errors = " ".join(q.validate_geometry())
    assert "unknown object 'Nope'" in errors and "max_size must be positive" in errors
    assert "regions" not in two_blocks().mesh.to_dict()                # files without regions are unchanged
