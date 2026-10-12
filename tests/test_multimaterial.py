"""Several materials in one model: conforming interfaces and correct coupling."""
import numpy as np

from solidocean.project import Feature, Material, MeshSettings, Project
from solidocean.lattice import Lattice
from solidocean.physics.solid.model import SolidPhysics, SolidStudy
from solidocean.physics.solid.solver import Solver

A = {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3"}
B = {"model": "isotropic", "rho": "5000[kg/m^3]", "E": "200[GPa]", "nu": "0.25"}


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def laminate():
    """Bulk laminate: two layers of equal thickness stacked along x, periodic in x, y, z."""
    return Project(
        variables={"a": "1[um]", "kx": "0.02*2*pi/a"},
        materials=[Material("A", dict(A)), Material("B", dict(B))],
        geometry=[Feature("block", "layerA", material="A",
                          params={"center": ["-a/4", "0", "0"], "size": ["a/2", "a", "a"]}),
                  Feature("block", "layerB", material="B",
                          params={"center": ["a/4", "0", "0"], "size": ["a/2", "a", "a"]})],
        mesh=MeshSettings(max_size="a/4"),
        lattice=Lattice(periodic={"x": "a", "y": "a", "z": "a"}),
        solid=SolidPhysics(study=SolidStudy(k=["kx", "0", "0"])),
    )


def test_layers_share_interface_nodes():
    p = laminate()
    mesh = Solver(p, log=lambda m: None).model(p.evaluator()).mesh
    assert mesh.material_names == ["A", "B"]
    # every interface node exists once and is used by elements of both materials
    key = np.round(mesh.points / 1e-12).astype(np.int64)
    assert len(np.unique(key, axis=0)) == mesh.n_nodes
    on_a = np.unique(mesh.cells[mesh.cell_domain == 0])
    on_b = np.unique(mesh.cells[mesh.cell_domain == 1])
    shared = np.intersect1d(on_a, on_b)
    assert len(shared) > 0 and np.allclose(mesh.points[shared, 0], 0, atol=1e-15)


def test_laminate_effective_velocities():
    """Long waves across the layers: series (Reuss) average of the moduli, mean density."""
    p = laminate()
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    (l1, m1), (l2, m2) = lame(100e9, 0.3), lame(200e9, 0.25)
    rho = (2000 + 5000) / 2
    M = 1 / (0.5 / (l1 + 2 * m1) + 0.5 / (l2 + 2 * m2))   # P-wave modulus across layers
    G = 1 / (0.5 / m1 + 0.5 / m2)                         # shear modulus across layers
    f = s.solve_point(ev, s.k_of(ev), n=4)
    v = 2 * np.pi * f[:3] / ev("kx")
    np.testing.assert_allclose(v, [np.sqrt(G / rho)] * 2 + [np.sqrt(M / rho)], rtol=2e-3)


def test_aluminium_in_library_and_pillar_template():
    from solidocean import templates
    from solidocean.materials import LIBRARY
    assert "Al" in LIBRARY
    p = templates.create("pillars")
    assert p.validate() == []
    p.mesh.max_size = "a/4"
    s = Solver(p, log=lambda m: None)
    mesh = s.model(p.evaluator()).mesh
    assert set(mesh.material_names) == {"Si", "Al"}
    on_si = np.unique(mesh.cells[mesh.cell_domain == mesh.material_names.index("Si")])
    on_al = np.unique(mesh.cells[mesh.cell_domain == mesh.material_names.index("Al")])
    assert len(np.intersect1d(on_si, on_al)) > 0        # the pillar is bonded to the slab
    f = s.solve_point(p.evaluator(), np.zeros(3), n=6)
    assert np.all(f[:3] < 1e-3 * f[3])                  # 3 rigid modes only: one connected body


def test_library_materials_are_physical():
    """Every library material: positive-definite stiffness and sound velocities in the known range."""
    import pytest
    from solidocean.expressions import Evaluator
    from solidocean.materials import LIBRARY, evaluate
    known_vL = {"Si": 8433, "Diamond": 17496, "Ge": 4913, "3C-SiC": 11023, "Al": 6420, "Au": 3240}  # m/s, [100] / bulk
    for name in LIBRARY:
        C, rho = evaluate({"library": name}, Evaluator({}))
        assert np.all(np.linalg.eigvalsh(C) > 0), name
        vL = np.sqrt(C[0, 0] / rho)
        if name in known_vL:
            assert vL == pytest.approx(known_vL[name], rel=0.01), name
    C, rho = evaluate({"library": "4H-SiC"}, Evaluator({}))
    assert np.sqrt(C[2, 2] / rho) == pytest.approx(13054, rel=0.01)   # longitudinal along the c-axis


def test_material_sweep_reuses_mesh():
    """Elastic sweep over a stiffness variable: one mesh, frequencies scale as sqrt(E)."""
    from solidocean.project import VariableSweep
    p = laminate()
    p.variables["EA"] = "100[GPa]"
    p.materials = [Material("A", {**A, "E": "EA"}), Material("B", {**A, "E": "EA"})]
    p.study.sweep = VariableSweep("EA", "[100, 400] * 1[GPa]")
    logs = []
    r = Solver(p, log=logs.append).run()
    assert sum(m.startswith("Mesh:") for m in logs) == 1
    np.testing.assert_allclose(r.freqs[1][:3], 2 * r.freqs[0][:3], rtol=1e-9)
