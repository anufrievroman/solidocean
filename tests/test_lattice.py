"""Non-orthogonal (hexagonal) lattices: reciprocal vectors, Bloch conditions, band folding."""
import numpy as np
import pytest

from solidocean import lattice as lat
from solidocean.lattice import Lattice
from solidocean.expressions import Evaluator
from solidocean.project import Feature, Material, MeshSettings, Project
from solidocean.physics.solid.solver import Solver

ISO = {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3"}
HEX = [["a", "0", "0"], ["a/2", "sqrt(3)*a/2", "0"]]


def test_reciprocal_and_hexagonal_path():
    a1, a2 = np.array([1.0, 0, 0]), np.array([0.5, np.sqrt(3) / 2, 0])
    b1, b2, b3 = lat.reciprocal([a1, a2, None])
    assert b3 is None
    np.testing.assert_allclose([[b1 @ a1, b1 @ a2], [b2 @ a1, b2 @ a2]], 2 * np.pi * np.eye(2), atol=1e-12)
    points, path = lat.KPATHS["hexagonal"]
    K = points["K"][0] * b1 + points["K"][1] * b2
    M = points["M"][0] * b1 + points["M"][1] * b2
    assert np.linalg.norm(K) == pytest.approx(4 * np.pi / 3)          # BZ corner
    assert np.linalg.norm(M) == pytest.approx(2 * np.pi / np.sqrt(3))  # BZ edge centre
    bz = lat.brillouin_zone_2d(b1, b2)
    assert len(bz) == 6 and np.min(np.linalg.norm(bz - K[:2], axis=1)) < 1e-9
    phys = Lattice(vectors=HEX)
    assert lat.classify(phys, Evaluator({"a": "1"})) == "hexagonal"


def test_bulk_isotropic_in_hexagonal_cell():
    """Homogeneous solid in a hexagonal prism cell: sound velocities along oblique k directions."""
    p = Project(
        variables={"a": "1[um]", "c": "1[um]"},
        materials=[Material("iso", dict(ISO))],
        objects=[Feature("cell", "cell", material="iso", params={"center": ["0", "0", "0"], "height": "c"})],
        mesh=MeshSettings(max_size="a/3"),
        lattice=Lattice(vectors=HEX + [["0", "0", "c"]]),
    )
    assert p.validate() == []
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    E, nu, rho = 100e9, 0.3, 2000.0
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    vT, vL = np.sqrt(mu / rho), np.sqrt((lam + 2 * mu) / rho)
    b = lat.reciprocal(lat.slots(p.lattice, ev))
    for name in ("K", "M"):
        red = lat.KPATHS["hexagonal"][0][name]
        k = 0.05 * (red[0] * b[0] + red[1] * b[1])                       # small k along G-K / G-M
        f = s.solve_point(ev, k, n=4)
        v = 2 * np.pi * f[:3] / np.linalg.norm(k)
        np.testing.assert_allclose(v, [vT, vT, vL], rtol=2e-4)


def hole_slab(lattice_vectors, holes, cell):
    params = {"a": "1[um]", "t": "0.3[um]", "r": "0.18*a"}
    geometry = [cell]
    for n, (x, y) in enumerate(holes):
        geometry.append(Feature("cylinder", f"hole{n}", op="subtract",
                                params={"center": [x, y, "-t/2"], "radius": "r", "height": "t"}))
    return Project(variables=params, materials=[Material("iso", dict(ISO))], objects=geometry,
                   mesh=MeshSettings(max_size="a/6"), lattice=Lattice(vectors=lattice_vectors))


def test_hexagonal_cell_equals_folded_rectangular_supercell():
    """A hexagonal hole lattice computed in its primitive (rhombic) cell and in an a x sqrt(3)a
    rectangular supercell (2 holes) must give the same bands, folded by the supercell."""
    hexa = hole_slab(HEX, [("0", "0")],
                     Feature("cell", "cell", material="iso", params={"center": ["0", "0", "0"], "height": "t"}))
    rect = hole_slab([["a", "0", "0"], ["0", "sqrt(3)*a", "0"]],
                     [("-a/4", "-sqrt(3)*a/4"), ("a/4", "sqrt(3)*a/4")],
                     Feature("block", "slab", material="iso",
                             params={"center": ["0", "0", "0"], "size": ["a", "sqrt(3)*a", "t"]}))
    assert hexa.validate() == [] and rect.validate() == []
    ev = hexa.evaluator()
    a = ev("a")
    B2 = np.array([0, 2 * np.pi / (np.sqrt(3) * a), 0])  # supercell reciprocal vector missing in hex
    sh, sr = Solver(hexa, log=lambda m: None), Solver(rect, log=lambda m: None)
    n = 8
    for k in (np.array([0.3, 0.1, 0]) * 2 * np.pi / a, np.array([0.1, 0.25, 0]) * 2 * np.pi / a):
        folded = np.sort(np.r_[sh.solve_point(ev, k, n=n), sh.solve_point(ev, k + B2, n=n)])[:n]
        supercell = sr.solve_point(rect.evaluator(), k, n=n)
        np.testing.assert_allclose(supercell, folded, rtol=5e-3)
