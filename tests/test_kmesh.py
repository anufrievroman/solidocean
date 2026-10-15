"""Wavevector meshes of the Brillouin zone (coherent heat conduction): the mesh itself and the spectrum it
computes, checked against the exact Rayleigh-Lamb solution of a free plate."""
import sys
from pathlib import Path

import numpy as np
import pytest

from solidocean import lattice as lat
from solidocean import physics
from solidocean.expressions import Evaluator
from solidocean.lattice import Lattice
from solidocean.physics.solid.model import KMesh, SolidPhysics, SolidProblem
from solidocean.project import Feature, Material, MeshSettings, Project

SQ = (np.array([2 * np.pi, 0.0, 0.0]), np.array([0.0, 2 * np.pi, 0.0]))
HEX_CELL = [["a", "0", "0"], ["a/2", "sqrt(3)*a/2", "0"]]
NU = 0.3            # the plate of validation/lamb_plate.py
RHO, E = 2700.0, 70e9


def hex_reciprocal():
    return [b for b in lat.reciprocal(lat.slots(Lattice(vectors=HEX_CELL), Evaluator({"a": "1"}))) if b is not None]


def area(points, tris) -> float:
    """Total area of a triangulation, in the (reduced) coordinates of its points."""
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    u, v = b - a, c - a
    return float(0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum())


@pytest.mark.parametrize("region, fraction", [("irreducible", 1 / 8), ("full", 1.0)])
def test_square_zone_mesh_covers_its_region(region, fraction):
    """A square lattice: the wedge is an eighth of the zone, which itself has unit area in reduced
    coordinates. The mesh is conforming (no point repeated) and starts at Gamma."""
    points, tris = lat.zone_mesh(*SQ, 8, region, "square")
    assert area(points, tris) == pytest.approx(fraction)
    assert len(points) == len(np.unique(np.round(points, 9), axis=0))
    np.testing.assert_allclose(points[0], [0, 0], atol=1e-15)
    assert tris.max() == len(points) - 1


def test_wedge_point_count_follows_the_divisions():
    """A triangular wedge divided n times per edge: (n + 1)(n + 2) / 2 points and n**2 triangles (the
    113 points per edge of a published calculation give its ~6,400 k points)."""
    for n in (1, 3, 12, 112):
        points, tris = lat.zone_mesh(*SQ, n, "irreducible", "square")
        assert len(points) == (n + 1) * (n + 2) // 2
        assert len(tris) == n * n
    assert len(lat.zone_mesh(*SQ, 112, "irreducible", "square")[0]) == 6441


def test_hexagonal_wedge_sits_on_the_zone():
    """The corners of the hexagonal wedge are Gamma, the zone edge centre M and the zone corner K, and the
    wedge is a twelfth of the zone."""
    b1, b2 = hex_reciprocal()
    points, tris = lat.zone_mesh(b1, b2, 6, "irreducible", "hexagonal")
    assert area(points, tris) == pytest.approx(1 / 12)
    cartesian = points[:, :1] * b1 + points[:, 1:] * b2
    radius = np.linalg.norm(cartesian, axis=1)
    corner = lat.brillouin_zone_2d(b1, b2)
    assert radius.max() == pytest.approx(np.linalg.norm(corner, axis=1).max())          # K
    assert np.linalg.norm(cartesian[points[:, 1] == 0], axis=1).max() == pytest.approx(np.pi / np.sqrt(3) * 2)  # M
    assert area(*lat.zone_mesh(b1, b2, 6, "full", "hexagonal")) == pytest.approx(1.0)


def test_zone_mesh_rejects_a_lattice_without_a_wedge():
    with pytest.raises(ValueError, match="no standard irreducible wedge"):
        lat.zone_mesh(*SQ, 4, "irreducible", "oblique")


def plate(n: int = 2, a_frac: float = 0.5, h: str = "t/5", num_eigs: int = 12) -> Project:
    """Free isotropic plate with an artificial square cell, meshed over the irreducible wedge of its zone."""
    return Project(
        name="Plate over the zone", physics_kind="solid",
        variables={"t": "1[um]", "a": f"{a_frac}*t"},
        materials=[Material("M", {"model": "isotropic", "rho": f"{RHO}[kg/m^3]", "E": f"{E / 1e9}[GPa]",
                                  "nu": str(NU)})],
        objects=[Feature("block", "Plate", material="M", params={"center": ["0", "0", "0"],
                                                                 "size": ["a", "a", "t"]})],
        mesh=MeshSettings(max_size=h, order=2),
        lattice=Lattice(periodic={"x": "a", "y": "a"}),
        solid=SolidPhysics(problem=SolidProblem(type="conductance", num_eigs=num_eigs, kmesh=KMesh(n=n))))


def lamb_reference():
    """validation/lamb_plate.lamb_frequencies: the exact Rayleigh-Lamb and shear-horizontal spectrum."""
    directory = str(Path(__file__).resolve().parent.parent / "validation")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from lamb_plate import lamb_frequencies
    return lamb_frequencies


def test_plate_over_the_zone_matches_lamb_theory():
    """Every eigenfrequency computed on the wedge of a homogeneous plate is a guided mode of that plate.

    The cell is artificial, so the spectrum at k is the exact one at every |k + G| folded into the zone; the
    whole set must be accounted for. Since the plate is isotropic the exact frequencies depend on |k + G|
    alone, so this also checks that the zone is sampled consistently in every direction (Gamma-X rotated
    around the zone).
    """
    lamb_frequencies = lamb_reference()
    p = plate()
    assert p.validate() == []
    results = physics.of(p).solver(p, log=lambda m: None).run()
    assert results.is_kmesh and results.ktris.shape == (4, 3) and len(results.kred) == 6

    c_t = np.sqrt(E / (2 * (1 + NU)) / RHO)
    thickness, b = 1e-6, 2 * np.pi / (0.5e-6)
    window = 10.0                       # compared range of omega t / c_T
    compared, worst = 0, 0.0
    for j, k in enumerate(results.k[0]):
        exact = set()
        for i in range(-2, 3):
            for m in range(-2, 3):
                folded = np.linalg.norm(k + np.array([i * b, m * b, 0.0])) * thickness
                exact.update(lamb_frequencies(folded, window + 1.0, NU).tolist())
        exact = np.array(sorted(exact))
        omega = 2 * np.pi * results.freqs[0, j] * thickness / c_t
        for w in omega[(omega > 1e-6) & (omega < window)]:
            compared += 1
            worst = max(worst, float(np.min(np.abs(exact - w)) / w))
    assert compared > 50                # not a vacuous comparison
    assert worst < 0.01                 # discretisation error of t/5 quadratic elements


def test_kmesh_round_trip(tmp_path):
    """The mesh and the problem type survive the model file; a disabled mesh is kept but not run."""
    p = plate(n=5)
    p.solid.problem.kmesh.region = "full"
    q = Project.load(p.save(tmp_path / "zone.toml"))
    assert q.solid.problem.type == "conductance"
    assert q.solid.problem.kmesh == KMesh(n=5, region="full")
    assert q.solid.problem.active_kmesh is q.solid.problem.kmesh
    q.solid.problem.kmesh.enabled = False
    r = Project.load(q.save(tmp_path / "off.toml"))
    assert r.solid.problem.kmesh.enabled is False and r.solid.problem.active_kmesh is None


def test_a_mesh_belongs_to_a_coherent_problem_and_a_plane_lattice():
    """The mesh runs only for a coherent heat conduction problem, which in turn needs one."""
    p = plate()
    p.solid.problem.type = "eigenfrequency"
    assert p.solid.problem.active_kmesh is None            # an ordinary eigenfrequency run follows its k-path
    p.solid.problem.type, p.solid.problem.kmesh = "conductance", None
    assert any("needs a wavevector mesh" in e for e in p.validate())
    p = plate()
    p.lattice = Lattice(periodic={"x": "a"})
    assert any("periodic in exactly two directions" in e for e in p.validate())
    p = plate()
    p.lattice = Lattice(vectors=[["a", "0", "0"], ["a/3", "a", "0"]])
    assert any("no standard irreducible wedge" in e for e in p.validate())
    p.solid.problem.kmesh.region = "full"
    assert p.validate() == []


def test_the_mesh_has_a_panel_and_a_node_in_the_window():
    """Switching to coherent heat conduction gives the model tree a wavevector mesh node with its panel, and
    the project a band map; the mesh panel reports how many wavevectors the run will solve."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    from solidocean.gui.panels.solid import set_problem_type

    w = MainWindow()
    w.set_project(templates.create("square"))
    set_problem_type(w, "conductance")
    app.processEvents()
    assert w.project.solid.problem.active_kmesh is not None
    assert [pl.kind for pl in w.project.plots].count("kmap") == 1
    w.select_node("kmesh")
    app.processEvents()
    texts = " ".join(lbl.text() for lbl in w.settings.widget().findChildren(QLabel))
    assert "91 wavevectors" in texts          # the default 12 divisions of a triangular wedge
    w.dirty = False
    w.close()
