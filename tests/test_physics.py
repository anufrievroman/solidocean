"""End-to-end checks against analytic results."""
import numpy as np
import pytest

from solidocean.project import Feature, Material, MeshSettings, Project
from solidocean.lattice import Lattice
from solidocean.physics.solid.model import SolidCondition, SolidPhysics, SolidStudy
from solidocean.physics.solid.solver import Solver

RHO, C11, C44 = 2329.0, 165.7e9, 79.6e9


def cube(order=2, periodic=None, k=("0", "0", "0"), fixed=None, **solid):
    """A Si cube of side a; periodic: lattice periods, k: Bloch wave vector, fixed: clamped faces,
    solid: further SolidPhysics settings (symmetry, antisymmetry)."""
    conditions = [SolidCondition("fixed", list(fixed))] if fixed else []
    return Project(
        variables={"a": "1[um]", "kx": "0.1*pi/a"},
        materials=[Material("Si", {"library": "Si"})],
        geometry=[Feature("block", "cell", material="Si",
                          params={"center": ["a/2", "a/2", "a/2"], "size": ["a", "a", "a"]})],
        mesh=MeshSettings(max_size="a/3", order=order),
        lattice=Lattice(periodic=dict(periodic or {})),
        solid=SolidPhysics(conditions=conditions, study=SolidStudy(k=list(k)), **solid),
    )


def test_bulk_silicon_sound_velocities():
    p = cube(periodic={"x": "a", "y": "a", "z": "a"}, k=["kx", "0", "0"])
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    f = s.solve_point(ev, s.k_of(ev), n=4)
    v = 2 * np.pi * f[:3] / ev("kx")
    np.testing.assert_allclose(v, [np.sqrt(C44 / RHO)] * 2 + [np.sqrt(C11 / RHO)], rtol=1e-5)


@pytest.mark.parametrize("order", [1, 2])
def test_free_body_has_six_rigid_modes(order):
    p = cube(order=order)
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    f = s.solve_point(ev, np.zeros(3), n=8)
    assert np.all(f[:6] < 1e-4 * f[6])
    assert f[6] > 1e9


def test_clamped_face_removes_rigid_modes():
    p = cube(fixed=["zmin"])
    s = Solver(p, log=lambda m: None)
    f = s.solve_point(p.evaluator(), np.zeros(3), n=3)
    assert f[0] > 1e8


def test_mode_shapes_are_bloch_periodic():
    p = cube(periodic={"x": "a"}, k=["kx", "0", "0"])
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    f, u = s.solve_point(ev, s.k_of(ev), n=2, modes=True)
    pts = s._model.mesh.points
    a = ev("a")
    hi = np.nonzero(np.isclose(pts[:, 0], a))[0]
    for i in hi[:5]:
        j = np.argmin(np.linalg.norm(pts - (pts[i] - [a, 0, 0]), axis=1))
        np.testing.assert_allclose(u[0, i], u[0, j] * np.exp(1j * ev("kx") * a), atol=1e-12)


def test_rotated_silicon_stiffness_and_velocity():
    from solidocean.materials import evaluate
    from solidocean.expressions import Evaluator
    C12 = 63.9e9
    C, _ = evaluate({"library": "Si", "rotation_deg": ["0", "0", "45"]}, Evaluator({}))
    np.testing.assert_allclose(C[0, 0], (C11 + C12) / 2 + C44, rtol=1e-12)
    np.testing.assert_allclose(C[0, 1], (C11 + C12) / 2 - C44, rtol=1e-12)
    np.testing.assert_allclose(C[5, 5], (C11 - C12) / 2, rtol=1e-12)
    np.testing.assert_allclose(C[3, 3], C44, rtol=1e-12)
    C90, _ = evaluate({"library": "Si", "rotation_deg": ["90", "0", "90"]}, Evaluator({}))
    C0, _ = evaluate({"library": "Si"}, Evaluator({}))
    np.testing.assert_allclose(C90, C0, atol=1e-3)  # cubic symmetry
    # longitudinal velocity along x of the rotated crystal = along Si [110]
    p = cube(periodic={"x": "a", "y": "a", "z": "a"}, k=["kx", "0", "0"])
    p.materials[0].props["rotation_deg"] = ["0", "0", "45"]
    s = Solver(p, log=lambda m: None)
    ev = p.evaluator()
    f = s.solve_point(ev, s.k_of(ev), n=4)
    vL = 2 * np.pi * f[2] / ev("kx")
    np.testing.assert_allclose(vL, np.sqrt(((C11 + C12) / 2 + C44) / RHO), rtol=1e-5)


def test_isotropic_is_rotation_invariant():
    from solidocean.materials import evaluate
    from solidocean.expressions import Evaluator
    props = {"model": "isotropic", "rho": "1", "E": "100[GPa]", "nu": "0.3"}
    C0, _ = evaluate(props, Evaluator({}))
    C1, _ = evaluate({**props, "rotation_deg": ["17", "-33", "71"]}, Evaluator({}))
    np.testing.assert_allclose(C1, C0, rtol=1e-12, atol=1e-3)


@pytest.mark.parametrize("kind, n_zero", [("symmetry", 3), ("antisymmetry", 3)])
def test_symmetry_planes_remove_rigid_modes(kind, n_zero):
    # plane z = 0. symmetry keeps x/y translation + z rotation;
    # antisymmetry keeps z translation + x/y rotations (complementary: 3 + 3 = 6)
    p = cube(**{kind: ["zmin"]})
    s = Solver(p, log=lambda m: None)
    f = s.solve_point(p.evaluator(), np.zeros(3), n=n_zero + 1)
    assert np.all(f[:n_zero] < 1e-4 * f[n_zero])
    assert f[n_zero] > 1e8


def test_half_slab_with_symmetry_reproduces_full_slab():
    def slab(z_center, height, **physics):
        p = cube(periodic={"x": "a", "y": "a"}, k=["kx", "0", "0"], **physics)
        p.geometry[0].params.update(center=["a/2", "a/2", z_center], size=["a", "a", height])
        p.mesh.max_size = "a/4"
        return p

    def freqs(p, n):
        s = Solver(p, log=lambda m: None)
        ev = p.evaluator()
        return s.solve_point(ev, s.k_of(ev), n=n)

    full = freqs(slab("0", "a"), 8)
    half = np.sort(np.r_[freqs(slab("a/4", "a/2", symmetry=["zmin"]), 8),
                         freqs(slab("a/4", "a/2", antisymmetry=["zmin"]), 8)])[:8]
    np.testing.assert_allclose(half, full, rtol=5e-3)


def test_periodic_face_cannot_be_constrained():
    p = cube(periodic={"x": "a"}, symmetry=["xmin"])
    assert any("periodic" in e for e in p.validate())
