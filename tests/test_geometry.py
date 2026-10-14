import numpy as np
import pytest

from solidocean import meshing
from solidocean.project import Feature, Material, Project


def slab(*extra):
    return Project(
        variables={"a": "1[um]"},
        materials=[Material("Si", {"library": "Si"})],
        objects=[Feature("block", "slab", material="Si", params={"center": ["0", "0", "0"], "size": ["a", "a", "a"]}),
                  *extra],
    )


def test_block_is_centered():
    pre = meshing.preview_geometry(p := slab(), p.evaluator())
    np.testing.assert_allclose(pre.points.min(axis=0), [-5e-7] * 3, atol=1e-12)
    np.testing.assert_allclose(pre.points.max(axis=0), [5e-7] * 3, atol=1e-12)


def test_bypassed_feature_is_ignored_and_roundtrips(tmp_path):
    hole = Feature("cylinder", "hole", op="subtract", enabled=False,
                   params={"center": ["0", "0", "-a/2"], "radius": "0.3*a", "height": "a"})
    p = slab(hole)
    from solidocean.fields import Quadrature
    a = p.evaluator()("a")
    m1 = meshing.generate(p, p.evaluator())
    p.objects[1].enabled = True
    m2 = meshing.generate(p, p.evaluator())
    v1, v2 = Quadrature.of(m1).volume, Quadrature.of(m2).volume     # the bypassed hole removes nothing
    assert v1 == pytest.approx(a ** 3, rel=1e-9)
    assert v1 - v2 == pytest.approx(np.pi * (0.3 * a) ** 2 * a, rel=0.03)    # the hole goes through
    p.objects[1].enabled = False
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.objects[1].enabled is False and q.objects[0].enabled is True


def test_tiny_feature_is_rejected_not_hung():
    hole = Feature("cylinder", "hole", op="subtract",
                   params={"center": ["0", "0", "-a/2"], "radius": "a*1e-9", "height": "a"})
    p = slab(hole)
    with pytest.raises(ValueError, match="too small"):
        meshing.preview_geometry(p, p.evaluator())


def test_default_features_scale_with_model():
    from solidocean.gui.panels.model import default_feature
    p = slab()
    ev = p.evaluator()
    for t in ("block", "cylinder", "cone", "sphere", "prism"):
        f = default_feature(p, t)
        for key in ("radius", "height"):
            if key in f.params:
                assert 1e-8 < ev(f.params[key]) <= 1.0001e-6
        f.op = "subtract"
        q = slab(f)
        meshing.preview_geometry(q, q.evaluator())  # builds without error


def _volume_box(p):
    from solidocean.fields import Quadrature
    m = meshing.generate(p, p.evaluator())
    lo, hi = m.bounds()
    return Quadrature.of(m).volume, lo, hi


def test_triangular_block():
    tri = Feature("triangle", "Triangle", material="Si",
                  params={"center": ["0", "0", "0"], "base": "2*a", "height": "a", "thickness": "a/2"})
    p = slab()
    p.objects = [tri]
    v, lo, hi = _volume_box(p)
    assert v == pytest.approx(0.5 * 2e-6 * 1e-6 * 0.5e-6, rel=1e-9)
    np.testing.assert_allclose(np.r_[lo, hi], [-1e-6, -0.5e-6, -0.25e-6, 1e-6, 0.5e-6, 0.25e-6], atol=1e-15)


def test_rotation_of_features(tmp_path):
    """Rotations keep the volume and turn the shape: a 2:1 block by 90 deg about z swaps its x and y extents,
    a cylinder along z rotated 90 deg about x points along -y. Only rotated features store a rotation."""
    blk = Feature("block", "Block", material="Si",
                  params={"center": ["0", "0", "0"], "size": ["2*a", "a", "a"], "rotation": ["0", "0", "90"]})
    p = slab()
    p.objects = [blk]
    v, lo, hi = _volume_box(p)
    assert v == pytest.approx(2e-18, rel=1e-9)
    np.testing.assert_allclose(hi - lo, [1e-6, 2e-6, 1e-6], rtol=1e-9)
    cyl = Feature("cylinder", "Rod", material="Si", params={"center": ["0", "0", "0"], "axis": ["0", "0", "1"],
                                                          "radius": "0.2*a", "height": "a", "rotation": ["90", "0", "0"]})
    p.objects = [cyl]
    _, lo, hi = _volume_box(p)
    np.testing.assert_allclose(np.r_[lo[1], hi[1]], [-1e-6, 0.0], atol=1e-12)
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.objects[0].params["rotation"] == ["90", "0", "0"]
    assert "rotation" not in Feature("block", "B", material="Si").to_dict()
    assert "rotation" not in Feature("cell", "C", material="Si").params     # the lattice cell is never rotated


def test_array_of_features(tmp_path):
    """An array is one object: 3 x 2 holes through the slab remove six holes' volume and the copies of a touching
    pillar array merge into one feature region; only arrays store `array` / `spacing`; bad counts are reported."""
    from solidocean.fields import Quadrature
    hole = Feature("cylinder", "Holes", op="subtract",
                   params={"center": ["-a/3", "-a/4", "-a/2"], "radius": "0.1*a", "height": "a",
                           "array": ["3", "2", "1"], "spacing": ["a/3", "a/2", "0"]})
    p = slab(hole)
    a = 1e-6
    v, lo, hi = _volume_box(p)
    assert a ** 3 - v == pytest.approx(6 * np.pi * (0.1 * a) ** 2 * a, rel=0.03)
    pillars = Feature("block", "Pillars", material="Si",
                      params={"center": ["-a/2 + a/8", "0", "a/2 + a/8"], "size": ["a/4", "a/4", "a/4"],
                              "array": ["4", "1", "1"], "spacing": ["a/4", "0", "0"]})
    p = slab(pillars)
    m = meshing.generate(p, p.evaluator())
    q = Quadrature.of(m)
    assert q.volume == pytest.approx(a ** 3 + 4 * (a / 4) ** 3, rel=1e-9)
    assert m.region("Pillars").sum() > 0
    np.testing.assert_allclose(m.bounds()[1], [a / 2, a / 2, 3 * a / 4], rtol=1e-9)
    r = Project.load(p.save(tmp_path / "m.toml"))
    assert r.objects[1].params["array"] == ["4", "1", "1"] and r.objects[1].array_label == "4×1×1"
    assert "array" not in r.objects[0].to_dict() and "spacing" not in r.objects[0].to_dict()
    r.objects[1].params["array"] = ["2.5", "1", "1"]
    assert any("whole numbers" in e for e in r.validate_geometry())
