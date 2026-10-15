"""Reductions over part of the outer boundary: mean(f, "Film:zmax"), integral(f, "all") with the outward
normal nx, ny, nz, checked against the exact fields of a heat-conducting bar and a bar in tension."""
import numpy as np
import pytest

from solidocean import fields
from solidocean.expressions import ExpressionError
from solidocean.physics.heat.model import HeatInterface
from solidocean.physics.heat.solver import HeatSolver, compute_quantity
from solidocean.physics.solid.model import SolidCondition
from solidocean.physics.solid.solver import Solver

from test_heat import K1, bar, hot_cold
import test_static

W, L = 0.2e-6, 1e-6
Q = K1 * 100 / L                       # heat flux along +x of the bar of test_heat (W/m²)


def heat(quantities, order=2, **kw):
    return HeatSolver(bar(hot_cold(), order=order, **kw), log=lambda s: None).run(list(quantities))


@pytest.mark.parametrize("order", [1, 2])
def test_face_values_areas_and_the_heat_through_them(order):
    out_x, out_all = "integral(qx*nx + qy*ny + qz*nz, \"xmax\")", "integral(qx*nx + qy*ny + qz*nz, \"all\")"
    qs = ['mean(T, "xmin")', 'mean(T, "A:xmax")', 'integral(1, "ymax")', 'integral(1, "all")',
          out_x, 'integral(qx*nx, "xmin")', out_all, 'max(T, "A:ymin")', 'mean(T, "A")']
    q = {k: v[0, 0] for k, v in heat(qs, order).quantities.items()}
    assert q['mean(T, "xmin")'] == pytest.approx(400, rel=1e-9)
    assert q['mean(T, "A:xmax")'] == pytest.approx(300, rel=1e-9)
    assert q['integral(1, "ymax")'] == pytest.approx(L * W, rel=1e-9)
    assert q['integral(1, "all")'] == pytest.approx(2 * W * W + 4 * L * W, rel=1e-9)
    assert q[out_x] == pytest.approx(Q * W * W, rel=1e-9)              # leaves through the cold end
    assert q['integral(qx*nx, "xmin")'] == pytest.approx(-Q * W * W, rel=1e-9)   # enters at the hot end
    assert abs(q[out_all]) < 1e-9 * Q * W * W                          # nothing is stored
    assert 399 < q['max(T, "A:ymin")'] <= 400          # at the quadrature point nearest the hot end
    assert q['mean(T, "A")'] == pytest.approx(350, rel=1e-9)           # an object is still its volume


def test_the_normal_points_out_of_a_closed_surface():
    """Divergence theorem on the whole boundary: the integral of x nx is the volume, of nx nothing."""
    q = {k: v[0, 0] for k, v in heat(['integral(x*nx, "all")', 'integral(nx, "all")',
                                      'integral(z*nz, "A:all")']).quantities.items()}
    assert q['integral(x*nx, "all")'] == pytest.approx(L * W * W, rel=1e-9)
    assert q['integral(z*nz, "A:all")'] == pytest.approx(L * W * W, rel=1e-9)
    assert abs(q['integral(nx, "all")']) < 1e-9 * W * W


def test_a_thermal_contact_is_not_outer_boundary():
    """The nodes split at a contact leave the outer surface of each object without its interface."""
    p = bar(hot_cold(), two_materials=True)
    p.heat.interfaces = [HeatInterface("resistance", ["A", "B"], "5e-9")]
    res = HeatSolver(p, log=lambda s: None).run(['integral(1, "A:all")', 'integral(1, "all")'])
    assert res.quantities['integral(1, "A:all")'][0, 0] == pytest.approx(W * W + 4 * L / 2 * W, rel=1e-9)
    assert res.quantities['integral(1, "all")'][0, 0] == pytest.approx(2 * W * W + 4 * L * W, rel=1e-9)
    with pytest.raises(ExpressionError, match="selects no outer boundary"):
        compute_quantity(res, p, 'mean(T, "A:xmax")')


def test_stress_on_a_face_gives_the_force():
    """A bar stretched by 2 nm: the traction integrated over its loaded end is E strain times the area."""
    p = test_static.bar([SolidCondition("displacement", ["xmax"], displacement=["2[nm]", "", ""])])
    r = Solver(p, log=lambda m: None).run(['mean(ux, "xmax")', 'integral(sxx*nx, "Bar:xmax")'])
    w = 0.5e-6
    assert r.quantities['mean(ux, "xmax")'][0, 0] == pytest.approx(2e-9, rel=1e-9)
    assert r.quantities['integral(sxx*nx, "Bar:xmax")'][0, 0] == pytest.approx(test_static.E * 1e-3 * w * w,
                                                                               rel=1e-9)


def test_names_and_errors():
    """Normals exist only on a surface; a bad face is reported; check() knows the normals."""
    fields.check('integral(qx*nx, "Film:zmax")', {}, ("qx",), ())
    res = heat([])
    p = bar(hot_cold())
    with pytest.raises(ExpressionError, match="exists only on a surface"):
        compute_quantity(res, p, "mean(nx)")
    with pytest.raises(ExpressionError, match="bad face"):
        compute_quantity(res, p, 'mean(T, "A:top")')
    with pytest.raises(ExpressionError, match="unknown object"):
        compute_quantity(res, p, 'mean(T, "Nope:zmax")')
    np.testing.assert_allclose(compute_quantity(res, p, 'mean(T, "xmax")'), 300, rtol=1e-9)
