import numpy as np
import pytest

from solidocean.expressions import Evaluator, ExpressionError


def test_units_and_dependencies():
    ev = Evaluator({"r": "0.3*a", "a": "500[nm]", "rho": "2.329[g/cm^3]", "k": "pi/a"})
    assert ev.values["a"] == pytest.approx(5e-7)
    assert ev.values["r"] == pytest.approx(1.5e-7)
    assert ev.values["rho"] == pytest.approx(2329)
    assert ev("2*a + 10[nm]") == pytest.approx(1.01e-6)
    assert ev("if(a > 1[um], 1, 2)") == 2


def test_cycle_and_unknown():
    with pytest.raises(ExpressionError, match="circular"):
        Evaluator({"a": "b", "b": "a"})
    with pytest.raises(ExpressionError, match="unknown"):
        Evaluator({"a": "c"})
    with pytest.raises(ExpressionError):
        Evaluator({"a": "__import__('os')"})


def test_overrides_and_arrays():
    ev = Evaluator({"a": "1", "b": "2*a"}).with_overrides({"a": 3})
    assert ev.values["b"] == 6
    np.testing.assert_allclose(ev.array("range(0, 0.25, 1)"), [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(ev.array("linspace(0, b, 3)"), [0, 3, 6])
    np.testing.assert_allclose(ev.array("[1, a, b]"), [1, 3, 6])


def test_units_bind_to_the_value_before_them():
    import pytest
    ev = Evaluator({"x": "3"})
    assert ev("x / 2[um]") == pytest.approx(1.5e6)            # x / (2 um), not x / 2 * 1e-6
    assert ev("2[um]^2") == pytest.approx(4e-12)
    assert ev("-2[um]^2") == pytest.approx(-4e-12)
    assert ev("10[GPa] / 2[GPa]") == pytest.approx(5)
    assert ev("1e-9[m]") == pytest.approx(1e-9)
    assert ev("sqrt(4)[um]") == pytest.approx(2e-6)


def test_resolve_each_reports_only_the_broken_variables():
    from solidocean.expressions import resolve_each
    values, errors = resolve_each({"a": "500[nm]", "t": "0.4*a", "r": "1/(", "R": "2*r", "k": "pi/a",
                                   "u": "b + 1", "p": "q", "q": "p"})
    assert set(values) == {"a", "t", "k"}
    assert set(errors) == {"r", "R", "u", "p", "q"}
    assert "depends on 'r'" in errors["R"] and "unknown name 'b'" in errors["u"]
    assert "circular" in errors["p"] or "circular" in errors["q"]
