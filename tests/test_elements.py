import numpy as np
import pytest

from solidocean.fem.elements import quadrature, shape_functions


@pytest.mark.parametrize("order", [1, 2])
def test_partition_of_unity(order):
    xi = np.random.default_rng(0).dirichlet(np.ones(4), 20)[:, 1:]
    N, dN = shape_functions(order, xi)
    np.testing.assert_allclose(N.sum(axis=1), 1)
    np.testing.assert_allclose(dN.sum(axis=1), 0, atol=1e-12)


def test_p2_matches_gmsh_ordering_and_reproduces_quadratics():
    import gmsh
    gmsh.initialize(interruptible=False)
    try:
        nodes = np.array(gmsh.model.mesh.getElementProperties(11)[4]).reshape(10, 3)
    finally:
        gmsh.finalize()
    N, _ = shape_functions(2, nodes)
    np.testing.assert_allclose(N, np.eye(10), atol=1e-12)  # Kronecker property at gmsh nodes
    f = lambda p: 1 + p[:, 0] - 2 * p[:, 1] * p[:, 2] + p[:, 0] ** 2  # noqa: E731
    xi = np.random.default_rng(1).dirichlet(np.ones(4), 10)[:, 1:]
    N, _ = shape_functions(2, xi)
    np.testing.assert_allclose(N @ f(nodes), f(xi))


def test_quadrature_exact():
    P, W = quadrature(3)
    assert W.sum() == pytest.approx(1 / 6)
    # integral of x^2 y z over the unit tet = 2! 1! 1! / 7! = 2/5040
    assert np.sum(W * P[:, 0] ** 2 * P[:, 1] * P[:, 2]) == pytest.approx(2 / 5040)
