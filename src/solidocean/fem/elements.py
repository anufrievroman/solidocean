"""Tetrahedral Lagrange elements (gmsh node ordering) and quadrature."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import roots_jacobi

# gmsh tet10 mid-edge nodes 4..9 sit on these vertex pairs
TET10_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]

# d(barycentric L0..L3)/d(xi, eta, zeta)
_DL = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def barycentric(xi: np.ndarray) -> np.ndarray:
    """(Q, 3) reference coords -> (Q, 4) barycentric coords."""
    return np.column_stack([1.0 - xi.sum(axis=1), xi])


def shape_functions(order: int, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape values (Q, nn) and reference gradients (Q, nn, 3) at points xi (Q, 3)."""
    L = barycentric(xi)
    Q = len(xi)
    if order == 1:
        return L, np.broadcast_to(_DL, (Q, 4, 3)).copy()
    if order != 2:
        raise ValueError("order must be 1 or 2")
    N = np.empty((Q, 10))
    dN = np.empty((Q, 10, 3))
    for a in range(4):
        N[:, a] = L[:, a] * (2 * L[:, a] - 1)
        dN[:, a] = (4 * L[:, a] - 1)[:, None] * _DL[a]
    for m, (a, b) in enumerate(TET10_EDGES):
        N[:, 4 + m] = 4 * L[:, a] * L[:, b]
        dN[:, 4 + m] = 4 * (L[:, a, None] * _DL[b] + L[:, b, None] * _DL[a])
    return N, dN


@lru_cache(maxsize=8)
def quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Conical-product (Stroud) rule with n^3 points, exact to degree 2n-1.

    Returns points (Q, 3) on the reference tet and weights (Q,) summing to 1/6.
    """
    pts, wts = [], []
    for alpha in (2, 1, 0):
        t, w = roots_jacobi(n, alpha, 0)
        pts.append((t + 1) / 2)
        wts.append(w / 2 ** (alpha + 1))
    u, v, w = np.meshgrid(*pts, indexing="ij")
    wu, wv, ww = np.meshgrid(*wts, indexing="ij")
    x = u
    y = v * (1 - u)
    z = w * (1 - u) * (1 - v)
    P = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    W = (wu * wv * ww).ravel()
    return P, W


def default_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    # mass matrix integrand has degree 2*order; allow curved (isoparametric) P2 cells
    return quadrature(2 if order == 1 else 3)
