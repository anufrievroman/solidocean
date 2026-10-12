"""Vectorised assembly of the global stiffness and mass matrices.

DOF numbering is node-major: dof = 3 * node + component.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .elements import default_quadrature, shape_functions

CHUNK = 4096  # elements per vectorised batch (bounds peak memory)


def _strain_operator(G: np.ndarray) -> np.ndarray:
    """B matrices (E, 6, 3nn) from physical gradients G (E, nn, 3). Voigt: xx yy zz yz xz xy."""
    E, nn, _ = G.shape
    B = np.zeros((E, 6, 3 * nn))
    gx, gy, gz = G[..., 0], G[..., 1], G[..., 2]
    B[:, 0, 0::3] = gx
    B[:, 1, 1::3] = gy
    B[:, 2, 2::3] = gz
    B[:, 3, 1::3] = gz
    B[:, 3, 2::3] = gy
    B[:, 4, 0::3] = gz
    B[:, 4, 2::3] = gx
    B[:, 5, 0::3] = gy
    B[:, 5, 1::3] = gx
    return B


def element_matrices(X: np.ndarray, C: np.ndarray, rho: float, order: int):
    """Element stiffness (E, 3nn, 3nn) and scalar mass (E, nn, nn) for coords X (E, nn, 3)."""
    qp, qw = default_quadrature(order)
    N, dN = shape_functions(order, qp)
    E, nn, _ = X.shape
    Ke = np.zeros((E, 3 * nn, 3 * nn))
    Me = np.zeros((E, nn, nn))
    for q in range(len(qw)):
        J = np.einsum("eai,aj->eij", X, dN[q])
        det = np.linalg.det(J)
        if np.any(det == 0):
            raise ValueError("degenerate element (zero Jacobian)")
        invJ = np.linalg.inv(J)
        G = np.einsum("aj,eji->eai", dN[q], invJ)
        B = _strain_operator(G)
        wdet = qw[q] * np.abs(det)
        Ke += wdet[:, None, None] * np.einsum("eki,kl,elj->eij", B, C, B, optimize=True)
        Me += (wdet * rho)[:, None, None] * np.outer(N[q], N[q])[None]
    return Ke, Me


def assemble_weighted(points, cells, cell_domain, C_list, weights, order):
    """Stiffness sum_e w_e K_e (CSR) with a weight per element, e.g. loss factors: i * this is the damping part
    of a complex (hysteretic) stiffness. Elements of zero weight are skipped."""
    n_nodes = len(points)
    nn = cells.shape[1]
    K = sp.csr_matrix((3 * n_nodes, 3 * n_nodes))
    comp = np.arange(3)
    for d, C in enumerate(C_list):
        sel = np.flatnonzero((cell_domain == d) & (weights != 0))
        for start in range(0, len(sel), CHUNK):
            e = sel[start:start + CHUNK]
            c = cells[e]
            Ke, _ = element_matrices(points[c], C, 0.0, order)
            Ke *= weights[e][:, None, None]
            dofs = (3 * c[:, :, None] + comp).reshape(len(c), 3 * nn)
            rows = np.repeat(dofs, 3 * nn, axis=1).ravel()
            cols = np.tile(dofs, (1, 3 * nn)).ravel()
            K = K + sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=K.shape)
    return (0.5 * (K + K.T)).tocsr()


def assemble(points: np.ndarray, cells: np.ndarray, cell_domain: np.ndarray,
             C_list: list[np.ndarray], rho_list: list[float], order: int):
    """Global stiffness K and mass M (scipy CSR, real symmetric)."""
    n_nodes = len(points)
    nn = cells.shape[1]
    K = sp.csr_matrix((3 * n_nodes, 3 * n_nodes))
    Ms = sp.csr_matrix((n_nodes, n_nodes))
    comp = np.arange(3)
    for d, (C, rho) in enumerate(zip(C_list, rho_list)):
        dcells = cells[cell_domain == d]
        for start in range(0, len(dcells), CHUNK):
            c = dcells[start:start + CHUNK]
            Ke, Me = element_matrices(points[c], C, rho, order)
            dofs = (3 * c[:, :, None] + comp).reshape(len(c), 3 * nn)
            rows = np.repeat(dofs, 3 * nn, axis=1).ravel()
            cols = np.tile(dofs, (1, 3 * nn)).ravel()
            K = K + sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=K.shape)
            rows = np.repeat(c, nn, axis=1).ravel()
            cols = np.tile(c, (1, nn)).ravel()
            Ms = Ms + sp.csr_matrix((Me.ravel(), (rows, cols)), shape=Ms.shape)
    M = sp.kron(Ms, sp.identity(3), format="csr")
    K = 0.5 * (K + K.T)  # remove round-off asymmetry
    return K.tocsr(), M
