"""Hydrodynamic phonon heat transport (Guyer-Krumhansl), steady state, on the same tetrahedral meshes.

The heat flux q is an unknown next to the temperature T:

    q / k + grad T = a (lap q + 2 grad div q),      div q = Q,       a = l^2 / k

with the viscous length l (the 2D/3D Guyer-Krumhansl equation with the drag q/tau_R and viscosity eta
folded into l^2 = eta tau_R; for l -> 0 it is Fourier's law q = -k grad T). Like Stokes flow (q: velocity,
T: pressure) it needs a stable pair of elements: quadratic q on all the nodes of the mesh, linear T on its
vertices (Taylor-Hood).

Weak form, with test functions w (vector, quadratic) and v (linear), T integrated by parts:
    int q.w / k + a int (grad q : grad w + 2 div q div w) - int T div w = - int_S T_D w.n
    int v div q = int v Q
The surface term with the prescribed temperature T_D is the natural condition of a heat bath (q free there);
a prescribed heat flux and a no-slip wall are conditions on q (essential), see physics/heat/hydro.py.

Unknowns: [q_x (N), q_y (N), q_z (N), T on the vertices (Nv)].
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .elements import TET10_EDGES, default_quadrature, shape_functions

CHUNK = 4096


def vertices(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vertex nodes (Nv,) sorted, position of every node among them (N,): -1 for mid-edge nodes)."""
    vert = np.unique(cells[:, :4])
    pos = np.full(int(cells.max()) + 1, -1, dtype=np.int64)
    pos[vert] = np.arange(len(vert))
    return vert, pos


def assemble(points, cells, cell_domain, k_list, a_list):
    """A (3N x 3N) and G (3N x Nv) of the weak form above, for the conductivity k and the coefficient
    a = l^2 / k of every domain (isotropic, constant), on quadratic elements."""
    if cells.shape[1] != 10:
        raise ValueError("hydrodynamic heat transport needs quadratic elements (mesh order 2)")
    qp, qw = default_quadrature(2)
    N, dN = shape_functions(2, qp)
    L = np.column_stack([1.0 - qp.sum(axis=1), qp])                 # linear shape functions of the vertices
    n = len(points)
    vert, pos = vertices(cells)
    nv = len(vert)
    kinv = 1.0 / np.asarray(k_list, float)
    a = np.asarray(a_list, float)
    rows_a, cols_a, vals_a, rows_g, cols_g, vals_g = [], [], [], [], [], []
    for start in range(0, len(cells), CHUNK):
        sl = slice(start, min(start + CHUNK, len(cells)))
        c = cells[sl]
        X = points[c]
        dom = cell_domain[sl]
        Ae = np.zeros((len(c), 3, 10, 3, 10))
        Ge = np.zeros((len(c), 3, 10, 4))
        for q in range(len(qw)):
            J = np.einsum("eai,aj->eij", X, dN[q])
            w = qw[q] * np.abs(np.linalg.det(J))
            G = np.einsum("aj,eji->eai", dN[q], np.linalg.inv(J))      # (e, 10, 3) physical gradients
            mass = (w * kinv[dom])[:, None, None] * np.outer(N[q], N[q])[None]
            lap = (w * a[dom])[:, None, None] * np.einsum("eai,ebi->eab", G, G)
            for comp in range(3):
                Ae[:, comp, :, comp, :] += mass + lap
            # row (test component i, node a), column (component j, node b): 2 a int d_i w_i d_j q_j
            Ae += (2 * w * a[dom])[:, None, None, None, None] * np.einsum("eai,ebj->eiajb", G, G)
            Ge -= w[:, None, None, None] * np.einsum("eac,j->ecaj", G, L[q])
        ra = (np.arange(3)[None, :, None] * n + c[:, None, :]).reshape(len(c), 30)
        rows_a.append(np.repeat(ra, 30, axis=1).ravel())
        cols_a.append(np.tile(ra, (1, 30)).ravel())
        vals_a.append(Ae.reshape(len(c), 30, 30).ravel())
        cg = pos[c[:, :4]]
        rows_g.append(np.repeat(ra, 4, axis=1).ravel())
        cols_g.append(np.tile(cg, (1, 30)).ravel())
        vals_g.append(Ge.reshape(len(c), 30, 4).ravel())
    A = sp.csr_matrix((np.concatenate(vals_a), (np.concatenate(rows_a), np.concatenate(cols_a))), shape=(3 * n, 3 * n))
    Gm = sp.csr_matrix((np.concatenate(vals_g), (np.concatenate(rows_g), np.concatenate(cols_g))), shape=(3 * n, nv))
    return (0.5 * (A + A.T)).tocsr(), Gm


def source_vector(points, cells, cell_domain, q_list, profile=None) -> np.ndarray:
    """F_i = int Q M_i over the domains with a volumetric heat source, M_i the linear functions of the vertices
    (Nv,); profile: optional factor of the position, a function of the quadrature points (e, 3) -> (e,)."""
    qp, qw = default_quadrature(2)
    N, dN = shape_functions(2, qp)
    L = np.column_stack([1.0 - qp.sum(axis=1), qp])
    vert, pos = vertices(cells)
    F = np.zeros(len(vert))
    for d, Q in enumerate(q_list):
        if not Q:
            continue
        c = cells[cell_domain == d]
        X = points[c]
        fe = np.zeros((len(c), 4))
        for q in range(len(qw)):
            J = np.einsum("eai,aj->eij", X, dN[q])
            w = qw[q] * np.abs(np.linalg.det(J)) * Q
            if profile is not None:
                w = w * profile(np.einsum("a,eai->ei", N[q], X))
            fe += w[:, None] * L[q][None]
        np.add.at(F, pos[c[:, :4]], fe)
    return F


def embed_temperature(cells: np.ndarray, n_nodes: int, T_vertices: np.ndarray) -> np.ndarray:
    """The linear temperature of the vertices as a quadratic field (the mid-edge nodes take the mean of their
    two vertices): nodal values (N,), exact for the linear function."""
    vert, pos = vertices(cells)
    T = np.zeros(n_nodes, dtype=T_vertices.dtype)
    T[vert] = T_vertices
    for m, (a, b) in enumerate(TET10_EDGES):
        T[cells[:, 4 + m]] = 0.5 * (T[cells[:, a]] + T[cells[:, b]])
    return T


def solve(A, G, f_q, f_T, fixed: np.ndarray, values: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Solve [[A, G], [G^T, 0]] [q; T] = [f_q; f_T] with x[fixed] = values (indices into [q; T]). The system is
    scaled first (A and G differ by many orders of magnitude in SI units at small sizes) and solved directly."""
    from .eigen import _factorize
    n, nv = A.shape[0], G.shape[1]
    K = sp.bmat([[A, G], [G.T, None]], format="csr")
    b = np.concatenate([f_q, f_T])
    sq = 1.0 / np.sqrt(np.mean(np.abs(A.diagonal())))
    gmean = float(np.mean(np.abs(G.data))) if G.nnz else 1.0
    sT = 1.0 / (sq * gmean)
    d = np.concatenate([np.full(n, sq), np.full(nv, sT)])
    D = sp.diags(d)
    Ks, bs = (D @ K @ D).tocsr(), d * b
    x = np.zeros(n + nv)
    x[fixed] = values / d[fixed]
    free = np.setdiff1d(np.arange(n + nv), fixed)
    rhs = bs[free] - Ks[free][:, fixed] @ x[fixed]
    x[free] = _factorize(Ks[free][:, free].tocsc(), backend, permc="COLAMD")(rhs)
    return d * x
