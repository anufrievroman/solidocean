"""Boundary surfaces of tetrahedral meshes: outer triangles, their selection, and surface integrals;
interior faces and node splitting for interfaces with a jump of the field (thermal contact).

Shared by heat transfer (fluxes, convection) and solid mechanics (boundary loads, constraints).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.special import roots_jacobi

from .elements import TET10_EDGES

AXES = "xyz"
ALL = "all"   # selection keyword: the whole outer surface (of the model or of one object)

# tet faces (vertex triples, outward orientation irrelevant here) and their mid-edge nodes (tet10)
_FACES = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
_EDGE = {tuple(sorted(e)): 4 + n for n, e in enumerate(TET10_EDGES)}


def _face_locals(order: int) -> np.ndarray:
    """(4, 3|6) local node positions of the tet faces, gmsh triangle order (vertices, then edges 01 12 20)."""
    out = []
    for a, b, c in _FACES:
        f = [a, b, c]
        if order == 2:
            f += [_EDGE[tuple(sorted((a, b)))], _EDGE[tuple(sorted((b, c)))], _EDGE[tuple(sorted((a, c)))]]
        out.append(f)
    return np.array(out)


def interior_faces(cells: np.ndarray, order: int):
    """Faces shared by two tetrahedra: (tris (F, 3|6) as seen from the first, e1 (F,), e2 (F,), local face
    of each (F,), (F,))."""
    loc = _face_locals(order)
    E = len(cells)
    key = np.sort(cells[:, loc[:, :3]].reshape(-1, 3), axis=1)          # (4E, 3), index e*4 + face
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order_ = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
    a, b = order_[starts], order_[starts + 1]
    e1, f1, e2, f2 = a // 4, a % 4, b // 4, b % 4
    return cells[e1[:, None], loc[f1]], e1, e2, f1, f2


def _slots(cells: np.ndarray, elements: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Local positions of `nodes` (F, k) in the cells `elements` (F,)."""
    return np.argmax(cells[elements][:, None, :] == nodes[:, :, None], axis=2)


def renumber_faces(tris, owner, old_cells, new_cells) -> np.ndarray:
    """Triangles of the cells `owner` after a renumbering of their nodes (old_cells -> new_cells)."""
    return new_cells[owner[:, None], _slots(old_cells, owner, tris)]


def split_nodes(cells: np.ndarray, order: int, cut: np.ndarray, faces=None):
    """Duplicate the nodes of the interior faces `cut` (a mask over interior_faces) so the field may jump across
    them: around each node, the cells connected through uncut faces keep one copy. Returns (new cells, source
    node of every new node (M,), (tri1, tri2) of the cut faces: the same nodes in the same order, as seen from
    the first and second cell)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    tris, e1, e2, f1, f2 = faces if faces is not None else interior_faces(cells, order)
    loc = _face_locals(order)
    E, nn = cells.shape
    nodes = np.unique(tris[cut])
    on = np.zeros(int(cells.max()) + 1, dtype=bool)
    on[nodes] = True
    # graph of (cell, local node) incidences of interface nodes, joined across uncut faces
    keep = ~cut
    t, a, b = tris[keep], e1[keep], e2[keep]
    sa = loc[f1[keep]]
    sb = _slots(cells, b, t)
    pick = on[t]
    pa = (a[:, None] * nn + sa)[pick]
    pb = (b[:, None] * nn + sb)[pick]
    graph = coo_matrix((np.ones(len(pa)), (pa, pb)), shape=(E * nn, E * nn))
    _, label = connected_components(graph, directed=False)
    pos = np.flatnonzero(on[cells.ravel()])
    node = cells.ravel()[pos]
    pairs, which = np.unique(np.c_[node, label[pos]], axis=0, return_inverse=True)
    which = which.ravel()
    first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]           # the first copy of each node keeps its index
    n = int(cells.max()) + 1
    new_id = np.where(first, pairs[:, 0], n + np.cumsum(~first) - 1)
    out = cells.ravel().copy()
    out[pos] = new_id[which]
    out = out.reshape(E, nn)
    t1 = out[e1[cut][:, None], loc[f1[cut]]]
    t2 = out[e2[cut][:, None], _slots(cells, e2[cut], tris[cut])]
    return out, pairs[~first, 0], (t1, t2)


def boundary_faces(cells: np.ndarray, order: int, owners: bool = False):
    """Triangles (F, 3|6) on the outer boundary, gmsh triangle node order (vertices, then edges 01 12 20).
    With owners=True also the index of the tetrahedron each triangle belongs to."""
    tris = []
    for a, b, c in _FACES:
        f = [cells[:, a], cells[:, b], cells[:, c]]
        if order == 2:
            f += [cells[:, _EDGE[tuple(sorted((a, b)))]], cells[:, _EDGE[tuple(sorted((b, c)))]],
                  cells[:, _EDGE[tuple(sorted((a, c)))]]]
        tris.append(np.column_stack(f))
    tris = np.vstack(tris)
    key = np.sort(tris[:, :3], axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    idx = idx[counts == 1]  # faces belonging to one tet only
    return (tris[idx], idx % len(cells)) if owners else tris[idx]


def faces_on(points: np.ndarray, tris: np.ndarray, face: str, tol: float, box=None) -> np.ndarray:
    """Boundary triangles lying on a bounding-box face such as 'xmin' (of `box` = (lo, hi) if given,
    else of all points)."""
    i = AXES.index(face[0])
    if box is not None:
        plane = box[0][i] if face.endswith("min") else box[1][i]
    else:
        plane = points[:, i].min() if face.endswith("min") else points[:, i].max()
    on = np.all(np.abs(points[tris[:, :3], i] - plane) < tol, axis=1)
    return tris[on]


@lru_cache(maxsize=4)
def _tri_rule(n: int):
    """Collapsed Gauss rule on the reference triangle (exact to degree 2n-1); weights sum to 1/2."""
    t1, w1 = roots_jacobi(n, 1, 0)
    t0, w0 = roots_jacobi(n, 0, 0)
    u, v = (t1 + 1) / 2, (t0 + 1) / 2
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.outer(w1 / 4, w0 / 2)
    return np.column_stack([U.ravel(), (V * (1 - U)).ravel()]), W.ravel()


def _tri_shapes(order: int, pts: np.ndarray):
    L = np.column_stack([1 - pts.sum(axis=1), pts])
    dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    if order == 1:
        return L, np.broadcast_to(dL, (len(pts), 3, 2))
    N = np.empty((len(pts), 6))
    dN = np.empty((len(pts), 6, 2))
    for a in range(3):
        N[:, a] = L[:, a] * (2 * L[:, a] - 1)
        dN[:, a] = (4 * L[:, a] - 1)[:, None] * dL[a]
    for m, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)]):
        N[:, 3 + m] = 4 * L[:, a] * L[:, b]
        dN[:, 3 + m] = 4 * (L[:, a, None] * dL[b] + L[:, b, None] * dL[a])
    return N, dN


def surface_at_points(points, tris, order, nodal: np.ndarray) -> np.ndarray:
    """Nodal values interpolated to the quadrature points of surface_terms: (F, Q)."""
    qp, _ = _tri_rule(3)
    N, _ = _tri_shapes(order, qp)
    return nodal[tris] @ np.asarray(N).T


def surface_terms(points, tris, order, coeff=0.0, load=0.0, n_nodes: int | None = None):
    """Surface integrals over triangles: matrix int coeff N N and vector int load N. coeff and load are
    numbers or values at the quadrature points (F, Q), see surface_at_points."""
    n = n_nodes or len(points)
    if len(tris) == 0:
        return sp.csr_matrix((n, n)), np.zeros(n)
    Me, fe = _surface_elements(points, tris, order, coeff, load)
    nt = tris.shape[1]
    rows = np.repeat(tris, nt, axis=1).ravel()
    cols = np.tile(tris, (1, nt)).ravel()
    M = sp.csr_matrix((Me.ravel(), (rows, cols)), shape=(n, n))
    f = np.zeros(n)
    np.add.at(f, tris, fe)
    return M, f


def interface_terms(points, tri1, tri2, order, coeff, n_nodes: int | None = None):
    """Matrix of int coeff (T1 - T2)(v1 - v2) over interface triangles with the nodes tri1 on one side and tri2
    on the other (same positions); coeff a number or values at the quadrature points (F, Q)."""
    n = n_nodes or len(points)
    if len(tri1) == 0:
        return sp.csr_matrix((n, n))
    Me, _ = _surface_elements(points, tri1, order, coeff, 0.0)
    both = np.hstack([tri1, tri2])
    Mb = np.block([[Me, -Me], [-Me, Me]])
    nt = both.shape[1]
    rows = np.repeat(both, nt, axis=1).ravel()
    cols = np.tile(both, (1, nt)).ravel()
    return sp.csr_matrix((Mb.ravel(), (rows, cols)), shape=(n, n))


def face_normals(points, tris) -> np.ndarray:
    """(F, 3) unit normals of the triangles (from their corners; orientation arbitrary)."""
    X = points[tris[:, :3]]
    nrm = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def _surface_elements(points, tris, order, coeff, load):
    """Element matrices int coeff N N (F, nt, nt) and vectors int load N (F, nt)."""
    qp, qw = _tri_rule(3)
    N, dN = _tri_shapes(order, qp)
    X = points[tris]                                            # (F, nt, 3)
    nt = tris.shape[1]
    Me = np.zeros((len(tris), nt, nt))
    fe = np.zeros((len(tris), nt))
    for q in range(len(qw)):
        t = np.einsum("fai,aj->fji", X, dN[q])                  # tangents (F, 2, 3)
        w = qw[q] * np.linalg.norm(np.cross(t[:, 0], t[:, 1]), axis=1)
        cq = coeff[:, q] if np.ndim(coeff) == 2 else coeff
        lq = load[:, q] if np.ndim(load) == 2 else load
        Me += (w * cq)[:, None, None] * np.outer(N[q], N[q])[None]
        fe += (w * lq)[:, None] * N[q][None]
    return Me, fe




def select(points, tris, faces, tol: float, box=None, within=None) -> np.ndarray:
    """Boolean mask of the boundary triangles in a selection.

    faces: bounding-box faces ('xmin', ...) of `box` (lo, hi; default: of all points), or 'all' for every
    triangle; within: optional (lo, hi) box that the triangles' corners must lie in (a box selection)."""
    mask = np.zeros(len(tris), dtype=bool)
    if ALL in faces:
        mask[:] = True
    else:
        for face in faces:
            i = AXES.index(face[0])
            if box is not None:
                plane = box[0][i] if face.endswith("min") else box[1][i]
            else:
                plane = points[:, i].min() if face.endswith("min") else points[:, i].max()
            mask |= np.all(np.abs(points[tris[:, :3], i] - plane) < tol, axis=1)
    if within is not None:
        lo, hi = np.minimum(within[0], within[1]), np.maximum(within[0], within[1])
        corners = points[tris[:, :3]]
        mask &= np.all((corners >= lo - tol) & (corners <= hi + tol), axis=(1, 2))
    return mask


def normal_terms(points, tris, owner_cells, order, n_nodes: int | None = None) -> np.ndarray:
    """(n_nodes, 3) integrals int N_a n dA over triangles with the outward unit normal n (pointing away from
    the tetrahedron each triangle belongs to, `owner_cells` (F, 4+) node indices). A pressure p gives the
    nodal forces -p times this."""
    n = n_nodes or len(points)
    out = np.zeros((n, 3))
    if len(tris) == 0:
        return out
    qp, qw = _tri_rule(3)
    N, dN = _tri_shapes(order, qp)
    X = points[tris]                                            # (F, nt, 3)
    inward = points[owner_cells[:, :4]].mean(axis=1) - X[:, :3].mean(axis=1)
    fe = np.zeros((len(tris), tris.shape[1], 3))
    for q in range(len(qw)):
        t = np.einsum("fai,aj->fji", X, dN[q])
        a = np.cross(t[:, 0], t[:, 1])                          # normal scaled by the area element
        a *= np.where(np.einsum("fi,fi->f", a, inward) > 0, -1.0, 1.0)[:, None]
        fe += qw[q] * N[q][None, :, None] * a[:, None, :]
    np.add.at(out, tris, fe)
    return out
